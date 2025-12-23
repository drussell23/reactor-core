"""
Sandbox Executor for Safe Scout.

Provides:
- Dockerized Playwright browser execution
- Network isolation and resource limits
- Timeout enforcement
- Screenshot capture for debugging
- Clean output extraction
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Browser execution mode."""
    DOCKER = "docker"           # Full Docker isolation (recommended)
    SUBPROCESS = "subprocess"   # Local subprocess (fallback)
    DIRECT = "direct"           # Direct Playwright (dev only)


class PageLoadStatus(Enum):
    """Status of page load."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    NAVIGATION_FAILED = "navigation_failed"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class SandboxConfig:
    """Configuration for sandbox executor."""
    # Execution mode
    mode: ExecutionMode = field(
        default_factory=lambda: ExecutionMode(
            os.getenv("NIGHTSHIFT_SANDBOX_MODE", "docker")
        )
    )

    # Docker settings
    docker_image: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_SANDBOX_IMAGE",
            "mcr.microsoft.com/playwright:v1.40.0-jammy"
        )
    )
    docker_network: str = "bridge"  # Use bridge for limited network access
    memory_limit: str = "512m"
    cpu_limit: float = 0.5

    # Browser settings
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 NightShift-Scout/1.0"
        )
    )

    # Timeouts (milliseconds)
    page_load_timeout: int = 30000
    script_timeout: int = 10000
    navigation_timeout: int = 30000

    # Resource blocking
    block_images: bool = True
    block_fonts: bool = True
    block_media: bool = True
    block_stylesheets: bool = False  # Keep for layout

    # JavaScript
    enable_javascript: bool = True
    wait_for_network_idle: bool = True
    network_idle_timeout: int = 5000

    # Output
    capture_screenshots: bool = False
    screenshot_path: Optional[Path] = None

    # Temp directory for Docker mounts
    temp_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "nightshift-sandbox"
    )


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    url: str
    status: PageLoadStatus
    html_content: str = ""
    text_content: str = ""
    title: str = ""
    final_url: str = ""  # After redirects
    http_status: int = 0
    response_headers: Dict[str, str] = field(default_factory=dict)
    load_time_ms: float = 0.0
    error_message: str = ""
    screenshot_path: Optional[Path] = None
    executed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "status": self.status.value,
            "html_content_length": len(self.html_content),
            "text_content_length": len(self.text_content),
            "title": self.title,
            "final_url": self.final_url,
            "http_status": self.http_status,
            "response_headers": self.response_headers,
            "load_time_ms": self.load_time_ms,
            "error_message": self.error_message,
            "screenshot_path": str(self.screenshot_path) if self.screenshot_path else None,
            "executed_at": self.executed_at.isoformat(),
            "metadata": self.metadata,
        }


class SandboxExecutor:
    """
    Executes web page fetching in an isolated sandbox.

    Supports:
    - Docker isolation (recommended for production)
    - Local subprocess fallback
    - Direct Playwright for development
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
    ):
        self.config = config or SandboxConfig()

        # Ensure temp directory exists
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        # Check Docker availability
        self._docker_available: Optional[bool] = None

    async def is_docker_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._docker_available = proc.returncode == 0
        except Exception:
            self._docker_available = False

        if not self._docker_available:
            logger.warning("Docker not available, falling back to subprocess mode")

        return self._docker_available

    async def execute(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        extract_selectors: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """
        Execute page fetch in sandbox.

        Args:
            url: URL to fetch
            wait_for_selector: CSS selector to wait for before extracting
            extract_selectors: Dict of name->selector for targeted extraction

        Returns:
            SandboxResult with page content
        """
        start_time = datetime.now()

        # Determine execution mode
        mode = self.config.mode

        if mode == ExecutionMode.DOCKER:
            if not await self.is_docker_available():
                mode = ExecutionMode.SUBPROCESS

        try:
            if mode == ExecutionMode.DOCKER:
                result = await self._execute_docker(url, wait_for_selector, extract_selectors)
            elif mode == ExecutionMode.SUBPROCESS:
                result = await self._execute_subprocess(url, wait_for_selector, extract_selectors)
            else:
                result = await self._execute_direct(url, wait_for_selector, extract_selectors)

            # Calculate load time
            result.load_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            return result

        except asyncio.TimeoutError:
            return SandboxResult(
                url=url,
                status=PageLoadStatus.TIMEOUT,
                error_message=f"Page load timed out after {self.config.page_load_timeout}ms",
            )
        except Exception as e:
            logger.error(f"Sandbox execution error for {url}: {e}")
            return SandboxResult(
                url=url,
                status=PageLoadStatus.ERROR,
                error_message=str(e),
            )

    async def _execute_docker(
        self,
        url: str,
        wait_for_selector: Optional[str],
        extract_selectors: Optional[Dict[str, str]],
    ) -> SandboxResult:
        """Execute in Docker container."""
        # Create unique run ID
        run_id = hashlib.md5(f"{url}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        work_dir = self.config.temp_dir / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create Playwright script
            script = self._generate_playwright_script(
                url, wait_for_selector, extract_selectors
            )
            script_path = work_dir / "fetch.js"
            script_path.write_text(script)

            # Create output file
            output_path = work_dir / "output.json"

            # Build Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",
                "--network", self.config.docker_network,
                "--memory", self.config.memory_limit,
                f"--cpus={self.config.cpu_limit}",
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "-v", f"{work_dir}:/work:rw",
                "--security-opt", "no-new-privileges",
                self.config.docker_image,
                "node", "/work/fetch.js",
            ]

            # Execute with timeout
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.page_load_timeout / 1000 + 10,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Docker execution failed: {error_msg}")
                return SandboxResult(
                    url=url,
                    status=PageLoadStatus.ERROR,
                    error_message=error_msg,
                )

            # Read output
            if output_path.exists():
                output_data = json.loads(output_path.read_text())
                return self._parse_output(url, output_data)
            else:
                return SandboxResult(
                    url=url,
                    status=PageLoadStatus.ERROR,
                    error_message="No output file generated",
                )

        finally:
            # Cleanup
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _execute_subprocess(
        self,
        url: str,
        wait_for_selector: Optional[str],
        extract_selectors: Optional[Dict[str, str]],
    ) -> SandboxResult:
        """Execute as local subprocess."""
        # Create unique run ID
        run_id = hashlib.md5(f"{url}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        work_dir = self.config.temp_dir / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create Playwright script
            script = self._generate_playwright_script(
                url, wait_for_selector, extract_selectors
            )
            script_path = work_dir / "fetch.js"
            script_path.write_text(script)

            # Execute with Node.js
            proc = await asyncio.create_subprocess_exec(
                "node", str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.page_load_timeout / 1000 + 10,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return SandboxResult(
                    url=url,
                    status=PageLoadStatus.ERROR,
                    error_message=error_msg,
                )

            # Parse stdout as JSON
            try:
                output_data = json.loads(stdout.decode())
                return self._parse_output(url, output_data)
            except json.JSONDecodeError:
                return SandboxResult(
                    url=url,
                    status=PageLoadStatus.ERROR,
                    error_message="Failed to parse output",
                )

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _execute_direct(
        self,
        url: str,
        wait_for_selector: Optional[str],
        extract_selectors: Optional[Dict[str, str]],
    ) -> SandboxResult:
        """Execute directly with Playwright (dev mode only)."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright required. Install with: pip install playwright && playwright install"
            )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.config.headless)

            try:
                context = await browser.new_context(
                    viewport={
                        "width": self.config.viewport_width,
                        "height": self.config.viewport_height,
                    },
                    user_agent=self.config.user_agent,
                )

                page = await context.new_page()

                # Set up resource blocking
                if any([
                    self.config.block_images,
                    self.config.block_fonts,
                    self.config.block_media,
                ]):
                    await page.route("**/*", self._route_handler)

                # Navigate
                response = await page.goto(
                    url,
                    timeout=self.config.navigation_timeout,
                    wait_until="networkidle" if self.config.wait_for_network_idle else "load",
                )

                # Wait for selector if specified
                if wait_for_selector:
                    await page.wait_for_selector(
                        wait_for_selector,
                        timeout=self.config.script_timeout,
                    )

                # Get content
                html = await page.content()
                text = await page.evaluate("() => document.body.innerText")
                title = await page.title()

                # Screenshot if enabled
                screenshot_path = None
                if self.config.capture_screenshots:
                    screenshot_path = self.config.screenshot_path or (
                        self.config.temp_dir / f"screenshot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                    )
                    await page.screenshot(path=str(screenshot_path))

                # Get response info
                http_status = response.status if response else 0
                headers = dict(response.headers) if response else {}

                return SandboxResult(
                    url=url,
                    status=PageLoadStatus.SUCCESS,
                    html_content=html,
                    text_content=text,
                    title=title,
                    final_url=page.url,
                    http_status=http_status,
                    response_headers=headers,
                    screenshot_path=screenshot_path,
                )

            finally:
                await browser.close()

    async def _route_handler(self, route):
        """Handle resource blocking."""
        resource_type = route.request.resource_type

        if self.config.block_images and resource_type == "image":
            await route.abort()
        elif self.config.block_fonts and resource_type == "font":
            await route.abort()
        elif self.config.block_media and resource_type in ["media", "video"]:
            await route.abort()
        elif self.config.block_stylesheets and resource_type == "stylesheet":
            await route.abort()
        else:
            await route.continue_()

    def _generate_playwright_script(
        self,
        url: str,
        wait_for_selector: Optional[str],
        extract_selectors: Optional[Dict[str, str]],
    ) -> str:
        """Generate Node.js Playwright script."""
        script = f'''
const {{ chromium }} = require('playwright');
const fs = require('fs');

(async () => {{
    const browser = await chromium.launch({{
        headless: {str(self.config.headless).lower()},
    }});

    const context = await browser.newContext({{
        viewport: {{
            width: {self.config.viewport_width},
            height: {self.config.viewport_height}
        }},
        userAgent: {json.dumps(self.config.user_agent)},
    }});

    const page = await context.newPage();

    // Block unnecessary resources
    await page.route('**/*', (route) => {{
        const type = route.request().resourceType();
        const blocked = [];
        {'blocked.push("image");' if self.config.block_images else ''}
        {'blocked.push("font");' if self.config.block_fonts else ''}
        {'blocked.push("media", "video");' if self.config.block_media else ''}
        {'blocked.push("stylesheet");' if self.config.block_stylesheets else ''}

        if (blocked.includes(type)) {{
            route.abort();
        }} else {{
            route.continue();
        }}
    }});

    let result = {{
        url: {json.dumps(url)},
        status: 'success',
        html: '',
        text: '',
        title: '',
        finalUrl: '',
        httpStatus: 0,
        headers: {{}},
        error: ''
    }};

    try {{
        const response = await page.goto({json.dumps(url)}, {{
            timeout: {self.config.navigation_timeout},
            waitUntil: {'networkidle' if self.config.wait_for_network_idle else 'load'}
        }});

        {'await page.waitForSelector(' + json.dumps(wait_for_selector) + ', { timeout: ' + str(self.config.script_timeout) + ' });' if wait_for_selector else ''}

        result.html = await page.content();
        result.text = await page.evaluate(() => document.body.innerText);
        result.title = await page.title();
        result.finalUrl = page.url();
        result.httpStatus = response ? response.status() : 0;
        result.headers = response ? await response.allHeaders() : {{}};

    }} catch (err) {{
        result.status = 'error';
        result.error = err.message;
    }}

    await browser.close();

    // Write to file for Docker mode, stdout for subprocess
    const output = JSON.stringify(result);
    try {{
        fs.writeFileSync('/work/output.json', output);
    }} catch {{
        console.log(output);
    }}
}})();
'''
        return script

    def _parse_output(self, url: str, data: Dict[str, Any]) -> SandboxResult:
        """Parse output from sandbox execution."""
        status_map = {
            "success": PageLoadStatus.SUCCESS,
            "timeout": PageLoadStatus.TIMEOUT,
            "error": PageLoadStatus.ERROR,
            "blocked": PageLoadStatus.BLOCKED,
        }

        return SandboxResult(
            url=url,
            status=status_map.get(data.get("status", "error"), PageLoadStatus.ERROR),
            html_content=data.get("html", ""),
            text_content=data.get("text", ""),
            title=data.get("title", ""),
            final_url=data.get("finalUrl", url),
            http_status=data.get("httpStatus", 0),
            response_headers=data.get("headers", {}),
            error_message=data.get("error", ""),
        )

    async def execute_batch(
        self,
        urls: List[str],
        max_concurrent: int = 3,
    ) -> List[SandboxResult]:
        """Execute multiple URLs with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_limit(url: str) -> SandboxResult:
            async with semaphore:
                return await self.execute(url)

        results = await asyncio.gather(
            *[execute_with_limit(url) for url in urls],
            return_exceptions=True,
        )

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SandboxResult(
                    url=urls[i],
                    status=PageLoadStatus.ERROR,
                    error_message=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results


# Convenience exports
__all__ = [
    "SandboxExecutor",
    "SandboxConfig",
    "SandboxResult",
    "ExecutionMode",
    "PageLoadStatus",
]
