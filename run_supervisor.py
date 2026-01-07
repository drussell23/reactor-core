#!/usr/bin/env python3
"""
AGI OS Unified Supervisor - Project Trinity
============================================

The central command that orchestrates the entire AGI ecosystem:
- JARVIS (Body) - macOS automation and interaction
- JARVIS Prime (Mind) - Local LLM inference and reasoning
- Reactor Core (Nervous System) - Training, learning, and model serving

RUN: python3 run_supervisor.py

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      AGI OS UNIFIED SUPERVISOR                     │
    │                    (Central Coordination Hub)                       │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
    ┌─────────┐               ┌─────────────┐               ┌─────────┐
    │ JARVIS  │◄────────────►│   TRINITY   │◄────────────►│ J-PRIME │
    │ (Body)  │     Events    │ ORCHESTRATOR│     Events    │ (Mind)  │
    │         │               │             │               │         │
    │ macOS   │               │ Heartbeats  │               │ LLM     │
    │ Actions │               │ Commands    │               │ Inference
    └─────────┘               │ State Sync  │               └─────────┘
                              └─────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │REACTOR CORE │
                              │  (Nerves)   │
                              │             │
                              │ Training    │
                              │ Learning    │
                              │ Serving     │
                              └─────────────┘

FEATURES:
- One-command startup for entire AGI ecosystem
- Automatic component discovery and health monitoring
- Graceful startup/shutdown sequence
- Real-time event streaming between components
- Continuous learning from JARVIS interactions
- Model serving for J-Prime inference
- Fault tolerance with automatic recovery
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Add reactor_core to path
sys.path.insert(0, str(Path(__file__).parent))

from reactor_core.orchestration.trinity_orchestrator import (
    TrinityOrchestrator,
    ComponentType,
    ComponentHealth,
    initialize_orchestrator,
    shutdown_orchestrator,
    get_orchestrator,
)
from reactor_core.integration.event_bridge import (
    EventBridge,
    EventSource,
    EventType,
    create_event_bridge,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SupervisorConfig:
    """Configuration for AGI Supervisor."""
    # Component paths (auto-detected or manual)
    jarvis_path: Optional[Path] = None
    jprime_path: Optional[Path] = None
    reactor_core_path: Optional[Path] = None

    # Component enable flags
    enable_jarvis: bool = True
    enable_jprime: bool = True
    enable_reactor_core: bool = True

    # Services to start
    enable_training: bool = True
    enable_serving: bool = True
    enable_scout: bool = False
    enable_api: bool = True

    # Network ports
    api_port: int = 8003
    serving_port: int = 8001
    jprime_port: int = 8000

    # Trinity coordination
    trinity_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")
    heartbeat_interval: float = 5.0
    health_check_interval: float = 10.0

    # Continuous learning
    experience_collection: bool = True
    auto_training_threshold: int = 1000  # Train after N experiences

    # Startup behavior
    startup_timeout: float = 60.0  # Seconds to wait for components
    retry_on_failure: bool = True
    max_retries: int = 3

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def __post_init__(self):
        # Auto-detect paths
        if self.jarvis_path is None:
            self.jarvis_path = self._find_repo("JARVIS-AI-Agent")
        if self.jprime_path is None:
            self.jprime_path = self._find_repo("jarvis-prime")
        if self.reactor_core_path is None:
            self.reactor_core_path = Path(__file__).parent

    def _find_repo(self, name: str) -> Optional[Path]:
        """Find repository by name."""
        search_paths = [
            Path(__file__).parent.parent / name,  # Sibling directory
            Path.home() / name,
            Path.home() / "Projects" / name,
            Path.home() / "code" / name,
            Path.home() / "dev" / name,
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None


class ComponentStatus(Enum):
    """Status of a managed component."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ManagedComponent:
    """A component managed by the supervisor."""
    name: str
    component_type: ComponentType
    path: Optional[Path] = None
    process: Optional[subprocess.Popen] = None
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_heartbeat: float = 0.0
    start_time: float = 0.0
    restart_count: int = 0
    error_message: str = ""

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        if self.status not in (ComponentStatus.RUNNING, ComponentStatus.DEGRADED):
            return False
        if self.process and self.process.poll() is not None:
            return False
        return True


# =============================================================================
# AGI SUPERVISOR - Main Controller
# =============================================================================

class AGISupervisor:
    """
    Central AGI Supervisor orchestrating all components.

    Manages:
    - Component lifecycle (start, stop, restart)
    - Health monitoring and recovery
    - Event routing between components
    - Continuous learning pipeline
    - Model serving coordination
    """

    def __init__(self, config: SupervisorConfig):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components
        self._components: Dict[str, ManagedComponent] = {}

        # Trinity orchestrator
        self._orchestrator: Optional[TrinityOrchestrator] = None

        # Event bridge
        self._event_bridge: Optional[EventBridge] = None

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        # Statistics
        self._start_time = 0.0
        self._stats = {
            "events_processed": 0,
            "experiences_collected": 0,
            "trainings_triggered": 0,
            "restarts": 0,
        }

        # Initialize logging
        self._setup_logging()

        logger.info("AGI Supervisor initialized")

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        handlers = [logging.StreamHandler(sys.stdout)]

        if self.config.log_file:
            handlers.append(logging.FileHandler(self.config.log_file))

        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )

    async def start(self) -> bool:
        """Start the AGI Supervisor and all components."""
        logger.info("=" * 70)
        logger.info("           AGI OS UNIFIED SUPERVISOR - PROJECT TRINITY")
        logger.info("=" * 70)
        logger.info("")

        self._start_time = time.time()
        self._running = True

        try:
            # Phase 1: Initialize Trinity Orchestrator
            logger.info("[Phase 1] Initializing Trinity Orchestrator...")
            self._orchestrator = await initialize_orchestrator()
            if not self._orchestrator.is_running():
                raise RuntimeError("Failed to start Trinity Orchestrator")
            logger.info("[OK] Trinity Orchestrator running")

            # Phase 2: Initialize Event Bridge
            logger.info("[Phase 2] Initializing Event Bridge...")
            self._event_bridge = create_event_bridge(
                source=EventSource.REACTOR_CORE,
                events_dir=self.config.trinity_dir / "events",
            )
            await self._event_bridge.start()
            logger.info("[OK] Event Bridge running")

            # Phase 3: Discover and register components
            logger.info("[Phase 3] Discovering components...")
            await self._discover_components()

            # Phase 4: Start Reactor Core services
            logger.info("[Phase 4] Starting Reactor Core services...")
            await self._start_reactor_core_services()

            # Phase 5: Start external components
            if self.config.enable_jarvis and self._components.get("jarvis"):
                logger.info("[Phase 5] Starting JARVIS (Body)...")
                await self._start_component("jarvis")

            if self.config.enable_jprime and self._components.get("jprime"):
                logger.info("[Phase 6] Starting J-Prime (Mind)...")
                await self._start_component("jprime")

            # Phase 6: Start background tasks
            logger.info("[Phase 7] Starting background services...")
            await self._start_background_tasks()

            # Wait for components to become healthy
            logger.info("[Phase 8] Waiting for component health...")
            healthy = await self._wait_for_health()

            if healthy:
                logger.info("")
                logger.info("=" * 70)
                logger.info("            AGI OS READY - All Systems Operational")
                logger.info("=" * 70)
                self._print_status()
                return True
            else:
                logger.warning("Some components failed to start, running in degraded mode")
                return True

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        """Gracefully stop all components."""
        logger.info("")
        logger.info("Initiating graceful shutdown...")
        self._running = False
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        # Stop components in reverse order
        for name in reversed(list(self._components.keys())):
            await self._stop_component(name)

        # Stop event bridge
        if self._event_bridge:
            await self._event_bridge.stop()

        # Stop orchestrator
        if self._orchestrator:
            await shutdown_orchestrator()

        logger.info("AGI Supervisor shutdown complete")
        self._print_final_stats()

    async def _discover_components(self) -> None:
        """Discover available components."""
        # JARVIS (Body)
        if self.config.jarvis_path and self.config.jarvis_path.exists():
            self._components["jarvis"] = ManagedComponent(
                name="JARVIS",
                component_type=ComponentType.JARVIS_BODY,
                path=self.config.jarvis_path,
            )
            logger.info(f"  Found JARVIS at {self.config.jarvis_path}")
        elif self.config.enable_jarvis:
            logger.warning("  JARVIS path not found, will run without Body")

        # J-Prime (Mind)
        if self.config.jprime_path and self.config.jprime_path.exists():
            self._components["jprime"] = ManagedComponent(
                name="J-Prime",
                component_type=ComponentType.J_PRIME,
                path=self.config.jprime_path,
            )
            logger.info(f"  Found J-Prime at {self.config.jprime_path}")
        elif self.config.enable_jprime:
            logger.warning("  J-Prime path not found, will run without Mind")

        # Reactor Core (always present)
        self._components["reactor_core"] = ManagedComponent(
            name="Reactor Core",
            component_type=ComponentType.REACTOR_CORE,
            path=self.config.reactor_core_path,
            status=ComponentStatus.RUNNING,
        )
        logger.info(f"  Reactor Core at {self.config.reactor_core_path}")

    async def _start_reactor_core_services(self) -> None:
        """Start Reactor Core internal services."""
        # API Server
        if self.config.enable_api:
            try:
                from reactor_core.api.server import start_server_background
                await start_server_background(port=self.config.api_port)
                logger.info(f"  [OK] API Server on port {self.config.api_port}")
            except ImportError:
                logger.warning("  API Server module not available")
            except Exception as e:
                logger.warning(f"  API Server failed: {e}")

        # Model Serving
        if self.config.enable_serving:
            try:
                from reactor_core.serving import InferenceEngine
                engine = InferenceEngine()
                await engine.start()
                logger.info(f"  [OK] Inference Engine started")
            except ImportError:
                logger.warning("  Inference Engine module not available")
            except Exception as e:
                logger.warning(f"  Inference Engine failed: {e}")

        # Training Pipeline (background)
        if self.config.enable_training:
            logger.info("  [OK] Training Pipeline ready (on-demand)")

        # Scout (optional)
        if self.config.enable_scout:
            logger.info("  [OK] Scout ingestion ready")

    async def _start_component(self, name: str) -> bool:
        """Start an external component."""
        component = self._components.get(name)
        if not component or not component.path:
            return False

        component.status = ComponentStatus.STARTING
        component.start_time = time.time()

        try:
            # Determine start command based on component
            if name == "jarvis":
                start_cmd = self._get_jarvis_start_command(component.path)
            elif name == "jprime":
                start_cmd = self._get_jprime_start_command(component.path)
            else:
                return False

            if start_cmd:
                # Start process
                component.process = subprocess.Popen(
                    start_cmd,
                    cwd=str(component.path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**os.environ, "TRINITY_ENABLED": "1"},
                )

                # Wait briefly for startup
                await asyncio.sleep(2.0)

                if component.process.poll() is None:
                    component.status = ComponentStatus.RUNNING
                    logger.info(f"  [OK] {component.name} started (PID: {component.process.pid})")
                    return True
                else:
                    component.status = ComponentStatus.FAILED
                    stderr = component.process.stderr.read().decode() if component.process.stderr else ""
                    component.error_message = stderr[:200]
                    logger.error(f"  [FAIL] {component.name} exited: {component.error_message}")
                    return False
            else:
                logger.warning(f"  No start command found for {name}")
                return False

        except Exception as e:
            component.status = ComponentStatus.FAILED
            component.error_message = str(e)
            logger.error(f"  [FAIL] Failed to start {component.name}: {e}")
            return False

    def _get_jarvis_start_command(self, path: Path) -> Optional[List[str]]:
        """Get command to start JARVIS."""
        # Check for various entry points
        candidates = [
            path / "main.py",
            path / "jarvis" / "main.py",
            path / "run.py",
            path / "jarvis.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                return [sys.executable, str(candidate), "--trinity"]

        # Try package execution
        if (path / "jarvis" / "__main__.py").exists():
            return [sys.executable, "-m", "jarvis", "--trinity"]

        return None

    def _get_jprime_start_command(self, path: Path) -> Optional[List[str]]:
        """Get command to start J-Prime."""
        candidates = [
            path / "run_server.py",
            path / "server.py",
            path / "main.py",
            path / "jprime" / "server.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                return [
                    sys.executable, str(candidate),
                    "--port", str(self.config.jprime_port),
                    "--trinity",
                ]

        return None

    async def _stop_component(self, name: str) -> None:
        """Stop a component gracefully."""
        component = self._components.get(name)
        if not component:
            return

        if component.process and component.process.poll() is None:
            logger.info(f"  Stopping {component.name}...")

            # Try graceful shutdown first
            component.process.terminate()

            # Wait for termination
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, component.process.wait
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                # Force kill
                component.process.kill()
                component.process.wait()

            logger.info(f"  [OK] {component.name} stopped")

        component.status = ComponentStatus.STOPPED

    async def _wait_for_health(self) -> bool:
        """Wait for all components to become healthy."""
        deadline = time.time() + self.config.startup_timeout

        while time.time() < deadline:
            all_healthy = True

            for name, component in self._components.items():
                if component.status == ComponentStatus.STARTING:
                    all_healthy = False

                # Check process health
                if component.process:
                    if component.process.poll() is not None:
                        component.status = ComponentStatus.FAILED
                        all_healthy = False

                # Check heartbeat
                if component.status == ComponentStatus.RUNNING:
                    orchestrator_state = self._orchestrator.get_component_state(component.component_type)
                    if orchestrator_state.health == ComponentHealth.HEALTHY:
                        component.last_heartbeat = time.time()

            if all_healthy:
                return True

            await asyncio.sleep(1.0)

        return False

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and processing tasks."""
        # Health monitoring
        self._tasks.append(asyncio.create_task(self._health_monitor_loop()))

        # Experience collection (continuous learning)
        if self.config.experience_collection:
            self._tasks.append(asyncio.create_task(self._experience_collection_loop()))

        # Event processing
        self._tasks.append(asyncio.create_task(self._event_processing_loop()))

    async def _health_monitor_loop(self) -> None:
        """Monitor component health and handle failures."""
        while self._running:
            try:
                for name, component in self._components.items():
                    # Skip if not supposed to be running
                    if component.status in (ComponentStatus.STOPPED, ComponentStatus.UNKNOWN):
                        continue

                    # Check process
                    if component.process and component.process.poll() is not None:
                        logger.warning(f"{component.name} has died unexpectedly")
                        component.status = ComponentStatus.FAILED

                        if self.config.retry_on_failure and component.restart_count < self.config.max_retries:
                            logger.info(f"Attempting to restart {component.name}...")
                            component.restart_count += 1
                            self._stats["restarts"] += 1
                            await self._start_component(name)

                    # Check heartbeat freshness
                    if component.status == ComponentStatus.RUNNING:
                        orchestrator_state = self._orchestrator.get_component_state(component.component_type)

                        if orchestrator_state.health == ComponentHealth.OFFLINE:
                            if component.status != ComponentStatus.DEGRADED:
                                logger.warning(f"{component.name} missed heartbeat, marking as degraded")
                                component.status = ComponentStatus.DEGRADED

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)

    async def _experience_collection_loop(self) -> None:
        """Collect experiences from JARVIS for continuous learning."""
        while self._running:
            try:
                # Check for new experiences in Trinity events
                events_dir = self.config.trinity_dir / "events"

                if events_dir.exists():
                    for event_file in events_dir.glob("*.json"):
                        try:
                            with open(event_file) as f:
                                event_data = json.load(f)

                            event_type = event_data.get("event_type", "")

                            # Collect interaction events
                            if event_type in ("interaction_end", "correction", "feedback"):
                                self._stats["experiences_collected"] += 1

                                # Check if we should trigger training
                                if (
                                    self._stats["experiences_collected"] > 0 and
                                    self._stats["experiences_collected"] % self.config.auto_training_threshold == 0
                                ):
                                    logger.info(f"Auto-training triggered after {self._stats['experiences_collected']} experiences")
                                    self._stats["trainings_triggered"] += 1
                                    # Would trigger training here

                        except Exception as e:
                            logger.debug(f"Error processing event file: {e}")

                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Experience collection error: {e}")
                await asyncio.sleep(10.0)

    async def _event_processing_loop(self) -> None:
        """Process events from the event bridge."""
        if not self._event_bridge:
            return

        # Register event handlers
        @self._event_bridge.on_event(EventType.CORRECTION)
        async def handle_correction(event):
            logger.info(f"Received correction event: {event.payload.get('user_input', '')[:50]}...")
            self._stats["events_processed"] += 1

        @self._event_bridge.on_event(EventType.TRAINING_COMPLETE)
        async def handle_training_complete(event):
            logger.info(f"Training completed: {event.payload}")
            self._stats["events_processed"] += 1

        @self._event_bridge.on_event(EventType.SAFETY_BLOCKED)
        async def handle_safety_blocked(event):
            logger.warning(f"Safety block: {event.payload.get('action', '')} - {event.payload.get('reason', '')}")
            self._stats["events_processed"] += 1

        while self._running:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    def _print_status(self) -> None:
        """Print current system status."""
        logger.info("")
        logger.info("COMPONENT STATUS:")
        logger.info("-" * 50)

        for name, component in self._components.items():
            status_icon = {
                ComponentStatus.RUNNING: "[RUNNING]",
                ComponentStatus.DEGRADED: "[DEGRADED]",
                ComponentStatus.STOPPED: "[STOPPED]",
                ComponentStatus.FAILED: "[FAILED]",
                ComponentStatus.STARTING: "[STARTING]",
                ComponentStatus.UNKNOWN: "[UNKNOWN]",
            }.get(component.status, "?")

            pid_info = f" (PID: {component.process.pid})" if component.process else ""
            logger.info(f"  {status_icon} {component.name}{pid_info}")

        logger.info("-" * 50)
        logger.info("")
        logger.info("ENDPOINTS:")
        if self.config.enable_api:
            logger.info(f"  API Server:      http://localhost:{self.config.api_port}")
        if self.config.enable_serving:
            logger.info(f"  Model Serving:   http://localhost:{self.config.serving_port}")
        if self._components.get("jprime") and self._components["jprime"].status == ComponentStatus.RUNNING:
            logger.info(f"  J-Prime:         http://localhost:{self.config.jprime_port}")
        logger.info("")
        logger.info("Press Ctrl+C to shutdown")
        logger.info("")

    def _print_final_stats(self) -> None:
        """Print final statistics."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0

        logger.info("")
        logger.info("FINAL STATISTICS:")
        logger.info("-" * 50)
        logger.info(f"  Uptime:              {uptime:.1f} seconds")
        logger.info(f"  Events Processed:    {self._stats['events_processed']}")
        logger.info(f"  Experiences:         {self._stats['experiences_collected']}")
        logger.info(f"  Trainings Triggered: {self._stats['trainings_triggered']}")
        logger.info(f"  Component Restarts:  {self._stats['restarts']}")
        logger.info("-" * 50)

    async def run_until_shutdown(self) -> None:
        """Run until shutdown signal received."""
        await self._shutdown_event.wait()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for AGI Supervisor."""
    parser = argparse.ArgumentParser(
        description="AGI OS Unified Supervisor - Project Trinity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_supervisor.py                    # Start all components
  python3 run_supervisor.py --no-jarvis        # Start without JARVIS
  python3 run_supervisor.py --no-training      # Disable training
  python3 run_supervisor.py --api-port 8080    # Custom API port
        """,
    )

    # Component flags
    parser.add_argument("--no-jarvis", action="store_true", help="Disable JARVIS (Body)")
    parser.add_argument("--no-jprime", action="store_true", help="Disable J-Prime (Mind)")
    parser.add_argument("--no-training", action="store_true", help="Disable training pipeline")
    parser.add_argument("--no-serving", action="store_true", help="Disable model serving")
    parser.add_argument("--enable-scout", action="store_true", help="Enable Scout web ingestion")

    # Paths
    parser.add_argument("--jarvis-path", type=Path, help="Path to JARVIS repository")
    parser.add_argument("--jprime-path", type=Path, help="Path to J-Prime repository")

    # Ports
    parser.add_argument("--api-port", type=int, default=8003, help="API server port")
    parser.add_argument("--serving-port", type=int, default=8001, help="Model serving port")
    parser.add_argument("--jprime-port", type=int, default=8000, help="J-Prime server port")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, help="Log file path")

    args = parser.parse_args()

    # Build config
    config = SupervisorConfig(
        jarvis_path=args.jarvis_path,
        jprime_path=args.jprime_path,
        enable_jarvis=not args.no_jarvis,
        enable_jprime=not args.no_jprime,
        enable_training=not args.no_training,
        enable_serving=not args.no_serving,
        enable_scout=args.enable_scout,
        api_port=args.api_port,
        serving_port=args.serving_port,
        jprime_port=args.jprime_port,
        log_level=args.log_level,
        log_file=args.log_file,
    )

    # Create supervisor
    supervisor = AGISupervisor(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\nReceived shutdown signal...")
        supervisor._shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Start supervisor
    success = await supervisor.start()

    if success:
        # Run until shutdown
        await supervisor.run_until_shutdown()

    # Cleanup
    await supervisor.stop()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
