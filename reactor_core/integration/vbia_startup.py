"""
Reactor Core VBIA Startup Integration
======================================

Initializes Reactor Core's connection to the JARVIS cross-repo VBIA system
during startup for event analytics and threat monitoring.

Features:
- Cross-repo state directory monitoring
- Reactor Core state file initialization
- VBIA event ingestion for analytics
- Threat pattern analysis
- Heartbeat registration
- Async startup integration

Author: Reactor Core Team
Version: 6.2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

logger = logging.getLogger("reactor-core.vbia.startup")


# =============================================================================
# Configuration
# =============================================================================

def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return _get_env(key, str(default)).lower() in ("true", "1", "yes", "on")


@dataclass
class VBIAStartupConfig:
    """Configuration for VBIA startup integration."""
    # Cross-repo directory
    cross_repo_dir: Path = field(
        default_factory=lambda: Path(os.path.expanduser(
            _get_env("JARVIS_CROSS_REPO_DIR", "~/.jarvis/cross_repo")
        ))
    )

    # Event ingestion settings
    enable_event_ingestion: bool = field(
        default_factory=lambda: _get_env_bool("REACTOR_CORE_INGEST_VBIA_EVENTS", True)
    )
    event_poll_interval: float = field(
        default_factory=lambda: float(_get_env("REACTOR_CORE_EVENT_POLL_INTERVAL", "1.0"))
    )

    # Analytics settings
    enable_threat_analytics: bool = field(
        default_factory=lambda: _get_env_bool("REACTOR_CORE_THREAT_ANALYTICS", True)
    )
    threat_analysis_interval: float = field(
        default_factory=lambda: float(_get_env("REACTOR_CORE_THREAT_ANALYSIS_INTERVAL", "60.0"))
    )

    # State update settings
    heartbeat_interval: float = field(
        default_factory=lambda: float(_get_env("REACTOR_CORE_HEARTBEAT_INTERVAL", "10.0"))
    )
    state_update_interval: float = field(
        default_factory=lambda: float(_get_env("REACTOR_CORE_STATE_UPDATE_INTERVAL", "5.0"))
    )

    # Capabilities
    vbia_analytics_enabled: bool = field(
        default_factory=lambda: _get_env_bool("REACTOR_CORE_VBIA_ANALYTICS", True)
    )
    visual_threat_monitoring: bool = field(
        default_factory=lambda: _get_env_bool("REACTOR_CORE_VISUAL_THREAT_MONITORING", True)
    )


# =============================================================================
# Enums
# =============================================================================

class ReactorStatus(str, Enum):
    """Reactor Core status."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    OFFLINE = "offline"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ReactorState:
    """Reactor Core state for cross-repo system."""
    repo_type: str = "reactor_core"
    status: ReactorStatus = ReactorStatus.INITIALIZING
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "6.2.0"
    capabilities: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ReactorHeartbeat:
    """Reactor Core heartbeat."""
    repo_type: str = "reactor_core"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: ReactorStatus = ReactorStatus.READY
    uptime_seconds: float = 0.0
    events_processed: int = 0


@dataclass
class ThreatAnalysis:
    """Threat analysis results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_events: int = 0
    visual_threats_detected: int = 0
    auth_failures: int = 0
    threat_rate: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# Reactor Core VBIA Startup Integrator
# =============================================================================

class ReactorCoreVBIAStartup:
    """
    Manages Reactor Core's startup integration with the cross-repo VBIA system.
    """

    def __init__(self, config: Optional[VBIAStartupConfig] = None):
        self.config = config or VBIAStartupConfig()
        self._initialized = False
        self._start_time = time.time()
        self._running = False

        # State files
        self._reactor_state_file = self.config.cross_repo_dir / "reactor_state.json"
        self._heartbeat_file = self.config.cross_repo_dir / "heartbeat.json"
        self._vbia_events_file = self.config.cross_repo_dir / "vbia_events.json"

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._state_update_task: Optional[asyncio.Task] = None
        self._event_ingestion_task: Optional[asyncio.Task] = None
        self._threat_analysis_task: Optional[asyncio.Task] = None

        # State
        self._reactor_state = ReactorState(
            capabilities={
                "vbia_analytics": self.config.vbia_analytics_enabled,
                "visual_threat_monitoring": self.config.visual_threat_monitoring,
                "event_ingestion": self.config.enable_event_ingestion,
                "threat_analytics": self.config.enable_threat_analytics,
            }
        )

        # Event metrics
        self._events_processed = 0
        self._visual_threats_detected = 0
        self._auth_failures = 0
        self._event_counts_by_type: Dict[str, int] = defaultdict(int)

        # Event handlers (can be registered by application)
        self._event_handlers: Dict[str, List[callable]] = {}

    async def initialize(self) -> bool:
        """
        Initialize Reactor Core's cross-repo VBIA connection.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            logger.info("[VBIA Startup] Already initialized")
            return True

        try:
            logger.info("[VBIA Startup] Starting initialization...")

            # Ensure cross-repo directory exists
            await self._ensure_cross_repo_directory()

            # Initialize Reactor Core state file
            await self._initialize_reactor_state()

            # Update heartbeat
            await self._update_heartbeat()

            # Start background tasks
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._state_update_task = asyncio.create_task(self._state_update_loop())

            if self.config.enable_event_ingestion:
                self._event_ingestion_task = asyncio.create_task(self._event_ingestion_loop())

            if self.config.enable_threat_analytics:
                self._threat_analysis_task = asyncio.create_task(self._threat_analysis_loop())

            # Update state to ready
            self._reactor_state.status = ReactorStatus.READY
            await self._write_reactor_state()

            self._initialized = True
            logger.info("[VBIA Startup] ✅ Initialization complete")
            logger.info(f"[VBIA Startup]    Cross-repo dir: {self.config.cross_repo_dir}")
            logger.info(f"[VBIA Startup]    VBIA analytics: {self.config.vbia_analytics_enabled}")
            logger.info(f"[VBIA Startup]    Visual threat monitoring: {self.config.visual_threat_monitoring}")
            logger.info(f"[VBIA Startup]    Event ingestion: {self.config.enable_event_ingestion}")
            logger.info(f"[VBIA Startup]    Threat analytics: {self.config.enable_threat_analytics}")

            return True

        except Exception as e:
            logger.error(f"[VBIA Startup] ❌ Initialization failed: {e}", exc_info=True)
            self._reactor_state.status = ReactorStatus.ERROR
            self._reactor_state.errors.append(str(e))
            return False

    async def shutdown(self) -> None:
        """Shutdown the VBIA startup integration."""
        logger.info("[VBIA Startup] Shutting down...")

        self._running = False

        # Cancel background tasks
        for task in [
            self._heartbeat_task,
            self._state_update_task,
            self._event_ingestion_task,
            self._threat_analysis_task,
        ]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Update state to offline
        self._reactor_state.status = ReactorStatus.OFFLINE
        await self._write_reactor_state()

        logger.info("[VBIA Startup] ✅ Shutdown complete")

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    async def _ensure_cross_repo_directory(self) -> None:
        """Ensure cross-repo directory exists."""
        try:
            self.config.cross_repo_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[VBIA Startup] Cross-repo directory ready: {self.config.cross_repo_dir}")
        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to create cross-repo directory: {e}")
            raise

    async def _initialize_reactor_state(self) -> None:
        """Initialize Reactor Core state file."""
        await self._write_reactor_state()
        logger.info("[VBIA Startup] ✓ reactor_state.json initialized")

    async def _update_heartbeat(self) -> None:
        """Update heartbeat in cross-repo system."""
        try:
            # Read existing heartbeats
            heartbeats = await self._read_json_file(self._heartbeat_file, default={})

            # Update Reactor Core heartbeat
            heartbeats["reactor_core"] = asdict(ReactorHeartbeat(
                status=self._reactor_state.status,
                uptime_seconds=time.time() - self._start_time,
                events_processed=self._events_processed,
            ))

            # Write back
            await self._write_json_file(self._heartbeat_file, heartbeats)

        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to update heartbeat: {e}")

    # =========================================================================
    # Event Handling
    # =========================================================================

    def register_event_handler(self, event_type: str, handler: callable) -> None:
        """
        Register an event handler for VBIA events.

        Args:
            event_type: Event type to handle (e.g., "vbia_visual_threat")
            handler: Async callable to handle the event
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.info(f"[VBIA Startup] Event handler registered for: {event_type}")

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """
        Handle a VBIA event from the cross-repo system.

        Args:
            event: Event dictionary
        """
        event_type = event.get("event_type")
        if not event_type:
            return

        # Update metrics
        self._events_processed += 1
        self._event_counts_by_type[event_type] += 1

        # Track specific metrics
        if event_type == "vbia_visual_threat":
            self._visual_threats_detected += 1
        elif event_type == "vbia_auth_failed":
            self._auth_failures += 1

        # Call registered handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"[VBIA Startup] Event handler error for {event_type}: {e}")

    # =========================================================================
    # Threat Analytics
    # =========================================================================

    async def _analyze_threats(self) -> ThreatAnalysis:
        """
        Analyze VBIA events for threat patterns.

        Returns:
            Threat analysis results
        """
        total_events = self._events_processed
        threat_rate = (
            self._visual_threats_detected / total_events
            if total_events > 0 else 0.0
        )

        # Determine risk level
        if threat_rate >= 0.3:
            risk_level = "critical"
        elif threat_rate >= 0.1:
            risk_level = "high"
        elif threat_rate >= 0.05:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Generate recommendations
        recommendations = []
        if self._visual_threats_detected > 0:
            recommendations.append(
                f"Visual threats detected ({self._visual_threats_detected} total) - "
                "review screen security settings"
            )
        if self._auth_failures > 10:
            recommendations.append(
                f"High authentication failure rate ({self._auth_failures} failures) - "
                "check for unauthorized access attempts"
            )
        if risk_level in ["high", "critical"]:
            recommendations.append(
                "Consider increasing VBIA security thresholds or enabling additional verification"
            )

        return ThreatAnalysis(
            total_events=total_events,
            visual_threats_detected=self._visual_threats_detected,
            auth_failures=self._auth_failures,
            threat_rate=threat_rate,
            risk_level=risk_level,
            recommendations=recommendations,
        )

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _heartbeat_loop(self) -> None:
        """Background task that updates heartbeat."""
        logger.info("[VBIA Startup] Heartbeat loop started")

        while self._running:
            try:
                await self._update_heartbeat()
                self._reactor_state.last_heartbeat = datetime.now().isoformat()
                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VBIA Startup] Heartbeat loop error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)

    async def _state_update_loop(self) -> None:
        """Background task that updates Reactor Core state."""
        logger.info("[VBIA Startup] State update loop started")

        while self._running:
            try:
                await self._write_reactor_state()
                await asyncio.sleep(self.config.state_update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VBIA Startup] State update loop error: {e}")
                await asyncio.sleep(self.config.state_update_interval)

    async def _event_ingestion_loop(self) -> None:
        """Background task that ingests VBIA events."""
        logger.info("[VBIA Startup] Event ingestion loop started")

        last_event_index = 0

        while self._running:
            try:
                # Read events file
                events = await self._read_json_file(self._vbia_events_file, default=[])

                # Process new events
                if len(events) > last_event_index:
                    new_events = events[last_event_index:]
                    for event in new_events:
                        await self._handle_event(event)
                    last_event_index = len(events)

                await asyncio.sleep(self.config.event_poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VBIA Startup] Event ingestion error: {e}")
                await asyncio.sleep(self.config.event_poll_interval)

    async def _threat_analysis_loop(self) -> None:
        """Background task that analyzes threat patterns."""
        logger.info("[VBIA Startup] Threat analysis loop started")

        while self._running:
            try:
                analysis = await self._analyze_threats()

                # Log threat analysis
                if analysis.risk_level in ["high", "critical"]:
                    logger.warning(
                        f"[VBIA Threat Analysis] Risk level: {analysis.risk_level.upper()}, "
                        f"threat rate: {analysis.threat_rate:.1%}, "
                        f"threats: {analysis.visual_threats_detected}"
                    )
                    for rec in analysis.recommendations:
                        logger.warning(f"  • {rec}")

                await asyncio.sleep(self.config.threat_analysis_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VBIA Startup] Threat analysis error: {e}")
                await asyncio.sleep(self.config.threat_analysis_interval)

    # =========================================================================
    # State Management
    # =========================================================================

    async def _write_reactor_state(self) -> None:
        """Write Reactor Core state to file."""
        self._reactor_state.last_update = datetime.now().isoformat()
        self._reactor_state.metrics.update({
            "uptime_seconds": time.time() - self._start_time,
            "events_processed": self._events_processed,
            "visual_threats_detected": self._visual_threats_detected,
            "auth_failures": self._auth_failures,
            "event_counts_by_type": dict(self._event_counts_by_type),
        })
        await self._write_json_file(self._reactor_state_file, asdict(self._reactor_state))

    async def update_status(self, status: ReactorStatus, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update Reactor Core status.

        Args:
            status: New status
            metrics: Optional metrics to update
        """
        self._reactor_state.status = status
        if metrics:
            self._reactor_state.metrics.update(metrics)
        await self._write_reactor_state()

    async def get_threat_analysis(self) -> ThreatAnalysis:
        """
        Get current threat analysis.

        Returns:
            Threat analysis results
        """
        return await self._analyze_threats()

    # =========================================================================
    # File I/O
    # =========================================================================

    async def _read_json_file(self, file_path: Path, default: Any = None) -> Any:
        """Read JSON file asynchronously."""
        try:
            if not file_path.exists():
                return default

            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                return json.loads(content) if content else default

        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to read {file_path}: {e}")
            return default

    async def _write_json_file(self, file_path: Path, data: Any) -> None:
        """Write JSON file asynchronously."""
        try:
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to write {file_path}: {e}")
            raise


# =============================================================================
# Global Singleton
# =============================================================================

_vbia_startup: Optional[ReactorCoreVBIAStartup] = None


async def get_vbia_startup(
    config: Optional[VBIAStartupConfig] = None
) -> ReactorCoreVBIAStartup:
    """
    Get or create the global VBIA startup instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The VBIA startup instance
    """
    global _vbia_startup

    if _vbia_startup is None:
        _vbia_startup = ReactorCoreVBIAStartup(config)

    return _vbia_startup


async def initialize_vbia_startup(
    config: Optional[VBIAStartupConfig] = None
) -> bool:
    """
    Initialize Reactor Core's VBIA cross-repo connection.

    This is the main entry point for Reactor Core startup integration.

    Args:
        config: Optional configuration

    Returns:
        True if initialization succeeded, False otherwise
    """
    startup = await get_vbia_startup(config)
    return await startup.initialize()


async def shutdown_vbia_startup() -> None:
    """Shutdown Reactor Core's VBIA cross-repo connection."""
    global _vbia_startup

    if _vbia_startup:
        await _vbia_startup.shutdown()
        _vbia_startup = None
