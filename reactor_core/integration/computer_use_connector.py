"""
Computer Use Connector for Reactor Core
========================================

Ingests Computer Use events from JARVIS for learning and analysis.

Features:
- Action Chaining optimization tracking
- Vision analysis result ingestion
- OmniParser UI parsing data
- Cross-repo Computer Use metrics

Author: Reactor Core Team
Version: 10.1.0 - Computer Use Intelligence
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

COMPUTER_USE_STATE_DIR = Path.home() / ".jarvis" / "cross_repo"
COMPUTER_USE_STATE_FILE = COMPUTER_USE_STATE_DIR / "computer_use_state.json"
COMPUTER_USE_EVENTS_FILE = COMPUTER_USE_STATE_DIR / "computer_use_events.json"


# ============================================================================
# Enums
# ============================================================================

class ComputerUseEventType(Enum):
    """Types of Computer Use events."""
    ACTION_EXECUTED = "action_executed"
    BATCH_COMPLETED = "batch_completed"
    VISION_ANALYSIS = "vision_analysis"
    ERROR = "error"


class ActionType(Enum):
    """Computer action types."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    KEY_PRESS = "key_press"
    SCREENSHOT = "screenshot"
    DRAG = "drag"
    SCROLL = "scroll"
    WAIT = "wait"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ComputerUseEvent:
    """A Computer Use event from JARVIS."""
    event_id: str
    timestamp: datetime
    event_type: ComputerUseEventType

    # Action/batch data
    action_type: Optional[str] = None
    batch_size: int = 1
    goal: str = ""

    # Execution metrics
    execution_time_ms: float = 0.0
    time_saved_ms: float = 0.0
    tokens_saved: int = 0

    # Vision data
    vision_analysis: Optional[Dict[str, Any]] = None
    used_omniparser: bool = False

    # Metadata
    session_id: str = ""
    repo_source: str = "jarvis"
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "action_type": self.action_type,
            "batch_size": self.batch_size,
            "goal": self.goal,
            "execution_time_ms": self.execution_time_ms,
            "time_saved_ms": self.time_saved_ms,
            "tokens_saved": self.tokens_saved,
            "vision_analysis": self.vision_analysis,
            "used_omniparser": self.used_omniparser,
            "session_id": self.session_id,
            "repo_source": self.repo_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputerUseEvent":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        event_type_str = data.get("event_type", "action_executed")
        try:
            event_type = ComputerUseEventType(event_type_str)
        except ValueError:
            event_type = ComputerUseEventType.ACTION_EXECUTED

        return cls(
            event_id=data.get("event_id", ""),
            timestamp=timestamp,
            event_type=event_type,
            action_type=data.get("action_type"),
            batch_size=data.get("batch_size", 1),
            goal=data.get("goal", ""),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            time_saved_ms=data.get("time_saved_ms", 0.0),
            tokens_saved=data.get("tokens_saved", 0),
            vision_analysis=data.get("vision_analysis"),
            used_omniparser=data.get("used_omniparser", False),
            session_id=data.get("session_id", ""),
            repo_source=data.get("repo_source", "jarvis"),
            raw_data=data,
        )


@dataclass
class ComputerUseConnectorConfig:
    """Configuration for Computer Use connector."""
    # State paths
    state_dir: Path = COMPUTER_USE_STATE_DIR
    state_file: Path = COMPUTER_USE_STATE_FILE
    events_file: Path = COMPUTER_USE_EVENTS_FILE

    # Filtering
    min_batch_size: int = 1  # Include single actions
    only_batches: bool = False
    only_omniparser: bool = False

    # Time window
    lookback_hours: int = 24  # Last 24 hours by default

    # Event filtering
    include_event_types: List[ComputerUseEventType] = field(
        default_factory=lambda: list(ComputerUseEventType)
    )


# ============================================================================
# Computer Use Connector
# ============================================================================

class ComputerUseConnector:
    """
    Connects to JARVIS Computer Use system for event ingestion.

    Reads Computer Use events from shared state files for:
    - Action Chaining optimization analysis
    - Vision analysis result learning
    - OmniParser UI parsing data ingestion
    - Cross-repo Computer Use metrics aggregation
    """

    def __init__(
        self,
        config: Optional[ComputerUseConnectorConfig] = None,
    ):
        """
        Initialize Computer Use connector.

        Args:
            config: Optional configuration
        """
        self.config = config or ComputerUseConnectorConfig()

        # Validate paths
        if not self.config.state_dir.exists():
            logger.info(
                f"Computer Use state directory not found at {self.config.state_dir}. "
                f"Will be created when JARVIS runs Computer Use tasks."
            )

        self._events_cache: List[ComputerUseEvent] = []
        self._last_sync: Optional[datetime] = None

    async def get_events(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[ComputerUseEvent]:
        """
        Get Computer Use events from JARVIS.

        Args:
            since: Start time (default: lookback_hours ago)
            until: End time (default: now)
            limit: Maximum events to return

        Returns:
            List of ComputerUseEvent objects
        """
        if since is None:
            since = datetime.now() - timedelta(hours=self.config.lookback_hours)
        if until is None:
            until = datetime.now()

        # Load events from file
        events = await self._load_events()

        # Filter by time
        filtered = [
            e for e in events
            if since <= e.timestamp <= until
        ]

        # Apply additional filters
        filtered = self._filter_events(filtered)

        return filtered[:limit]

    async def get_batch_events(
        self,
        since: Optional[datetime] = None,
        min_batch_size: int = 2,
    ) -> List[ComputerUseEvent]:
        """Get batch execution events only."""
        events = await self.get_events(since=since)
        return [
            e for e in events
            if e.event_type == ComputerUseEventType.BATCH_COMPLETED
            and e.batch_size >= min_batch_size
        ]

    async def get_omniparser_events(
        self,
        since: Optional[datetime] = None,
    ) -> List[ComputerUseEvent]:
        """Get events that used OmniParser."""
        events = await self.get_events(since=since)
        return [e for e in events if e.used_omniparser]

    async def get_parser_mode_breakdown(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Get breakdown of parser modes used.

        Returns:
            Dictionary with counts per parser mode (omniparser, claude_vision, ocr, etc.)
        """
        events = await self.get_events(since=since)

        mode_counts = {}
        for event in events:
            if event.vision_analysis and isinstance(event.vision_analysis, dict):
                mode = event.vision_analysis.get("parser_mode", "unknown")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

        return mode_counts

    async def get_optimization_metrics(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get Computer Use optimization metrics.

        Returns:
            Dictionary with optimization statistics
        """
        events = await self.get_events(since=since)

        total_actions = sum(e.batch_size for e in events)
        total_batches = sum(
            1 for e in events
            if e.event_type == ComputerUseEventType.BATCH_COMPLETED
        )
        total_time_saved_ms = sum(e.time_saved_ms for e in events)
        total_tokens_saved = sum(e.tokens_saved for e in events)

        omniparser_events = [e for e in events if e.used_omniparser]

        return {
            "total_events": len(events),
            "total_actions": total_actions,
            "total_batches": total_batches,
            "avg_batch_size": (
                total_actions / total_batches if total_batches > 0 else 0
            ),
            "total_time_saved_ms": total_time_saved_ms,
            "total_time_saved_seconds": total_time_saved_ms / 1000,
            "total_tokens_saved": total_tokens_saved,
            "omniparser_usage_count": len(omniparser_events),
            "omniparser_usage_percent": (
                len(omniparser_events) / len(events) * 100 if events else 0
            ),
            "time_window_hours": self.config.lookback_hours,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
        }

    async def get_jarvis_state(self) -> Optional[Dict[str, Any]]:
        """Read JARVIS Computer Use bridge state."""
        try:
            if self.config.state_file.exists():
                content = self.config.state_file.read_text()
                return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to read Computer Use state: {e}")
        return None

    def _filter_events(self, events: List[ComputerUseEvent]) -> List[ComputerUseEvent]:
        """Apply configured filters to events."""
        filtered = []

        for event in events:
            # Filter by event type
            if (
                self.config.include_event_types
                and event.event_type not in self.config.include_event_types
            ):
                continue

            # Filter by batch size
            if event.batch_size < self.config.min_batch_size:
                continue

            # Filter batches only
            if (
                self.config.only_batches
                and event.event_type != ComputerUseEventType.BATCH_COMPLETED
            ):
                continue

            # Filter OmniParser only
            if self.config.only_omniparser and not event.used_omniparser:
                continue

            filtered.append(event)

        return filtered

    async def _load_events(self) -> List[ComputerUseEvent]:
        """Load events from shared state file."""
        try:
            if not self.config.events_file.exists():
                return []

            content = self.config.events_file.read_text()
            events_data = json.loads(content)

            events = [
                ComputerUseEvent.from_dict(e)
                for e in events_data
            ]

            self._events_cache = events
            self._last_sync = datetime.now()

            logger.info(f"Loaded {len(events)} Computer Use events from JARVIS")
            return events

        except Exception as e:
            logger.warning(f"Failed to load Computer Use events: {e}")
            return []

    async def watch_for_events(
        self,
        callback: callable,
        interval_seconds: float = 5.0,
    ) -> None:
        """
        Watch for new Computer Use events and call callback.

        Args:
            callback: Async function to call with new events
            interval_seconds: Polling interval
        """
        last_event_count = 0

        while True:
            try:
                events = await self._load_events()

                # Check for new events
                if len(events) > last_event_count:
                    new_events = events[last_event_count:]
                    logger.info(f"Detected {len(new_events)} new Computer Use events")

                    if asyncio.iscoroutinefunction(callback):
                        await callback(new_events)
                    else:
                        callback(new_events)

                    last_event_count = len(events)

            except Exception as e:
                logger.error(f"Error watching Computer Use events: {e}")

            await asyncio.sleep(interval_seconds)


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    "ComputerUseConnector",
    "ComputerUseConnectorConfig",
    "ComputerUseEvent",
    "ComputerUseEventType",
    "ActionType",
]
