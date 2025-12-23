"""
Cron-based pipeline scheduling.

Provides:
- Cron expression scheduling
- Async scheduler loop
- Run history tracking
- Schedule management
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for the scheduler."""
    # Cron expression (default: Sunday 2 AM)
    cron_expression: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_SCHEDULE",
            "0 2 * * 0"  # Sunday 2 AM
        )
    )

    # Timezone
    timezone: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_TIMEZONE", "UTC")
    )

    # Retry configuration
    max_retries: int = 3
    retry_delay_minutes: int = 30

    # History
    history_file: Optional[Path] = None
    max_history: int = 100

    # Control
    enabled: bool = True
    run_on_start: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cron_expression": self.cron_expression,
            "timezone": self.timezone,
            "max_retries": self.max_retries,
            "retry_delay_minutes": self.retry_delay_minutes,
            "history_file": str(self.history_file) if self.history_file else None,
            "max_history": self.max_history,
            "enabled": self.enabled,
            "run_on_start": self.run_on_start,
        }


@dataclass
class ScheduledRun:
    """Record of a scheduled run."""
    run_id: str
    scheduled_time: datetime
    actual_start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: bool = False
    retry_count: int = 0
    error: Optional[str] = None
    result_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.actual_start_time and self.end_time:
            return (self.end_time - self.actual_start_time).total_seconds()
        return 0.0

    @property
    def status(self) -> str:
        if self.end_time is None:
            if self.actual_start_time:
                return "running"
            return "pending"
        return "success" if self.success else "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scheduled_time": self.scheduled_time.isoformat(),
            "actual_start_time": (
                self.actual_start_time.isoformat()
                if self.actual_start_time else None
            ),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "retry_count": self.retry_count,
            "error": self.error,
            "result_summary": self.result_summary,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledRun":
        return cls(
            run_id=data["run_id"],
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            actual_start_time=(
                datetime.fromisoformat(data["actual_start_time"])
                if data.get("actual_start_time") else None
            ),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time") else None
            ),
            success=data.get("success", False),
            retry_count=data.get("retry_count", 0),
            error=data.get("error"),
            result_summary=data.get("result_summary", {}),
        )


class PipelineScheduler:
    """
    Cron-based scheduler for the Night Shift pipeline.

    Runs the training pipeline on a schedule with retry support.
    """

    def __init__(
        self,
        pipeline_runner: Callable,
        config: Optional[ScheduleConfig] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            pipeline_runner: Async function to run the pipeline
            config: Scheduler configuration
        """
        self.pipeline_runner = pipeline_runner
        self.config = config or ScheduleConfig()

        self._running = False
        self._current_run: Optional[ScheduledRun] = None
        self._history: List[ScheduledRun] = []
        self._task: Optional[asyncio.Task] = None

        # Load history
        if self.config.history_file and self.config.history_file.exists():
            self._load_history()

    def _get_next_run_time(self) -> datetime:
        """Calculate next run time from cron expression."""
        try:
            from croniter import croniter
        except ImportError:
            raise ImportError(
                "croniter required for scheduling. Install with: pip install croniter"
            )

        now = datetime.now()
        cron = croniter(self.config.cron_expression, now)
        return cron.get_next(datetime)

    def _load_history(self) -> None:
        """Load run history from file."""
        try:
            with open(self.config.history_file) as f:
                data = json.load(f)
            self._history = [ScheduledRun.from_dict(r) for r in data]
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save run history to file."""
        if not self.config.history_file:
            return

        # Trim history
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

        self.config.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.history_file, "w") as f:
            json.dump([r.to_dict() for r in self._history], f, indent=2)

    async def _execute_run(self, scheduled_run: ScheduledRun) -> None:
        """Execute a scheduled pipeline run."""
        scheduled_run.actual_start_time = datetime.now()
        self._current_run = scheduled_run

        logger.info(f"Starting scheduled run: {scheduled_run.run_id}")

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self.pipeline_runner()

                scheduled_run.end_time = datetime.now()
                scheduled_run.success = result.get("success", False)
                scheduled_run.result_summary = result
                scheduled_run.retry_count = attempt

                if scheduled_run.success:
                    logger.info(f"Scheduled run completed: {scheduled_run.run_id}")
                    break
                else:
                    scheduled_run.error = result.get("error", "Unknown error")

            except Exception as e:
                scheduled_run.error = str(e)
                logger.error(f"Run failed (attempt {attempt + 1}): {e}")

            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_minutes * 60
                logger.info(f"Retrying in {self.config.retry_delay_minutes} minutes...")
                await asyncio.sleep(delay)

        if not scheduled_run.end_time:
            scheduled_run.end_time = datetime.now()

        self._history.append(scheduled_run)
        self._save_history()
        self._current_run = None

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info(f"Scheduler started with cron: {self.config.cron_expression}")

        if self.config.run_on_start:
            run = ScheduledRun(
                run_id=f"startup-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                scheduled_time=datetime.now(),
            )
            await self._execute_run(run)

        while self._running:
            try:
                next_run = self._get_next_run_time()
                wait_seconds = (next_run - datetime.now()).total_seconds()

                if wait_seconds > 0:
                    logger.info(
                        f"Next run scheduled for: {next_run.isoformat()} "
                        f"(in {wait_seconds / 3600:.1f} hours)"
                    )
                    await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                run = ScheduledRun(
                    run_id=f"sched-{next_run.strftime('%Y%m%d%H%M%S')}",
                    scheduled_time=next_run,
                )
                await self._execute_run(run)

            except asyncio.CancelledError:
                logger.info("Scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        if not self.config.enabled:
            logger.info("Scheduler is disabled")
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Scheduler stopped")

    async def trigger_now(self) -> ScheduledRun:
        """Trigger an immediate run."""
        run = ScheduledRun(
            run_id=f"manual-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            scheduled_time=datetime.now(),
        )
        await self._execute_run(run)
        return run

    def get_next_run_time(self) -> datetime:
        """Get the next scheduled run time."""
        return self._get_next_run_time()

    def get_current_run(self) -> Optional[ScheduledRun]:
        """Get currently executing run."""
        return self._current_run

    def get_history(
        self,
        limit: int = 10,
        success_only: bool = False,
    ) -> List[ScheduledRun]:
        """Get run history."""
        history = self._history
        if success_only:
            history = [r for r in history if r.success]
        return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        if not self._history:
            return {"total_runs": 0}

        successful = [r for r in self._history if r.success]
        failed = [r for r in self._history if not r.success]

        durations = [r.duration_seconds for r in self._history if r.duration_seconds > 0]

        return {
            "total_runs": len(self._history),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "success_rate": len(successful) / len(self._history),
            "avg_duration_minutes": (
                sum(durations) / len(durations) / 60 if durations else 0
            ),
            "last_run": self._history[-1].to_dict() if self._history else None,
            "next_run": self._get_next_run_time().isoformat() if self._running else None,
        }

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Convenience exports
__all__ = [
    "PipelineScheduler",
    "ScheduleConfig",
    "ScheduledRun",
]
