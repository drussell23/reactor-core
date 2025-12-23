"""
Training callbacks for Night Shift Training Engine.

Provides:
- Async-compatible progress callbacks
- Real-time metrics streaming
- Checkpoint management callbacks
- Early stopping with patience
- GPU memory monitoring
- Rich console progress display
- Webhook notifications
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TrainingEvent:
    """Event emitted during training."""
    event_type: str
    step: int
    epoch: float
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class BaseCallback(ABC):
    """
    Abstract base class for training callbacks.

    Callbacks receive events during training and can perform
    actions like logging, checkpointing, or early stopping.
    """

    @abstractmethod
    def on_train_begin(self, total_steps: int, **kwargs) -> None:
        """Called at the start of training."""
        pass

    @abstractmethod
    def on_train_end(self, **kwargs) -> None:
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_step_begin(self, step: int, **kwargs) -> None:
        """Called at the start of each training step."""
        pass

    @abstractmethod
    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        """Called at the end of each training step."""
        pass

    @abstractmethod
    def on_evaluate(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called after evaluation."""
        pass

    @abstractmethod
    def on_save(self, step: int, checkpoint_path: Path, **kwargs) -> None:
        """Called after saving a checkpoint."""
        pass

    @abstractmethod
    def should_stop(self) -> bool:
        """Check if training should stop early."""
        pass


class CallbackMixin:
    """Default implementations for callback methods."""

    def on_train_begin(self, total_steps: int, **kwargs) -> None:
        pass

    def on_train_end(self, **kwargs) -> None:
        pass

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        pass

    def on_step_begin(self, step: int, **kwargs) -> None:
        pass

    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        pass

    def on_evaluate(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        pass

    def on_save(self, step: int, checkpoint_path: Path, **kwargs) -> None:
        pass

    def should_stop(self) -> bool:
        return False


class ProgressCallback(CallbackMixin, BaseCallback):
    """
    Training progress callback with async event streaming.

    Emits TrainingEvents that can be consumed by async handlers.
    """

    def __init__(
        self,
        event_handler: Optional[Callable[[TrainingEvent], None]] = None,
        async_event_handler: Optional[Callable[[TrainingEvent], Any]] = None,
        log_interval: int = 10,
    ):
        """
        Initialize progress callback.

        Args:
            event_handler: Sync handler for events
            async_event_handler: Async handler for events
            log_interval: Log every N steps
        """
        self.event_handler = event_handler
        self.async_event_handler = async_event_handler
        self.log_interval = log_interval

        self._total_steps = 0
        self._current_step = 0
        self._current_epoch = 0.0
        self._start_time: Optional[float] = None
        self._step_times: List[float] = []
        self._losses: List[float] = []

    def _emit_event(self, event: TrainingEvent) -> None:
        """Emit an event to handlers."""
        if self.event_handler:
            self.event_handler(event)

        if self.async_event_handler:
            # Schedule async handler
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.async_event_handler(event))
            except RuntimeError:
                # No event loop running, skip async handler
                pass

    def on_train_begin(self, total_steps: int, **kwargs) -> None:
        self._total_steps = total_steps
        self._start_time = time.time()

        event = TrainingEvent(
            event_type="train_begin",
            step=0,
            epoch=0.0,
            metadata={"total_steps": total_steps, **kwargs},
        )
        self._emit_event(event)
        logger.info(f"Training started: {total_steps} total steps")

    def on_train_end(self, **kwargs) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0
        avg_loss = sum(self._losses) / len(self._losses) if self._losses else 0

        event = TrainingEvent(
            event_type="train_end",
            step=self._current_step,
            epoch=self._current_epoch,
            metrics={"final_loss": avg_loss, "total_time_seconds": elapsed},
            metadata=kwargs,
        )
        self._emit_event(event)
        logger.info(f"Training completed in {elapsed:.1f}s, final loss: {avg_loss:.4f}")

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        self._current_epoch = float(epoch)
        event = TrainingEvent(
            event_type="epoch_begin",
            step=self._current_step,
            epoch=float(epoch),
            metadata=kwargs,
        )
        self._emit_event(event)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        event = TrainingEvent(
            event_type="epoch_end",
            step=self._current_step,
            epoch=float(epoch),
            metrics=metrics,
            metadata=kwargs,
        )
        self._emit_event(event)
        logger.info(f"Epoch {epoch} completed: {metrics}")

    def on_step_begin(self, step: int, **kwargs) -> None:
        self._step_start = time.time()

    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        self._current_step = step
        self._losses.append(loss)

        # Track step time
        step_time = time.time() - getattr(self, "_step_start", time.time())
        self._step_times.append(step_time)

        # Calculate progress
        progress = step / self._total_steps if self._total_steps > 0 else 0
        avg_step_time = sum(self._step_times[-100:]) / len(self._step_times[-100:])
        remaining_steps = self._total_steps - step
        eta_seconds = remaining_steps * avg_step_time

        # Update epoch estimate
        self._current_epoch = step / (self._total_steps / kwargs.get("num_epochs", 1))

        event = TrainingEvent(
            event_type="step_end",
            step=step,
            epoch=self._current_epoch,
            metrics={
                "loss": loss,
                "learning_rate": learning_rate,
                "step_time": step_time,
                "progress": progress,
                "eta_seconds": eta_seconds,
            },
            metadata=kwargs,
        )
        self._emit_event(event)

        # Log periodically
        if step % self.log_interval == 0:
            logger.info(
                f"Step {step}/{self._total_steps} ({progress:.1%}) | "
                f"Loss: {loss:.4f} | LR: {learning_rate:.2e} | "
                f"ETA: {eta_seconds / 60:.1f}min"
            )

    def on_evaluate(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        event = TrainingEvent(
            event_type="evaluate",
            step=step,
            epoch=self._current_epoch,
            metrics=metrics,
            metadata=kwargs,
        )
        self._emit_event(event)
        logger.info(f"Evaluation at step {step}: {metrics}")

    def on_save(self, step: int, checkpoint_path: Path, **kwargs) -> None:
        event = TrainingEvent(
            event_type="checkpoint_saved",
            step=step,
            epoch=self._current_epoch,
            metadata={"checkpoint_path": str(checkpoint_path), **kwargs},
        )
        self._emit_event(event)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def get_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        avg_loss = sum(self._losses[-100:]) / len(self._losses[-100:]) if self._losses else 0

        return {
            "current_step": self._current_step,
            "total_steps": self._total_steps,
            "progress": self._current_step / self._total_steps if self._total_steps > 0 else 0,
            "current_epoch": self._current_epoch,
            "elapsed_seconds": elapsed,
            "average_loss": avg_loss,
        }


class EarlyStoppingCallback(CallbackMixin, BaseCallback):
    """
    Early stopping callback with patience.

    Monitors a metric and stops training if no improvement
    is seen for `patience` evaluations.
    """

    def __init__(
        self,
        monitor: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of evals without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" - whether lower or higher is better
            restore_best: Whether to restore best checkpoint on stop
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self._best_value: Optional[float] = None
        self._best_step: int = 0
        self._best_checkpoint: Optional[Path] = None
        self._counter = 0
        self._should_stop = False

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self._best_value is None:
            return True

        if self.mode == "min":
            return current < self._best_value - self.min_delta
        else:
            return current > self._best_value + self.min_delta

    def on_evaluate(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        if self.monitor not in metrics:
            logger.warning(f"Early stopping: metric '{self.monitor}' not found in metrics")
            return

        current = metrics[self.monitor]

        if self._is_improvement(current):
            self._best_value = current
            self._best_step = step
            self._counter = 0
            logger.info(
                f"Early stopping: new best {self.monitor}={current:.4f} at step {step}"
            )
        else:
            self._counter += 1
            logger.info(
                f"Early stopping: no improvement for {self._counter}/{self.patience} evals "
                f"(best: {self._best_value:.4f} at step {self._best_step})"
            )

            if self._counter >= self.patience:
                self._should_stop = True
                logger.info(
                    f"Early stopping triggered at step {step}. "
                    f"Best {self.monitor}={self._best_value:.4f} at step {self._best_step}"
                )

    def on_save(self, step: int, checkpoint_path: Path, **kwargs) -> None:
        if step == self._best_step:
            self._best_checkpoint = checkpoint_path

    def should_stop(self) -> bool:
        return self._should_stop

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        return self._best_checkpoint

    def get_best_value(self) -> Optional[float]:
        """Get best metric value."""
        return self._best_value


class CheckpointCallback(CallbackMixin, BaseCallback):
    """
    Checkpoint management callback.

    Handles saving and cleanup of training checkpoints.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        save_steps: int = 500,
        max_checkpoints: int = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ):
        """
        Initialize checkpoint callback.

        Args:
            output_dir: Directory for checkpoints
            save_steps: Save every N steps
            max_checkpoints: Maximum checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.output_dir = Path(output_dir)
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        self._checkpoints: List[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        if step > 0 and step % self.save_steps == 0:
            self._request_save(step, kwargs.get("trainer"))

    def _request_save(self, step: int, trainer: Any) -> None:
        """Request checkpoint save from trainer."""
        if trainer is not None and hasattr(trainer, "save_checkpoint"):
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
            trainer.save_checkpoint(checkpoint_dir)

    def on_save(self, step: int, checkpoint_path: Path, **kwargs) -> None:
        self._checkpoints.append(checkpoint_path)

        # Cleanup old checkpoints
        while len(self._checkpoints) > self.max_checkpoints:
            old_checkpoint = self._checkpoints.pop(0)
            self._remove_checkpoint(old_checkpoint)

    def _remove_checkpoint(self, checkpoint_path: Path) -> None:
        """Remove a checkpoint directory."""
        if checkpoint_path.exists():
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed old checkpoint: {checkpoint_path}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if self._checkpoints:
            return self._checkpoints[-1]
        return None


class GPUMemoryCallback(CallbackMixin, BaseCallback):
    """
    GPU memory monitoring callback.

    Tracks memory usage and can trigger actions on high usage.
    """

    def __init__(
        self,
        log_interval: int = 100,
        warning_threshold: float = 0.9,
        clear_cache_threshold: float = 0.95,
    ):
        """
        Initialize GPU memory callback.

        Args:
            log_interval: Log memory every N steps
            warning_threshold: Log warning above this usage ratio
            clear_cache_threshold: Clear cache above this usage ratio
        """
        self.log_interval = log_interval
        self.warning_threshold = warning_threshold
        self.clear_cache_threshold = clear_cache_threshold

        self._peak_memory: float = 0.0
        self._memory_history: List[float] = []

    def _get_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get current GPU memory usage."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None

            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_ratio = allocated / total

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "usage_ratio": usage_ratio,
            }
        except Exception:
            return None

    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        if step % self.log_interval != 0:
            return

        memory = self._get_memory_usage()
        if memory is None:
            return

        self._memory_history.append(memory["allocated_gb"])
        self._peak_memory = max(self._peak_memory, memory["allocated_gb"])

        # Log periodically
        logger.debug(
            f"GPU Memory: {memory['allocated_gb']:.2f}GB / {memory['total_gb']:.2f}GB "
            f"({memory['usage_ratio']:.1%})"
        )

        # Warning on high usage
        if memory["usage_ratio"] > self.warning_threshold:
            logger.warning(
                f"High GPU memory usage: {memory['usage_ratio']:.1%} "
                f"({memory['allocated_gb']:.2f}GB / {memory['total_gb']:.2f}GB)"
            )

        # Clear cache on very high usage
        if memory["usage_ratio"] > self.clear_cache_threshold:
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache due to high memory usage")
            except Exception:
                pass

    def on_train_end(self, **kwargs) -> None:
        logger.info(f"Peak GPU memory usage: {self._peak_memory:.2f}GB")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self._memory_history:
            return {}

        return {
            "peak_memory_gb": self._peak_memory,
            "average_memory_gb": sum(self._memory_history) / len(self._memory_history),
            "current_memory": self._get_memory_usage(),
        }


class RichProgressCallback(CallbackMixin, BaseCallback):
    """
    Rich console progress display callback.

    Uses the Rich library for beautiful progress bars and tables.
    """

    def __init__(
        self,
        refresh_rate: int = 10,
        show_metrics: bool = True,
        show_gpu_memory: bool = True,
    ):
        """
        Initialize Rich progress callback.

        Args:
            refresh_rate: Refresh every N steps
            show_metrics: Show training metrics
            show_gpu_memory: Show GPU memory usage
        """
        self.refresh_rate = refresh_rate
        self.show_metrics = show_metrics
        self.show_gpu_memory = show_gpu_memory

        self._progress = None
        self._task_id = None
        self._metrics_table = None
        self._total_steps = 0
        self._current_metrics: Dict[str, float] = {}

    def _setup_progress(self, total_steps: int) -> None:
        """Setup Rich progress bar."""
        try:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                TextColumn,
                BarColumn,
                TaskProgressColumn,
                TimeRemainingColumn,
                TimeElapsedColumn,
            )
            from rich.console import Console
            from rich.live import Live
            from rich.table import Table

            self._console = Console()
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("| Loss: {task.fields[loss]:.4f}"),
                TextColumn("| LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
                refresh_per_second=4,
            )
            self._task_id = self._progress.add_task(
                "Training",
                total=total_steps,
                loss=0.0,
                lr=0.0,
            )
            self._progress.start()
        except ImportError:
            logger.warning("Rich not installed, falling back to basic progress")
            self._progress = None

    def on_train_begin(self, total_steps: int, **kwargs) -> None:
        self._total_steps = total_steps
        self._setup_progress(total_steps)

    def on_train_end(self, **kwargs) -> None:
        if self._progress:
            self._progress.stop()

    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=step,
                loss=loss,
                lr=learning_rate,
            )

        self._current_metrics["loss"] = loss
        self._current_metrics["learning_rate"] = learning_rate

    def on_evaluate(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        self._current_metrics.update(metrics)

        if self._progress:
            # Print evaluation results
            try:
                from rich.table import Table
                from rich.panel import Panel

                table = Table(title=f"Evaluation at Step {step}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                for name, value in metrics.items():
                    table.add_row(name, f"{value:.4f}")

                self._console.print(table)
            except ImportError:
                pass


class WebhookCallback(CallbackMixin, BaseCallback):
    """
    Webhook notification callback.

    Sends training events to external services (Slack, Discord, etc.)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        notify_on: Optional[List[str]] = None,
        include_metrics: bool = True,
    ):
        """
        Initialize webhook callback.

        Args:
            webhook_url: Webhook URL (or use NIGHTSHIFT_WEBHOOK_URL env var)
            notify_on: Event types to notify on (default: train_begin, train_end, evaluate)
            include_metrics: Include metrics in notifications
        """
        self.webhook_url = webhook_url or os.getenv("NIGHTSHIFT_WEBHOOK_URL")
        self.notify_on = notify_on or ["train_begin", "train_end", "evaluate"]
        self.include_metrics = include_metrics

        self._session = None

    async def _send_webhook(self, payload: Dict[str, Any]) -> None:
        """Send webhook notification."""
        if not self.webhook_url:
            return

        try:
            import aiohttp

            if self._session is None:
                self._session = aiohttp.ClientSession()

            async with self._session.post(
                self.webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    logger.warning(f"Webhook failed: {response.status}")
        except Exception as e:
            logger.warning(f"Webhook error: {e}")

    def _schedule_webhook(self, event_type: str, data: Dict[str, Any]) -> None:
        """Schedule webhook send."""
        if event_type not in self.notify_on:
            return

        payload = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_webhook(payload))
        except RuntimeError:
            # No event loop, skip
            pass

    def on_train_begin(self, total_steps: int, **kwargs) -> None:
        self._schedule_webhook("train_begin", {"total_steps": total_steps, **kwargs})

    def on_train_end(self, **kwargs) -> None:
        self._schedule_webhook("train_end", kwargs)

    def on_evaluate(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        data = {"step": step}
        if self.include_metrics:
            data["metrics"] = metrics
        self._schedule_webhook("evaluate", data)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()


class CompositeCallback(BaseCallback):
    """
    Composite callback that combines multiple callbacks.

    All child callbacks are called for each event.
    """

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        """
        Initialize composite callback.

        Args:
            callbacks: List of callbacks to combine
        """
        self.callbacks = callbacks or []

    def add(self, callback: BaseCallback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def remove(self, callback: BaseCallback) -> None:
        """Remove a callback."""
        self.callbacks.remove(callback)

    def on_train_begin(self, total_steps: int, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(total_steps, **kwargs)

    def on_train_end(self, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_train_end(**kwargs)

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics, **kwargs)

    def on_step_begin(self, step: int, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_step_begin(step, **kwargs)

    def on_step_end(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        for cb in self.callbacks:
            cb.on_step_end(step, loss, learning_rate, **kwargs)

    def on_evaluate(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_evaluate(step, metrics, **kwargs)

    def on_save(self, step: int, checkpoint_path: Path, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_save(step, checkpoint_path, **kwargs)

    def should_stop(self) -> bool:
        """Returns True if any callback wants to stop."""
        return any(cb.should_stop() for cb in self.callbacks)


def create_default_callbacks(
    output_dir: Union[str, Path],
    log_interval: int = 10,
    save_steps: int = 500,
    max_checkpoints: int = 3,
    early_stopping: bool = True,
    patience: int = 3,
    use_rich: bool = True,
    webhook_url: Optional[str] = None,
) -> CompositeCallback:
    """
    Create a default set of callbacks for training.

    Args:
        output_dir: Directory for checkpoints
        log_interval: Log every N steps
        save_steps: Save checkpoint every N steps
        max_checkpoints: Maximum checkpoints to keep
        early_stopping: Enable early stopping
        patience: Early stopping patience
        use_rich: Use Rich progress display
        webhook_url: Optional webhook URL

    Returns:
        CompositeCallback with all configured callbacks
    """
    callbacks = CompositeCallback()

    # Progress tracking
    callbacks.add(ProgressCallback(log_interval=log_interval))

    # Checkpointing
    callbacks.add(CheckpointCallback(
        output_dir=output_dir,
        save_steps=save_steps,
        max_checkpoints=max_checkpoints,
    ))

    # GPU memory monitoring
    callbacks.add(GPUMemoryCallback(log_interval=log_interval * 10))

    # Early stopping
    if early_stopping:
        callbacks.add(EarlyStoppingCallback(patience=patience))

    # Rich progress (optional)
    if use_rich:
        try:
            import rich
            callbacks.add(RichProgressCallback())
        except ImportError:
            pass

    # Webhook notifications (optional)
    if webhook_url:
        callbacks.add(WebhookCallback(webhook_url=webhook_url))

    return callbacks


# Convenience exports
__all__ = [
    # Events
    "TrainingEvent",
    # Base classes
    "BaseCallback",
    "CallbackMixin",
    # Callbacks
    "ProgressCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "GPUMemoryCallback",
    "RichProgressCallback",
    "WebhookCallback",
    "CompositeCallback",
    # Factory
    "create_default_callbacks",
]
