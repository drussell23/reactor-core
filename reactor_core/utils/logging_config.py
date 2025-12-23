"""
Structured logging configuration for Night Shift Training Engine.

Features:
- JSON structured logging for production
- Rich console logging for development
- Context propagation
- Performance metrics
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Context variables for log correlation
_run_id: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_stage: ContextVar[Optional[str]] = ContextVar("stage", default=None)
_extra_context: ContextVar[Dict[str, Any]] = ContextVar("extra_context", default={})

T = TypeVar("T")


def set_run_id(run_id: str) -> None:
    """Set the current run ID for log correlation."""
    _run_id.set(run_id)


def set_stage(stage: str) -> None:
    """Set the current pipeline stage."""
    _stage.set(stage)


def set_context(**kwargs: Any) -> None:
    """Set additional context fields."""
    current = _extra_context.get().copy()
    current.update(kwargs)
    _extra_context.set(current)


def clear_context() -> None:
    """Clear extra context."""
    _extra_context.set({})


class StructuredFormatter(logging.Formatter):
    """
    JSON structured log formatter for production.

    Output format:
    {
        "timestamp": "2024-12-22T02:15:30.123456Z",
        "level": "INFO",
        "logger": "reactor_core.training",
        "message": "Training started",
        "run_id": "nightshift-2024-12-22",
        "stage": "training",
        "extra": {...}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context vars
        run_id = _run_id.get()
        stage = _stage.get()
        extra = _extra_context.get()

        if run_id:
            log_data["run_id"] = run_id
        if stage:
            log_data["stage"] = stage
        if extra:
            log_data["context"] = extra

        # Add record extras (excluding standard fields)
        standard_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }

        record_extras = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_fields and not k.startswith("_")
        }
        if record_extras:
            log_data["extra"] = record_extras

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class RichConsoleFormatter(logging.Formatter):
    """
    Rich console formatter for development with colors and formatting.
    """

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        # Color code by level
        color = self.COLORS.get(record.levelname, "")

        # Build timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Build context string
        context_parts = []
        run_id = _run_id.get()
        stage = _stage.get()

        if run_id:
            context_parts.append(f"run={run_id[:12]}")
        if stage:
            context_parts.append(f"stage={stage}")

        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format message
        message = record.getMessage()

        # Build output
        output = (
            f"{self.DIM}{timestamp}{self.RESET} "
            f"{color}{self.BOLD}{record.levelname:8}{self.RESET} "
            f"{self.DIM}{record.name}{self.RESET}"
            f"{context_str}: "
            f"{message}"
        )

        # Add exception if present
        if record.exc_info:
            output += "\n" + self.formatException(record.exc_info)

        return output


@dataclass
class LoggingConfig:
    """Logging configuration."""

    # Output settings
    level: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_LOG_LEVEL", "INFO")
    )
    format: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_LOG_FORMAT", "rich")
    )  # "rich" or "json"

    # File logging
    log_file: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("NIGHTSHIFT_LOG_FILE", ""))
        if os.getenv("NIGHTSHIFT_LOG_FILE") else None
    )
    max_file_size_mb: int = 50
    backup_count: int = 5

    # Console settings
    console_enabled: bool = True
    console_rich_tracebacks: bool = True

    # Filtering
    quiet_loggers: list = field(
        default_factory=lambda: [
            "urllib3",
            "httpx",
            "httpcore",
            "openai",
            "anthropic",
        ]
    )


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure logging for Night Shift.

    Args:
        config: Logging configuration. Uses defaults if None.
    """
    if config is None:
        config = LoggingConfig()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stderr)

        if config.format == "json":
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(RichConsoleFormatter())

        root_logger.addHandler(console_handler)

    # File handler
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
        )
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Quiet noisy loggers
    for logger_name in config.quiet_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set reactor_core logger level
    logging.getLogger("reactor_core").setLevel(
        getattr(logging, config.level.upper())
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for temporary log context.

    Example:
        with LogContext(operation="training", model="llama-3"):
            logger.info("Started")  # Includes operation and model
        logger.info("Done")  # No longer includes operation and model
    """

    def __init__(self, **kwargs: Any):
        self._context = kwargs
        self._previous: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        self._previous = _extra_context.get().copy()
        new_context = self._previous.copy()
        new_context.update(self._context)
        _extra_context.set(new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _extra_context.set(self._previous)


def log_duration(
    logger: logging.Logger,
    level: int = logging.INFO,
    message: str = "Operation completed",
) -> Callable:
    """
    Decorator to log function duration.

    Example:
        @log_duration(logger, message="Training step")
        async def train_step():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{message} ({duration:.2f}ms)",
                    extra={"duration_ms": duration, "function": func.__name__},
                )
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{message} failed ({duration:.2f}ms): {e}",
                    extra={
                        "duration_ms": duration,
                        "function": func.__name__,
                        "error": str(e),
                    },
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{message} ({duration:.2f}ms)",
                    extra={"duration_ms": duration, "function": func.__name__},
                )
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{message} failed ({duration:.2f}ms): {e}",
                    extra={
                        "duration_ms": duration,
                        "function": func.__name__,
                        "error": str(e),
                    },
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class MetricsLogger:
    """
    Logger for training metrics with automatic aggregation.

    Example:
        metrics = MetricsLogger(logger)
        metrics.log("loss", 0.5)
        metrics.log("accuracy", 0.85)
        metrics.flush()  # Logs aggregated metrics
    """

    def __init__(
        self,
        logger: logging.Logger,
        flush_interval: int = 100,
        prefix: str = "metrics",
    ):
        self.logger = logger
        self.flush_interval = flush_interval
        self.prefix = prefix

        self._metrics: Dict[str, list] = {}
        self._count = 0

    def log(self, name: str, value: float) -> None:
        """Log a metric value."""
        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append(value)
        self._count += 1

        if self._count >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        """Flush and log aggregated metrics."""
        if not self._metrics:
            return

        aggregated = {}
        for name, values in self._metrics.items():
            if values:
                aggregated[f"{name}_mean"] = sum(values) / len(values)
                aggregated[f"{name}_min"] = min(values)
                aggregated[f"{name}_max"] = max(values)
                aggregated[f"{name}_count"] = len(values)

        self.logger.info(
            f"{self.prefix}: {aggregated}",
            extra={"metrics": aggregated},
        )

        self._metrics.clear()
        self._count = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get current unflushed metrics."""
        aggregated = {}
        for name, values in self._metrics.items():
            if values:
                aggregated[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "values": values,
                }
        return aggregated
