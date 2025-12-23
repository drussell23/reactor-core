"""
Utilities module for Night Shift Training Engine.

Provides:
- Environment detection (M1 Mac, GCP VM)
- Async helpers (rate limiting, batch processing)
- Structured logging configuration
"""

from reactor_core.utils.environment import (
    detect_environment,
    EnvironmentType,
    EnvironmentInfo,
    get_recommended_config,
    get_quantization_config,
    print_environment_info,
)

from reactor_core.utils.async_helpers import (
    AsyncSemaphore,
    TokenBucketRateLimiter,
    ParallelBatchProcessor,
    BatchResult,
    AsyncQueue,
    ProgressTracker,
    async_retry,
    gather_with_concurrency,
    run_with_timeout,
)

from reactor_core.utils.logging_config import (
    setup_logging,
    get_logger,
    LoggingConfig,
    LogContext,
    MetricsLogger,
    set_run_id,
    set_stage,
    set_context,
    clear_context,
    log_duration,
)

__all__ = [
    # Environment
    "detect_environment",
    "EnvironmentType",
    "EnvironmentInfo",
    "get_recommended_config",
    "get_quantization_config",
    "print_environment_info",
    # Async helpers
    "AsyncSemaphore",
    "TokenBucketRateLimiter",
    "ParallelBatchProcessor",
    "BatchResult",
    "AsyncQueue",
    "ProgressTracker",
    "async_retry",
    "gather_with_concurrency",
    "run_with_timeout",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggingConfig",
    "LogContext",
    "MetricsLogger",
    "set_run_id",
    "set_stage",
    "set_context",
    "clear_context",
    "log_duration",
]
