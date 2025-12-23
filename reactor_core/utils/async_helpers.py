"""
Async utility helpers for Night Shift Training Engine.

Provides:
- AsyncSemaphore: Bounded concurrency control
- TokenBucketRateLimiter: Rate limiting for API calls
- ParallelBatchProcessor: Parallel async batch processing
- AsyncRetry: Retry decorator with exponential backoff
- AsyncQueue: Bounded async producer/consumer queue
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class AsyncSemaphore:
    """
    Enhanced async semaphore with metrics and timeout support.

    Example:
        semaphore = AsyncSemaphore(max_concurrent=10)

        async with semaphore:
            await do_work()

        # Or with timeout
        async with semaphore.acquire_with_timeout(5.0):
            await do_work()
    """

    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._current_count = 0
        self._total_acquisitions = 0
        self._total_wait_time = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncSemaphore":
        start = time.monotonic()
        await self._semaphore.acquire()
        wait_time = time.monotonic() - start

        async with self._lock:
            self._current_count += 1
            self._total_acquisitions += 1
            self._total_wait_time += wait_time

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        async with self._lock:
            self._current_count -= 1
        self._semaphore.release()

    async def acquire_with_timeout(
        self, timeout: float
    ) -> "AsyncSemaphoreContext":
        """Acquire with timeout, raising TimeoutError if exceeded."""
        return AsyncSemaphoreContext(self, timeout)

    @property
    def current_usage(self) -> int:
        """Current number of acquired permits."""
        return self._current_count

    @property
    def available(self) -> int:
        """Number of available permits."""
        return self._max_concurrent - self._current_count

    def get_metrics(self) -> Dict[str, Any]:
        """Get semaphore metrics."""
        return {
            "max_concurrent": self._max_concurrent,
            "current_usage": self._current_count,
            "available": self.available,
            "total_acquisitions": self._total_acquisitions,
            "avg_wait_time": (
                self._total_wait_time / self._total_acquisitions
                if self._total_acquisitions > 0
                else 0.0
            ),
        }


class AsyncSemaphoreContext:
    """Context manager for semaphore with timeout."""

    def __init__(self, semaphore: AsyncSemaphore, timeout: float):
        self._semaphore = semaphore
        self._timeout = timeout

    async def __aenter__(self) -> AsyncSemaphore:
        try:
            await asyncio.wait_for(
                self._semaphore.__aenter__(),
                timeout=self._timeout,
            )
            return self._semaphore
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Failed to acquire semaphore within {self._timeout}s"
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._semaphore.__aexit__(exc_type, exc_val, exc_tb)


@dataclass
class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls.

    Supports:
    - Configurable requests per minute/second
    - Burst handling
    - Async-safe
    - Metrics tracking

    Example:
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        await limiter.acquire()  # Blocks if rate exceeded
        response = await api_call()
    """

    requests_per_minute: int = 60
    burst_size: Optional[int] = None  # Defaults to requests_per_minute

    # Internal state
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _total_requests: int = field(init=False, default=0)
    _total_wait_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()

    @property
    def rate(self) -> float:
        """Tokens per second."""
        return self.requests_per_minute / 60.0

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1).
        """
        async with self._lock:
            await self._wait_for_tokens(tokens)
            self._tokens -= tokens
            self._total_requests += tokens

    async def _wait_for_tokens(self, needed: int) -> None:
        """Wait until enough tokens are available."""
        while True:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.rate,
            )

            if self._tokens >= needed:
                return

            # Calculate wait time
            deficit = needed - self._tokens
            wait_time = deficit / self.rate

            self._total_wait_time += wait_time
            await asyncio.sleep(wait_time)

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Returns:
            True if tokens acquired, False otherwise.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.rate,
            )

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_requests += tokens
                return True

            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
            "current_tokens": self._tokens,
            "total_requests": self._total_requests,
            "total_wait_time": self._total_wait_time,
            "avg_wait_time": (
                self._total_wait_time / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }


@dataclass
class BatchResult(Generic[T, R]):
    """Result from batch processing."""

    item: T
    result: Optional[R] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None


class ParallelBatchProcessor(Generic[T, R]):
    """
    Process items in parallel batches with rate limiting and error handling.

    Example:
        processor = ParallelBatchProcessor(
            process_fn=api_call,
            max_concurrent=10,
            rate_limiter=TokenBucketRateLimiter(60),
        )

        async for result in processor.process(items):
            if result.success:
                print(result.result)
            else:
                print(f"Error: {result.error}")
    """

    def __init__(
        self,
        process_fn: Callable[[T], Awaitable[R]],
        max_concurrent: int = 10,
        batch_size: int = 100,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        retry_attempts: int = 0,
        retry_delay: float = 1.0,
        on_error: Optional[Callable[[T, Exception], Awaitable[None]]] = None,
    ):
        """
        Initialize batch processor.

        Args:
            process_fn: Async function to process each item.
            max_concurrent: Max concurrent processing tasks.
            batch_size: Items per batch for yielding results.
            rate_limiter: Optional rate limiter for API calls.
            retry_attempts: Number of retry attempts on failure.
            retry_delay: Delay between retries (exponential backoff).
            on_error: Optional error handler callback.
        """
        self.process_fn = process_fn
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.rate_limiter = rate_limiter
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.on_error = on_error

        self._semaphore = AsyncSemaphore(max_concurrent)
        self._processed_count = 0
        self._success_count = 0
        self._error_count = 0

    async def _process_single(self, item: T) -> BatchResult[T, R]:
        """Process a single item with rate limiting and retries."""
        start = time.monotonic()

        for attempt in range(self.retry_attempts + 1):
            try:
                # Rate limit
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                # Process with semaphore
                async with self._semaphore:
                    result = await self.process_fn(item)

                duration = (time.monotonic() - start) * 1000
                self._processed_count += 1
                self._success_count += 1

                return BatchResult(
                    item=item,
                    result=result,
                    duration_ms=duration,
                )

            except Exception as e:
                if attempt < self.retry_attempts:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.retry_attempts} after {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    duration = (time.monotonic() - start) * 1000
                    self._processed_count += 1
                    self._error_count += 1

                    if self.on_error:
                        await self.on_error(item, e)

                    return BatchResult(
                        item=item,
                        error=e,
                        duration_ms=duration,
                    )

        # Should never reach here, but satisfy type checker
        return BatchResult(item=item, error=Exception("Unexpected error"))

    async def process(
        self,
        items: Union[List[T], AsyncIterator[T]],
    ) -> AsyncIterator[BatchResult[T, R]]:
        """
        Process items in parallel batches.

        Args:
            items: List or async iterator of items to process.

        Yields:
            BatchResult for each processed item.
        """
        # Convert to list if needed for batching
        if isinstance(items, list):
            item_list = items
        else:
            item_list = [item async for item in items]

        # Process in batches
        for batch_start in range(0, len(item_list), self.batch_size):
            batch = item_list[batch_start : batch_start + self.batch_size]

            # Process batch in parallel
            tasks = [self._process_single(item) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    # This shouldn't happen since we catch exceptions in _process_single
                    yield BatchResult(
                        item=None,
                        error=result,
                    )
                else:
                    yield result

    async def process_all(
        self,
        items: Union[List[T], AsyncIterator[T]],
    ) -> List[BatchResult[T, R]]:
        """
        Process all items and return complete results.

        Args:
            items: Items to process.

        Returns:
            List of all BatchResults.
        """
        results = []
        async for result in self.process(items):
            results.append(result)
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        return {
            "processed_count": self._processed_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": (
                self._success_count / self._processed_count
                if self._processed_count > 0
                else 0.0
            ),
            "semaphore": self._semaphore.get_metrics(),
            "rate_limiter": (
                self.rate_limiter.get_metrics()
                if self.rate_limiter
                else None
            ),
        }


def async_retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for async retry with exponential backoff.

    Example:
        @async_retry(attempts=3, delay=1.0)
        async def api_call():
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{attempts} for {func.__name__} "
                            f"after {wait_time:.1f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)

            raise last_exception

        return wrapper

    return decorator


class AsyncQueue(Generic[T]):
    """
    Bounded async producer/consumer queue with backpressure.

    Example:
        queue = AsyncQueue(maxsize=100)

        # Producer
        await queue.put(item)

        # Consumer
        async for item in queue:
            process(item)

        # Or get single item
        item = await queue.get()
    """

    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue[Optional[T]] = asyncio.Queue(maxsize)
        self._closed = False
        self._put_count = 0
        self._get_count = 0

    async def put(self, item: T) -> None:
        """Put item in queue, blocking if full."""
        if self._closed:
            raise RuntimeError("Queue is closed")
        await self._queue.put(item)
        self._put_count += 1

    def put_nowait(self, item: T) -> None:
        """Put item without blocking, raises QueueFull if full."""
        if self._closed:
            raise RuntimeError("Queue is closed")
        self._queue.put_nowait(item)
        self._put_count += 1

    async def get(self) -> T:
        """Get item from queue, blocking if empty."""
        item = await self._queue.get()
        if item is None and self._closed:
            raise StopAsyncIteration()
        self._get_count += 1
        return item

    def get_nowait(self) -> T:
        """Get item without blocking, raises QueueEmpty if empty."""
        item = self._queue.get_nowait()
        if item is None and self._closed:
            raise StopAsyncIteration()
        self._get_count += 1
        return item

    def close(self) -> None:
        """Close queue. No more items can be added."""
        self._closed = True
        # Put sentinel to unblock consumers
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def __aiter__(self) -> "AsyncQueue[T]":
        return self

    async def __anext__(self) -> T:
        try:
            item = await self.get()
            return item
        except StopAsyncIteration:
            raise

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        return self._queue.full()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "current_size": self.qsize,
            "put_count": self._put_count,
            "get_count": self._get_count,
            "closed": self._closed,
        }


async def gather_with_concurrency(
    coros: List[Awaitable[T]],
    max_concurrent: int = 10,
) -> List[T]:
    """
    Like asyncio.gather but with concurrency limit.

    Example:
        results = await gather_with_concurrency(
            [api_call(x) for x in items],
            max_concurrent=5,
        )
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[bounded_coro(c) for c in coros])


async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Run coroutine with timeout, returning default on timeout.

    Example:
        result = await run_with_timeout(api_call(), timeout=5.0, default=None)
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


class ProgressTracker:
    """
    Track progress of async operations.

    Example:
        tracker = ProgressTracker(total=100)

        async for item in items:
            process(item)
            tracker.update()
            print(f"Progress: {tracker.percent:.1f}%")
    """

    def __init__(
        self,
        total: int,
        callback: Optional[Callable[[int, int, float], Awaitable[None]]] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items.
            callback: Optional async callback(current, total, percent).
        """
        self.total = total
        self.callback = callback
        self._current = 0
        self._start_time = time.monotonic()
        self._lock = asyncio.Lock()

    async def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        async with self._lock:
            self._current += n

            if self.callback:
                await self.callback(self._current, self.total, self.percent)

    @property
    def current(self) -> int:
        return self._current

    @property
    def percent(self) -> float:
        return (self._current / self.total * 100) if self.total > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self._current == 0:
            return None

        rate = self._current / self.elapsed
        remaining = self.total - self._current

        return remaining / rate if rate > 0 else None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "current": self._current,
            "total": self.total,
            "percent": self.percent,
            "elapsed": self.elapsed,
            "eta_seconds": self.eta_seconds,
            "rate": self._current / self.elapsed if self.elapsed > 0 else 0,
        }
