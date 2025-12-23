"""
Rate limiting utilities for API calls.

Provides:
- Token bucket rate limiter
- Request rate limiting
- Async-compatible rate limiting
- Multi-tier rate limiting (RPM, TPM)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Requests per minute
    requests_per_minute: int = 60

    # Tokens per minute (optional)
    tokens_per_minute: Optional[int] = None

    # Burst capacity (multiplier)
    burst_multiplier: float = 1.5

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Blocking behavior
    block_on_limit: bool = True
    max_wait_time: float = 60.0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls.

    Supports both request-based and token-based rate limiting.
    Thread-safe and async-compatible.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        requests_per_minute: int = 60,
        tokens_per_minute: Optional[int] = None,
        burst_multiplier: float = 1.5,
    ):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration (overrides other args)
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute (optional)
            burst_multiplier: Burst capacity multiplier
        """
        if config:
            self.requests_per_minute = config.requests_per_minute
            self.tokens_per_minute = config.tokens_per_minute
            self.burst_multiplier = config.burst_multiplier
            self.block_on_limit = config.block_on_limit
            self.max_wait_time = config.max_wait_time
        else:
            self.requests_per_minute = requests_per_minute
            self.tokens_per_minute = tokens_per_minute
            self.burst_multiplier = burst_multiplier
            self.block_on_limit = True
            self.max_wait_time = 60.0

        # Request bucket
        self._request_capacity = int(self.requests_per_minute * self.burst_multiplier)
        self._request_tokens = float(self._request_capacity)
        self._request_rate = self.requests_per_minute / 60.0  # tokens per second

        # Token bucket (if configured)
        if self.tokens_per_minute:
            self._token_capacity = int(self.tokens_per_minute * self.burst_multiplier)
            self._token_tokens = float(self._token_capacity)
            self._token_rate = self.tokens_per_minute / 60.0
        else:
            self._token_capacity = None
            self._token_tokens = None
            self._token_rate = None

        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_waits = 0
        self._total_wait_time = 0.0

    def _refill_buckets(self) -> None:
        """Refill token buckets based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Refill request bucket
        self._request_tokens = min(
            self._request_capacity,
            self._request_tokens + elapsed * self._request_rate,
        )

        # Refill token bucket
        if self._token_rate is not None:
            self._token_tokens = min(
                self._token_capacity,
                self._token_tokens + elapsed * self._token_rate,
            )

    def _calculate_wait_time(
        self,
        request_cost: float = 1.0,
        token_cost: float = 0.0,
    ) -> float:
        """Calculate wait time needed for request."""
        wait_time = 0.0

        # Request bucket wait
        if self._request_tokens < request_cost:
            needed = request_cost - self._request_tokens
            wait_time = max(wait_time, needed / self._request_rate)

        # Token bucket wait
        if self._token_rate is not None and token_cost > 0:
            if self._token_tokens < token_cost:
                needed = token_cost - self._token_tokens
                wait_time = max(wait_time, needed / self._token_rate)

        return wait_time

    async def acquire(
        self,
        request_cost: float = 1.0,
        token_cost: float = 0.0,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire permission to make a request.

        Args:
            request_cost: Number of request tokens to consume
            token_cost: Number of API tokens to consume
            timeout: Maximum wait time (overrides config)

        Returns:
            True if acquired, raises if timeout exceeded

        Raises:
            RateLimitExceeded: If timeout exceeded and blocking disabled
        """
        timeout = timeout or self.max_wait_time
        start_time = time.monotonic()

        async with self._lock:
            while True:
                self._refill_buckets()

                wait_time = self._calculate_wait_time(request_cost, token_cost)

                if wait_time <= 0:
                    # Consume tokens
                    self._request_tokens -= request_cost
                    if self._token_tokens is not None and token_cost > 0:
                        self._token_tokens -= token_cost

                    self._total_requests += 1
                    self._total_tokens += int(token_cost)
                    return True

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    if not self.block_on_limit:
                        raise RateLimitExceeded(
                            f"Rate limit exceeded, retry after {wait_time:.2f}s",
                            retry_after=wait_time,
                        )
                    # Truncate wait time
                    wait_time = max(0.1, timeout - elapsed)

                # Wait
                self._total_waits += 1
                self._total_wait_time += wait_time
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")

                # Release lock while waiting
                self._lock.release()
                await asyncio.sleep(wait_time)
                await self._lock.acquire()

                # Check if timeout exceeded after wait
                if time.monotonic() - start_time > timeout:
                    raise RateLimitExceeded(
                        f"Rate limit timeout after {timeout:.2f}s",
                        retry_after=wait_time,
                    )

    async def acquire_with_retry(
        self,
        request_cost: float = 1.0,
        token_cost: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ) -> bool:
        """
        Acquire with automatic retry on rate limit.

        Args:
            request_cost: Number of request tokens
            token_cost: Number of API tokens
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay
            retry_backoff: Backoff multiplier

        Returns:
            True if acquired

        Raises:
            RateLimitExceeded: If all retries exhausted
        """
        delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                return await self.acquire(request_cost, token_cost)
            except RateLimitExceeded as e:
                if attempt == max_retries:
                    raise

                wait = e.retry_after or delay
                logger.warning(
                    f"Rate limit exceeded, retry {attempt + 1}/{max_retries} "
                    f"after {wait:.2f}s"
                )
                await asyncio.sleep(wait)
                delay *= retry_backoff

        return False

    def get_available(self) -> dict:
        """Get current available capacity."""
        self._refill_buckets()
        return {
            "requests_available": self._request_tokens,
            "requests_capacity": self._request_capacity,
            "tokens_available": self._token_tokens,
            "tokens_capacity": self._token_capacity,
        }

    def get_statistics(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_waits": self._total_waits,
            "total_wait_time": self._total_wait_time,
            "avg_wait_time": (
                self._total_wait_time / self._total_waits
                if self._total_waits > 0
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        self._request_tokens = float(self._request_capacity)
        if self._token_capacity:
            self._token_tokens = float(self._token_capacity)
        self._last_update = time.monotonic()

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._total_requests = 0
        self._total_tokens = 0
        self._total_waits = 0
        self._total_wait_time = 0.0


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter for complex API limits.

    Handles multiple rate limit tiers (e.g., per-minute, per-day).
    """

    def __init__(
        self,
        tiers: Optional[dict] = None,
    ):
        """
        Initialize multi-tier rate limiter.

        Args:
            tiers: Dictionary of tier name to RateLimitConfig
        """
        self._limiters: dict[str, TokenBucketRateLimiter] = {}

        if tiers:
            for name, config in tiers.items():
                if isinstance(config, RateLimitConfig):
                    self._limiters[name] = TokenBucketRateLimiter(config)
                else:
                    self._limiters[name] = TokenBucketRateLimiter(**config)

    def add_tier(
        self,
        name: str,
        config: RateLimitConfig,
    ) -> None:
        """Add a rate limit tier."""
        self._limiters[name] = TokenBucketRateLimiter(config)

    async def acquire(
        self,
        request_cost: float = 1.0,
        token_cost: float = 0.0,
    ) -> bool:
        """
        Acquire from all tiers.

        Args:
            request_cost: Request tokens to consume
            token_cost: API tokens to consume

        Returns:
            True if acquired from all tiers
        """
        # First check all tiers can be acquired
        for limiter in self._limiters.values():
            available = limiter.get_available()
            if available["requests_available"] < request_cost:
                # One tier would block, wait on it
                break

        # Acquire from all tiers
        for limiter in self._limiters.values():
            await limiter.acquire(request_cost, token_cost)

        return True

    def get_statistics(self) -> dict:
        """Get statistics from all tiers."""
        return {
            name: limiter.get_statistics()
            for name, limiter in self._limiters.items()
        }


# Convenience exports
__all__ = [
    "TokenBucketRateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    "MultiTierRateLimiter",
]
