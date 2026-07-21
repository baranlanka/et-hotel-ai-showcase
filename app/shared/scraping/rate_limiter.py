"""Sliding window rate limiter for domain-specific request throttling.

This module provides a SimpleRateLimiter that enforces per-domain rate limits
using a sliding window algorithm. Each domain tracks its own request history
independently.

Usage:
    from app.shared.scraping.rate_limiter import get_rate_limiter

    limiter = get_rate_limiter()
    await limiter.acquire("example.com")  # Blocks until request allowed
    # Make request to example.com
"""
import asyncio
import time
from collections import deque
from typing import Dict, Deque, Optional

# Global singleton instance
_rate_limiter: Optional["SimpleRateLimiter"] = None


class SimpleRateLimiter:
    """Sliding window rate limiter for domain-specific throttling.

    Implements sliding window algorithm to enforce requests-per-minute
    limits on a per-domain basis. Each domain maintains independent
    request history.

    Attributes:
        requests_per_minute: Maximum requests allowed per domain per minute
        window_seconds: Time window for rate limiting (60 seconds)
        _domain_windows: Dict mapping domain to deque of request timestamps
        _lock: Async lock for thread-safe operations
    """

    def __init__(self, requests_per_minute: int = 10) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per domain per minute (default: 10)
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._domain_windows: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, domain: str) -> None:
        """Block until request is allowed within rate limit.

        Uses sliding window algorithm:
        1. Remove timestamps older than window_seconds
        2. Check if at limit
        3. If at limit, wait until oldest timestamp expires
        4. Add current timestamp to window

        Args:
            domain: Domain to rate limit (e.g., "example.com")
        """
        async with self._lock:
            now = time.time()

            # Initialize domain window if not exists
            if domain not in self._domain_windows:
                self._domain_windows[domain] = deque()

            window = self._domain_windows[domain]

            # Remove expired timestamps (outside window)
            cutoff_time = now - self.window_seconds
            while window and window[0] < cutoff_time:
                window.popleft()

            # Check if at limit
            if len(window) >= self.requests_per_minute:
                # Calculate wait time until oldest request expires
                oldest_timestamp = window[0]
                wait_time = oldest_timestamp + self.window_seconds - now

                if wait_time > 0:
                    # Release lock during sleep to allow other domains
                    self._lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        await self._lock.acquire()

                    # Re-clean window after sleep
                    now = time.time()
                    cutoff_time = now - self.window_seconds
                    while window and window[0] < cutoff_time:
                        window.popleft()

            # Add current request timestamp
            window.append(now)

    async def reset(self, domain: Optional[str] = None) -> None:
        """Reset rate limit tracking.

        Args:
            domain: Specific domain to reset, or None to reset all domains
        """
        async with self._lock:
            if domain is None:
                # Reset all domains
                self._domain_windows.clear()
            elif domain in self._domain_windows:
                # Reset specific domain
                self._domain_windows[domain].clear()

    async def get_remaining(self, domain: str) -> int:
        """Get remaining requests available for domain.

        Args:
            domain: Domain to check

        Returns:
            Number of requests remaining in current window
        """
        async with self._lock:
            if domain not in self._domain_windows:
                return self.requests_per_minute

            now = time.time()
            window = self._domain_windows[domain]

            # Remove expired timestamps
            cutoff_time = now - self.window_seconds
            while window and window[0] < cutoff_time:
                window.popleft()

            return max(0, self.requests_per_minute - len(window))

    async def get_window_size(self, domain: str) -> int:
        """Get current window size for domain.

        Args:
            domain: Domain to check

        Returns:
            Number of requests in current window
        """
        async with self._lock:
            if domain not in self._domain_windows:
                return 0

            now = time.time()
            window = self._domain_windows[domain]

            # Remove expired timestamps
            cutoff_time = now - self.window_seconds
            while window and window[0] < cutoff_time:
                window.popleft()

            return len(window)


def get_rate_limiter() -> SimpleRateLimiter:
    """Get or create singleton SimpleRateLimiter instance.

    Uses lazy initialization to create limiter on first access.
    Thread-safe through Python's GIL for module-level assignment.

    Returns:
        Singleton SimpleRateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SimpleRateLimiter(requests_per_minute=10)
    return _rate_limiter
