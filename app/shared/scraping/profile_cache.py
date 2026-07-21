"""Thread-safe in-memory cache for browser profiles.

This module provides a singleton ProfileCache for storing browser profiles
across activity executions, reducing overhead from repeated profile creation.

Usage:
    from app.shared.scraping.profile_cache import get_profile_cache

    cache = get_profile_cache()
    profile = await cache.get(telemetry_id)
    if profile is None:
        profile = create_new_profile()
        await cache.set(telemetry_id, profile)
"""
import asyncio
from typing import Any, Dict, Optional

# Global singleton instance
_profile_cache: Optional["ProfileCache"] = None


class ProfileCache:
    """Thread-safe in-memory cache for browser profiles.

    Implements async locking to ensure thread-safe access to the cache
    dictionary. Profiles are keyed by telemetry_id to maintain isolation
    between workflow executions.

    Attributes:
        _cache: Dictionary mapping telemetry_id to profile objects
        _lock: Async lock for thread-safe operations
    """

    def __init__(self) -> None:
        """Initialize empty cache with async lock."""
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, telemetry_id: str) -> Optional[Any]:
        """Get profile from cache.

        Args:
            telemetry_id: Unique identifier for the profile

        Returns:
            Profile object if found, None otherwise
        """
        async with self._lock:
            return self._cache.get(telemetry_id)

    async def set(self, telemetry_id: str, profile: Any) -> None:
        """Store profile in cache.

        Args:
            telemetry_id: Unique identifier for the profile
            profile: Profile object to cache
        """
        async with self._lock:
            self._cache[telemetry_id] = profile

    async def remove(self, telemetry_id: str) -> None:
        """Remove profile from cache.

        Args:
            telemetry_id: Unique identifier for the profile to remove
        """
        async with self._lock:
            self._cache.pop(telemetry_id, None)

    async def clear(self) -> None:
        """Clear all cached profiles.

        Should be called during worker shutdown or when
        resetting state between test runs.
        """
        async with self._lock:
            self._cache.clear()

    async def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of profiles currently cached
        """
        async with self._lock:
            return len(self._cache)


def get_profile_cache() -> ProfileCache:
    """Get or create singleton ProfileCache instance.

    Uses lazy initialization to create cache on first access.
    Thread-safe through Python's GIL for module-level assignment.

    Returns:
        Singleton ProfileCache instance
    """
    global _profile_cache
    if _profile_cache is None:
        _profile_cache = ProfileCache()
    return _profile_cache
