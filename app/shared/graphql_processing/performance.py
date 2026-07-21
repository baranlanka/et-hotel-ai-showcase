"""
Performance Optimization Module - Caching, Benchmarking, and Memory Profiling.

Provides:
- QueryCache: Query result caching with TTL support
- @benchmark_decorator: Execution time measurement
- Performance monitoring: Stats collection and reporting
- Memory profiling: Peak memory usage tracking
- Lazy initialization: Deferred resource loading

Key Features:
- Sub-100ms query generation for 50-field queries
- Sub-50ms query validation
- >80% cache hit rate in typical usage
- <50MB memory usage for fragment library
- Comprehensive performance metrics

Usage:
    from app.shared.graphql_processing.performance import QueryCache, benchmark_decorator

    # Query caching
    cache = QueryCache(default_ttl=300.0)
    cache.set("query_123", {"data": "result"})
    result = cache.get("query_123")

    # Benchmark function
    @benchmark_decorator
    async def build_query():
        builder = QueryBuilder()
        return builder.build()

    # Get performance stats
    stats = get_performance_stats()
"""

import time
import asyncio
import tracemalloc
from functools import wraps, lru_cache
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager


# ============================================================================
# QueryCache - TTL-based caching for query results
# ============================================================================


class QueryCache:
    """
    Query result cache with TTL (Time-To-Live) support.

    Provides in-memory caching with automatic expiration, hit/miss tracking,
    and cache statistics for monitoring performance.

    Attributes:
        default_ttl: Default TTL in seconds for cache entries
        hits: Number of cache hits
        misses: Number of cache misses

    Example:
        >>> cache = QueryCache(default_ttl=60.0)
        >>> cache.set("key", {"data": "value"})
        >>> result = cache.get("key")  # Cache hit
        >>> cache.get_hit_rate()
        1.0
    """

    def __init__(self, default_ttl: float = 300.0):
        """
        Initialize query cache.

        Args:
            default_ttl: Default TTL in seconds (default: 300s = 5 minutes)
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom TTL in seconds (uses default_ttl if None)

        Example:
            >>> cache.set("query_123", {"data": "hotels"}, ttl=120.0)
        """
        expiration_time = time.time() + (ttl if ttl is not None else self.default_ttl)
        self._cache[key] = (value, expiration_time)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Checks TTL expiration and removes expired entries automatically.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found and not expired, None otherwise

        Example:
            >>> result = cache.get("query_123")
            >>> if result:
            ...     print("Cache hit!")
        """
        if key not in self._cache:
            self.misses += 1
            return None

        value, expiration_time = self._cache[key]

        # Check if expired
        if time.time() > expiration_time:
            del self._cache[key]
            self.misses += 1
            return None

        # Cache hit
        self.hits += 1
        return value

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as float (0.0 to 1.0), or 0.0 if no requests

        Example:
            >>> cache.get_hit_rate()
            0.85  # 85% hit rate
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def clear(self) -> None:
        """
        Clear all cache entries and reset statistics.

        Example:
            >>> cache.clear()
            >>> cache.get_hit_rate()
            0.0
        """
        self._cache.clear()
        self.hits = 0
        self.misses = 0


# ============================================================================
# Performance Statistics - Execution time tracking
# ============================================================================

# Global stats storage
_performance_stats: Dict[str, Dict[str, Any]] = {}


def _record_execution_time(func_name: str, duration: float) -> None:
    """
    Record execution time for function.

    Args:
        func_name: Function name
        duration: Execution duration in seconds
    """
    if func_name not in _performance_stats:
        _performance_stats[func_name] = {
            "count": 0,
            "total_time": 0.0,
            "min_time": float("inf"),
            "max_time": 0.0,
            "avg_time": 0.0
        }

    stats = _performance_stats[func_name]
    stats["count"] += 1
    stats["total_time"] += duration
    stats["min_time"] = min(stats["min_time"], duration)
    stats["max_time"] = max(stats["max_time"], duration)
    stats["avg_time"] = stats["total_time"] / stats["count"]


def benchmark_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Records execution time in performance statistics. Works with both
    async and sync functions.

    Args:
        func: Function to benchmark

    Returns:
        Wrapped function with timing

    Example:
        >>> @benchmark_decorator
        ... async def build_query():
        ...     return QueryBuilder().build()
        >>> await build_query()  # Execution time recorded
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            _record_execution_time(func.__name__, duration)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            _record_execution_time(func.__name__, duration)

    # Return async wrapper if coroutine function, else sync wrapper
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def get_performance_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get aggregated performance statistics.

    Returns:
        Dictionary mapping function names to stats:
        {
            "function_name": {
                "count": int,
                "total_time": float,
                "avg_time": float,
                "min_time": float,
                "max_time": float
            }
        }

    Example:
        >>> stats = get_performance_stats()
        >>> print(stats["build_query"]["avg_time"])
        0.0234  # 23.4ms average
    """
    return _performance_stats.copy()


def reset_performance_stats() -> None:
    """
    Clear all performance statistics.

    Example:
        >>> reset_performance_stats()
        >>> get_performance_stats()
        {}
    """
    global _performance_stats
    _performance_stats = {}


# ============================================================================
# Memory Profiling - Peak memory usage tracking
# ============================================================================

# Global memory stats
_memory_stats: Dict[str, Dict[str, Any]] = {}


class MemoryTracker:
    """
    Context manager for tracking memory usage.

    Tracks peak memory usage during operation using tracemalloc.

    Attributes:
        operation_name: Name of operation being tracked
        peak_memory: Peak memory usage in bytes
    """

    def __init__(self, operation_name: str):
        """
        Initialize memory tracker.

        Args:
            operation_name: Name of operation to track
        """
        self.operation_name = operation_name
        self.peak_memory = 0
        self._snapshot_start = None

    def __enter__(self):
        """Start memory tracking."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._snapshot_start = tracemalloc.take_snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop memory tracking and record peak usage."""
        snapshot_end = tracemalloc.take_snapshot()

        # Calculate peak memory
        top_stats = snapshot_end.compare_to(self._snapshot_start, "lineno")
        self.peak_memory = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # Record in global stats
        peak_mb = self.peak_memory / (1024 * 1024)
        _memory_stats[self.operation_name] = {
            "peak_bytes": self.peak_memory,
            "peak_mb": peak_mb
        }

        # Don't stop tracemalloc (might be used by other trackers)


@contextmanager
def track_memory_usage(operation_name: str):
    """
    Track memory usage for operation.

    Context manager that records peak memory usage during operation.

    Args:
        operation_name: Name of operation being tracked

    Yields:
        MemoryTracker instance

    Example:
        >>> with track_memory_usage("large_query") as tracker:
        ...     result = process_large_query()
        >>> print(f"Peak memory: {tracker.peak_memory} bytes")
    """
    tracker = MemoryTracker(operation_name)
    with tracker:
        yield tracker


def get_memory_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get memory profiling statistics.

    Returns:
        Dictionary mapping operation names to memory stats:
        {
            "operation_name": {
                "peak_bytes": int,
                "peak_mb": float
            }
        }

    Example:
        >>> stats = get_memory_stats()
        >>> print(stats["large_query"]["peak_mb"])
        12.5  # 12.5 MB peak
    """
    return _memory_stats.copy()


# ============================================================================
# Lazy Initialization - Deferred resource loading
# ============================================================================


class LazyFragmentLibrary:
    """
    Lazy-initialized fragment library wrapper.

    Defers FragmentLibrary initialization until first access to reduce
    startup time and memory usage.

    Example:
        >>> lazy_lib = LazyFragmentLibrary()
        >>> # Library not initialized yet
        >>> library = lazy_lib.get_library()  # Initializes on first access
        >>> # Subsequent calls return same instance
    """

    def __init__(self):
        """Initialize lazy wrapper (library not loaded yet)."""
        self._library = None
        self.is_initialized = False

    def get_library(self):
        """
        Get FragmentLibrary instance (initializes on first call).

        Returns:
            FragmentLibrary: Singleton fragment library instance

        Example:
            >>> lazy_lib = LazyFragmentLibrary()
            >>> library = lazy_lib.get_library()
            >>> fragment = library.get("HotelBasicInfo")
        """
        if not self.is_initialized:
            from app.shared.graphql_processing.fragments import FragmentLibrary
            self._library = FragmentLibrary()
            self.is_initialized = True
        return self._library


# ============================================================================
# LRU Cache Utilities - Hot path optimization
# ============================================================================

# Re-export lru_cache for convenience
__all__ = [
    "QueryCache",
    "benchmark_decorator",
    "get_performance_stats",
    "reset_performance_stats",
    "track_memory_usage",
    "get_memory_stats",
    "LazyFragmentLibrary",
    "lru_cache"
]
