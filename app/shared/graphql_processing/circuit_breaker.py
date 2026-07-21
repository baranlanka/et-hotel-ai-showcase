"""
Circuit breaker pattern for fault tolerance in GraphQL scraping.

Implements the circuit breaker pattern to prevent cascading failures by failing fast
when a service is experiencing repeated errors. The circuit breaker has three states:
- closed: Normal operation, requests pass through
- open: Service is failing, requests fail immediately without execution
- half-open: Testing if service has recovered after timeout

Architecture:
    - Failure threshold: 5 consecutive errors trigger circuit open
    - Recovery timeout: 60 seconds before attempting half-open state
    - State transitions logged with structured metrics
    - Async-compatible for use in async workflows

References:
    - Story 20.5: Production Error Handling (Production Readiness)
    - Pattern: Circuit Breaker (Martin Fowler)

Examples:
    >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60, name="external_api")
    >>> try:
    ...     result = await breaker.call(scrape_function, region_id="123")
    ... except CircuitBreakerOpenError:
    ...     logger.warning("Circuit breaker is open, skipping request")
"""

from datetime import datetime, timedelta
from typing import Callable, Any, TypeVar
import asyncio
from common.observability import trace
from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("graphql-processing", context="activity")
logger = _obs.logger

T = TypeVar('T')


class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open, preventing execution."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Implements the circuit breaker pattern to detect repeated failures and fail fast
    instead of attempting doomed requests. After a timeout period, the circuit
    transitions to half-open to test if the service has recovered.

    States:
        - closed: Normal operation, all requests execute
        - open: Too many failures, all requests fail immediately
        - half-open: Testing recovery, one request allowed

    State Transitions:
        closed → (failure_threshold failures) → open
        open → (timeout elapsed) → half-open
        half-open → (success) → closed
        half-open → (failure) → open

    Args:
        failure_threshold: Number of consecutive failures before opening circuit (default: 5)
        timeout: Seconds to wait before attempting recovery (default: 60)
        name: Circuit breaker name for logging (default: "default")

    Examples:
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60, name="external_api")
        >>>
        >>> async def fetch_data():
        ...     # Function that may fail
        ...     return await api_client.get_hotels()
        >>>
        >>> try:
        ...     hotels = await breaker.call(fetch_data)
        ... except CircuitBreakerOpenError:
        ...     # Circuit is open, use cached data or return error
        ...     hotels = []
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening (default: 5)
            timeout: Seconds before attempting half-open state (default: 60)
            name: Circuit breaker name for metrics/logging (default: "default")
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "closed"  # closed, open, half-open

    @trace(span_name="circuit_breaker.call")
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Wraps an async function call with circuit breaker logic. If the circuit is
        open, the call fails immediately with CircuitBreakerOpenError. If closed or
        half-open, the function executes and the circuit state updates based on
        success or failure.

        Args:
            func: Async callable to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            T: Result from func if execution succeeds

        Raises:
            CircuitBreakerOpenError: Circuit is open and timeout has not elapsed
            Exception: Any exception raised by func (after updating circuit state)

        State Updates:
            - Success in half-open: Close circuit and reset failure count
            - Failure in any state: Increment failure count
            - Failure count >= threshold: Open circuit

        Metrics:
            All state transitions are logged as structured JSON for observability:
            - Transition to half-open: info level
            - Circuit closed after recovery: info level
            - Circuit opened: error level
            - Individual failures: warning level

        Examples:
            >>> # Circuit is closed, call succeeds
            >>> result = await breaker.call(fetch_hotels, region_id="123")
            >>>
            >>> # After 5 failures, circuit opens
            >>> # Next call fails immediately without executing fetch_hotels
            >>> try:
            ...     await breaker.call(fetch_hotels, region_id="123")
            ... except CircuitBreakerOpenError:
            ...     logger.warning("Service unavailable, circuit open")
        """
        # Check if circuit is open
        if self.state == "open":
            # Check if timeout elapsed
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).total_seconds() > self.timeout:
                self.state = "half-open"
                logger.info(
                    f"Circuit breaker '{self.name}' transitioning to half-open",
                    extra={
                        "circuit_breaker": self.name,
                        "state": "half-open",
                        "previous_state": "open",
                        "timeout_seconds": self.timeout
                    }
                )
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )

        try:
            result = await func(*args, **kwargs)

            # Success - reset circuit if in half-open state
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' closed after successful call",
                    extra={
                        "circuit_breaker": self.name,
                        "state": "closed",
                        "previous_state": "half-open"
                    }
                )

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            logger.warning(
                f"Circuit breaker '{self.name}' failure {self.failure_count}/{self.failure_threshold}",
                extra={
                    "circuit_breaker": self.name,
                    "state": self.state,
                    "failure_count": self.failure_count,
                    "failure_threshold": self.failure_threshold,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

            # Open circuit if threshold reached
            if self.failure_count >= self.failure_threshold:
                previous_state = self.state
                self.state = "open"
                logger.error(
                    f"Circuit breaker '{self.name}' opened after {self.failure_count} failures",
                    extra={
                        "circuit_breaker": self.name,
                        "state": "open",
                        "previous_state": previous_state,
                        "failure_count": self.failure_count,
                        "failure_threshold": self.failure_threshold
                    }
                )

            raise
