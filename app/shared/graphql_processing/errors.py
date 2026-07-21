"""
Error handling framework for GraphQL scraping system.

Provides production-grade exception hierarchy with retry logic and exponential backoff.
Supports structured error context and classification for retryable vs. non-retryable errors.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import functools
import asyncio
from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("graphql-processing", context="activity")
logger = _obs.logger


class GraphQLScraperError(Exception):
    """Base exception for GraphQL scraping errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize GraphQLScraperError.

        Args:
            message: Human-readable error message
            context: Additional error context (backend, operation, parameters, etc.)
        """
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.utcnow()

    def is_retryable(self) -> bool:
        """
        Determine if this error should trigger retry logic.

        Override in subclasses to indicate retry behavior.

        Returns:
            bool: False for base class (conservative default)
        """
        return False


class NetworkError(GraphQLScraperError):
    """Network-related errors (connection, timeout)."""

    def is_retryable(self) -> bool:
        """Network errors are retryable (transient failures)."""
        return True


class RateLimitError(GraphQLScraperError):
    """API rate limit exceeded."""

    def is_retryable(self) -> bool:
        """Rate limit errors are retryable (wait and retry)."""
        return True


class AuthenticationError(GraphQLScraperError):
    """Authentication failed."""

    def is_retryable(self) -> bool:
        """Authentication errors are not retryable (need manual intervention)."""
        return False


class EmptyResultError(GraphQLScraperError):
    """Raised when a request fetched cleanly (HTTP 200, parsed fine) but returned ZERO records.

    A successful round-trip with no data can be a transient upstream hiccup rather
    than a genuinely empty result set. Treated as retryable so the caller can requeue
    with a fresh proxy/session; a caller may choose to accept the result as genuinely
    empty only after several diverse attempts.
    """

    def is_retryable(self) -> bool:
        """Empty results are retryable (rotate proxy/session and retry)."""
        return True


class QueryBuildError(GraphQLScraperError):
    """Query construction failed."""

    def is_retryable(self) -> bool:
        """Query build errors are not retryable (logic error)."""
        return False


class BackendResponseError(GraphQLScraperError):
    """Intermediate base for errors produced by a remote backend response.

    Raised when the HTTP round-trip succeeded but the backend's payload indicates
    a definitive or structurally unexpected failure.  Subclasses should include a
    consistent set of context keys so that log queries and dashboards stay uniform
    across backends:

    Suggested context key set:
        operation_name (str): GraphQL operation name.
        record_id      (str): Identifier of the record being fetched.
        status_code    (int): HTTP status code returned by the backend.
        api_message    (str): Error text from the backend payload (e.g. an error
                              page title or GraphQL ``errors[0].message``).
        body_preview   (str): First 200 characters of the raw response body.
        response_type  (str): Discriminator such as "graphql_error",
                              "schema_drift", or a GraphQL ``__typename`` value.

    FORBIDDEN key:
        ``message`` — this name is a reserved ``LogRecord`` attribute in Python's
        ``logging`` module.  Using it as an ``extra`` key silently shadows the log
        record's own message field and produces malformed log lines.  Always use
        ``api_message`` instead.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Backend response error carrying a consistent context dict."""
        super().__init__(message, context)

    def is_retryable(self) -> bool:
        """Backend response errors are not retryable (definitive server-side rejection)."""
        return False


class BackendBlockedError(BackendResponseError):
    """Raised when the backend explicitly rejects the request (an error status or a
    GraphQL ``errors`` array), signalling access was refused rather than a transient
    network fault."""

    def is_retryable(self) -> bool:
        """Backend-blocked errors are not retryable (explicit rejection by the server)."""
        return False


class BackendSchemaDriftError(BackendResponseError):
    """Raised when a 200-OK response is missing the expected root container, indicating schema drift."""

    def is_retryable(self) -> bool:
        """Schema drift errors are not retryable (structural mismatch requiring code update)."""
        return False


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0):
    """
    Retry decorator with exponential backoff for async functions.

    Automatically retries functions that raise retryable GraphQLScraperError exceptions.
    Uses exponential backoff: delay = base_delay * (2 ** attempt)

    Args:
        max_attempts: Maximum number of execution attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)

    Returns:
        Decorated async function with retry logic

    Example:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        async def scrape_region(region_id: int):
            # This will retry on NetworkError or RateLimitError
            # but fail immediately on AuthenticationError or QueryBuildError
            pass
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except GraphQLScraperError as e:
                    # Check if error is retryable and if we have attempts remaining
                    if not e.is_retryable() or attempt == max_attempts - 1:
                        logger.error(
                            f"Non-retryable error or max attempts reached",
                            extra={"error": str(e), "context": e.context},
                        )
                        raise

                    # Calculate exponential backoff delay
                    delay = base_delay * (2**attempt)

                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_attempts} after {delay}s",
                        extra={"delay": delay, "attempt": attempt + 1},
                    )

                    # Use asyncio.sleep for async-compatible delay
                    await asyncio.sleep(delay)

        return wrapper

    return decorator
