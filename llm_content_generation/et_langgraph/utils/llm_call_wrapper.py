"""Robust LLM call wrapper with timeout, retry, and empty response handling.

Designed for DeepSeek R1 and other slow-responding LLMs in Temporal activities.
Ensures activities can be restarted cleanly on failures.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMCallError(Exception):
    """Base exception for LLM call failures - retryable by default."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class LLMEmptyResponseError(LLMCallError):
    """Raised when LLM returns empty or invalid response."""

    def __init__(self, message: str = "LLM returned empty response"):
        super().__init__(message, retryable=True)


class LLMTimeoutError(LLMCallError):
    """Raised when LLM call times out."""

    def __init__(self, timeout_seconds: int, message: str = None):
        msg = message or f"LLM call timed out after {timeout_seconds}s"
        super().__init__(msg, retryable=True)
        self.timeout_seconds = timeout_seconds


async def invoke_llm_with_validation(
    llm: Any,
    prompt: Any,
    config: Optional[dict] = None,
    timeout_seconds: int = 300,
    min_response_length: int = 10,
    operation_name: str = "llm_call",
) -> str:
    """Invoke LLM with timeout, empty response validation, and error handling.

    Designed for Temporal activities - raises retryable errors on failure so
    the activity can be restarted by Temporal's retry mechanism.

    Args:
        llm: LangChain LLM instance (ChatOpenAI or similar)
        prompt: Prompt to send to LLM
        config: Optional LangChain config dict (metadata, callbacks, etc.)
        timeout_seconds: Max time to wait for LLM response (default 5 min for DeepSeek R1)
        min_response_length: Minimum acceptable response length (default 10 chars)
        operation_name: Name for logging/tracing

    Returns:
        str: The LLM response content (stripped)

    Raises:
        LLMTimeoutError: If LLM call times out (retryable)
        LLMEmptyResponseError: If response is empty/too short (retryable)
        LLMCallError: For other LLM failures (retryable)
    """
    logger.info(
        f"[{operation_name}] Starting LLM call with {timeout_seconds}s timeout"
    )

    try:
        # Wrap the LLM call with asyncio timeout
        response = await asyncio.wait_for(
            llm.ainvoke(prompt, config=config),
            timeout=timeout_seconds,
        )

        # Extract content from response
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Strip and validate
        content = content.strip() if content else ""

        # Check for empty response
        if not content:
            logger.error(f"[{operation_name}] LLM returned empty response")
            raise LLMEmptyResponseError(
                f"[{operation_name}] LLM returned empty response"
            )

        # Check minimum length
        if len(content) < min_response_length:
            logger.error(
                f"[{operation_name}] LLM response too short: {len(content)} chars "
                f"(min: {min_response_length})"
            )
            raise LLMEmptyResponseError(
                f"[{operation_name}] Response too short ({len(content)} chars)"
            )

        logger.info(
            f"[{operation_name}] LLM call successful, response length: {len(content)}"
        )
        return content

    except asyncio.TimeoutError:
        logger.error(
            f"[{operation_name}] LLM call timed out after {timeout_seconds}s"
        )
        raise LLMTimeoutError(
            timeout_seconds,
            f"[{operation_name}] LLM call timed out after {timeout_seconds}s - "
            "activity will be retried",
        )

    except (LLMEmptyResponseError, LLMTimeoutError):
        # Re-raise our custom exceptions
        raise

    except Exception as e:
        # Log and wrap other exceptions as retryable
        error_type = type(e).__name__
        error_msg = str(e)

        # Check for common error patterns
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            logger.warning(f"[{operation_name}] Rate limited: {error_msg}")
            raise LLMCallError(
                f"[{operation_name}] Rate limited - will retry: {error_msg}",
                retryable=True,
            )

        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.warning(f"[{operation_name}] Connection error: {error_msg}")
            raise LLMCallError(
                f"[{operation_name}] Connection error - will retry: {error_msg}",
                retryable=True,
            )

        if "invalid_api_key" in error_msg.lower() or "401" in error_msg:
            logger.error(f"[{operation_name}] Auth error (non-retryable): {error_msg}")
            raise LLMCallError(
                f"[{operation_name}] Authentication error: {error_msg}",
                retryable=False,
            )

        # Default: retryable error
        logger.error(f"[{operation_name}] LLM call failed: {error_type}: {error_msg}")
        raise LLMCallError(
            f"[{operation_name}] {error_type}: {error_msg}",
            retryable=True,
        )


# Default timeout constants for different model types
DEFAULT_TIMEOUT_FAST = 120  # 2 min for fast models (GPT-4, Claude)
DEFAULT_TIMEOUT_SLOW = 300  # 5 min for slow models (DeepSeek R1)
DEFAULT_TIMEOUT_EXTRACTION = 180  # 3 min for extraction (smaller prompts)
DEFAULT_TIMEOUT_VALIDATION = 120  # 2 min for validation
