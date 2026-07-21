"""Lazy Langfuse prompt fetch for the leadgen-outreach activities (F3).

The 4 outreach prompts (miner / opener-fill / qualifier / router) live in
Langfuse as `langchain/outreach_*` chat prompts with the model pinned in
``config.model``. Langfuse is the SINGLE source of truth for these prompts — there
are no in-code prompt copies. This helper fetches + compiles one of them inside an
activity body.

Why a helper (not inline): all four activities share the exact same
fetch→compile→resolve-model dance. Centralising it keeps the callers DRY and gives
every activity the same contract: on a Langfuse miss the caller raises a retryable
error rather than falling back to a local prompt.

Sandbox safety: every import is local to the function — this module does no I/O
and no SDK init at import time, so it is safe to import lazily from inside a
Temporal activity. Never import it at module scope in an activity file.
"""
from __future__ import annotations

from typing import Any


def fetch_outreach_prompt(
    prompt_name: str, variables: dict[str, Any]
) -> tuple[list[Any] | None, str | None]:
    """Fetch + compile a Langfuse outreach prompt.

    Args:
        prompt_name: Langfuse prompt name, e.g. ``langchain/outreach_router``.
        variables:   Template variables to compile into the prompt.

    Returns:
        ``(messages, model_name)`` where ``messages`` is a list of LangChain
        message objects ready to pass to an LLM, and ``model_name`` is the slug
        pinned in the prompt's ``config.model`` (may be None if the prompt has
        no model in config).

        Returns ``(None, None)`` when Langfuse is disabled / unreachable (the
        fetch layer returns a ``"fallback"`` sentinel version). The caller MUST
        treat ``(None, None)`` as "use my hardcoded prompt + create_for_extraction()"
        — we never send the fetch layer's dummy JSON-dump payload to a hotel.
    """
    from llm_content_generation.shared.singletons import get_shared_prompt_manager

    pm = get_shared_prompt_manager()
    prompt_obj, metadata = pm.get_prompt(prompt_name)

    # Langfuse disabled / fetch failed → fetch layer returns version "fallback".
    if metadata.get("langfuse_prompt_version") == "fallback":
        return None, None

    messages = pm.compile_prompt(prompt_obj, variables)
    model = pm.extract_model_from_config(metadata)
    return messages, model
