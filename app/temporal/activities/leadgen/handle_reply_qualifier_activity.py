"""Temporal activity: Agent C — reply handler / qualifier (F3).

Handles the "no site / interested" funnel branch. Confirms the hotel's
situation, answers questions, and pitches the free-site offer in
Stitch-bridge prose. All copy must be place-sourced and review-sourced —
no invented operator biographical facts (ADR-OUTREACH-004).

Decorator stack (mandatory — conventions.md §1):
    @activity.defn → @trace → @log_io → @inject_telemetry_id

All app.* imports are lazy (inside the function body) — never at module
level in Temporal activities.

References:
    - Conventions: docs/features/conversational-agents-b-e/conventions.md §1–5
    - ADR-OUTREACH-004: Honest place-sourced prose (Stitch-bridge); no fabricated facts
    - ADR-OUTREACH-006: Eval-set-as-parity-contract (prototype→eval→port)
    - FR-I: Reply-handler / qualifier agent
    - Canonical pattern: app/temporal/activities/leadgen/resolve_hotel_enrichment_activity.py

Traceability:
    - Feature: F3 conversational-agents-b-e
    - Component: C2 (Agent C)
    - Draft confirms situation, no invented operator facts, Stitch-bridge prose
"""
from __future__ import annotations

from temporalio import activity
from temporalio.exceptions import ApplicationError

from common.observability import log_io, trace
from app.core.observability.factory import ObservabilityFactory
from app.core.observability.telemetry_context import inject_telemetry_id
from app.temporal.workflows.leadgen.models import QualifierResult

# ---------------------------------------------------------------------------
# Lazy observability — NEVER initialise at module level in activities.
# ---------------------------------------------------------------------------

_obs = None


def get_obs():
    """Lazy initialization of the unified observability manager."""
    global _obs
    if _obs is None:
        try:
            _obs = ObservabilityFactory.create_unified(
                service_name="leadgen-outreach", context="activity"
            )
        except Exception:
            pass  # Fallback: will use activity.logger
    return _obs


# Fallback when no review hook is available — generic offer framing.
_GENERIC_EVIDENCE = "guests appreciate the unique character and warm hospitality of this property"
_GENERIC_HOOK = "a property with real character worth showcasing online"


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


@activity.defn(name="handle_reply_qualifier_activity")
@trace(span_name="handle_reply_qualifier_activity", include=["ota_hotel_id", "telemetry_id"])
@log_io(logger=get_obs().logger if get_obs() else activity.logger)
@inject_telemetry_id
async def handle_reply_qualifier_activity(
    ota_hotel_id: str,
    inbound_message: str,
    review_hook_json: str,
    telemetry_id: str = None,
    inbound_datamarked: str | None = None,
    current_stage: str = "opener_sent",
    intent: str = "interested",
    conversation_history: str = "",
) -> QualifierResult:
    """Draft the Agent C reply for the qualify / free-site-pitch branch.

    Flow:
        1. Input validation — raises non-retryable on empty fields.
        2. Parse review_hook_json to extract evidence_quote and hook_text.
        3. Build Stitch-bridge prompt (place-sourced, no invented facts).
        4. Call LLMFactory.create_for_extraction() via invoke_llm_with_validation.
        5. Parse JSON response, extract draft + offered_sample fields.
        6. Return QualifierResult(draft, offered_sample).

    Args:
        ota_hotel_id: The hotel being contacted (for logging/tracing).
        inbound_message:  The hotel's reply text that triggered this branch.
        review_hook_json: JSON-serialised ReviewHook from Agent A (F1) for
                          sourcing honest copy. Must NOT invent facts beyond
                          what is present here.
        telemetry_id:     Injected by @inject_telemetry_id.
        inbound_datamarked: OWASP LLM01 — the datamarked (whitespace→marker) form
                          of the reply from the input guard, sent to the LLM so it
                          treats the hotel reply as DATA, not instructions. None →
                          fall back to ``inbound_message``.
        current_stage:    Current funnel stage (e.g. "opener_sent"). Passed as
                          ``funnel_state`` in template_vars for context-aware
                          prompting. Defaults to "opener_sent".
        intent:           Router-classified intent for this reply (e.g.
                          "reservation_request"). Passed as ``intent`` in template_vars
                          so Agent C can select the correct response path.
                          Defaults to "interested".
        conversation_history: Prior turns formatted as "You: ...\nThem: ..."
                          (fetched by fetch_conversation_history_activity). Empty
                          string when no history is available; the prompt receives
                          the literal "(no prior messages yet)" in that case so the
                          variable is always present.

    Returns:
        QualifierResult with:
          - ``draft``: reply text (email-ready) that confirms the situation,
            answers likely questions, and pitches the free site offer in
            Stitch-bridge prose — no fabricated operator biographical facts.
          - ``offered_sample``: True when this reply actually makes/repeats the
            free sample-page offer. The workflow advances the funnel to
            ``qualifier_sent`` on this signal (NOT on the router intent), so a
            subsequent acceptance routes to send_D instead of another qualifier.

    Raises:
        ApplicationError(non_retryable=True, type="ValidationError"):
            ota_hotel_id or inbound_message is empty.
        ApplicationError(non_retryable=False, type="LLMCallError"):
            Transient LLM failure — Temporal retries.

    Traceability:
        - Confirms situation, no invented facts, Stitch-bridge prose
    """
    # ── Lazy imports (CRITICAL: never at module level in Temporal activities) ──
    import json  # noqa: PLC0415

    from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (  # noqa: PLC0415
        invoke_llm_with_validation,
    )
    from llm_content_generation.services.llm_factory import LLMFactory  # noqa: PLC0415

    from app.temporal.activities.leadgen._langfuse_prompt import (  # noqa: PLC0415
        fetch_outreach_prompt,
    )

    logger = get_obs().logger if get_obs() else activity.logger

    # ── Input validation guard ────────────────────────────────────────────────
    if not ota_hotel_id or not ota_hotel_id.strip():
        raise ApplicationError(
            "ota_hotel_id is required",
            type="ValidationError",
            non_retryable=True,
        )
    if not inbound_message or not inbound_message.strip():
        raise ApplicationError(
            "inbound_message is required",
            type="ValidationError",
            non_retryable=True,
        )

    # ── Parse ReviewHook for place-sourced evidence ───────────────────────────
    #   [ASSUMPTION]: review_hook_json may be empty / None when the hotel was
    #   not yet mined.  Fall back to generic evidence copy in that case.
    evidence_quote = _GENERIC_EVIDENCE
    hook_text = _GENERIC_HOOK
    hotel_name = ota_hotel_id  # fallback if not in hook

    if review_hook_json:
        try:
            hook_dict = json.loads(review_hook_json)
            evidence_quote = hook_dict.get("evidence_quote") or _GENERIC_EVIDENCE
            hook_text = hook_dict.get("hook_text") or _GENERIC_HOOK
            hotel_name = hook_dict.get("hotel_name") or ota_hotel_id
        except (json.JSONDecodeError, TypeError, AttributeError):
            logger.warning(
                "handle_reply_qualifier_activity: could not parse review_hook_json; "
                "using generic evidence",
                extra={"telemetry_id": telemetry_id, "ota_hotel_id": ota_hotel_id},
            )

    logger.info(
        "handle_reply_qualifier_activity: drafting qualifier reply",
        extra={
            "telemetry_id": telemetry_id,
            "ota_hotel_id": ota_hotel_id,
            "evidence_source": "review_hook" if review_hook_json else "generic",
        },
    )

    # ── Build prompt messages (Langfuse-sourced; model pinned in config) ──────
    # langchain/outreach_qualifier carries the system + user templates — Langfuse
    # is the SINGLE source of truth for the prompt. When it is disabled/unreachable
    # the fetch returns (None, None); we RAISE a retryable error rather than draft
    # on a stale/absent prompt (Temporal retries; conversation pauses).
    # OWASP LLM01: the LLM sees the datamarked reply (spotlighting) so it treats
    # the hotel's text as DATA, not instructions.
    # Derive channel hint: "chat" if either the inbound text or the review hook
    # mentions it (case-insensitive), otherwise "email". Used in the reservation_request
    # behavior rule so Agent C knows which channel to name when re-asking.
    _channel = (
        "chat"
        if "chat" in (inbound_message or "").lower()
        or "chat" in (review_hook_json or "").lower()
        else "email"
    )
    template_vars = {
        "hotel_name": hotel_name,
        "ota_hotel_id": ota_hotel_id,
        "inbound_message": inbound_datamarked or inbound_message,
        "evidence_quote": evidence_quote,
        "hook_text": hook_text,
        "funnel_state": current_stage,
        "intent": intent,
        "channel": _channel,
        "conversation_so_far": conversation_history or "(no prior messages yet)",
    }
    messages, model = fetch_outreach_prompt("langchain/outreach_qualifier", template_vars)
    if messages is None:
        raise ApplicationError(
            "Langfuse prompt 'langchain/outreach_qualifier' is unavailable "
            "(disabled or unreachable) — retrying.",
            type="PromptUnavailableError",
            non_retryable=False,  # retryable: Temporal retries; conversation pauses
                                  # rather than drafting on an absent/stale prompt
        )

    # ── LLM call ─────────────────────────────────────────────────────────────
    try:
        llm = (
            LLMFactory().create_from_prompt_config(model)
            if model
            else LLMFactory().create_for_extraction()
        )
        raw_response: str = await invoke_llm_with_validation(llm, messages)
    except Exception as exc:
        logger.error(
            "handle_reply_qualifier_activity: LLM call failed",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "error": str(exc),
            },
        )
        raise ApplicationError(
            f"LLM call failed: {exc}",
            type="LLMCallError",
            non_retryable=False,
        ) from exc

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        # Strip optional markdown fences (some models ignore prompt instructions)
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()
        result = json.loads(cleaned)
        draft: str = result["draft"]
        # offered_sample: the funnel-advance signal. Optional in the JSON for
        # backward-compat with a prompt version that has not yet been updated —
        # a missing/false value simply means "did not advance the stage this turn"
        # (the pre-fix behavior), so an old prompt degrades safely rather than
        # crashing the parse.
        offered_sample: bool = bool(result.get("offered_sample", False))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error(
            "handle_reply_qualifier_activity: could not parse LLM JSON response",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "raw_response": raw_response[:200],
                "error": str(exc),
            },
        )
        # OWASP LLM09 fail-closed: re-running the same call rarely fixes malformed
        # JSON (it burns tokens on the same bad output), so fail fast and let the
        # workflow send a canned holding reply + escalate to a human instead of
        # retrying. non_retryable=True is what surfaces this to the workflow's
        # ActivityError handler after a single attempt.
        raise ApplicationError(
            f"LLM response parse error: {exc}",
            type="LLMCallError",
            non_retryable=True,
        ) from exc

    logger.info(
        "handle_reply_qualifier_activity: draft complete",
        extra={
            "telemetry_id": telemetry_id,
            "ota_hotel_id": ota_hotel_id,
            "draft_length": len(draft),
            "offered_sample": offered_sample,
        },
    )

    return QualifierResult(draft=draft, offered_sample=offered_sample)
