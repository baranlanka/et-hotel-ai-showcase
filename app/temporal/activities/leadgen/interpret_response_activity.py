"""Temporal activity: Agent E — response interpreter / router (F3).

Fires on EVERY inbound hotel reply. Classifies the message into a closed
``InterpretedResponse`` schema and routes to the correct next action.
Owns STOP compliance (intent==stop → next_action==close; opted_out flag set
by the workflow). Human-escalation routing is included for ambiguous or
sensitive replies.

Decorator stack (mandatory — conventions.md §1):
    @activity.defn → @trace → @log_io → @inject_telemetry_id

All app.* imports are lazy (inside the function body) — never at module
level in Temporal activities.

References:
    - Conventions: docs/features/conversational-agents-b-e/conventions.md §3
    - ADR-OUTREACH-001: Structured LLM call; LangGraph only if genuine branching
    - FR-K: Response interpreter / router
    - Canonical pattern: app/temporal/activities/leadgen/audit_hotel_website_activity.py

Traceability:
    - Feature: F3 conversational-agents-b-e
    - Component: C4 (Agent E)
    - STOP → intent==stop, next_action==close; workflow marks opted_out
    - Output validates against closed InterpretedResponse enums;
                 human-needed replies route to next_action==escalate
"""
from __future__ import annotations

import asyncio

from temporalio import activity
from temporalio.exceptions import ApplicationError

from common.observability import log_io, trace
from app.core.observability.factory import ObservabilityFactory
from app.core.observability.telemetry_context import inject_telemetry_id
from app.temporal.workflows.leadgen.models import (
    IntentEnum,
    InterpretedResponse,
    NextActionEnum,
    SentimentEnum,
)

# Hard timeout (seconds) on the router LLM call. Gives Agent E timeout parity
# with Agent C (whose invoke_llm_with_validation wraps asyncio.wait_for): a hung
# call fails fast as a retryable error instead of relying solely on the Temporal
# start_to_close_timeout — and, after retries, the workflow fails closed to
# escalate (OWASP LLM10/LLM09).
_ROUTER_TIMEOUT_SECONDS: int = 60

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


# ---------------------------------------------------------------------------
# STOP keyword guard — evaluated BEFORE the LLM call ()
# ---------------------------------------------------------------------------

_STOP_KEYWORDS: frozenset[str] = frozenset(
    {
        "stop",
        "unsubscribe",
        "do not contact",
        "don't contact",
        "dont contact",
        "remove me",
        "opt out",
        "opt-out",
        "no more messages",
        "please stop",
    }
)


def _is_stop_signal(text: str) -> bool:
    """Return True if the message is an unambiguous STOP / opt-out request.

    Why: STOP compliance is a hard legal requirement (GDPR / chat policy).
    The keyword gate runs BEFORE the LLM call so it can never be overridden by
    an LLM mis-classification. The LLM then handles the nuanced mid-funnel cases.

    Args:
        text: Raw inbound message, any case / whitespace.

    Returns:
        True when the normalised text matches any known STOP keyword.
    """
    normalised = text.strip().lower()
    # Exact match first (e.g. "STOP")
    if normalised in _STOP_KEYWORDS:
        return True
    # Substring match for multi-word phrases (e.g. "Please stop contacting me")
    return any(kw in normalised for kw in _STOP_KEYWORDS)


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


@activity.defn(name="interpret_response_activity")
@trace(span_name="interpret_response_activity", include=["ota_hotel_id", "telemetry_id"])
@log_io(logger=get_obs().logger if get_obs() else activity.logger)
@inject_telemetry_id
async def interpret_response_activity(
    ota_hotel_id: str,
    inbound_message: str,
    funnel_state: str,
    telemetry_id: str = None,
    inbound_datamarked: str | None = None,
) -> InterpretedResponse:
    """Classify an inbound hotel reply and return a routing decision.

    This activity fires on EVERY inbound message. It must never swallow a
    STOP signal — STOP replies take absolute precedence over any other
    classification ().

    Flow:
        1. Validate inputs (non-retryable ApplicationError on empty strings).
        2. STOP keyword gate: if the message is unambiguously a STOP, return
           immediately without an LLM call. This is the only path that avoids
           the LLM entirely — safety over latency.
        3. Structured LLM call via ``LLMFactory().create_for_extraction()``
           bound to ``InterpretedResponse`` via ``with_structured_output``.
        4. Post-LLM STOP guard: if the LLM returned intent==stop, enforce
           next_action==close (prevents prompt-injection drift).
        5. Return validated InterpretedResponse.

    Args:
        ota_hotel_id: The hotel that replied (for logging/tracing).
        inbound_message:  Raw text from the hotel's inbound email reply.
        funnel_state:     Current funnel stage (passed in by the workflow
                          so the classifier has context).
        telemetry_id:     Injected by @inject_telemetry_id.
        inbound_datamarked: OWASP LLM01 — the datamarked (whitespace→marker)
                          form of the reply from the input guard. Sent to the
                          LLM so it can tell hotel DATA from instructions. The
                          deterministic STOP gate keeps reading the CLEAN
                          ``inbound_message`` (datamarking would defeat
                          multi-word phrase matching). None → fall back to
                          ``inbound_message``.

    Returns:
        InterpretedResponse with closed ``intent`` and ``next_action`` enums.
        A STOP reply MUST yield intent==stop and next_action==close.
        Ambiguous / sensitive replies MUST yield next_action==escalate.

    Raises:
        ApplicationError(non_retryable=True, type="ValidationError"):
            ota_hotel_id or inbound_message is empty.
        ApplicationError(non_retryable=False, type="LLMCallError"):
            Transient LLM failure — Temporal retries.

    Traceability:
        - STOP → intent==stop, next_action==close
        - Closed-enum validation; escalate for human-needed replies
    """
    # ── Lazy imports (CRITICAL: never at module level in Temporal activities) ──
    from llm_content_generation.services.llm_factory import LLMFactory  # noqa: PLC0415

    from app.temporal.activities.leadgen._langfuse_prompt import (      # noqa: PLC0415
        fetch_outreach_prompt,
    )

    logger = get_obs().logger if get_obs() else activity.logger

    # ── 1. Input validation (non-retryable) ──────────────────────────────────
    if not ota_hotel_id or not ota_hotel_id.strip():
        raise ApplicationError(
            "ota_hotel_id must be a non-empty string",
            type="ValidationError",
            non_retryable=True,
        )
    if not inbound_message or not inbound_message.strip():
        raise ApplicationError(
            "inbound_message must be a non-empty string",
            type="ValidationError",
            non_retryable=True,
        )

    # ── 2. STOP keyword gate () — before LLM, unconditional ─────────
    if _is_stop_signal(inbound_message):
        logger.info(
            "STOP keyword gate triggered — skipping LLM",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "funnel_state": funnel_state,
            },
        )
        return InterpretedResponse(
            intent=IntentEnum.stop,
            sentiment=SentimentEnum.neutral,
            funnel_state=funnel_state,
            next_action=NextActionEnum.close,
        )

    # ── 3. Structured LLM classification ─────────────────────────────────────
    # Prompt + model come from Langfuse (langchain/outreach_router, model pinned
    # in config) — Langfuse is the SINGLE source of truth for the prompt. When it
    # is disabled/unreachable the fetch returns (None, None); we RAISE a retryable
    # error rather than send a dummy payload, so the STOP-critical router pauses
    # (Temporal retries) instead of misclassifying on a stale/absent prompt.
    # The LLM sees the datamarked form (OWASP LLM01 spotlighting); the STOP gate
    # above already ran on the clean `inbound_message`.
    template_vars = {
        "funnel_state": funnel_state or "unknown",
        "inbound_message": inbound_datamarked or inbound_message,
    }
    try:
        messages, model = fetch_outreach_prompt("langchain/outreach_router", template_vars)
        if messages is None:
            raise ApplicationError(
                "Langfuse prompt 'langchain/outreach_router' is unavailable "
                "(disabled or unreachable) — retrying.",
                type="PromptUnavailableError",
                non_retryable=False,  # retryable: Temporal retries; the conversation
                                      # pauses rather than routing on a stale prompt
            )

        llm_base = (
            LLMFactory().create_from_prompt_config(model)
            if model
            else LLMFactory().create_for_extraction()
        )
        # OWASP LLM09: pin deterministic routing. The factory derives temperature
        # from LLMConfig (it takes no kwarg), so set it on the instance before
        # binding the structured-output schema.
        try:
            llm_base.temperature = 0
        except (AttributeError, ValueError):
            pass  # model without a settable temperature — keep the factory default
        structured_llm = llm_base.with_structured_output(InterpretedResponse)

        # OWASP LLM10: hard wall-clock timeout (parity with Agent C, whose
        # invoke_llm_with_validation wraps asyncio.wait_for). A hung call raises
        # asyncio.TimeoutError → caught below → retryable LLMCallError → (after
        # retries) the workflow fails closed to escalate.
        result: InterpretedResponse = await asyncio.wait_for(
            structured_llm.ainvoke(messages),
            timeout=_ROUTER_TIMEOUT_SECONDS,
        )
        if result is None:
            raise ApplicationError(
                "Agent E returned an empty classification",
                type="LLMCallError",
                non_retryable=False,
            )
    except ApplicationError:
        raise  # Propagate non-retryable errors from nested calls
    except Exception as exc:
        msg = str(exc)
        logger.error(
            "LLM classification failed",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "error": msg,
            },
        )
        raise ApplicationError(
            f"LLM call failed: {msg}",
            type="LLMCallError",
            non_retryable=False,
        )

    # ── 4. Post-LLM STOP enforcement ( hard guard) ──────────────────
    # [ASSUMPTION]: If the LLM classifies intent==stop, next_action MUST be close
    # regardless of what the LLM suggested. This prevents prompt-injection drift.
    if result.intent == IntentEnum.stop and result.next_action != NextActionEnum.close:
        logger.warning(
            "LLM returned intent=stop with non-close next_action — enforcing close",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "llm_next_action": result.next_action.value,
            },
        )
        result = InterpretedResponse(
            intent=result.intent,
            sentiment=result.sentiment,
            funnel_state=result.funnel_state,
            next_action=NextActionEnum.close,
            confidence=result.confidence,
        )

    # ── 5. Deterministic acceptance guard (accept-after-offer reveal) ────────
    # At qualifier_sent, an "interested" reply IS the hotel accepting the sample
    # offer we just made — it must REVEAL (send_D), never loop another qualifier.
    # Router rule 4 says exactly this, but the LLM applies it unreliably for softer
    # acceptances ("sounds interesting, send it over" routed to send_C ~half the
    # time while "yes please, show me" got send_D), so CODE enforces it — same
    # pattern as the STOP guard above. We key off the AUTHORITATIVE input
    # funnel_state (not the LLM's echoed one). Safe: send_D is human-gated
    # (_AUTO_SEND_D_ENABLED=False) → this fires an escalate + ops hand-off,
    # never an auto-spend; and a low-confidence forced send_D is still downgraded to
    # escalate by the workflow's confidence gate. Both page a human either way.
    if (
        result.intent == IntentEnum.interested
        and (funnel_state or "").strip() == "qualifier_sent"
        and result.next_action == NextActionEnum.send_c
    ):
        logger.info(
            "interested@qualifier_sent — forcing send_C->send_D (acceptance reveal)",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "confidence": result.confidence,
            },
        )
        result = InterpretedResponse(
            intent=result.intent,
            sentiment=result.sentiment,
            funnel_state=result.funnel_state,
            next_action=NextActionEnum.send_d,
            confidence=result.confidence,
        )

    logger.info(
        "response classified",
        extra={
            "telemetry_id": telemetry_id,
            "ota_hotel_id": ota_hotel_id,
            "intent": result.intent.value,
            "next_action": result.next_action.value,
            "funnel_state": result.funnel_state,
        },
    )
    return result
