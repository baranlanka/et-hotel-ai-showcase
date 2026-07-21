"""Temporal activity: Agent B — write the cold-outreach opener (F3).

Given a ReviewHook from Agent A + the hotel name, Agent B writes the FULL opener
email with the LLM (prompt: Langfuse ``langchain/outreach_opener_fill`` — Langfuse
is the single source of truth; if it is unavailable the activity raises a retryable
``PromptUnavailableError`` rather than sending a stale prompt). Voice: a casual acquaintance
recommended the place and complimented a REAL named staffer for a specific thing,
then ONE open question about taking reservations directly vs through OTA. No
company name, no opt-out footer. Honesty: only the real staff name + real praise
(ADR-OUTREACH-004); an empty hook → a deterministic generic opener (no LLM call).
Returns the opener STRING; the workflow sends it verbatim — channel-agnostic (email
now, chat DMs later). The old M1 ``{{1}}/{{2}}`` template mechanic was removed.

Decorator stack (mandatory — conventions.md §1):
    @activity.defn → @trace → @log_io → @inject_telemetry_id

All app.* imports are lazy (inside the function body) — never at module level
in Temporal activities.

References:
    - Spec: docs/features/agentic-outreach/spec.md
    - ADR-OUTREACH-004: Honest place-sourced prose (no fabricated facts)
    - Canonical pattern: app/temporal/activities/leadgen/mine_reviews_activity.py

Traceability:
    - Feature: F3 conversational-agents-b-e
    - Component: C1 (Agent B)
"""

from __future__ import annotations

import logging

from temporalio import activity
from temporalio.exceptions import ApplicationError

from common.observability import log_io, trace
from app.core.observability.factory import ObservabilityFactory
from app.core.observability.telemetry_context import inject_telemetry_id

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
# Generic opener fallback (used when no review hook is available AND when the
# output guard fires on a poisoned personalized opener — §3 defense)
# ---------------------------------------------------------------------------

_GENERIC_OPENER: str = (
    "An acquaintance of mine recently told me about your place and really "
    "recommended it — they had lovely things to say about how welcoming and "
    "helpful your team is. I'm curious how you handle your reservations: do you "
    "take them directly through your own website, or mostly through "
    "OTA and the other platforms?\n\nAlex"
)

# Stage-aware re-engagement nudges (silent-window bump). A follow-up must (a) never
# re-send the cold opener verbatim, and (b) resurface the LAST thing we actually
# asked — which depends on where the conversation is. Sending the opener's
# book-direct question when the thread already moved on to the free-site offer reads
# as a bot that isn't following along. Keyed by funnel_stage; unknown / human-owned
# stages get a topic-neutral bump. Deterministic + hook-free by design: a bump should
# not re-cite the staffer/scene, and for a one-line nudge, saying the RIGHT thing
# beats LLM personalization. Each is shorter/lighter than the opener and never verbatim.
_REENGAGEMENT_BY_STAGE: dict[str, str] = {
    # opener_sent — we asked whether they'd take a reservation directly.
    "opener_sent": (
        "Apologies if my earlier note got buried — just floating it back up in case "
        "it wasn't the right moment. I'm still hoping to come and stay, and I'd "
        "genuinely rather book with you directly than through the usual sites. Is "
        "that something we could sort out? No worries at all if it's not the right "
        "time.\n\nAlex"
    ),
    # qualifier_sent — we offered to build them a free sample website.
    "qualifier_sent": (
        "Apologies if my last note slipped by — just floating it back up. No pressure "
        "at all, but the offer still stands: I'm happy to put together a quick sample "
        "of a site for you, free and with nothing owed either way, so you can actually "
        "see it. Want me to go ahead?\n\nAlex"
    ),
    # site_revealed — we sent a live sample page and asked what they thought.
    "site_revealed": (
        "Just circling back on the sample page I put together for you — no rush at "
        "all. I'd genuinely love your honest thoughts whenever you get a minute, even "
        "if the honest answer is \"not for us.\"\n\nAlex"
    ),
}

# Topic-neutral bump for any stage without dedicated copy (e.g. escalated/unknown):
# resurfaces the last note without naming a specific new ask.
_REENGAGEMENT_NEUTRAL: str = (
    "Apologies if my earlier note got buried — just floating it back up in case it "
    "wasn't the right moment. No worries at all if now isn't the time; happy to pick "
    "things up whenever suits you.\n\nAlex"
)


def _reengagement_for_stage(funnel_stage: str) -> str:
    """Return the silent-window nudge whose topic matches the funnel stage.

    Why: The re-engagement bump must resurface the last thing we asked; the topic
    is chosen by funnel_stage so a nudge never re-asks a question the thread already
    moved past. Unknown / human-owned stages fall back to the topic-neutral bump.
    """
    return _REENGAGEMENT_BY_STAGE.get(funnel_stage, _REENGAGEMENT_NEUTRAL)

# ---------------------------------------------------------------------------
# M1 character constraints (spec.md §1)
# ---------------------------------------------------------------------------

# Character cap for the full opener email Agent B writes — 1100 allows
# 2-3 short paragraphs + a sign-off.
M1_VAR2_MAX_CHARS: int = 1100


# ---------------------------------------------------------------------------
# Lazy logger (mirrors mine_reviews_activity._LazyLogger pattern)
# ---------------------------------------------------------------------------


class _LazyLogger(logging.Logger):
    """Proxy logger that resolves the unified _obs logger at call time.

    Why: log_io() captures the logger object by value at module-load time.
    At that point get_obs() returns None (Temporal sandbox constraint).
    This proxy delegates every call to the real logger once get_obs() succeeds.

    Trace:
        feature: conversational-agents-b-e (F3)
        component: C1 (Agent B)
    """

    def __init__(self) -> None:
        super().__init__("leadgen-outreach-lazy")

    def _real(self) -> logging.Logger:
        obs = get_obs()
        return obs.logger if obs is not None else activity.logger  # type: ignore[return-value]

    def log(self, level: int, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self._real().log(level, msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self._real().info(msg, *args, **kwargs)

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self._real().debug(msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self._real().warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self._real().error(msg, *args, **kwargs)

    def exception(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self._real().exception(msg, *args, **kwargs)


_lazy_logger = _LazyLogger()


# ---------------------------------------------------------------------------
# Pure helpers (no I/O — safe to test in isolation)
# ---------------------------------------------------------------------------


def _truncate_var2(phrase: str) -> str:
    """Strip trailing punctuation and enforce VAR2 character limit.

    Trace:
        feature: conversational-agents-b-e (F3)
        component: C1 (Agent B)
        reqs: []

    Why: Strips a trailing period/whitespace from the multi-paragraph email
    and hard-caps length at M1_VAR2_MAX_CHARS to avoid a reject-and-retry
    on messages that exceed the channel limit.

    Args:
        phrase: Raw LLM output (the full opener email).

    Returns:
        Phrase with trailing ". " stripped and hard-capped at VAR2 limit.
    """
    stripped = phrase.strip().rstrip(".")
    return stripped[:M1_VAR2_MAX_CHARS]


def _validate_hook_text_present(var2: str, hook_text: str) -> bool:
    """Return True if the hook_text's key tokens appear in var2.

    Trace:
        feature: conversational-agents-b-e (F3)
        component: C1 (Agent B)
        reqs: []

    Why: ADR-OUTREACH-004 requires that the hook phrase is review-sourced
    and grounded in the evidence the Miner extracted. A token-overlap check
    (not a strict substring match) tolerates LLM rephrasing while catching
    completely fabricated content.

    The check is intentionally permissive: at least 2 tokens from
    hook_text must appear in var2. The LLM-judge (harness) applies a
    stricter honesty gate at eval time.

    Args:
        var2:      The LLM-produced {{2}} phrase.
        hook_text: The review-sourced hook_text from ReviewHook.

    Returns:
        True if enough token overlap exists; False otherwise.
    """
    hook_tokens = set(hook_text.lower().split())
    var2_tokens = set(var2.lower().split())
    # Exclude very common stopwords from the overlap count
    stopwords = {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "is",
        "at",
        "by",
        "and",
        "or",
        "for",
    }
    meaningful = (hook_tokens - stopwords) & (var2_tokens - stopwords)
    return len(meaningful) >= 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


@activity.defn(name="fill_opener_template_activity")
@trace(
    span_name="fill_opener_template_activity",
    include=["ota_hotel_id", "telemetry_id"],
)
@log_io(logger=_lazy_logger)
@inject_telemetry_id
async def fill_opener_template_activity(
    ota_hotel_id: str,
    hotel_name: str,
    review_hook_json: str,
    telemetry_id: str = None,
    is_reengagement: bool = False,
    funnel_stage: str = "opener_sent",
) -> str:
    """Write the cold-outreach opener email for Agent B.

    Trace:
        feature: conversational-agents-b-e (F3)
        component: C1 (Agent B)
        reqs: [, ]

    Why: Agent B takes a ReviewHook + hotel name and writes a FULL free-form
    cold-outreach email via the LLM (Langfuse ``langchain/outreach_opener_fill``
    prompt, deepseek). Returns a plain capped string (≤ M1_VAR2_MAX_CHARS).
    Falls back to the deterministic _GENERIC_OPENER when the hook is absent
    or trips the input/output guards. The brand-echo guard (ADR-003) and
    character-limit enforcement prevent two known failure modes from reaching
    the send layer.

    Flow:
        1. Validate inputs — raise non-retryable on empty hotel_name or
           malformed (non-empty but unparseable) review_hook_json.
        2. If review_hook_json is empty/blank: return the deterministic
           _GENERIC_OPENER (no LLM call). Handles re-engagement timer fires
           where no review hook is available.
        3. Parse ReviewHook from JSON.
        4. Build the full opener email via LLM call:
           a. System prompt + conditional brand-guard instruction.
           b. User prompt carries hook_text / staff_name / evidence_quote /
              highlights.
           c. Strip trailing punctuation; enforce M1_VAR2_MAX_CHARS cap.
        5. Return the opener string.

    Args:
        ota_hotel_id: The hotel being contacted (for logging/tracing).
        hotel_name:       Display name used for tone context in the prompt.
        review_hook_json: JSON-serialised ReviewHook from Agent A / F1.
                          Empty string is ACCEPTED — triggers generic fallback.
                          Non-empty but malformed JSON raises a ValidationError.
        telemetry_id:     Injected by @inject_telemetry_id; do not pass explicitly.
        is_reengagement:  When True, produce a silent-window re-engagement nudge
                          rather than the full opener. The nudge is a deterministic,
                          hook-free, stage-aware bump (see funnel_stage) — never a
                          verbatim re-send of the opener (anti-spam) and never a
                          question the thread already moved past. Defaults to False so
                          the normal opener call is unchanged.
        funnel_stage:     Conversation stage at the re-engagement timer fire
                          (opener_sent / qualifier_sent / site_revealed / ...). Only
                          consulted when is_reengagement=True; selects which pending
                          topic the nudge resurfaces (unknown stage → neutral bump).

    Returns:
        The full opener email string, max M1_VAR2_MAX_CHARS chars,
        no trailing full-stop, review-sourced, complying with
        ADR-OUTREACH-003/004. The generic fallback is returned
        deterministically when review_hook_json is empty.

    Raises:
        ApplicationError(non_retryable=True, type="ValidationError"):
            review_hook_json is non-empty but cannot be parsed, required
            fields are empty, or hotel_name is blank. Empty review_hook_json
            is NOT a validation error — it produces a generic fallback.
        ApplicationError(non_retryable=False, type="TemplateFillError"):
            LLM call or response processing failed — Temporal retries.

    Traceability:
        - opener email within M1_VAR2_MAX_CHARS, review-sourced
        - Brand-echo guard when name_matches_brand==True
    """
    # ── Lazy imports (CRITICAL: never at module level in Temporal activities) ──
    import json  # noqa: PLC0415

    from app.temporal.activities.leadgen.models import ReviewHook  # noqa: PLC0415
    from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (  # noqa: PLC0415
        invoke_llm_with_validation,
    )
    from llm_content_generation.services.llm_factory import LLMFactory  # noqa: PLC0415

    from app.temporal.activities.leadgen._langfuse_prompt import (  # noqa: PLC0415
        fetch_outreach_prompt,
    )

    logger = get_obs().logger if get_obs() else activity.logger

    # ── Step 1: Input validation ──────────────────────────────────────────────
    if not hotel_name or not hotel_name.strip():
        raise ApplicationError(
            "hotel_name is required and must not be blank",
            type="ValidationError",
            non_retryable=True,
        )

    if not ota_hotel_id or not ota_hotel_id.strip():
        raise ApplicationError(
            "ota_hotel_id is required and must not be blank",
            type="ValidationError",
            non_retryable=True,
        )

    # ── Step 1b: Re-engagement nudge — stage-aware, deterministic, hook-free ──
    # A silent-window bump short-circuits the whole opener machinery: it must NOT
    # re-send the cold opener and must resurface the topic that matches funnel_stage
    # (book-direct / free-site offer / sample page). It is intentionally deterministic
    # and hook-free — a bump should not re-cite the staffer/scene, and for a one-line
    # nudge saying the RIGHT thing beats LLM personalization. review_hook_json is
    # ignored here on purpose.
    if is_reengagement:
        nudge = _truncate_var2(_reengagement_for_stage(funnel_stage))
        logger.info(
            "fill_opener_template_activity: stage-aware re-engagement nudge",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "funnel_stage": funnel_stage,
                "nudge_len": len(nudge),
            },
        )
        return nudge

    # ── Step 2: Parse ReviewHook (or use Tier-3 generic fallback) ────────────
    # An empty/blank review_hook_json means no review hook was available (e.g.
    # re-engagement timer fire). In this case we fall back to a generic Tier-3
    # opener — mirroring handle_reply_qualifier_activity's missing-hook path.
    # A non-empty but unparseable JSON is still a hard ValidationError (caller bug).
    hook: ReviewHook | None = None
    if not review_hook_json or not review_hook_json.strip():
        # [ASSUMPTION]: empty hook → Tier-3 generic fallback; no LLM call needed
        # because the generic phrase is deterministic (no evidence to shape).
        hook = None
        logger.info(
            "fill_opener_template_activity: review_hook_json is empty — "
            "using Tier-3 generic fallback",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
            },
        )
    else:
        try:
            hook_dict = json.loads(review_hook_json)
            hook = ReviewHook.model_validate(hook_dict)
        except Exception as exc:
            raise ApplicationError(
                f"Failed to parse review_hook_json for hotel {ota_hotel_id}: {exc}",
                type="ValidationError",
                non_retryable=True,
            ) from exc

    if hook is not None:
        logger.info(
            "fill_opener_template_activity: hook parsed",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "tier": hook.tier,
                "name_matches_brand": hook.name_matches_brand,
            },
        )

    # ── Step 4: Build opener — LLM call (with hook) or Tier-3 generic fallback ─
    # When hook is None (empty review_hook_json), we produce the canonical Tier-3
    # generic phrase deterministically rather than calling the LLM without evidence.
    # This mirrors handle_reply_qualifier_activity's missing-hook fallback path.
    if hook is None:
        # No review hook (Agent A found nothing) — send the deterministic generic
        # opener (no LLM call, no invented staff name). Re-engagement never reaches
        # here: it short-circuited to a stage-aware nudge in Step 1b above.
        var_2 = _truncate_var2(_GENERIC_OPENER)
        logger.info(
            "fill_opener_template_activity: Tier-3 generic fallback applied",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "var_2_len": len(var_2),
            },
        )
        return var_2

    # Format highlights as a readable inline list for the prompt (or "(none)" when empty).
    # The model-layer validator already dropped unsafe highlights from hook.highlights,
    # but we still run the activity-level input guard below for defense-in-depth.
    highlights_text = (
        "; ".join(hook.highlights) if hook.highlights else "(none)"
    )

    # Brand-guard instruction TEXT. Langfuse is the single source of truth for the
    # opener prompt TEMPLATE (langchain/outreach_opener_fill, which carries a
    # {{brand_guard_instruction}} placeholder); this string is the runtime VALUE the
    # activity substitutes into that variable — supplied only when the staff name
    # echoes the hotel brand. The deterministic staff_name="N/A" forcing below is the
    # primary brand-echo defense; this is the advisory prompt-level guard.
    _brand_guard_value = (
        "7. IMPORTANT — BRAND GUARD: The staff name matches the hotel brand name. In BEAT 2,\n"
        "   Do NOT phrase the praise as \"your [name] looked after us\" — it reads as if the\n"
        "   hotel's own name is a staff member. For this email, BEAT 2 drops the named staffer\n"
        "   entirely: speak to the team and the welcome in general (grounded in the evidence\n"
        "   quote), and let one of the property/experience highlights carry the warmth. The\n"
        "   beat ORDER stays the same.\n"
    )
    # Prompt + model from Langfuse (langchain/outreach_opener_fill). The brand guard
    # stays conditional in code (it depends on hook.name_matches_brand) and is injected
    # as the {{brand_guard_instruction}} template variable; var2_max is passed the same
    # way. When Langfuse is unavailable the fetch returns (None, None) and we raise.
    template_vars = {
        "var2_max": str(M1_VAR2_MAX_CHARS),
        "brand_guard_instruction": (
            _brand_guard_value if hook.name_matches_brand else ""
        ),
        "hotel_name": hotel_name,
        "tier": f"Tier {hook.tier}",
        "hook_text": hook.hook_text,
        # Brand-guard (deterministic): when the staff name echoes the hotel brand,
        # never hand the name to the LLM. The prompt-level guard alone is advisory —
        # deepseek still wrote "your Rasa looked after us" — and the brand-echo
        # post-check only matched the hotel name, so the staff token slipped through.
        # Forcing N/A makes the opener lean on the team + highlights instead.
        "staff_name": "N/A" if hook.name_matches_brand else (hook.staff_name or "N/A"),
        "role": "N/A" if hook.name_matches_brand else (hook.role or "N/A"),
        "evidence_quote": hook.evidence_quote,
        "highlights": highlights_text,
        "is_reengagement": is_reengagement,
    }
    # §3: INPUT guard — hook_text/evidence_quote/highlights come from UNTRUSTED
    # scraped review text. If they carry injection / markup / PII, skip the LLM entirely
    # and use the generic opener. The model-layer validator (_sanitize_highlights) already
    # dropped unsafe highlights; this activity-level check is defense-in-depth.
    from app.leadgen.outreach.output_guard import detect_unsafe_opener  # noqa: PLC0415

    hook_input_unsafe = (
        detect_unsafe_opener(hook.hook_text or "")
        or detect_unsafe_opener(hook.evidence_quote or "")
        or next(
            (detect_unsafe_opener(h) for h in hook.highlights if detect_unsafe_opener(h)),
            None,
        )
    )
    if hook_input_unsafe is not None:
        logger.warning(
            "fill_opener_template_activity: review hook tripped INPUT guard — "
            "using generic opener (no LLM personalization)",
            extra={
                "reason": hook_input_unsafe,
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
            },
        )
        return _truncate_var2(_GENERIC_OPENER)

    messages, model = fetch_outreach_prompt("langchain/outreach_opener_fill", template_vars)
    if messages is None:
        raise ApplicationError(
            "Langfuse prompt 'langchain/outreach_opener_fill' is unavailable "
            "(disabled or unreachable) — retrying.",
            type="PromptUnavailableError",
            non_retryable=False,  # retryable: Temporal retries rather than writing
                                  # the opener from an absent/stale prompt
        )

    try:
        llm_factory = LLMFactory()
        llm = (
            llm_factory.create_from_prompt_config(model)
            if model
            else llm_factory.create_for_extraction()
        )

        config = {
            "metadata": {
                "operation": "fill_opener_template",
                "ota_hotel_id": ota_hotel_id,
                "tier": hook.tier,
                "feature": "conversational-agents-b-e",
                "component": "C1",
                "telemetry_id": telemetry_id,
            }
        }

        raw_response: str = await invoke_llm_with_validation(
            llm=llm,
            prompt=messages,
            config=config,
            timeout_seconds=60,
            min_response_length=5,
            operation_name=f"fill_opener_template:{ota_hotel_id}",
        )
    except Exception as exc:
        logger.error(
            "fill_opener_template_activity: LLM call failed",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "error": str(exc),
            },
        )
        raise ApplicationError(
            f"LLM call failed for hotel {ota_hotel_id}: {exc}",
            type="TemplateFillError",
            non_retryable=False,
        ) from exc

    # ── Step 5: Post-process and enforce constraints ──────────────────────────
    var_2 = _truncate_var2(raw_response)

    if not var_2:
        raise ApplicationError(
            f"LLM returned empty {{2}} phrase for hotel {ota_hotel_id}",
            type="TemplateFillError",
            non_retryable=False,
        )

    # review-sourced grounding check (deterministic post-check).
    # The phrase must stay anchored in the Miner's hook_text — ADR-OUTREACH-004
    # (no fabricated praise). The token-overlap test is intentionally permissive
    # (it tolerates honest LLM rephrasing — e.g. "Ask for Nimal" → "special
    # mention to Nimal"), so a weak overlap is a SIGNAL for review, not a hard
    # block: the strict honesty gate is the LLM-judge at eval time. We log a
    # warning rather than fabricate-block so a legitimate rephrase is never
    # silently downgraded to the Tier-3 generic in production.
    if not _validate_hook_text_present(var_2, hook.hook_text):
        logger.warning(
            "fill_opener_template_activity: var_2 weakly grounded in hook_text — "
            "possible LLM drift from review source (flagged for eval-time review)",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "tier": hook.tier,
                "hook_text": hook.hook_text,
                "var_2": var_2,
            },
        )

    # Brand-echo guard verification (deterministic post-check).
    # When name_matches_brand is True, the brand-guard prompt forces Tier-3
    # voice. Log a warning if the hotel name still appears in the phrase so
    # operators can review — the prompt-level guard is the primary defence.
    if hook.name_matches_brand and hotel_name.lower() in var_2.lower():
        logger.warning(
            "fill_opener_template_activity: brand echo detected in var_2 despite guard",
            extra={
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "hotel_name": hotel_name,
                "var_2": var_2,
            },
        )

    # §3: Output guard — review supply-chain injection defense.
    # The opener was built from an UNTRUSTED ReviewHook extracted from scraped
    # review text. A poisoned review may have smuggled markup, prompt-injection
    # vocabulary, or guest PII through the miner's LLM. The deterministic guard
    # is the code-level backstop: if it fires, discard the personalized opener
    # and fall back to the safe generic phrase. The poisoned content is NEVER
    # returned to the caller.
    from app.leadgen.outreach.output_guard import detect_unsafe_opener  # noqa: PLC0415

    reason = detect_unsafe_opener(var_2)
    if reason is not None:
        logger.warning(
            "fill_opener_template_activity: opener tripped output guard — using generic fallback",
            extra={
                "reason": reason,
                "telemetry_id": telemetry_id,
                "ota_hotel_id": ota_hotel_id,
                "tier": hook.tier,
            },
        )
        var_2 = _truncate_var2(_GENERIC_OPENER)

    logger.info(
        "fill_opener_template_activity: opener email filled",
        extra={
            "telemetry_id": telemetry_id,
            "ota_hotel_id": ota_hotel_id,
            "tier": hook.tier,
            "var_2_len": len(var_2),
        },
    )

    return var_2
