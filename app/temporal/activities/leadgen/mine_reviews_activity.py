"""Temporal activity: extract a ReviewHook from hotel reviews via LLM (F1 — ).

Ports the frozen prompt + schema from the lab prototype (C3) into a Temporal
activity. Decorator stack, lazy-import, and observability patterns mirror
audit_hotel_website_activity.py exactly (ADR-OUTREACH-001).

Decorator stack (MANDATORY order per conventions.md §1):
    @activity.defn → @trace → @log_io → @inject_telemetry_id

All DB, storage, settings, and LLM imports are LAZY (inside the function body)
to preserve Temporal sandbox safety (conventions.md §2).

Parity gate: re-run C4 golden set via harness against this activity; tier
accuracy and tier-1 precision must stay within 5% of the prototype baseline
( / ADR-OUTREACH-006).

Feature: review-miner-agent-a (F1)
Release: agentic-outreach
Component: ACs: , 
"""
from __future__ import annotations

from temporalio import activity
from temporalio.exceptions import ApplicationError

from common.observability import log_io, trace
from app.core.observability.factory import ObservabilityFactory
from app.core.observability.telemetry_context import inject_telemetry_id
from app.temporal.activities.leadgen.models import MineReviewsInput, ReviewHook

# ---------------------------------------------------------------------------
# Lazy observability — NEVER initialise at module level in activities.
# (Temporal sandbox violation risk — mirrors audit_hotel_website_activity.py:37-50)
# ---------------------------------------------------------------------------

_obs = None


def get_obs():
    """Lazy initialization of the unified observability manager."""
    global _obs
    if _obs is None:
        try:
            _obs = ObservabilityFactory.create_unified(
                service_name="leadgen-review-miner", context="activity"
            )
        except Exception:
            pass  # Fallback: will use activity.logger
    return _obs


import logging  # noqa: E402  (standard library — safe at module level)


class _LazyLogger(logging.Logger):
    """Proxy logger that resolves the unified _obs logger at call time.

    Why: log_io() captures the logger object by value at module-load time.
    At that point get_obs() returns None (Temporal sandbox constraint —
    ObservabilityFactory must not run at import time). Passing the plain
    `get_obs().logger if get_obs() else activity.logger` expression means
    the unified logger is NEVER wired (it is always None at decoration time).

    This proxy is passed to @log_io at module load and delegates every
    log() / info() / debug() call to the real unified logger once get_obs()
    succeeds. Safe to construct at module level because it does no I/O or
    SDK init itself.
    """

    def __init__(self) -> None:
        # logging.Logger requires a name; use a sentinel that matches the service.
        super().__init__("leadgen-review-miner-lazy")

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
# Pure helper functions (no I/O — safe to test in isolation without Temporal)
# ---------------------------------------------------------------------------


def _extract_reviews_list(data: dict) -> list[dict]:
    """Extract review records from a B2 reviews JSON blob.

    Trace:
        feature: review-miner-agent-a (F1)
        component: reqs: []

    Why: B2 reviews JSON uses two shapes depending on the scraper version.
    Primary path data["data"]["reviews"] is the current writer; legacy uses
    data["reviews"] directly.

    Args:
        data: Parsed JSON dict from B2 reviews file.

    Returns:
        List of raw review dicts (may be empty).
    """
    try:
        return data["data"]["reviews"]
    except (KeyError, TypeError):
        pass
    return data.get("reviews", [])


def _filter_english_reviews(reviews: list[dict]) -> list[dict]:
    """Return reviews whose textDetails.lang starts with 'en'.

    Trace:
        feature: review-miner-agent-a (F1)
        component: reqs: []

    Why: Only English reviews feed the extraction prompt to avoid hallucination
    from language mismatch. lang field uses ISO 639-1 prefix (conventions.md §5).

    Args:
        reviews: Raw list of review dicts.

    Returns:
        Filtered list — only English reviews.
    """
    return [
        r for r in reviews
        if (r.get("textDetails") or {}).get("lang", "").lower().startswith("en")
    ]


def _format_reviews_for_prompt(reviews: list[dict], max_reviews: int = 30) -> str:
    """Serialise review text for the extraction prompt.

    Trace:
        feature: review-miner-agent-a (F1)
        component: reqs: []

    Why: Combines positiveText + negativeText + title into a numbered list capped
    at max_reviews to stay within the LLM token budget.

    Args:
        reviews: List of review dicts (already English-filtered).
        max_reviews: Maximum number of reviews to include.

    Returns:
        Multi-line string with numbered review excerpts, or a placeholder.
    """
    lines: list[str] = []
    for i, r in enumerate(reviews[:max_reviews], start=1):
        td = r.get("textDetails") or {}
        # Use `or ""` (not get-default): real B2 reviews carry explicit JSON
        # null for absent text fields, so get("title", "") still returns None
        # and .strip() would raise AttributeError. Synthetic tests never hit
        # this because they always populate the fields.
        title = (td.get("title") or "").strip()
        pos = (td.get("positiveText") or "").strip()
        neg = (td.get("negativeText") or "").strip()
        parts = [p for p in [title, pos, neg] if p]
        if parts:
            lines.append(f"[{i}] " + " | ".join(parts))
    return "\n".join(lines) if lines else "(no review text available)"


def _check_name_matches_brand(hook_dict: dict, hotel_name: str | None) -> bool:
    """Return True if hook staff_name equals (or is contained in) the hotel brand.

    Trace:
        feature: review-miner-agent-a (F1)
        component: reqs: []

    Why: Some hotels are named after a person (e.g. "Rasa" hotel, staff "Rasa").
    Setting this flag deterministically (never trusting LLM output) lets the
    outreach copy layer suppress echo phrases that read as tautological.

    Args:
        hook_dict: Parsed hook dict (contains staff_name key).
        hotel_name: Optional hotel brand name for comparison.

    Returns:
        True when staff_name and hotel_name share a token (case-insensitive).
    """
    staff_name: str | None = hook_dict.get("staff_name")
    if not staff_name or not hotel_name:
        return False
    # Tokenize on ANY non-alphanumeric character so possessive forms like
    # "Nimal's" or "Rasa's" are split into {"nimal", "s"} and {"rasa", "s"}
    # respectively, allowing a match against the plain brand token "Nimal" /
    # "Rasa". Using str.split() alone leaves "nimal's" intact and misses the
    # overlap. No new imports needed — str.isalnum() is stdlib.
    staff_tokens = set("".join(c if c.isalnum() else " " for c in staff_name.lower()).split())
    brand_tokens = set("".join(c if c.isalnum() else " " for c in hotel_name.lower()).split())
    return bool(staff_tokens & brand_tokens)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Trace:
        feature: review-miner-agent-a (F1)
        component: Why: Despite prompt instructions some models wrap JSON in ```json fences.
    Strip them before parse to avoid json.JSONDecodeError.

    Args:
        text: Raw LLM response string.

    Returns:
        String with any leading/trailing markdown fences removed.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Temporal activity
# ---------------------------------------------------------------------------


@activity.defn(name="mine_reviews_activity")
@trace(span_name="mine_reviews_activity", include=["ota_hotel_id", "telemetry_id"])
@log_io(logger=_lazy_logger)  # _LazyLogger resolves to unified obs.logger at call time (C1 fix)
@inject_telemetry_id
async def mine_reviews_activity(
    input: MineReviewsInput,
    telemetry_id: str = None,
) -> ReviewHook:
    """Extract a ReviewHook from a hotel's English reviews via LLM.

    Ports the frozen prompt + schema from the C3 lab prototype into a production
    Temporal activity. Mirrors the decorator stack, lazy-import, and observability
    patterns from audit_hotel_website_activity.py (ADR-OUTREACH-001).

    Flow (all imports lazy — conventions.md §2):
        1. Validate/coerce Temporal dict payload into MineReviewsInput.
        2. Input guard — raises non-retryable on empty ota_hotel_id.
        3. Open DB session; call resolve_reviews_b2_path to get the geo-keyed B2
           reviews path (ADR-OUTREACH-002 — never operation_results stale key).
        4. Raise retryable ApplicationError when no reviews path is found (the
           hotel will be retried by Temporal; the caller may also choose to skip).
        5. Fetch reviews JSON from B2 via get_async_storage_adapter.
        6. Filter reviews where textDetails.lang startswith "en".
        7. Raise retryable error when no English reviews exist.
        8. Build extraction prompt (precision>recall, no invented names,
           ADR-OUTREACH-003 tier rules, ADR-OUTREACH-004 honesty rules).
        9. Call LLMFactory.create_for_extraction() with invoke_llm_with_validation.
        10. Parse raw JSON response, strip optional markdown fences.
        11. Apply deterministic name_matches_brand override ( — never
            trust LLM output for this flag).
        12. Validate parsed dict against ReviewHook via model_validate.
        13. Log completion telemetry; return ReviewHook.

    Retry safety (idempotency): the activity is read-only (no writes to DB or
    B2) — it is safe to retry from any step without side effects.

    Args:
        input: MineReviewsInput with ota_hotel_id and optional hotel_name.
        telemetry_id: Injected by @inject_telemetry_id; do not pass explicitly.

    Returns:
        ReviewHook with tier, hook_text, staff_name, role, evidence_quote,
        confidence, and name_matches_brand populated.

    Raises:
        ApplicationError(non_retryable=True, type="ValidationError"):
            Empty ota_hotel_id or malformed input payload.
        ApplicationError(non_retryable=False, type="MineReviewsNoPath"):
            No reviews B2 path found for hotel — Temporal retries until timeout.
        ApplicationError(non_retryable=False, type="MineReviewsNoEnglish"):
            Hotel has no English reviews — Temporal retries until timeout.
        ApplicationError(non_retryable=False, type="MineReviewsError"):
            LLM call, storage I/O, or JSON parse failure — Temporal retries.

    Traceability:
        parity gate — golden set re-run within 5% of prototype.
        mandatory decorator stack; lazy imports only (no module-level
                   settings/DB/LLM import); lazy get_obs() observability;
                   no print(), no logging.basicConfig.
    """
    # ── Showcase note ──────────────────────────────────────────────────────────
    # The pure helper functions above carry the review-mining logic and ARE
    # exercised on synthetic data by the eval harness
    # (scripts/eval/run_outreach_review_poisoning.py, which re-runs the miner
    # prompt over adversarial review fixtures). The LIVE data path — resolving a
    # hotel's reviews from the operational review store, then running the miner
    # LLM — depends on the business data layer that is intentionally NOT shipped
    # in this public showcase. The activity is retained as an architecture
    # artifact; invoking it live raises a clear, non-retryable error rather than
    # importing the excluded data layer.

    # ── Handle dict input from Temporal serialization ──
    if isinstance(input, dict):
        input = MineReviewsInput.model_validate(input)

    logger = get_obs().logger if get_obs() else activity.logger

    # ── Input validation guard ────────────────────────────────────────────────
    if not input.ota_hotel_id:
        raise ApplicationError(
            "ota_hotel_id is required",
            type="ValidationError",
            non_retryable=True,
        )

    logger.info(
        "mine_reviews_activity: live review-store retrieval is excluded from the "
        "showcase build",
        extra={"telemetry_id": telemetry_id, "ota_hotel_id": input.ota_hotel_id},
    )
    raise ApplicationError(
        "mine_reviews_activity live data path is not available in the showcase "
        "(operational review-store integration excluded). The review-mining logic "
        "is demonstrated on synthetic data by "
        "scripts/eval/run_outreach_review_poisoning.py.",
        type="NotAvailableInShowcase",
        non_retryable=True,
    )
