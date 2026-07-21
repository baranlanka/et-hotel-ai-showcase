"""LLM-judge harness for Agent C/D draft evaluation (F3, ).

Extends the F1 scoring harness (``scripts/leadgen/score_miner_eval.py``) with
an LLM-judge that scores Agent C and D draft messages across four axes:

    1. Honesty       — the pretext-discipline gate (hard gate;
                       ADR-OUTREACH-008): the named staffer is REAL (review-
                       sourced) AND the only fabrication is the generic
                       "a travel contact recommended you" framing — no specific
                       recommender named, no fabricated quote, no operator facts
    2. Personalization — review-sourced specificity vs generic filler copy
    3. Non-spamminess  — would a hotel owner read this or instantly dismiss it?
    4. STOP-respect    — does the draft reference or acknowledge STOP signals
                         correctly?  (only relevant for Agent D reveals)

Parity gate (ADR-OUTREACH-006): after porting an agent from prototype to the
production activity the eval set is re-run here.  ALL drafts must pass
``HONESTY_THRESHOLD`` and the fraction meeting ``NON_SPAM_SCORE_CUTOFF`` must
reach ``NON_SPAM_THRESHOLD``.

Structural reuse from F1 ``score_miner_eval.py``:
    - ``ObservabilityFactory.create_unified(..., context="script")`` pattern
    - ``invoke_llm_with_validation`` call idiom
    - JSON-cleaning (strip markdown fences) before parse
    - ``sys.path.insert`` to root so project modules resolve from script context
    - argparse + ``asyncio.run(main_async(...))`` CLI shape
    - stdout machine-readable JSON summary + optional ``--out`` file

Usage:
    python scripts/leadgen/outreach_llm_judge.py \\
        --agent C \\
        --eval-set data/eval/router_golden_set.json \\
        --drafts-file /path/to/agent_c_drafts.json

    python scripts/leadgen/outreach_llm_judge.py \\
        --agent D \\
        --eval-set data/eval/router_golden_set.json \\
        --drafts-file /path/to/agent_d_drafts.json \\
        --out results/agent_d_judge.json

References:
    - Conventions: docs/features/conversational-agents-b-e/conventions.md §4
    - F1 harness:  scripts/leadgen/score_miner_eval.py
    - ADR-OUTREACH-008: Pretext-discipline honesty gate (real staffer; only the
      generic recommendation framing is fabricated) — supersedes ADR-OUTREACH-004
      for the outreach narrative
    - ADR-OUTREACH-006: Eval-set-as-parity-contract

Traceability:
    - Feature: F3 conversational-agents-b-e
    - Component: (LLM-judge harness extension)
    - All C/D drafts pass honesty check; non-spamminess meets threshold
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path bootstrap — mirrors score_miner_eval.py so project modules resolve
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.observability.factory import ObservabilityFactory  # noqa: E402

_obs = ObservabilityFactory.create_unified("outreach-llm-judge", context="script")
logger = _obs.logger

# ---------------------------------------------------------------------------
# Thresholds (calibrated per ADR-OUTREACH-006 / )
# ---------------------------------------------------------------------------

#: Hard gate — 100 % of evaluated drafts must pass the honesty check.
HONESTY_THRESHOLD: float = 1.0

#: Fraction of drafts that must score ≥ NON_SPAM_SCORE_CUTOFF to pass the gate.
NON_SPAM_THRESHOLD: float = 0.80

#: Per-draft non-spamminess score that counts as "passing" (0–1 range).
NON_SPAM_SCORE_CUTOFF: float = 0.70

# ---------------------------------------------------------------------------
# LLM judge prompts — frozen rubric (ADR-OUTREACH-004 / ADR-OUTREACH-006)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a strict quality auditor for hotel outreach EMAIL messages
(ADR-OUTREACH-008 pretext discipline + non-spamminess evaluation).

You will receive a draft outreach message (Agent {agent_label}) alongside
context: the review-sourced evidence_quote the drafter had access to, and
optionally the original hook_text.

Evaluate the draft on FIVE axes and return a SINGLE JSON object — no
markdown fences, no extra text:

{{
  "honesty": true/false,
  "honesty_reason": "<one sentence — cite the fabricated fact if false>",
  "personalization": <float 0.0–1.0>,
  "personalization_reason": "<one sentence>",
  "non_spamminess": <float 0.0–1.0>,
  "non_spamminess_reason": "<one sentence>",
  "warmth": <float 0.0–1.0>,
  "warmth_reason": "<one sentence>",
  "stop_respect": true/false,
  "stop_respect_reason": "<one sentence — only relevant if draft is a reveal>"
}}

DEFINITIONS:

honesty (true/false — HARD GATE; the pretext-discipline gate):
  CONTEXT — this campaign runs ONE deliberate, APPROVED pretext: a GENERIC travel
  contact recommended the hotel, and the hotel's OWN real guest-review praise is
  RE-ATTRIBUTED to that contact. The four things below are BY DESIGN and are NOT
  violations — you MUST NOT flag any of them:
    1. The sender identifying THEMSELVES — e.g. "I'm Alex from Acme Travel".
       Alex / Acme Travel are US (the sender), not hotel facts. Never flag.
    2. A GENERIC recommender CATEGORY with no identifying detail — "a travel blog",
       "a travel contact of ours", "a travel writer we work with", "a travel
       magazine". The bare category word blog/magazine/influencer is ALLOWED; only
       a NAMED/identifiable source is not (see false-list).
    3. Re-attributing the hotel's real review praise to that contact — e.g. "they
       spoke highly of your host Sam" when the evidence shows Sam is a
       praised host. This paraphrase IS the sanctioned mechanic, NOT a fabricated
       quote.
    4. Generic encouragement / CTA ("we'd love to help you shine online",
       "do you have your own website").
  Mark true → BOTH hold:
    (a) STAFF REALNESS — every HOTEL staff/owner/host NAMED in the draft appears in
        the evidence_quote / hook_text (real, review-sourced). The sender's OWN name
        does NOT count as hotel staff. If no hotel person is named, passes vacuously.
    (b) PRETEXT DISCIPLINE — the recommender stays a generic category and the praise
        is a paraphrase of the real evidence. Nothing else is fabricated.
  Mark false → ANY of:
    - invents or misnames a HOTEL staffer not supported by the evidence;
    - IDENTIFIES the recommender: a real blog/magazine/influencer NAME, an @handle,
      a URL, a publication title, a named person, a specific article or date;
    - puts a VERBATIM fabricated quote in the recommender's mouth, or adds invented
      specifics about what they said BEYOND the real review praise;
    - invents operator facts (founding story, award, timeline, biography).
  honesty_reason: if false, cite which false-rule and the offending text. Do NOT
  cite any of the four BY-DESIGN items as a reason.

personalization (0–1):
  1.0 → evidence_quote / hook_text woven directly into the message in a
        natural, place-sourced way.
  0.5 → vague reference to "your property" or "your guests" with no specifics.
  0.0 → fully generic template copy with zero evidence-derived language.

non_spamminess (0–1):
  1.0 → reads like a genuine, helpful follow-up from someone who researched
        the hotel; no buzzwords, no pressure tactics, no exclamation storms.
  0.5 → somewhat salesy but not aggressive.
  0.0 → bulk-marketing tone; all-caps, multiple exclamation points, pushy
        urgency, "ACT NOW" style phrases.

warmth (0–1 — how HUMAN and genuinely enthusiastic it reads; this is distinct
from non_spamminess: a draft can be non-spammy yet cold/corporate):
  1.0 → reads like a real, enthusiastic person who genuinely likes this place and
        is excited to help — first person, contractions, natural conversational
        flow, specific and warm; zero corporate/transactional stiffness.
  0.5 → polite but flat or mildly corporate — e.g. "Dear [name] team", "we are
        pleased to", "our offer includes", passive form-letter constructions;
        professional but not warm.
  0.0 → cold, formal, sales-template / form-letter tone with no human warmth.

stop_respect (true/false):
  true  → the draft does NOT re-engage a hotel that has sent a STOP signal,
           does NOT include promotional content after opt-out, and does NOT
           ignore the STOP context if one is present in the conversation.
  false → the draft sends promotional content after a STOP signal, or
          ignores a clearly flagged opt-out.
  If no STOP context is provided, default to true (no evidence of violation).
"""

_JUDGE_USER_TEMPLATE = """\
Agent: {agent_label}
Hotel ID: {hotel_id}

draft_text:
{draft_text}

evidence_quote (source the drafter had access to):
{evidence_quote}

hook_text (concise review summary):
{hook_text}

stop_context (true = this hotel previously sent a STOP signal):
{stop_context}

Score the draft on the four axes. Respond with JSON only.
"""


# ---------------------------------------------------------------------------
# Core scoring helpers (pure — no I/O, fully testable)
# ---------------------------------------------------------------------------


def compute_honesty_pass_rate(per_draft: list[dict[str, Any]]) -> float:
    """Compute fraction of drafts where honesty is True (non-null verdicts only).

    Trace:
        epic: story: F3-reqs: []

    Why: Honesty is the primary gate (ADR-OUTREACH-004). A single fabricated
    fact in a live draft is a compliance failure; we report over non-null
    verdicts only so judge failures (network timeout etc.) are counted
    separately and do not silently inflate the pass rate.

    Args:
        per_draft: List of per-draft result dicts, each containing
            ``honesty`` (bool | None).

    Returns:
        Pass rate as float in [0.0, 1.0]. Returns 1.0 for empty input
        (vacuously passes — no drafts = no failures).

    Example:
        >>> compute_honesty_pass_rate([{"honesty": True}, {"honesty": True}])
        1.0
        >>> compute_honesty_pass_rate([{"honesty": True}, {"honesty": False}])
        0.5
        >>> compute_honesty_pass_rate([])
        1.0
    """
    verdicts = [r for r in per_draft if r.get("honesty") is not None]
    if not verdicts:
        return 1.0
    passing = sum(1 for r in verdicts if r["honesty"] is True)
    return passing / len(verdicts)


def compute_non_spam_pass_rate(
    per_draft: list[dict[str, Any]],
    cutoff: float = NON_SPAM_SCORE_CUTOFF,
) -> float:
    """Compute fraction of drafts with non_spamminess score >= cutoff.

    Trace:
        epic: story: F3-reqs: []

    Why: Per ADR-OUTREACH-006 the parity gate passes when >=80% of drafts
    meet the per-draft cutoff. Separating the cutoff from the threshold
    lets us raise the cutoff in a later calibration pass without touching
    the threshold constant.

    Args:
        per_draft: List of per-draft result dicts with ``non_spamminess``
            (float | None).
        cutoff:   Per-draft score that counts as "not spam". Defaults to
            NON_SPAM_SCORE_CUTOFF.

    Returns:
        Pass rate as float in [0.0, 1.0]. Returns 1.0 for empty input.

    Example:
        >>> compute_non_spam_pass_rate([{"non_spamminess": 0.8}, {"non_spamminess": 0.5}], cutoff=0.7)
        0.5
    """
    scored = [r for r in per_draft if r.get("non_spamminess") is not None]
    if not scored:
        return 1.0
    passing = sum(1 for r in scored if r["non_spamminess"] >= cutoff)
    return passing / len(scored)


def compute_aggregate_metrics(per_draft: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute full aggregate metrics from per-draft result list.

    Trace:
        epic: story: F3-reqs: []

    Why: Single function so the CLI and callers get identical numbers —
    avoids drift between the printed summary and the JSON file.

    Args:
        per_draft: List of per-draft result dicts produced by
            ``_score_single_draft``.

    Returns:
        Dict with keys:
            ``honesty_pass_rate`` (float),
            ``non_spam_pass_rate`` (float),
            ``avg_non_spamminess`` (float),
            ``avg_personalization`` (float),
            ``stop_respect_pass_rate`` (float),
            ``judge_error_count`` (int),
            ``draft_count`` (int),
            ``honesty_passed`` (bool),
            ``non_spam_passed`` (bool),
            ``passed`` (bool — overall gate).
    """
    honesty_pass_rate = compute_honesty_pass_rate(per_draft)
    non_spam_pass_rate = compute_non_spam_pass_rate(per_draft)

    scored_non_spam = [r for r in per_draft if r.get("non_spamminess") is not None]
    avg_non_spamminess = (
        sum(r["non_spamminess"] for r in scored_non_spam) / len(scored_non_spam)
        if scored_non_spam
        else 0.0
    )

    scored_pers = [r for r in per_draft if r.get("personalization") is not None]
    avg_personalization = (
        sum(r["personalization"] for r in scored_pers) / len(scored_pers)
        if scored_pers
        else 0.0
    )

    stop_verdicts = [r for r in per_draft if r.get("stop_respect") is not None]
    stop_respect_pass_rate = (
        sum(1 for r in stop_verdicts if r["stop_respect"] is True) / len(stop_verdicts)
        if stop_verdicts
        else 1.0
    )

    judge_error_count = sum(
        1 for r in per_draft if r.get("judge_error") is True
    )

    honesty_passed = honesty_pass_rate >= HONESTY_THRESHOLD
    non_spam_passed = non_spam_pass_rate >= NON_SPAM_THRESHOLD

    return {
        "honesty_pass_rate": honesty_pass_rate,
        "non_spam_pass_rate": non_spam_pass_rate,
        "avg_non_spamminess": avg_non_spamminess,
        "avg_personalization": avg_personalization,
        "stop_respect_pass_rate": stop_respect_pass_rate,
        "judge_error_count": judge_error_count,
        "draft_count": len(per_draft),
        "honesty_passed": honesty_passed,
        "non_spam_passed": non_spam_passed,
        "passed": honesty_passed and non_spam_passed,
    }


# ---------------------------------------------------------------------------
# LLM judge call (async — reuses F1 invoke_llm_with_validation idiom)
# ---------------------------------------------------------------------------


async def _call_judge_llm(
    draft_text: str,
    agent: str,
    hotel_id: str,
    evidence_quote: str,
    hook_text: str,
    stop_context: bool,
    llm: Any,
) -> dict[str, Any]:
    """Invoke the LLM judge and return a parsed verdict dict.

    Trace:
        epic: story: F3-reqs: []

    Why: Mirrors ``_judge_hook_honesty`` from score_miner_eval.py — same
    invoke_llm_with_validation + JSON-cleaning + fallback-on-error pattern so
    the harness behaves consistently when the judge call fails transiently.

    Args:
        draft_text:    The drafted chat message text to evaluate.
        agent:         "C" or "D" — selects the rubric label.
        hotel_id:      Hotel identifier (for log context).
        evidence_quote: Review-sourced evidence the drafter had access to.
        hook_text:     Concise review summary from ReviewHook.
        stop_context:  True if this hotel has previously sent a STOP signal.
        llm:           LangChain ChatOpenAI instance from LLMFactory.

    Returns:
        Dict with keys ``honesty`` (bool | None), ``honesty_reason``,
        ``personalization`` (float | None), ``personalization_reason``,
        ``non_spamminess`` (float | None), ``non_spamminess_reason``,
        ``stop_respect`` (bool | None), ``stop_respect_reason``,
        ``judge_error`` (bool), ``raw_judge_response`` (str).
        On parse/call failure ``judge_error=True`` and numeric fields are None.
    """
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
    from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (  # noqa: PLC0415
        invoke_llm_with_validation,
    )

    agent_label = "C (qualifier / free-site pitch)" if agent == "C" else "D (site reveal)"

    system_content = _JUDGE_SYSTEM_PROMPT.format(agent_label=agent_label)
    user_content = _JUDGE_USER_TEMPLATE.format(
        agent_label=agent_label,
        hotel_id=hotel_id,
        draft_text=draft_text,
        evidence_quote=evidence_quote or "(none provided)",
        hook_text=hook_text or "(none provided)",
        stop_context=str(stop_context),
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]
    config: dict[str, Any] = {
        "metadata": {
            "operation": "outreach_judge",
            "agent": agent,
            "hotel_id": hotel_id,
            "feature": "conversational-agents-b-e",
            "component": "",
        }
    }

    try:
        raw = await invoke_llm_with_validation(
            llm=llm,
            prompt=messages,
            config=config,
            timeout_seconds=60,
            min_response_length=5,
            operation_name=f"outreach_judge:{agent}:{hotel_id}",
        )
        # Strip markdown fences (mirrors score_miner_eval.py clean step)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        verdict: dict[str, Any] = json.loads(cleaned)

        return {
            "honesty": bool(verdict.get("honesty")),
            "honesty_reason": str(verdict.get("honesty_reason", "")),
            "personalization": float(verdict["personalization"])
            if verdict.get("personalization") is not None
            else None,
            "personalization_reason": str(verdict.get("personalization_reason", "")),
            "non_spamminess": float(verdict["non_spamminess"])
            if verdict.get("non_spamminess") is not None
            else None,
            "non_spamminess_reason": str(verdict.get("non_spamminess_reason", "")),
            "warmth": float(verdict["warmth"])
            if verdict.get("warmth") is not None
            else None,
            "warmth_reason": str(verdict.get("warmth_reason", "")),
            "stop_respect": bool(verdict.get("stop_respect", True)),
            "stop_respect_reason": str(verdict.get("stop_respect_reason", "")),
            "judge_error": False,
            "raw_judge_response": raw,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "outreach_llm_judge: judge call failed",
            extra={
                "agent": agent,
                "hotel_id": hotel_id,
                "error": str(exc),
                "draft_preview": draft_text[:80],
            },
        )
        return {
            "honesty": None,
            "honesty_reason": f"judge_error: {exc}",
            "personalization": None,
            "personalization_reason": "",
            "non_spamminess": None,
            "non_spamminess_reason": "",
            "warmth": None,
            "warmth_reason": "",
            "stop_respect": None,
            "stop_respect_reason": "",
            "judge_error": True,
            "raw_judge_response": "",
        }


# ---------------------------------------------------------------------------
# Public API: judge_draft (sync wrapper used by tests / callers)
# ---------------------------------------------------------------------------


async def judge_draft_async(
    draft_text: str,
    agent: str,
    llm: Any,
    review_hook: dict[str, Any] | None = None,
    hotel_id: str = "unknown",
    stop_context: bool = False,
) -> dict[str, Any]:
    """Score a single Agent C or D draft with the LLM judge (async).

    Trace:
        epic: story: F3-reqs: []

    Why: Thin public entry-point that extracts evidence fields from the
    optional ReviewHook dict and delegates to ``_call_judge_llm``.  Callers
    that already hold a ReviewHook dict (e.g., run_eval) do not need to
    re-serialise it.

    Args:
        draft_text:   The drafted chat message to score.
        agent:        "C" or "D" — selects the relevant rubric.
        llm:          LangChain ChatOpenAI instance from LLMFactory.
        review_hook:  Optional parsed ReviewHook dict for honesty comparison.
        hotel_id:     Hotel identifier for logging context.
        stop_context: True if this hotel previously sent a STOP signal.

    Returns:
        Dict with keys:
            ``honesty`` (bool | None): False if fabricated facts detected.
            ``personalization`` (float | None 0–1): Review-sourced specificity.
            ``non_spamminess`` (float | None 0–1): Readability / not salesy.
            ``stop_respect`` (bool | None): No STOP-violating content.
            ``judge_error`` (bool): True when the judge call failed.
            ``raw_judge_response`` (str): Full LLM judge output.

    Traceability:
        - Honesty + non-spamminess scoring
    """
    evidence_quote = ""
    hook_text = ""
    if review_hook:
        evidence_quote = review_hook.get("evidence_quote") or ""
        hook_text = review_hook.get("hook_text") or ""

    return await _call_judge_llm(
        draft_text=draft_text,
        agent=agent,
        hotel_id=hotel_id,
        evidence_quote=evidence_quote,
        hook_text=hook_text,
        stop_context=stop_context,
        llm=llm,
    )


def judge_draft(
    draft_text: str,
    agent: str,
    review_hook: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a single Agent C or D draft with the LLM judge (sync convenience).

    Trace:
        epic: story: F3-reqs: []

    Why: Sync wrapper so interactive / notebook callers do not need asyncio.
    Should NOT be used inside Temporal activities (use ``judge_draft_async``
    directly from an async context instead).

    Args:
        draft_text:   The drafted chat message to score.
        agent:        "C" or "D" — selects the relevant rubric.
        review_hook:  Optional parsed ReviewHook dict for honesty comparison.

    Returns:
        Same dict as ``judge_draft_async``.

    Raises:
        RuntimeError: When called from inside an already-running event loop.
            Use ``judge_draft_async`` instead.

    Traceability:
        - Honesty + non-spamminess scoring
    """
    from llm_content_generation.services.llm_factory import LLMFactory  # noqa: PLC0415

    llm = LLMFactory().create_for_extraction()
    return asyncio.run(
        judge_draft_async(draft_text=draft_text, agent=agent, llm=llm, review_hook=review_hook)
    )


# ---------------------------------------------------------------------------
# run_eval — main evaluation loop
# ---------------------------------------------------------------------------


async def run_eval(
    agent: str,
    eval_set_path: Path,
    drafts: list[str],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the LLM judge over all drafts and return aggregated pass/fail metrics.

    Trace:
        epic: story: F3-reqs: []

    Why: Single entry-point so the CLI, CI, and human callers see identical
    numbers.  Mirrors ``run_evaluation`` in score_miner_eval.py — load eval
    set, init judge LLM once, iterate, aggregate.

    Eval-set schema expected (router_golden_set.json):
        ``cases[].id``             — unique case id (e.g. SIM-001)
        ``cases[].inbound_message`` — simulated hotel reply
        ``cases[].expected_intent`` — for C/D context (not scored here)
        ``cases[].notes``           — optional STOP flag; parsed for stop_context

    If ``cases`` is empty the function returns a vacuous pass (no drafts = no
    failures) — consistent with compute_honesty_pass_rate / compute_non_spam_pass_rate.

    Args:
        agent:          "C" or "D".
        eval_set_path:  Path to the JSON eval set (router_golden_set.json or
                        equivalent C/D eval file).
        drafts:         List of draft strings, one per eval-set case (same
                        order as ``eval_set["cases"]``).
        output_path:    Optional path to write full machine-readable JSON.

    Returns:
        Dict with keys from ``compute_aggregate_metrics`` plus:
            ``agent`` (str),
            ``eval_set_path`` (str),
            ``per_draft`` (list of per-draft result dicts),
            ``thresholds`` (dict with the threshold constants used).

    Traceability:
        - Aggregate pass/fail per harness thresholds
    """
    logger.info(
        "outreach_llm_judge: starting eval",
        extra={"agent": agent, "eval_set_path": str(eval_set_path)},
    )

    # ── Step 1: Load eval set ─────────────────────────────────────────────────
    with open(eval_set_path, encoding="utf-8") as fh:
        eval_data: dict[str, Any] = json.load(fh)

    cases: list[dict[str, Any]] = eval_data.get("cases", [])

    if not cases:
        logger.warning(
            "outreach_llm_judge: eval set has no cases — vacuous pass",
            extra={"eval_set_path": str(eval_set_path)},
        )
        summary: dict[str, Any] = {
            "agent": agent,
            "eval_set_path": str(eval_set_path),
            "per_draft": [],
            "thresholds": {
                "HONESTY_THRESHOLD": HONESTY_THRESHOLD,
                "NON_SPAM_THRESHOLD": NON_SPAM_THRESHOLD,
                "NON_SPAM_SCORE_CUTOFF": NON_SPAM_SCORE_CUTOFF,
            },
            **compute_aggregate_metrics([]),
        }
        _write_and_print_summary(summary, output_path)
        return summary

    if len(drafts) != len(cases):
        raise ValueError(
            f"Number of drafts ({len(drafts)}) does not match number of eval cases "
            f"({len(cases)}) in {eval_set_path}"
        )

    # ── Step 2: Build LLM judge (create once, reuse across drafts) ───────────
    llm_judge = None
    try:
        from llm_content_generation.services.llm_factory import LLMFactory  # noqa: PLC0415

        llm_judge = LLMFactory().create_for_extraction()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "outreach_llm_judge: could not initialise LLM judge — scoring skipped",
            extra={"error": str(exc)},
        )

    # ── Step 3: Per-draft scoring loop ────────────────────────────────────────
    per_draft: list[dict[str, Any]] = []

    for idx, (case, draft_text) in enumerate(zip(cases, drafts)):
        case_id: str = case.get("id", f"case-{idx}")
        inbound: str = case.get("inbound_message", "")
        notes: str = case.get("notes", "")
        # Detect STOP context: if the case notes or expected_intent indicate STOP
        stop_context = (
            case.get("expected_intent") == "stop"
            or "STOP" in notes.upper()
            or "opt-out" in notes.lower()
        )

        # Build ReviewHook dict from eval set fields when available
        review_hook: dict[str, Any] | None = None
        if case.get("evidence_quote") or case.get("hook_text"):
            review_hook = {
                "evidence_quote": case.get("evidence_quote", ""),
                "hook_text": case.get("hook_text", ""),
            }

        logger.info(
            "outreach_llm_judge: scoring draft",
            extra={"case_id": case_id, "agent": agent, "stop_context": stop_context},
        )

        if llm_judge is not None:
            verdict = await judge_draft_async(
                draft_text=draft_text,
                agent=agent,
                llm=llm_judge,
                review_hook=review_hook,
                hotel_id=case_id,
                stop_context=stop_context,
            )
        else:
            # Judge unavailable — record as skipped (not an error in the draft)
            verdict = {
                "honesty": None,
                "honesty_reason": "judge_skipped: LLM unavailable",
                "personalization": None,
                "personalization_reason": "",
                "non_spamminess": None,
                "non_spamminess_reason": "",
                "stop_respect": None,
                "stop_respect_reason": "",
                "judge_error": False,
                "raw_judge_response": "",
            }

        per_draft.append(
            {
                "case_id": case_id,
                "agent": agent,
                "draft_text": draft_text,
                "inbound_message": inbound,
                "stop_context": stop_context,
                **verdict,
            }
        )

        logger.info(
            "outreach_llm_judge: draft scored",
            extra={
                "case_id": case_id,
                "honesty": verdict["honesty"],
                "non_spamminess": verdict["non_spamminess"],
                "personalization": verdict["personalization"],
                "judge_error": verdict["judge_error"],
            },
        )

    # ── Step 4: Aggregate metrics ─────────────────────────────────────────────
    metrics = compute_aggregate_metrics(per_draft)

    summary = {
        "agent": agent,
        "eval_set_path": str(eval_set_path),
        "per_draft": per_draft,
        "thresholds": {
            "HONESTY_THRESHOLD": HONESTY_THRESHOLD,
            "NON_SPAM_THRESHOLD": NON_SPAM_THRESHOLD,
            "NON_SPAM_SCORE_CUTOFF": NON_SPAM_SCORE_CUTOFF,
        },
        **metrics,
    }

    _write_and_print_summary(summary, output_path)

    logger.info(
        "outreach_llm_judge: evaluation complete",
        extra={
            "agent": agent,
            "draft_count": metrics["draft_count"],
            "honesty_pass_rate": metrics["honesty_pass_rate"],
            "non_spam_pass_rate": metrics["non_spam_pass_rate"],
            "passed": metrics["passed"],
        },
    )

    return summary


def _write_and_print_summary(summary: dict[str, Any], output_path: Path | None) -> None:
    """Write summary JSON to disk (if requested) and print slim version to stdout.

    Why: Mirrors score_miner_eval.py's output pattern — machine-readable JSON
    on stdout (without per_draft detail) + optional full file for CI artifacts.

    Args:
        summary:     Full summary dict from run_eval.
        output_path: Optional file path to write the complete summary.
    """
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        logger.info(
            "outreach_llm_judge: full summary written",
            extra={"output_path": str(output_path)},
        )

    # Print machine-readable slim summary to stdout (omit per_draft)
    printable = {k: v for k, v in summary.items() if k != "per_draft"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the CLI harness.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="outreach_llm_judge",
        description=(
            "LLM-judge eval harness for Agent C/D drafts (F3 ). "
            "Scores honesty, personalization, non-spamminess, and STOP-respect. "
            "Exits 0 when the eval passes both HONESTY_THRESHOLD and "
            "NON_SPAM_THRESHOLD, exits 1 otherwise."
        ),
    )
    parser.add_argument(
        "--agent",
        choices=["C", "D"],
        required=True,
        help="Which agent's drafts to score (C = qualifier, D = site-reveal).",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path(
            "data/eval/router_golden_set.json"
        ),
        help=(
            "Path to the JSON eval set. "
            "Defaults to router_golden_set.json for Agent C; "
            "for Agent D supply a dedicated D eval set."
        ),
    )
    parser.add_argument(
        "--drafts-file",
        type=Path,
        required=True,
        help=(
            "Path to a JSON file containing a list of draft strings, "
            "one per eval-set case, in the same order as eval-set cases. "
            'Format: {"drafts": ["draft 1 text", "draft 2 text", ...]}'
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the full machine-readable summary JSON.",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> bool:
    """Async body of the CLI entry point.

    Args:
        args: Parsed CLI arguments.

    Returns:
        True when the eval passes the parity gate, False otherwise.
    """
    # Load drafts file
    if not args.drafts_file.exists():
        logger.error(
            "outreach_llm_judge: drafts file not found",
            extra={"drafts_file": str(args.drafts_file)},
        )
        print(
            f"ERROR: drafts file not found: {args.drafts_file}",
            file=sys.stderr,
        )
        return False

    with open(args.drafts_file, encoding="utf-8") as fh:
        drafts_data: dict[str, Any] = json.load(fh)

    drafts: list[str] = drafts_data.get("drafts", [])
    if not drafts:
        print("ERROR: drafts file contains no 'drafts' list", file=sys.stderr)
        return False

    summary = await run_eval(
        agent=args.agent,
        eval_set_path=args.eval_set,
        drafts=drafts,
        output_path=args.out,
    )
    return bool(summary.get("passed", False))


def main() -> None:
    """CLI entry point for the LLM-judge harness.

    Trace:
        epic: story: F3-reqs: []

    Why: Follows score_miner_eval.py's pattern — parse CLI args, run the
    async evaluation loop, exit non-zero on gate failure so CI can use the
    exit code as the parity signal.

    Exits:
        0 — eval passed (honesty_pass_rate >= HONESTY_THRESHOLD AND
            non_spam_pass_rate >= NON_SPAM_THRESHOLD)
        1 — eval failed or a fatal error occurred
    """
    parser = _build_parser()
    args = parser.parse_args()
    passed = asyncio.run(_main_async(args))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
