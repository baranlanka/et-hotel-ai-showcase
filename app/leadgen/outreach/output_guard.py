"""Output guard for Agent C's outbound draft — OWASP LLM01 / LLM07 defense (§3).

A deterministic, non-LLM leak-check run in the workflow BETWEEN Agent C returning
its draft and ``send_outbound_message_activity``. If our own scaffolding, a chat
role marker, or the internal datamark char appears in the hotel-facing draft, the
workflow drops the draft and escalates to a human (fail closed) rather than send a
possibly-injected reply under our name.

``detect_unsafe_opener`` is a broader guard applied by Agent B (fill_opener_template_activity)
immediately after the LLM returns the opener. Agent B builds its opener from an UNTRUSTED
ReviewHook that was extracted from scraped OTA review text. A poisoned review can
smuggle HTML markup, prompt-injection vocabulary, or a guest's PII (email/phone) into the
miner's extraction output, which then flows into the opener. ``detect_unsafe_opener`` catches
all categories that ``detect_outbound_leak`` covers PLUS these review-supply-chain attack
vectors. When it fires, Agent B discards the personalised opener and falls back to the
deterministic generic phrase — so the injection is never sent to the hotel.

Deterministic on purpose: the regex layer is independent of the manipulable model
(the LLM judge is a Phase-2 second opinion, not the gate). Stdlib only (``re``);
sandbox-safe (the workflow imports this under
``workflow.unsafe.imports_passed_through()``).
"""
from __future__ import annotations

import re

from app.leadgen.outreach.input_guard import DATAMARK

# Chat role-turn markers that must never appear in a hotel email.
_ROLE_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?im)^\s*(system|assistant|user)\s*:"),
    re.compile(r"<\|.*?\|>", re.S),
)

# Distinctive strings from our own prompts / output schema. Their presence in a
# hotel-facing reply means the model leaked its scaffolding (or echoed an
# injection). Tuned to phrases that effectively never occur in genuine warm prose.
_SCAFFOLDING: re.Pattern[str] = re.compile(
    r"(?i)("
    r"Agent\s+[A-E]\b"
    r"|ADR-OUTREACH"
    r"|evidence_quote"
    r"|evidence_used"
    r"|funnel_state"
    r"|next_action"
    r"|interpret_response"
    r"|handle_reply_qualifier"
    r"|OUTPUT\s+—\s+a\s+single\s+JSON"
    r"|BANNED\s+corporate"
    r"|HONESTY\s+\(hard\s+rules"
    r"|ignore\s+(all\s+|the\s+)?(previous|prior|above)\s+(instruction|prompt)"
    r'|"draft"\s*:'
    r")"
)

# ---------------------------------------------------------------------------
# Opener-specific patterns (review supply-chain attack vectors — §3)
# ---------------------------------------------------------------------------

# Any HTML / XML markup tag. Reviews occasionally contain <script>, <img>, or
# HTML-template artifacts that should never reach an outbound email.
_MARKUP_TAG: re.Pattern[str] = re.compile(r"<[^>]+>")

# Prompt-injection vocabulary: classic "ignore instructions" phrasing, system
# prompt references, and direct injection keywords. Patterns are kept TIGHT to
# avoid false positives on warm prose ("you are a wonderful host" must not fire).
_INJECTION_VOCAB: re.Pattern[str] = re.compile(
    r"(?i)("
    r"ignore\s+(all\s+|the\s+|your\s+)?(previous|prior|above)?\s*(instruction|prompt|rule)s?"
    r"|disregard\s+(all|the|your|previous|prior)"
    r"|system\s+prompt"
    r"|your\s+(instructions|rules|guidelines|prompt)"
    r"|prompt\s+injection"
    r")"
)

# Email address pattern (guest PII smuggled through review text).
_EMAIL_PII: re.Pattern[str] = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Phone number pattern: 8+ digit runs with optional formatting (+, spaces, parens,
# dashes). The negative lookahead/lookbehind prevents matching naked integers
# that are part of longer words or identifiers (e.g. room numbers or hotel IDs
# with letters attached).
_PHONE_PII: re.Pattern[str] = re.compile(r"(?<!\w)\+?\d[\d ()\-]{7,}\d(?!\w)")


def detect_outbound_leak(draft: str) -> str | None:
    """Return the matched leak category, or None if the draft looks clean.

    Categories (checked in order): ``"datamark_echo"``, ``"role_marker"``,
    ``"scaffolding_phrase"``.
    """
    if not draft:
        return None
    if DATAMARK in draft:
        return "datamark_echo"
    if any(p.search(draft) for p in _ROLE_MARKERS):
        return "role_marker"
    if _SCAFFOLDING.search(draft):
        return "scaffolding_phrase"
    return None


def detect_unsafe_opener(opener: str) -> str | None:
    """Return the first safety-violation category found in *opener*, or None if clean.

    This guard is applied by Agent B (``fill_opener_template_activity``) immediately
    after the LLM returns the personalised opener. Agent B's opener is built from
    review-sourced hook text that came from UNTRUSTED scraped OTA text. A
    poisoned review can smuggle markup, prompt-injection vocabulary, or a guest's PII
    (email/phone) into the miner's ReviewHook and then into the opener. This function
    is the code-level backstop that catches those attacks deterministically, independent
    of whether the miner LLM resisted them.

    Checks (in order):
        1. All categories from ``detect_outbound_leak`` (datamark echo, role markers,
           scaffolding phrases) — re-uses the existing Agent C guard without
           duplication.
        2. ``"markup"``          — any HTML/XML tag (``<…>``).
        3. ``"injection_vocab"`` — prompt-injection phrases such as "ignore all previous
           instructions", "disregard the guidelines", "system prompt", etc.
        4. ``"email_pii"``       — an email address (guest PII from review text).
        5. ``"phone_pii"``       — a phone number (8+ digits with optional formatting).

    False-positive discipline: patterns are TIGHT. A warm opener such as
    "you are a wonderful host" must return None; only clearly adversarial or PII
    content should fire. When in doubt we accept the false-negative (a weak attacker
    gets through) rather than destroy legitimate personalisation.

    Args:
        opener: The LLM-produced opener string before it is returned by Agent B.

    Returns:
        The string category of the first violation found, or None if the opener is
        clean. The caller (Agent B) discards the personalised opener and substitutes
        the deterministic generic phrase whenever this returns non-None.
    """
    if not opener:
        return None

    # Re-use existing Agent C guard first (datamark, role markers, scaffolding).
    leak = detect_outbound_leak(opener)
    if leak is not None:
        return leak

    if _MARKUP_TAG.search(opener):
        return "markup"

    if _INJECTION_VOCAB.search(opener):
        return "injection_vocab"

    if _EMAIL_PII.search(opener):
        return "email_pii"

    if _PHONE_PII.search(opener):
        return "phone_pii"

    return None
