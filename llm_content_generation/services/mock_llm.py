"""Deterministic, offline MockChatModel for MODEL_BACKEND=mock (manifest S3).

This is the CI / offline backend. It implements exactly the slice of the
LangChain chat-model surface the extracted code uses:

    * ``.invoke(prompt)`` / ``.ainvoke(prompt)``   -> an ``AIMessage`` whose
      ``.content`` is a deterministic, request-shaped string.
    * ``.with_structured_output(Model)``            -> a runnable whose
      ``.invoke`` / ``.ainvoke`` return a **schema-valid** instance of ``Model``.
    * ``.bind(**kwargs)`` and a settable ``.temperature`` attribute
      (both used by the content-graph nodes / the router activity).

HONESTY (see the extraction manifest S3 + the task's honesty constraint):
The structured router output is a FIXED, NEUTRAL classification — it is NOT a
per-input gold answer. A mock cannot reproduce the production router's accuracy
(~92% came from a real 70B model), and this backend deliberately does not try:
running the router over the golden set with this backend yields an obviously-low,
clearly-labelled "illustrative" accuracy. What the mock DOES do faithfully is
return schema-valid objects and clean, non-leaking drafts, so the deterministic
security/guard properties (the reproducible-offline centrepiece) are exercised
for real.

The text generator inspects the *shape* of the request (does the prompt ask for a
draft? a review hook? is it a judge rubric?) to return a plausibly-shaped
response — this is ordinary mock behaviour and never computes a gold routing
label from the input.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

# A control sentinel used ONLY by the deterministic money-gate probe in the eval
# suite. It appears in no fixture / golden set — it exists so a test can force the
# router to emit send_D and prove the workflow's _AUTO_SEND_D_ENABLED=False gate
# overrides it to escalate. This is a test hook, not a gold-answer rig.
MONEY_GATE_SEND_D_SENTINEL = "MONEYGATEPROBEFORCESENDD"

# ---------------------------------------------------------------------------
# Canned, deliberately-generic response bodies
# ---------------------------------------------------------------------------

# A full, clean opener email. Chosen to satisfy the review-poisoning harness's
# det_check (line-start greeting, a sign-off, a direct-reservation question) and
# to leak nothing (no markup, no PII, no injection vocab, no brand name).
_OPENER_EMAIL = (
    "Hi there,\n\n"
    "A traveller I know recently stayed with you and spoke warmly about how "
    "welcoming your team is. I help small places win more reservations directly, "
    "and I was curious: do you take reservations mainly through your own website, "
    "or mostly via the usual OTA platforms?\n\n"
    "If it is useful, I would be glad to put together a quick sample page for you, "
    "free and with nothing owed either way.\n\n"
    "Warmly,\n"
    "Alex"
)

# A clean qualifier reply BODY (the value of the JSON "draft" field). Must pass
# detect_outbound_leak (no datamark char, no role markers, no scaffolding phrases).
_QUALIFIER_DRAFT = (
    "Hi there — thanks so much for getting back to us! We would love to help you "
    "take more reservations directly through your own website rather than only "
    "through the OTA platforms. If it helps, I can put together a quick sample page "
    "for you, free and with nothing owed either way. Would that be useful? "
    "Warmly, Alex"
)

# A short, clean opener inline phrase (Agent B path truncates raw output).
_OPENER_PHRASE = (
    "your guests clearly feel looked after — I would love to help you turn that "
    "into more direct reservations"
)

# A neutral room-description paragraph for the content-graph demo.
_ROOM_PARAGRAPH = (
    "A calm, comfortable room with an easy, uncluttered layout and plenty of "
    "natural light. Guests consistently mention how restful the space feels and how "
    "thoughtfully it is kept — a simple, welcoming base for a stay."
)


def _blob(x: Any) -> str:
    """Flatten any prompt shape (str / message / list of messages) to text."""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(_blob(i) for i in x)
    content = getattr(x, "content", None)
    if content is not None:
        return content if isinstance(content, str) else str(content)
    if isinstance(x, dict):
        return str(x.get("content", x))
    return str(x)


def _generate_text(prompt: Any) -> str:
    """Return a deterministic, request-shaped response string.

    Detection is by response-SHAPE keywords the baseline prompts / judge rubrics
    carry, in priority order. Judge rubrics are matched first because their
    embedded context can contain other markers.
    """
    blob = _blob(prompt).lower()

    # ---- Judge rubrics (return the judge's expected JSON verdict) ----
    if "non_spamminess" in blob:  # outreach_llm_judge (Agent B/C quality judge)
        return json.dumps(
            {
                "honesty": True,
                "honesty_reason": "mock backend — illustrative only",
                "personalization": 0.9,
                "personalization_reason": "mock backend — illustrative only",
                "non_spamminess": 0.9,
                "non_spamminess_reason": "mock backend — illustrative only",
                "warmth": 0.9,
                "warmth_reason": "mock backend — illustrative only",
                "stop_respect": True,
                "stop_respect_reason": "mock backend — illustrative only",
            }
        )
    if "resisted" in blob:  # red-team injection judge
        return json.dumps(
            {"resisted": True, "issue": "", "notes": "mock backend — illustrative only"}
        )
    # review-poisoning safety judge — checked BEFORE the generic "leaked" branch
    # because its rubric also contains the word "leaked" ("Leaked guest PII").
    if "final opener" in blob:
        return json.dumps(
            {"safe": True, "issue": "", "notes": "mock backend — illustrative only"}
        )
    if "leaked" in blob:  # multi-turn leak judge
        return json.dumps(
            {"leaked": False, "issue": "", "notes": "mock backend — illustrative only"}
        )

    # ---- Content-pipeline prompts (demo-graph) ----
    if "primary_type" in blob:  # langchain/hotel_classify
        return json.dumps(
            {
                "primary_type": "resort",
                "confidence": 0.7,
                "secondary_types": ["boutique"],
                "supporting_evidence": {},
                "signal_analysis": {"note": "mock backend — illustrative only"},
            }
        )
    if "aspectextractionresponse" in blob or ("amenities" in blob and "hotel_signals" in blob):
        # langchain/hotel_review_analyzer — a minimal valid AspectExtractionResponse.
        return json.dumps(
            {
                "rooms": [],
                "amenities": [
                    {"name": "pool", "sentiment": "positive", "evidence": "the pool area was lovely"}
                ],
                "service": [
                    {"name": "front desk", "sentiment": "positive", "evidence": "staff were welcoming and helpful"}
                ],
                "location": [],
                "other": [],
                "hotel_context": None,
                "hotel_signals": [
                    {"signal_type": "resort", "confidence": 0.8, "evidence": "a relaxed seaside retreat"}
                ],
            }
        )
    if "room description" in blob:  # langchain/room_description(+_brief)
        return _ROOM_PARAGRAPH

    # ---- Outreach agent prompts ----
    if '"draft"' in blob:  # langchain/outreach_qualifier
        return json.dumps({"draft": _QUALIFIER_DRAFT})
    if "name_matches_brand" in blob:  # langchain/outreach_miner (ReviewHook JSON)
        return json.dumps(
            {
                "tier": 3,
                "hook_text": "a warm and welcoming team",
                "staff_name": None,
                "role": None,
                "evidence_quote": "guests consistently praised the friendly, helpful team",
                "confidence": 0.8,
                "name_matches_brand": False,
                "highlights": ["comfortable rooms", "a handy location"],
            }
        )

    # ---- Default: the generic opener email (Agent B full-opener path) ----
    return _OPENER_EMAIL


# ---------------------------------------------------------------------------
# Structured-output builders — schema-valid NEUTRAL instances
# ---------------------------------------------------------------------------

def _build_structured(schema: Any, prompt: Any) -> Any:
    """Return a schema-valid, NEUTRAL instance of *schema*.

    Explicit builders for the models the extracted code binds via
    ``with_structured_output``; a generic best-effort fallback otherwise.
    """
    name = getattr(schema, "__name__", "")
    blob = _blob(prompt)

    if name == "InterpretedResponse":
        from app.temporal.workflows.leadgen.models import (
            InterpretedResponse,
            IntentEnum,
            NextActionEnum,
            SentimentEnum,
        )

        # Money-gate probe ONLY: force a send_D so the eval suite can verify the
        # _AUTO_SEND_D_ENABLED=False override. Never triggered by any fixture.
        if MONEY_GATE_SEND_D_SENTINEL.lower() in blob.lower():
            return InterpretedResponse(
                intent=IntentEnum.interested,
                sentiment=SentimentEnum.positive,
                funnel_state="qualifier_sent",
                next_action=NextActionEnum.send_d,
                confidence=1.0,
            )
        # Fixed, neutral classification (NOT a per-input gold answer).
        return InterpretedResponse(
            intent=IntentEnum.interested,
            sentiment=SentimentEnum.positive,
            funnel_state="opener_sent",
            next_action=NextActionEnum.send_c,
            confidence=1.0,
        )

    if name == "ReviewHook":
        from app.temporal.activities.leadgen.models import ReviewHook

        return ReviewHook(
            tier=3,
            hook_text="a warm and welcoming team",
            staff_name=None,
            role=None,
            evidence_quote="guests consistently praised the friendly, helpful team",
            confidence=0.8,
            name_matches_brand=False,
            highlights=["comfortable rooms", "a handy location"],
        )

    if name == "QualifierResult":
        from app.temporal.workflows.leadgen.models import QualifierResult

        return QualifierResult(draft=_QUALIFIER_DRAFT, offered_sample=True)

    # Generic fallback: try to construct from field defaults / neutral values.
    return _generic_instance(schema)


def _generic_instance(schema: Any) -> Any:
    """Best-effort construction of an arbitrary pydantic BaseModel."""
    try:
        fields = schema.model_fields  # pydantic v2
    except AttributeError:
        return schema()  # dataclass / no fields
    kwargs: dict[str, Any] = {}
    for fname, finfo in fields.items():
        if not finfo.is_required():
            continue
        ann = finfo.annotation
        kwargs[fname] = _neutral_for(ann)
    return schema(**kwargs)


def _neutral_for(ann: Any) -> Any:
    import enum
    import typing

    origin = typing.get_origin(ann)
    if origin in (list, tuple, set):
        return []
    if origin is dict:
        return {}
    if isinstance(ann, type):
        if issubclass(ann, enum.Enum):
            return list(ann)[0]
        if issubclass(ann, bool):
            return False
        if issubclass(ann, int):
            return 0
        if issubclass(ann, float):
            return 0.0
        if issubclass(ann, str):
            return ""
    return None


class _StructuredRunnable:
    """The object returned by ``MockChatModel.with_structured_output(schema)``."""

    def __init__(self, schema: Any) -> None:
        self._schema = schema

    def invoke(self, prompt: Any, config: Any = None, **kwargs: Any) -> Any:
        return _build_structured(self._schema, prompt)

    async def ainvoke(self, prompt: Any, config: Any = None, **kwargs: Any) -> Any:
        return _build_structured(self._schema, prompt)

    def bind(self, **kwargs: Any) -> "_StructuredRunnable":
        return self


class MockChatModel:
    """A deterministic, offline stand-in for ``ChatOpenAI`` (MODEL_BACKEND=mock)."""

    def __init__(self, temperature: float = 0.0, **kwargs: Any) -> None:
        self.temperature = temperature
        self.model = "mock-deterministic"

    # -- structured output -------------------------------------------------
    def with_structured_output(self, schema: Any, **kwargs: Any) -> _StructuredRunnable:
        return _StructuredRunnable(schema)

    # -- plain chat --------------------------------------------------------
    def invoke(self, prompt: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        return AIMessage(content=_generate_text(prompt))

    async def ainvoke(self, prompt: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        return AIMessage(content=_generate_text(prompt))

    # -- runnable composition helpers used by the content nodes ------------
    def bind(self, **kwargs: Any) -> "MockChatModel":
        return self

    def __or__(self, other: Any) -> Any:  # pragma: no cover - not exercised
        return other
