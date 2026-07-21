"""Pydantic contracts for the agentic-outreach conversation workflow.

Workflow + activity I/O for the durable cold-outreach conversation lifecycle
(``OutreachConversationWorkflow``). Activities must return pydantic models (the
worker uses ``pydantic_data_converter``).

Only the outreach conversation contracts live here in the showcase; the
non-outreach lead-gen business-wiring models (audit / provisioning / CMS-link)
were part of excluded workflows and are intentionally not shipped.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Outreach conversation workflow models (F2)
# ---------------------------------------------------------------------------


class OutreachConversationInput(BaseModel):
    """Input to OutreachConversationWorkflow — one outreach conversation to run.

    Also re-used as continue_as_new seed so that funnel_stage and turn_count
    survive history truncation.

    Determinism note: pure Pydantic v2 BaseModel; no I/O, no settings import,
    no non-deterministic defaults — safe for the Temporal sandbox.
    """

    ota_hotel_id: str = Field(..., description="Logical ref to the discovered-hotels store")
    crm_contact_id: int = Field(
        ...,
        description=(
            "CRM contact ID — the conversation's identity (webhook + poller "
            "lookup) and the recipient the message is emailed to via the CRM."
        ),
    )
    hotel_name: str = Field(..., description="Hotel display name — used in outreach messages")
    funnel_stage: str = Field(
        default="opener_sent",
        description="Current funnel stage; preserved across continue_as_new",
    )
    turn_count: int = Field(
        default=0,
        description="Number of inbound replies processed; preserved across continue_as_new",
    )
    review_hook_json: str = Field(
        default="",
        description=(
            "JSON-serialised ReviewHook (Agent A) for place-sourced Agent B/C copy. "
            "Empty string => agents use their generic fallback. Carried through "
            "continue_as_new so personalization survives history truncation."
        ),
    )
    et_registration_input_json: str = Field(
        default="",
        description=(
            "JSON-serialised provisioning input for Agent D's lazy site-reveal "
            "(the site-reveal activity requires a valid provisioning input or it "
            "raises ValidationError). Carried through continue_as_new."
        ),
    )
    m1_sent: bool = Field(
        default=False,
        description=(
            "True once the M1 opener has been sent. Gates the workflow's one-time "
            "opener send so it does not re-fire after continue_as_new. Carried over."
        ),
    )
    followup_count: int = Field(
        default=0,
        description=(
            "Number of silent-window re-engagement nudges already sent. Bounds "
            "over-messaging: the workflow stops nudging once it reaches the "
            "follow-up cap and waits indefinitely for an inbound reply. Carried "
            "through continue_as_new so the cap survives history truncation."
        ),
    )
    subject: Optional[str] = Field(
        default=None,
        description=(
            "Optional email subject for the opener (= the CRM task name). None "
            "→ a default subject. The test-drive sets a UNIQUE value per run so "
            "each test lands in its own email thread instead of collapsing into "
            "one email conversation. Carried through continue_as_new."
        ),
    )


def outreach_workflow_id(
    ota_hotel_id: str,
    *,
    crm_contact_id: int,
) -> str:
    """Build the deterministic OutreachConversationWorkflow id for a conversation.

    THE single source of truth for the id, shared by the starter, the reply
    poller, and the inbound webhook so they always agree:
    ``outreach-conversation-{ota_hotel_id}-{crm_contact_id}``.
    """
    return f"outreach-conversation-{ota_hotel_id}-{crm_contact_id}"


class OutreachConversationSignal(BaseModel):
    """Parsed inbound-reply signal payload — delivered via @workflow.signal.

    Represents one inbound reply forwarded from the CRM webhook into the running
    OutreachConversationWorkflow.

    Determinism note: pure Pydantic v2 BaseModel; optional received_at is a
    plain str (ISO-8601) so no datetime import is needed at module scope — safe
    for the Temporal sandbox.
    """

    text: str = Field(..., description="Sanitized, human-readable reply text (input-guard clean_text)")
    datamarked: Optional[str] = Field(
        None,
        description=(
            "Datamarked LLM-facing form of the reply (whitespace runs → marker token) "
            "from the input guard. None for legacy/internal signals → activities fall "
            "back to `text`."
        ),
    )
    suspected_injection: bool = Field(
        False,
        description=(
            "True when the input-guard tripwire fired (OWASP LLM01). The workflow "
            "fails closed: escalates to a human and skips the E/C/D dispatch."
        ),
    )
    intent: Optional[str] = Field(
        None,
        description="Parsed intent label, e.g. 'stop', 'escalate', 'interested', None=unknown",
    )
    raw_payload: Optional[str] = Field(
        None,
        description="Original JSON payload from the CRM, serialised as string for sandbox safety",
    )
    received_at: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp when the reply was received; None if not supplied",
    )


# ---------------------------------------------------------------------------
# Agent E — InterpretedResponse
# ---------------------------------------------------------------------------


class IntentEnum(str, Enum):
    """Closed intent enum for Agent E classifier output.

    No out-of-set value permitted; the closed enum rejects LLM drift.
    """

    interested = "interested"
    question = "question"
    has_site = "has_site"
    not_interested = "not_interested"
    stop = "stop"
    unclear = "unclear"
    reservation_request = "reservation_request"
    """The hotel engaged us AS A CUSTOMER — asked for our stay dates, or offered to
    take/key in our reservation directly (over email / chat / phone) — instead of
    answering whether they have their own direct-reservation site. This both reveals
    a manual (no real direct-reservation website) operation AND ignores our question,
    so the workflow deflects-and-re-qualifies via Agent C rather than pitching or
    going silent (routed to send_C; never unclear→wait)."""


class NextActionEnum(str, Enum):
    """Closed next-action enum for Agent E routing output.

    stop intent → close action; never send_C/send_D/wait.
    Human-needed / ambiguous replies → escalate.
    """

    send_c = "send_C"
    send_d = "send_D"
    escalate = "escalate"
    close = "close"
    wait = "wait"


class SentimentEnum(str, Enum):
    """Closed sentiment enum for Agent E classifier output.

    The closed enum rejects LLM drift on the sentiment field; it prevents
    free-form values like "very positive" or "mixed" from silently passing
    Pydantic validation.
    """

    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class InterpretedResponse(BaseModel):
    """Agent E classifier output — closed schema; no out-of-set value accepted.

    All fields are required. The workflow reads ``next_action`` to dispatch to
    the correct funnel branch and checks ``intent == stop`` to set opted_out.

    Determinism note: pure Pydantic v2 BaseModel; no I/O, no settings import —
    safe for the Temporal sandbox and as a structured-output binding target.

    Invariants: intent==stop MUST pair with next_action==close; closed enums
    reject out-of-set values; human-escalation routing for ambiguous replies.
    """

    intent: IntentEnum = Field(..., description="Classified intent of the inbound reply")
    sentiment: SentimentEnum = Field(
        ..., description="Sentiment label: positive / neutral / negative"
    )
    funnel_state: str = Field(..., description="Current funnel stage label")
    next_action: NextActionEnum = Field(
        ..., description="Routing instruction for the workflow"
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Agent E self-reported confidence in this classification (0-1). The "
            "workflow overrides a costly action (send_C/send_D) to escalate when "
            "this is below threshold — fail closed to a human (OWASP LLM09). "
            "Defaults to 1.0 so the STOP fast-path and prompts that do not emit a "
            "score stay valid (the gate simply never fires until a score is given)."
        ),
    )


# ---------------------------------------------------------------------------
# Agent C — QualifierResult
# ---------------------------------------------------------------------------


class QualifierResult(BaseModel):
    """Agent C (qualifier) output — the draft reply plus a funnel-advance signal.

    ``offered_sample`` is what advances the funnel to ``qualifier_sent``: the
    workflow bumps the stage whenever Agent C actually puts (or repeats) the free
    sample-page offer on the table — REGARDLESS of the router intent. This
    decouples stage advancement from the inbound intent and fixes the off-by-one
    where an offer made IN REPLY to a ``reservation_request`` (which never advanced
    on intent) left the hotel's next acceptance stranded at ``opener_sent`` and
    mis-routed to another qualifier instead of ``send_D``.

    Determinism note: pure Pydantic v2 BaseModel; safe as an activity return
    type (worker uses pydantic_data_converter) and inside the Temporal sandbox.
    """

    draft: str = Field(..., description="Email-ready reply text sent to the hotel")
    offered_sample: bool = Field(
        default=False,
        description=(
            "True if THIS reply makes or repeats the free sample-page offer. Drives "
            "the workflow's advance to funnel_stage=qualifier_sent so a subsequent "
            "acceptance routes to send_D. False when the reply only re-qualifies / "
            "deflects / answers a question without offering."
        ),
    )
