"""OutreachConversationWorkflow — durable cold-outreach conversation lifecycle (F2/F3).

Manages a single hotel-contact conversation from first touch through to
opted-out / escalated / fully-engaged state.  The workflow loops on an inbound
signal OR a follow-up sleep timer, and calls continue_as_new before Temporal
history exceeds safe limits, preserving funnel_stage + turn_count across the
boundary.

All DB mutations are performed via activity calls — the workflow itself never
writes to the database directly (Temporal determinism invariant).

CRITICAL — Temporal determinism rules enforced here:
    - NEVER import settings / get_settings() at module or workflow scope.
    - All non-stdlib imports MUST be wrapped in
      ``with workflow.unsafe.imports_passed_through():``.
    - Use workflow.sleep(), NOT asyncio.sleep().
    - Use workflow.now(),  NOT datetime.now().
    - Reference activities by STRING name only.
    - Every execute_activity call MUST supply start_to_close_timeout + retry_policy.

References:
    - Models:   app/temporal/workflows/leadgen/models.py (C3)
    - DB model: app/database/models/outreach_conversation.py (C1)
    - Worker:   app/workers/run_leadgen_outreach_worker.py (/C7)
    - Plan:     docs/features/outreach-lifecycle-skeleton/plan.md (C4)
    - Plan:     docs/features/conversational-agents-b-e/plan.md (C7, )
    - Pattern:  app/temporal/workflows/leadgen/leadgen_et_registration_workflow.py

Traceability:
    - workflow init creates outreach_conversations row; blocks on
                 wait_condition for first inbound signal
    - workflow.sleep follow-up timer fires; re-engagement emitted;
                 wait_condition re-entered with funnel state intact
    - continue_as_new called on history threshold; funnel_stage +
                 turn_count preserved in next-run input
    - STOP signal → opted_out=True; persistence activity called
                 atomically; no further timers/messages scheduled
    - on inbound reply, Agent E (interpret_response_activity) is called
                 FIRST; next_action dispatches to send_C / send_D / escalate /
                 close / wait; four F3 activities registered in worker
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from app.leadgen.outreach.output_guard import detect_outbound_leak
    from app.temporal.base.base_temporal_workflow import BaseTemporalWorkflow
    from app.temporal.workflows.leadgen.models import (
        OutreachConversationInput,
        OutreachConversationSignal,
        NextActionEnum,
        InterpretedResponse,
        QualifierResult,
    )
    from app.temporal.activities.leadgen.models import MineReviewsInput, ReviewHook

# ---------------------------------------------------------------------------
# Constants (deterministic — no settings import)
# ---------------------------------------------------------------------------

_HISTORY_THRESHOLD: int = 500
"""continue_as_new when workflow history (events) exceeds this count."""

_FOLLOWUP_INTERVAL: timedelta = timedelta(hours=24)
"""Grace window between outreach touches (B2 cadence fix).

After each outbound (opener or nudge) we wait this long for the hotel to reply
before taking the next step. 24h keeps the cadence human and non-spammy (was
3 days) while still closing an ignored thread promptly.
"""

_MAX_FOLLOWUPS: int = 1
"""Number of gentle re-engagement nudges sent while the hotel stays silent.

Cadence (B2): opener → wait 24h → nudge → wait 24h → CLOSE + a ops
"hotel did not respond, here's what we did" alert. With this set to 1 there
are exactly two 24h grace windows and one nudge before close; bump to 2 for an
extra nudge + window. CRITICAL: after the final window the workflow ALWAYS
terminates — it never drops the timeout to wait indefinitely, so a silent
thread can no longer sit open forever (fixes the orphaned-thread / opener-spam
class). Carried across continue_as_new via input.followup_count.
"""

_ESCALATED_STAGE: str = "escalated"
"""funnel_stage value once the conversation is handed to a human.

Set by every escalate branch (router decision, low confidence, Agent C failure,
output-guard leak, send_D approval, max-turns). When the follow-up timer fires while
the thread is in this stage, the workflow SKIPS the automated re-engagement nudge —
a human was already alerted at escalation and is handling the thread, so an automated
bump would step on them. The thread is NOT closed: it stays alive so the human's
continued replies still route through Agent E (bounded by the 30-day
execution_timeout). This keeps main's keep-alive-on-escalation intent while adding
'don't nudge a human-owned thread'.
"""

_MAX_TURNS: int = 12
"""Hard ceiling on inbound conversation turns (OWASP LLM10 — unbounded consumption).

After this many inbound replies the workflow escalates to a human and CLOSES,
skipping the Agent E/C/D dispatch — so a contact who floods the thread cannot
burn unbounded LLM spend or keep the workflow alive forever. ``turn_count`` is
already carried across continue_as_new, so the cap survives history truncation.
Pairs with the 30-day ``execution_timeout`` (wall-clock leg) on the start call.
"""

_ROUTER_MIN_CONFIDENCE: float = 0.6
"""Agent-E confidence floor for a costly action (OWASP LLM09 — misinformation).

When Agent E proposes send_C/send_D with self-reported ``confidence`` below this,
the workflow overrides the action to ``escalate`` — fail closed to a human rather
than act on a low-confidence classification of an untrusted reply.
"""

_HOLDING_REPLY: str = (
    "Thanks so much for getting back to us! We want to make sure we give you a "
    "proper answer, so a member of our team will follow up with you shortly."
)
"""Deterministic canned reply sent when Agent C fails unrecoverably, so the
thread is never left silent; always paired with an escalation to a human."""

_AUTO_SEND_D_ENABLED: bool = False
"""Master gate for the costly send_D "money action" (OWASP LLM06 — excessive agency).

send_D triggers a real, billable ET-registration + CMS site-gen. While False
(the pilot default), the workflow refuses to let the LLM authorize that spend:
EVERY send_D intent is routed to escalate + a staff ops approval instead.
The LLM emits an intent; CODE authorizes the spend. Flipping this to True (to
enable auto-reveal once volume justifies it) is a deliberate code change — at
which point a deterministic per-conversation reveal cap is added alongside it
(today every reveal is human-approved, the strongest possible qualified gate).
A module constant, not a setting: workflow code cannot read settings, and the
money gate should change only via reviewed code, never an env var.
"""

_ACTIVITY_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=["ValidationError"],
)
"""Standard retry policy for outreach persistence activities."""

_LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    maximum_interval=timedelta(seconds=60),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=["ValidationError"],
)
"""Retry policy for LLM-backed activities (B, C, D, E)."""

def _render_reveal(site_url: str) -> str:
    """Render the Agent-D site-reveal message. Deterministic — workflow-safe."""
    return (
        "Okay — I went ahead and put a little something together so you could "
        "actually see it rather than just imagine it 🙂\n\n"
        "Here's a quick sample page I made for you, free, nothing owed either "
        f"way: {site_url}\n\n"
        "Have a look whenever you get a minute — and tell me honestly what you "
        "think, even if the honest answer is \"not for us.\" I'd genuinely "
        "rather know.\n\n"
        "Warmly,\n"
        "Alex"
    )


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------

@workflow.defn(name="OutreachConversationWorkflow")
class OutreachConversationWorkflow(BaseTemporalWorkflow):
    """Durable conversation lifecycle for cold-outreach (F2/F3).

    Workflow ID convention (enforces at-most-one in-flight per contact):
        ``"outreach-conversation-{ota_hotel_id}-{crm_contact_id}"``

    Signal handlers
    ---------------
    on_inbound_reply — fired by the FastAPI webhook () when CRM delivers
        a reply.  Sets ``_inbound_pending`` flag so ``wait_condition`` unblocks.

    Loop (F3-wired, )
    --------------------------
    1. Create conversation row via persistence activity.
    2. Block on wait_condition OR follow-up timer (, ).
    3. On signal: call Agent E (interpret_response_activity) FIRST → route by
       next_action:
         - close    → persist opt-out + return (STOP / not-interested)
         - escalate → escalate_to_human_activity, re-enter loop
         - send_C   → handle_reply_qualifier_activity (Agent C), update stage
         - send_D   → site_reveal_activity (Agent D), update stage
         - wait     → no outbound, re-enter loop
    4. On timer: call Agent B (fill_opener_template_activity) for re-engagement
       template fill, then send_outbound_message_activity; re-enter loop ().
    5. After each iteration: check history length; continue_as_new if over
       threshold, carrying funnel_stage + turn_count ().

    Activities referenced (string names — ADR-OUTREACH-001):
        Persistence / infra (F2 seams):
        - ``"create_outreach_conversation_activity"``
        - ``"send_outbound_message_activity"``  — unified outbound seam (opener,
          qualifier, re-engagement, reveal); log-only in Phase 4a
        - ``"persist_opt_out_activity"``
        - ``"escalate_to_human_activity"``
        F3 Agents ():
        - ``"fill_opener_template_activity"``     — Agent B (M1 template fill)
        - ``"handle_reply_qualifier_activity"``   — Agent C (qualify / pitch)
        - ``"site_reveal_activity"``              — Agent D (lazy site-gen)
        - ``"interpret_response_activity"``       — Agent E (router, fires FIRST)

    Traceability:
        - , , , 
        - 
    """

    def __init__(self) -> None:
        """Initialise per-run signal state.

        Note: __init__ must be cheap and deterministic — no I/O.
        """
        # Chain to WorkflowAlertMixin.__init__ so it can set _sent_alert_ids.
        # send_workflow_alert() (the wait-branch heads-up + the no-response-close
        # alert) reads that set; without this super() call it raises
        # AttributeError and every ops alert silently fails (prod 2026-06-30).
        super().__init__()
        self._inbound_pending: bool = False
        self._pending_signal: OutreachConversationSignal | None = None

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    @workflow.signal
    async def on_inbound_reply(self, signal: OutreachConversationSignal) -> None:
        """Receive an inbound CRM reply and unblock the wait_condition.

        Called by the FastAPI webhook handler () via Temporal client.signal().
        Stores the parsed signal payload so the main loop can inspect intent.

        Args:
            signal: Parsed inbound reply from OutreachConversationSignal.

        Traceability:
            - wait_condition unblocks when _inbound_pending is True
            - STOP intent text stored here; routed in run() via Agent E
        """
        self._pending_signal = signal
        self._inbound_pending = True

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, input: OutreachConversationInput) -> None:
        """Execute durable outreach conversation loop (F2 + F3 wired).

        Implements the full /003/004/005 +  lifecycle:
          1. Persist the conversation row at funnel_stage=opener_sent ().
          2. Enter a loop: wait for inbound reply OR follow-up timer (,
             ).
          3. On reply: call Agent E (interpret_response_activity) FIRST ();
             dispatch by next_action (close/escalate/send_C/send_D/wait).
          4. On timer: call Agent B (fill_opener_template) for re-engagement template
             fill, then send_outbound_message_activity; re-enter loop ().
          5. After each iteration: check history length; continue_as_new if over
             threshold, carrying funnel_stage + turn_count ().

        Args:
            input: OutreachConversationInput with ota_hotel_id,
                   crm_contact_id, hotel_name, funnel_stage, turn_count.

        Traceability:
            - creates outreach_conversations row; blocks on
                         wait_condition for inbound signal
            - workflow.sleep follow-up timer; re-engagement on fire;
                         funnel state intact after re-entering wait_condition
            - continue_as_new when history threshold exceeded;
                         funnel_stage + turn_count carried to next run
            - STOP signal → opted_out persisted; no further
                         timers/messages scheduled
            - Agent E called first on every inbound; dispatch per
                         next_action to B/C/D/escalate/close/wait
        """
        telemetry_id = self.setup_telemetry(None, "OutreachConversation")
        self.log_workflow_start(
            "OutreachConversationWorkflow",
            {
                "ota_hotel_id": input.ota_hotel_id,
                "crm_contact_id": input.crm_contact_id,
                "funnel_stage": input.funnel_stage,
                "turn_count": input.turn_count,
                "telemetry_id": telemetry_id,
            },
        )

        # --- Step 1: persist conversation row () ---
        # Creates / upserts the outreach_conversations row. Re-runs on each
        # continue_as_new, refreshing funnel_stage/turn_count from the seed.
        await workflow.execute_activity(
            "create_outreach_conversation_activity",
            args=[input, telemetry_id],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_ACTIVITY_RETRY_POLICY,
        )

        # Mutable loop state — carried across iterations and into continue_as_new.
        current_stage: str = input.funnel_stage
        turn_count: int = input.turn_count
        m1_sent: bool = input.m1_sent
        followup_count: int = input.followup_count

        # --- Step 1a: mine the personalization hook once (Agent A) ---
        # The conversation owns review mining. When no hook was supplied at start
        # (the starter passes none) and the opener hasn't been sent yet, run Agent
        # A to extract a real review-sourced hook — a named staffer when one is
        # present in the hotel's OTA reviews. The result is kept in
        # review_hook_json and carried across continue_as_new, so mining happens at
        # most once. A mining failure (no reviews / B2 miss / LLM error) degrades
        # to the Tier-3 generic opener (empty hook) rather than blocking the
        # conversation — personalization is best-effort, delivery is not.
        review_hook_json: str = input.review_hook_json
        if not m1_sent and not review_hook_json:
            try:
                hook = await workflow.execute_activity(
                    "mine_reviews_activity",
                    args=[
                        MineReviewsInput(
                            ota_hotel_id=input.ota_hotel_id,
                            hotel_name=input.hotel_name,
                        ),
                        telemetry_id,
                    ],
                    # result_type so the pydantic data converter reconstructs a
                    # ReviewHook — without it the activity result deserializes to a
                    # dict and hook.model_dump_json() raises (silent generic-opener
                    # fallback). Tests stub the activity with a real object, so this
                    # only bit in live Temporal execution.
                    result_type=ReviewHook,
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=_LLM_RETRY_POLICY,
                )
                review_hook_json = hook.model_dump_json() if hook is not None else ""
                workflow.logger.info(
                    "OutreachConversationWorkflow: mined review hook for opener",
                    extra={
                        "ota_hotel_id": input.ota_hotel_id,
                        "tier": getattr(hook, "tier", None),
                        "telemetry_id": telemetry_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — degrade, never block delivery
                workflow.logger.warning(
                    "OutreachConversationWorkflow: review mining failed — "
                    "falling back to the generic opener",
                    extra={
                        "ota_hotel_id": input.ota_hotel_id,
                        "error": str(exc),
                        "telemetry_id": telemetry_id,
                    },
                )
                review_hook_json = ""

        # --- Step 1b: send the M1 opener exactly once (B2 fix) ---
        # Agent B (fill_opener_template_activity) writes the full opener email
        # (empty review_hook_json => Tier-3 generic, no LLM). The returned string
        # is sent directly via the unified seam. Gated on m1_sent so
        # continue_as_new never re-sends it.
        if not m1_sent:
            opener_text = await workflow.execute_activity(
                "fill_opener_template_activity",
                args=[
                    input.ota_hotel_id,
                    input.hotel_name,
                    review_hook_json,
                    telemetry_id,
                ],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=_LLM_RETRY_POLICY,
            )
            await workflow.execute_activity(
                "send_outbound_message_activity",
                args=[
                    input.ota_hotel_id,
                    input.crm_contact_id,
                    opener_text,
                    telemetry_id,
                    # Opener creates the CRM task; its subject = the task name.
                    # input.subject (unique per test-drive run) gives each test its
                    # own email thread. None → OUTREACH_EMAIL_SUBJECT_DEFAULT (prod).
                    input.subject,
                ],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=_ACTIVITY_RETRY_POLICY,
            )
            m1_sent = True

        # --- Step 2: main conversation loop ---
        while True:
            # Reset signal flag before each wait so a previously consumed signal
            # does not immediately re-unblock the condition.
            self._inbound_pending = False
            self._pending_signal = None

            # Always wake after the follow-up interval: while under the nudge
            # cap a timeout sends the next gentle nudge; AT the cap it CLOSES the
            # thread with a "no response" ops alert. The wait NEVER drops its
            # timeout to None, so a silent thread can no longer sit open forever
            # re-engaging on every signal (B2 fix — orphaned-thread/opener-spam).
            followup_timeout = _FOLLOWUP_INTERVAL
            # workflow.wait_condition returns None when the condition is met and
            # RAISES asyncio.TimeoutError when the follow-up timer fires — it never
            # returns a bool. So never trust its return value: catch the timeout, then
            # read the flag the signal handler set. (The old code did
            # `signal_received = await wait_condition(...)`, captured None on every
            # real reply → falsy → the reply fell through to the re-engagement branch
            # and re-sent the opener instead of routing to Agent E.)
            try:
                await workflow.wait_condition(
                    lambda: self._inbound_pending,
                    timeout=followup_timeout,
                )
            except asyncio.TimeoutError:
                pass  # follow-up timer fired — no reply this window
            signal_received: bool = self._inbound_pending

            if signal_received:
                # --- Inbound reply path () ---
                signal = self._pending_signal
                turn_count += 1

                workflow.logger.info(
                    "OutreachConversationWorkflow: inbound reply received — calling Agent E",
                    extra={
                        "ota_hotel_id": input.ota_hotel_id,
                        "turn_count": turn_count,
                        "telemetry_id": telemetry_id,
                    },
                )

                # OWASP LLM10: hard turn ceiling. After _MAX_TURNS inbound replies,
                # escalate to a human and CLOSE — no further Agent E/C/D spend, no
                # timers. The 30-day execution_timeout is the complementary
                # wall-clock bound; turn_count is already carried across
                # continue_as_new so this survives history truncation.
                if turn_count > _MAX_TURNS:
                    workflow.logger.warning(
                        "OutreachConversationWorkflow: max-turns cap reached — "
                        "escalating and closing",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "turn_count": turn_count,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    await self._escalate_to_human(
                        input, telemetry_id, reason="max_turns"
                    )
                    return  # escalate-then-close: skip E/C/D dispatch entirely

                # OWASP LLM01: if the input guard flagged the reply, fail closed —
                # escalate to a human and skip Agent E/C/D entirely (no LLM spend
                # on a hostile message). Reuses the escalate dispatch branch below.
                if signal and signal.suspected_injection:
                    workflow.logger.warning(
                        "OutreachConversationWorkflow: input-guard tripwire — "
                        "escalating, skipping Agent E/C/D",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    next_action: NextActionEnum = NextActionEnum.escalate
                else:
                    # Agent E (interpret_response_activity) fires FIRST
                    # on every inbound. It classifies the reply and returns a
                    # routing decision (next_action). All branching is driven by
                    # next_action. Agent E gets the datamarked text for the LLM;
                    # the deterministic STOP gate inside the activity reads the
                    # clean text (signal.text) so datamarking can't defeat it.
                    #
                    # OWASP LLM09/LLM10 fail-closed: a terminal router failure
                    # (timeout / retries exhausted) or a low-confidence costly
                    # action must NEVER fall through to send_C/send_D — default to
                    # escalate so a human decides.
                    try:
                        interpreted = await workflow.execute_activity(
                            "interpret_response_activity",
                            args=[
                                input.ota_hotel_id,
                                signal.text if signal else "",
                                current_stage,
                                telemetry_id,
                                signal.datamarked if signal else None,
                            ],
                            # result_type so the result is a real InterpretedResponse,
                            # not a dict (else interpreted.next_action raises and the
                            # reply path crashes — same converter gotcha as the miner).
                            result_type=InterpretedResponse,
                            start_to_close_timeout=timedelta(seconds=60),
                            retry_policy=_LLM_RETRY_POLICY,
                        )
                        next_action = interpreted.next_action
                        if (
                            next_action
                            in (NextActionEnum.send_c, NextActionEnum.send_d)
                            and interpreted.confidence < _ROUTER_MIN_CONFIDENCE
                        ):
                            workflow.logger.warning(
                                "OutreachConversationWorkflow: Agent E confidence "
                                "below threshold — overriding to escalate",
                                extra={
                                    "ota_hotel_id": input.ota_hotel_id,
                                    "proposed_action": next_action.value,
                                    "confidence": interpreted.confidence,
                                    "telemetry_id": telemetry_id,
                                },
                            )
                            next_action = NextActionEnum.escalate
                        workflow.logger.info(
                            "OutreachConversationWorkflow: Agent E decision",
                            extra={
                                "ota_hotel_id": input.ota_hotel_id,
                                "next_action": next_action.value,
                                "intent": interpreted.intent.value,
                                "telemetry_id": telemetry_id,
                            },
                        )
                    except ActivityError:
                        workflow.logger.warning(
                            "OutreachConversationWorkflow: Agent E failed after "
                            "retries — failing closed to escalate",
                            extra={
                                "ota_hotel_id": input.ota_hotel_id,
                                "telemetry_id": telemetry_id,
                            },
                        )
                        next_action = NextActionEnum.escalate

                # --- Dispatch by next_action ---

                if next_action == NextActionEnum.close:
                    # STOP / opt-out / not-interested — persist and terminate
                    # (, ).
                    await self._persist_opt_out(input, telemetry_id)
                    workflow.logger.info(
                        "OutreachConversationWorkflow: next_action=close — opted out, "
                        "workflow ending",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    return  # No further timers or messages scheduled.

                elif next_action == NextActionEnum.escalate:
                    # Route to human; loop continues so further signals can arrive.
                    # Covers both the LLM's escalate decision and the input-guard
                    # tripwire (which set next_action=escalate above).
                    await self._escalate_to_human(
                        input, telemetry_id, reason="router_escalate"
                    )
                    current_stage = "escalated"

                elif next_action == NextActionEnum.send_c:
                    # Agent C — qualify / free-site pitch (). Uses the
                    # conversation's threaded ReviewHook (NOT signal.raw_payload,
                    # which is the raw webhook envelope). Capture C's draft and
                    # actually send it via the unified seam.

                    # Fetch conversation history for statefulness (best-effort:
                    # any failure defaults to "" and does NOT block the reply).
                    # workflow.patched() gates the NEW activity call so conversations
                    # already in-flight when this deploys replay deterministically:
                    # their recorded history has no fetch_conversation_history_activity
                    # event, so patched() returns False on replay and the fetch is
                    # skipped (history stays ""). New turns + new conversations run it.
                    # The extra trailing arg on the qualifier call below is replay-safe
                    # on its own (same activity type/position; args aren't part of the
                    # determinism check) — only the new activity needs the patch gate.
                    history = ""
                    if workflow.patched("outreach-conversation-history-v1"):
                        try:
                            history = await workflow.execute_activity(
                                "fetch_conversation_history_activity",
                                args=[
                                    input.ota_hotel_id,
                                    input.crm_contact_id,
                                    telemetry_id,
                                ],
                                start_to_close_timeout=timedelta(seconds=30),
                                retry_policy=_ACTIVITY_RETRY_POLICY,
                            )
                        except Exception:
                            history = ""

                    try:
                        # Pass current_stage + the router's intent so Agent C is
                        # NOT stage-blind: it must (a) never fabricate a prior
                        # offer it has not made (Case-B continuity-lapse fix), and
                        # (b) handle a reservation_request (hotel asked us for dates)
                        # by deflecting and re-posing the direct-reservation question
                        # instead of pitching.
                        #
                        # Agent C now returns a QualifierResult (draft +
                        # offered_sample). The result_type change is gated behind
                        # workflow.patched() for determinism: an in-flight
                        # conversation recorded a BARE-STRING draft, so on replay
                        # patched()==False reads it back as a str; only new turns
                        # deserialize the structured result. The legacy branch
                        # reproduces the OLD advance-on-intent rule EXACTLY, so the
                        # current_stage carried into the next router call is
                        # replay-identical (it is passed as an activity arg).
                        _qualifier_args = [
                            input.ota_hotel_id,
                            signal.text if signal else "",
                            review_hook_json,
                            telemetry_id,
                            signal.datamarked if signal else None,
                            current_stage,
                            interpreted.intent.value,
                            history,
                        ]
                        if workflow.patched("outreach-offer-handoff-v1"):
                            qualifier = await workflow.execute_activity(
                                "handle_reply_qualifier_activity",
                                args=_qualifier_args,
                                result_type=QualifierResult,
                                start_to_close_timeout=timedelta(seconds=90),
                                retry_policy=_LLM_RETRY_POLICY,
                            )
                            draft = qualifier.draft
                            offered_sample = qualifier.offered_sample
                        else:
                            draft = await workflow.execute_activity(
                                "handle_reply_qualifier_activity",
                                args=_qualifier_args,
                                start_to_close_timeout=timedelta(seconds=90),
                                retry_policy=_LLM_RETRY_POLICY,
                            )
                            # Legacy advance rule (pre-fix): any non-reservation_request
                            # reply advanced the stage. Preserved verbatim on replay.
                            offered_sample = interpreted.intent != "reservation_request"
                    except ActivityError:
                        # OWASP LLM09 fail-closed: Agent C unrecoverable (e.g.
                        # unparseable JSON). Send a deterministic holding line so
                        # the thread is never left silent, then escalate to a human.
                        workflow.logger.warning(
                            "OutreachConversationWorkflow: Agent C failed after "
                            "retries — sending holding reply and escalating",
                            extra={
                                "ota_hotel_id": input.ota_hotel_id,
                                "telemetry_id": telemetry_id,
                            },
                        )
                        await workflow.execute_activity(
                            "send_outbound_message_activity",
                            args=[
                                input.ota_hotel_id,
                                input.crm_contact_id,
                                _HOLDING_REPLY,
                                telemetry_id,
                            ],
                            start_to_close_timeout=timedelta(seconds=60),
                            retry_policy=_ACTIVITY_RETRY_POLICY,
                        )
                        await self._escalate_to_human(
                            input, telemetry_id, reason="agent_c_failed"
                        )
                        current_stage = "escalated"
                    else:
                        # OWASP LLM01/LLM07 output guard: a deterministic, non-LLM
                        # leak-check on C's draft BEFORE it is sent to the hotel. If
                        # our scaffolding / a role marker / the datamark char leaked
                        # into the draft, drop it and fail closed to a human rather
                        # than send a possibly-injected reply under our name.
                        leak = detect_outbound_leak(draft)
                        if leak:
                            workflow.logger.warning(
                                "OutreachConversationWorkflow: output-guard leak — "
                                "dropping draft, escalating",
                                extra={
                                    "ota_hotel_id": input.ota_hotel_id,
                                    "leak_category": leak,
                                    "telemetry_id": telemetry_id,
                                },
                            )
                            await self._escalate_to_human(
                                input, telemetry_id, reason="output_guard_leak"
                            )
                            current_stage = "escalated"
                        else:
                            await workflow.execute_activity(
                                "send_outbound_message_activity",
                                args=[
                                    input.ota_hotel_id,
                                    input.crm_contact_id,
                                    draft,
                                    telemetry_id,
                                ],
                                start_to_close_timeout=timedelta(seconds=60),
                                retry_policy=_ACTIVITY_RETRY_POLICY,
                            )
                            # Advance to qualifier_sent when Agent C ACTUALLY made
                            # the offer (offered_sample) — NOT keyed on the inbound
                            # intent. The pivot pitch is often made in the same
                            # breath as deflecting a reservation_request ("email works —
                            # btw I put together simple direct-reservation pages, want
                            # one?"), so an intent-based gate stranded the hotel's
                            # NEXT acceptance at opener_sent and mis-routed it to yet
                            # another qualifier instead of send_D (the accept-after-
                            # reservation-request bug). offered_sample carries the OLD
                            # advance-on-intent value on replay of pre-fix turns (see
                            # the patch gate above), so this stays determinism-safe.
                            if offered_sample:
                                current_stage = "qualifier_sent"

                elif next_action == NextActionEnum.send_d:
                    # OWASP LLM06 — deterministic gate on the costly "money action".
                    # send_D triggers a real, billable ET-registration + CMS
                    # site-gen. The LLM only emits the intent; CODE authorizes the
                    # spend. While _AUTO_SEND_D_ENABLED is False (pilot), EVERY
                    # reveal is routed to a human via escalate + ops approval,
                    # NOT auto-dispatched — so funnel_stage==qualifier_sent (which
                    # an attacker advances by merely replying) can never trigger a
                    # spend on its own.
                    if not _AUTO_SEND_D_ENABLED:
                        workflow.logger.info(
                            "OutreachConversationWorkflow: send_D intent — hotel "
                            "accepted the sample; handing off to a human and closing "
                            "(auto-D disabled, no auto-spend)",
                            extra={
                                "ota_hotel_id": input.ota_hotel_id,
                                "telemetry_id": telemetry_id,
                            },
                        )
                        await self._escalate_to_human(
                            input, telemetry_id, reason="send_d_approval"
                        )
                        current_stage = "escalated"
                        # Terminal hand-off: a WON lead (accepted the sample) is now a
                        # human's job to build + send. Returning here STOPS the
                        # follow-up nudge/close-as-opt-out machinery from pestering the
                        # hotel and mislabeling a won lead as a ghost. Gated by the same
                        # patch marker as the qualifier change so an in-flight
                        # conversation that already escalated-and-looped under the old
                        # code replays deterministically (no early return on replay).
                        if workflow.patched("outreach-offer-handoff-v1"):
                            return
                    else:
                        # Agent D — lazy site-gen trigger (/009, FR-J).
                        # Uses the conversation's threaded EtRegistrationInput — NOT
                        # signal.raw_payload, which would make site_reveal_activity
                        # raise a non-retryable ValidationError. Then reveal the
                        # live URL to the hotel via the unified seam.
                        site_url = await workflow.execute_activity(
                            "site_reveal_activity",
                            args=[
                                input.ota_hotel_id,
                                input.et_registration_input_json,
                                telemetry_id,
                            ],
                            start_to_close_timeout=timedelta(seconds=300),
                            retry_policy=_LLM_RETRY_POLICY,
                        )
                        await workflow.execute_activity(
                            "send_outbound_message_activity",
                            args=[
                                input.ota_hotel_id,
                                input.crm_contact_id,
                                _render_reveal(site_url),
                                telemetry_id,
                            ],
                            start_to_close_timeout=timedelta(seconds=60),
                            retry_policy=_ACTIVITY_RETRY_POLICY,
                        )
                        current_stage = "site_revealed"

                elif next_action == NextActionEnum.wait:
                    # The router read the reply but chose NO outbound (a bare
                    # ack / soft deferral). Per the no-response policy, surface
                    # EVERY such silent outcome to a human via a low-priority
                    # ops heads-up — an inbound reply must never be both
                    # unanswered AND invisible (B2). Then re-enter the wait loop
                    # with state intact (no auto-reply).
                    workflow.logger.info(
                        "OutreachConversationWorkflow: next_action=wait — "
                        "no outbound, alerting then re-entering wait loop",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "intent": interpreted.intent.value,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    await self.send_workflow_alert(
                        message=(
                            f"\U0001f914 Outreach no-reply decision — hotel "
                            f"'{input.hotel_name}' (ota_hotel_id="
                            f"{input.ota_hotel_id}) sent a reply the agent "
                            f"classified as needing no response "
                            f"(intent={interpreted.intent.value}, "
                            f"sentiment={interpreted.sentiment.value}). Left on "
                            f"'wait' — review if a human reply is warranted."
                        ),
                        level="low",
                        context={
                            "ota_hotel_id": input.ota_hotel_id,
                            "intent": interpreted.intent.value,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    # No outbound activity call — loop naturally continues.

                else:
                    # Fail-closed: an unhandled next_action (a future enum member,
                    # or deserialization drift after continue_as_new) must NEVER
                    # silently drop a reply. Escalate to a human (which alerts on
                    # ops) so no inbound is ever both unanswered and
                    # unsurfaced (B2 invariant: outbound message OR human alert).
                    workflow.logger.error(
                        "OutreachConversationWorkflow: unhandled next_action — "
                        "escalating to a human",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "next_action": getattr(
                                next_action, "value", str(next_action)
                            ),
                            "telemetry_id": telemetry_id,
                        },
                    )
                    await self._escalate_to_human(
                        input, telemetry_id, reason="router_no_response"
                    )
                    current_stage = "escalated"

            else:
                # --- Follow-up timer fired — no reply within _FOLLOWUP_INTERVAL. ---
                if current_stage == _ESCALATED_STAGE:
                    # A human owns this thread (already alerted at escalation). Skip
                    # the automated nudge so we don't step on them, but keep the
                    # conversation alive so their continued replies still route
                    # through Agent E — no outbound, no close. Bounded by the 30-day
                    # execution_timeout.
                    workflow.logger.info(
                        "OutreachConversationWorkflow: follow-up timer fired while "
                        "escalated — skipping nudge (human-owned thread)",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "funnel_stage": current_stage,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    # No activity call — loop naturally continues.
                elif followup_count < _MAX_FOLLOWUPS:
                    # Still under the nudge cap: send ONE gentle re-engagement.
                    workflow.logger.info(
                        "OutreachConversationWorkflow: follow-up timer fired, "
                        "sending re-engagement nudge via Agent B",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "funnel_stage": current_stage,
                            "followup_count": followup_count,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    # is_reengagement=True → Agent B returns a stage-aware nudge
                    # (never a verbatim opener re-send — B2 anti-spam). current_stage
                    # is passed so the bump resurfaces the LAST thing we asked
                    # (book-direct / free-site offer / sample page) rather than
                    # blindly re-asking the opener after the thread already moved on.
                    reopener_text = await workflow.execute_activity(
                        "fill_opener_template_activity",
                        args=[
                            input.ota_hotel_id,
                            input.hotel_name,
                            review_hook_json,
                            telemetry_id,
                            True,           # is_reengagement
                            current_stage,  # stage-aware nudge topic
                        ],
                        start_to_close_timeout=timedelta(seconds=60),
                        retry_policy=_LLM_RETRY_POLICY,
                    )
                    await workflow.execute_activity(
                        "send_outbound_message_activity",
                        args=[
                            input.ota_hotel_id,
                            input.crm_contact_id,
                            reopener_text,
                            telemetry_id,
                        ],
                        start_to_close_timeout=timedelta(seconds=60),
                        retry_policy=_ACTIVITY_RETRY_POLICY,
                    )
                    followup_count += 1
                    # Re-enter wait_condition with funnel state intact; the next
                    # silent window closes the thread (cap reached).
                else:
                    # Cap reached and this final grace window elapsed with no
                    # reply: the hotel ignored the opener + every nudge. Alert a
                    # human (what we did), then CLOSE the thread — no indefinite
                    # open wait, no orphaned pollable row (B2 fix).
                    workflow.logger.info(
                        "OutreachConversationWorkflow: no reply after opener + "
                        "nudges — alerting and closing the thread",
                        extra={
                            "ota_hotel_id": input.ota_hotel_id,
                            "funnel_stage": current_stage,
                            "followup_count": followup_count,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    await self.send_workflow_alert(
                        message=(
                            f"\U0001f4ed Outreach unanswered — hotel "
                            f"'{input.hotel_name}' (ota_hotel_id="
                            f"{input.ota_hotel_id}) never replied. We sent the "
                            f"opener + {followup_count} re-engagement nudge(s) "
                            f"across {followup_count + 1}× 24h windows with no "
                            f"response. Closing the thread."
                        ),
                        level="low",
                        context={
                            "ota_hotel_id": input.ota_hotel_id,
                            "followup_count": followup_count,
                            "telemetry_id": telemetry_id,
                        },
                    )
                    await self._persist_opt_out(input, telemetry_id)
                    return  # Terminal: thread closed, no further timers/messages.

            # --- continue_as_new guard () ---
            history_len: int = workflow.info().get_current_history_length()
            if history_len >= _HISTORY_THRESHOLD:
                next_input = OutreachConversationInput(
                    ota_hotel_id=input.ota_hotel_id,
                    crm_contact_id=input.crm_contact_id,
                    hotel_name=input.hotel_name,
                    funnel_stage=current_stage,
                    turn_count=turn_count,
                    # Carry the threaded context + opener flag across the boundary,
                    # else Agent C/D lose their inputs (send_d 422s) and M1 re-sends.
                    review_hook_json=review_hook_json,
                    et_registration_input_json=input.et_registration_input_json,
                    m1_sent=m1_sent,
                    # Carry the nudge counter so the follow-up cap survives history
                    # truncation — else continue_as_new resets it and re-engagement
                    # could restart indefinitely ( bounded re-engagement).
                    followup_count=followup_count,
                )
                workflow.logger.info(
                    "OutreachConversationWorkflow: history threshold reached, "
                    "continuing as new",
                    extra={
                        "history_len": history_len,
                        "funnel_stage": current_stage,
                        "turn_count": turn_count,
                        "telemetry_id": telemetry_id,
                    },
                )
                workflow.continue_as_new(next_input)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _persist_opt_out(
        self,
        input: OutreachConversationInput,
        telemetry_id: str,
    ) -> None:
        """Call persistence activity to set opted_out=True atomically.

        Why: All DB writes must go through activities; the workflow must not
        touch the database directly. This helper keeps the opt-out path
        readable in run() and isolates the activity call-site.

        Args:
            input: Current workflow input carrying ota_hotel_id.
            telemetry_id: Trace correlation token.

        Traceability:
            - opted_out persisted via activity, no further actions
        """
        await workflow.execute_activity(
            "persist_opt_out_activity",
            args=[input, telemetry_id],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_ACTIVITY_RETRY_POLICY,
        )

    async def _escalate_to_human(
        self,
        input: OutreachConversationInput,
        telemetry_id: str,
        reason: str = "unspecified",
    ) -> None:
        """Call escalation activity to route conversation to a human handler.

        Why: Escalation is side-effectful (updates DB row, sends a best-effort
        staff ops alert); it must run in an activity, not the workflow body.

        Args:
            input: Current workflow input.
            telemetry_id: Trace correlation token.
            reason: Why we escalated (max_turns / router_escalate /
                router_low_confidence / agent_c_failed / output_guard_leak /
                send_d_approval / ...). Surfaced in the staff ops alert.

        Traceability:
            - escalation path within the follow-up loop
            - next_action==escalate triggers this helper
        """
        await workflow.execute_activity(
            "escalate_to_human_activity",
            args=[input, telemetry_id, reason],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                backoff_coefficient=2.0,
                maximum_attempts=3,
                non_retryable_error_types=["ValidationError"],
            ),
        )
