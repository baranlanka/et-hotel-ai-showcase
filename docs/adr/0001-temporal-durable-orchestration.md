# 0001 — Temporal durable workflows for agent orchestration

**Status:** Accepted

## Context
The outreach engine runs long-lived, multi-turn conversations with hotels that can span days, must survive process restarts and deploys, need retries and circuit-breaking around flaky external calls, and must keep auditable, resumable state per conversation. A crash must not lose a conversation or double-send a message.

## Decision
Model each conversation as a **Temporal durable workflow**, with the agents as activities. Use Temporal's determinism discipline, automatic retries, `continue_as_new`, and signal handling for inbound replies.

## Alternatives considered
- **Celery / raw task queue + a `state` column.** Rejected: we would re-implement retries, idempotency, replay, and timers by hand, and still not get replayable history.
- **Ad-hoc async services.** Rejected: no durability or auditability across restarts.

## Consequences
- (+) Durable, resumable, replayable conversations; retries and timers for free; a clean audit trail.
- (−) Operational weight of a Temporal cluster, and the discipline that workflow code must be deterministic (side effects go in activities). Accepted as the price of correctness for long-lived agents.
