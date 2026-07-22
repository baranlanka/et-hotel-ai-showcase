# 0001 — Temporal durable workflows for agent orchestration

**Status:** Accepted

## Context
Temporal is the platform's orchestration backbone: **every activity — content generation, scraping, proxy management, and outreach — runs as a durable Temporal workflow/activity across ~10 worker services.** These are long-lived (outreach conversations span days), must survive process restarts and deploys, need retries and circuit-breaking around flaky external calls (LLM providers, proxies, scrape targets), and must keep auditable, resumable state. A crash must not lose work or double-send a message.

## Decision
Model **every unit of work** as a Temporal workflow with activities — each content-generation stage, each scrape/proxy operation, and the outreach conversation (agents as activities) — run across per-domain worker services. Use Temporal's determinism discipline, automatic retries, `continue_as_new`, timers, and signal handling for inbound replies, with OpenTelemetry tracing across all of it.

## Alternatives considered
- **Celery / raw task queue + a `state` column.** Rejected: we would re-implement retries, idempotency, replay, and timers by hand, and still not get replayable history.
- **Ad-hoc async services.** Rejected: no durability or auditability across restarts.

## Consequences
- (+) Durable, resumable, replayable conversations; retries and timers for free; a clean audit trail.
- (−) Operational weight of a Temporal cluster, and the discipline that workflow code must be deterministic (side effects go in activities). Accepted as the price of correctness for long-lived agents.
