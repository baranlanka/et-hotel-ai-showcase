# 0002 — Deterministic fail-closed money-action gate

**Status:** Accepted

## Context
The outreach agent can decide to take a costly, hard-to-reverse action (`send_D` — revealing/generating a paid site for a hotel). Letting an LLM authorize that autonomously is the OWASP **LLM08 Excessive Agency** risk: a prompt-injection or a confident hallucination could trigger real spend.

## Decision
Gate the money action with a **deterministic, fail-closed constant**: `_AUTO_SEND_D_ENABLED = False` in the workflow. Every `send_D` intent is force-routed to **human approval**, regardless of model confidence. The flag can change only via reviewed code — never an environment variable or runtime config. A router-confidence floor and a per-conversation flood cap back it up.

## Alternatives considered
- **Trust the model + a confidence threshold.** Rejected: safety of an irreversible action must not depend on a model score that injection can inflate.
- **Env-var toggle for flexibility.** Rejected explicitly: an ops-changeable switch is exactly the attack/mistake surface we want to remove.

## Consequences
- (+) The agent is provably never autonomous on the irreversible action; the guarantee is testable and reproduced offline (`make eval`).
- (−) A human is always in the loop for that step (by design). Throughput on that action is bounded by human review — an accepted, deliberate limit.
