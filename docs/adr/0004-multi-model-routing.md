# 0004 — Per-operation multi-model routing via an LLM factory

**Status:** Accepted

## Context
The content pipeline performs ~8 distinct LLM operations (aspect extraction, classification, room/hotel description, review summarization, translation, vision extraction). They have very different quality/cost sweet spots, and at production scale, using one frontier model everywhere is neither affordable nor optimal.

## Decision
Centralize model construction in an **`llm_factory`** that maps each operation to a configured, cost-appropriate model (e.g. a small instruct model for extraction, a vision model for images, a stronger model for long-form generation), across OpenRouter and local (LM Studio/Ollama). Enforce a **hard cost guardrail** that blocks expensive models unless explicitly allowed. Expose a single `MODEL_BACKEND` switch (`mock`/`ollama`/`openrouter`) so every node and the eval harness pick up the backend uniformly.

## Alternatives considered
- **One strong model for everything.** Rejected on cost and on over-serving simple operations.
- **Per-node hardcoded clients.** Rejected: no central cost control, no uniform mock/test path.

## Consequences
- (+) Large cost reduction; per-task fit; a single injection point that makes the whole system deterministically mockable for CI.
- (−) More configuration surface and one evaluation per operation to guard against silent quality regressions.
