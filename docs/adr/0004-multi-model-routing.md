# 0004 — Per-operation multi-model routing via an LLM factory

**Status:** Accepted

## Context
The content pipeline performs ~8 distinct LLM operations (aspect extraction, classification, room/hotel description, review summarization, translation, vision extraction). They have very different quality/cost sweet spots, and there was a deliberate product goal: **hit production quality on small, cheap, open models (DeepSeek / Qwen / Llama), not frontier models (GPT-5 / Claude)** — for cost and control at scale. Reaching for a frontier model everywhere would be neither affordable nor a demonstration of engineering.

## Decision
Centralize model construction in an **`llm_factory`** that maps each operation to a configured, cost-appropriate model (e.g. a small instruct model for extraction, a vision model for images, a stronger model for long-form generation), across OpenRouter and local (LM Studio/Ollama). Enforce a **hard cost guardrail** that blocks expensive models unless explicitly allowed. Expose a single `MODEL_BACKEND` switch (`mock`/`ollama`/`openrouter`) so every node and the eval harness pick up the backend uniformly.

## Alternatives considered
- **One strong model for everything.** Rejected on cost and on over-serving simple operations.
- **Per-node hardcoded clients.** Rejected: no central cost control, no uniform mock/test path.

## Consequences
- (+) Large cost reduction (a fraction of frontier-model cost); per-task fit; **it proves the quality came from engineering — routing, prompts, structured outputs, guardrails — not model size**; and a single injection point that makes the whole system deterministically mockable for CI.
- (−) More configuration surface and one evaluation per operation to guard against silent quality regressions.
