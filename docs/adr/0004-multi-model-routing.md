# 0004 — Per-operation multi-model routing via an LLM factory

**Status:** Accepted

## Context
The content pipeline performs ~8 distinct LLM operations (aspect extraction, classification, room/hotel description, review summarization, translation, vision extraction). They have very different quality/cost sweet spots, and there was a deliberate product goal: **hit production quality on cheap, open, non-frontier models (Llama-3.3-70B, DeepSeek-V3.2, Qwen-VL-32B, Mistral-Small-24B), not frontier models (GPT-5 / Claude)** — for cost and control at scale. Reaching for a frontier model everywhere would be neither affordable nor a demonstration of engineering.

## Decision
Centralize model construction in an **`llm_factory`** that maps each operation to a configured cheap/open model — **Llama-3.3-70B** for extraction/classification/routing, **DeepSeek-V3.2** for generation/translation/outreach, **Qwen3-VL-32B** for vision, **Mistral-Small-24B** for the description validator — across OpenRouter and local (LM Studio/Ollama). The model for each operation is set in its **Langfuse prompt config** (tunable without a deploy). Enforce a **hard cost guardrail** that blocks expensive models unless explicitly allowed. Expose a single `MODEL_BACKEND` switch (`mock`/`ollama`/`openrouter`) so every node and the eval harness pick up the backend uniformly.

## Alternatives considered
- **One strong model for everything.** Rejected on cost and on over-serving simple operations.
- **Per-node hardcoded clients.** Rejected: no central cost control, no uniform mock/test path.

## Consequences
- (+) Large cost reduction (a fraction of frontier-model cost); per-task fit; **it proves the quality came from engineering — routing, prompts, structured outputs, guardrails — not model size**; and a single injection point that makes the whole system deterministically mockable for CI.
- (−) More configuration surface and one evaluation per operation to guard against silent quality regressions.
