# et-hotel-ai-showcase

> **Work in progress.** A sanitized, runnable showcase extracted from a production
> multi-service AI/data platform I built — a LangGraph multi-model content
> pipeline, a Temporal-orchestrated multi-agent LLM outreach engine with
> OWASP-LLM-Top-10 guardrails and a reproducible eval/red-team harness, and a
> resilient distributed fetching engine.

**What's shown vs. withheld (by design):**

- ✅ The architecture, the engineering, and a **runnable eval harness** you can
  reproduce offline with a deterministic mock model — no API keys, no cost.
- 🚫 **Curated production prompts** live in [Langfuse](https://langfuse.com) and
  are withheld; this repo ships generic **baseline** prompts in
  `prompts/baseline/` that exercise the same code paths.
- 🚫 **No secrets, no real customer/lead data** — all fixtures under
  `data/synthetic/` are fictional.
- 🚫 The scraping subsystem is presented as a **generic resilient-fetcher engine**
  demonstrated against neutral/self-hosted targets — it ships **no**
  site-specific scraper.

_A polished README, architecture diagrams, ADRs, and reproduced eval metrics land
during the presentation phase._
