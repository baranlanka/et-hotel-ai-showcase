<div align="center">

# et-hotel-ai

**A production hotel/travel AI platform: it turns raw hotel data into published multilingual listings, and runs safe, autonomous cold-outreach to recruit hotels — built ground-up at Effective Tours.**

[▶ Reproduce the eval](#results--evaluation) · [Architecture](docs/architecture/ARCHITECTURE.md) · [Methods & case study](METHODS.md) · [Decision records](docs/adr)

[![CI](https://img.shields.io/github/actions/workflow/status/baranlanka/et-hotel-ai-showcase/ci.yml?branch=main&label=CI)](https://github.com/baranlanka/et-hotel-ai-showcase/actions)
[![eval](https://img.shields.io/badge/eval-reproducible%20offline-brightgreen)](#results--evaluation)
![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/github/license/baranlanka/et-hotel-ai-showcase)](LICENSE)

</div>

> **This is a sanitized, runnable *showcase* slice** of a ~97k-LOC production platform I built at **Effective Tours**. It demonstrates the architecture and engineering — and lets you **reproduce the safety evaluation offline, with no API keys** — while deliberately withholding the company's curated prompts, real data, and secrets. See [What's shown vs. withheld](#whats-shown-vs-withheld).

```text
$ make eval        # deterministic, offline, no API keys
A. DETERMINISTIC SECURITY / GUARD ASSERTIONS  (gate the exit code)
  [PASS] input-guard catches injection tripwires            10/10 flagged, 0/4 benign false-positives
  [PASS] input-guard datamarks untrusted text (spotlighting)
  [PASS] output-guard catches outbound leaks                5/5 caught, clean draft passes
  [PASS] opener guard catches markup / injection / PII      4/4 caught
  [PASS] ReviewHook validators null poisoned staff names + drop unsafe highlights
  [PASS] money-gate: send_D forced -> escalate  (_AUTO_SEND_D_ENABLED=False)
  [PASS] structured outputs validate (InterpretedResponse / ReviewHook / QualifierResult)
  [PASS] all 4 red-team / eval harnesses run to completion
 RESULT: deterministic assertions ALL PASS  ->  exit 0
```

## Overview

Effective Tours needed two things a small team can't do by hand at scale: **(1)** produce rich, accurate, *multilingual* content for tens of thousands of hotels, and **(2)** find and recruit new hotels through personalized outreach — without a human writing every message, and **without ever letting an autonomous agent take a money-committing action on its own.**

This repo is the engineering answer to both, as two coupled product lines:

- **A LangGraph multi-model content pipeline** — ingest hotel data → extract aspects (a fine-tuned ABSA model + LLM) → generate overviews, room descriptions, and review summaries → translate → publish. Each operation is routed to a **different, cost-appropriate model** via an LLM factory.
- **A Temporal-orchestrated, 5-agent cold-outreach engine** — it mines personalization hooks from reviews, opens a conversation, qualifies replies, and drives a funnel *autonomously* — hardened against the OWASP **LLM Top-10** (prompt injection, output leakage, excessive agency) with a **deterministic, fail-closed "money-action" gate** and a **reproducible red-team + eval harness**.

**Constraints it was built under:** real production scale and cost pressure (so: multi-model routing + guardrails, not one big model everywhere); durable, resilient execution (Temporal); an autonomous agent that must **never** self-authorize an irreversible action; and strict data-provenance discipline.

**Boundaries (what this showcase intentionally excludes):** the curated production prompts (they live in [Langfuse](https://langfuse.com), not the repo), all real customer/hotel data (replaced with synthetic fixtures), all credentials, and any site-specific scraper (the fetcher ships as a *generic* engine — see [ethics](#whats-shown-vs-withheld)).

## What's inside

| Subsystem | What it demonstrates | Runnable here |
|---|---|---|
| **Content pipeline** (`llm_content_generation/`) | LangGraph `StateGraph` of subgraphs, **per-operation multi-model routing** (`llm_factory`), cost guardrails, ABSA aspect extraction, Langfuse-managed prompts with a local fallback | `make demo-graph` — runs the aspect→content graph on a synthetic hotel with a mock model |
| **Outreach engine + eval** (`app/leadgen/`, `app/temporal/…/leadgen/`, `scripts/eval/`) | 5-agent durable **Temporal** state machine, **OWASP-LLM** input/output guards (spotlighting + datamarking), a **fail-closed money-gate**, structured Pydantic outputs, and a **red-team + eval harness** | `make eval` — reproduces the deterministic safety properties offline |
| **Resilient fetcher** (`app/shared/`) | A generic, fault-tolerant distributed HTTP/GraphQL engine: **proxy failover, circuit breaker, token-bucket rate limiting, backoff-with-jitter**, browser-profile management | `make demo-scrape` — runs the engine against a neutral demo backend |

## Tech stack

**Orchestration & agents:** Temporal · LangGraph · LangChain · Pydantic (structured outputs)
**Models & prompts:** OpenRouter (DeepSeek / Qwen-VL / Mistral / Llama) · local LM Studio/Ollama · a fine-tuned ABSA model · **Langfuse** (prompt management + tracing)
**Backend & data:** FastAPI · CockroachDB (SQLAlchemy 2.0 + Alembic) · Redis · Backblaze B2/CDN
**Platform:** Docker · GitHub Actions → GHCR → Komodo CD · self-hosted **OpenTelemetry → Grafana / Tempo / Loki / Prometheus**

## Architecture

```mermaid
flowchart LR
    OTA[(OTA / web data)] --> F
    subgraph fetch[Resilient fetcher]
      F[proxy failover · circuit breaker<br/>rate limiting · backoff+jitter]
    end
    subgraph pipeline[Content pipeline — LangGraph]
      A[aspect extraction<br/>ABSA + LLM] --> G[multi-model generation<br/>per-operation routing]
    end
    subgraph agents[Outreach engine — Temporal]
      GU[OWASP-LLM guards<br/>input · output] --- R[router agent]
      R --> Q[qualifier] --> MG{{money-gate<br/>fail-closed → human}}
    end
    F --> A
    G --> CMS[(CMS + CDN)]
    LF[(Langfuse<br/>prompts · tracing)] -. prompts .-> G
    LF -. prompts .-> R
    DB[(CockroachDB)] --- pipeline
    DB --- agents
    pipeline --> OBS[OTel → Grafana / Tempo / Loki / Prometheus]
    agents --> OBS
```

One diagram; the full container/runtime views and the 5-agent sequence diagram are in **[docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)**.

## Design decisions & trade-offs

| Decision | Options considered | Choice | Trade-off accepted |
|---|---|---|---|
| Orchestrating autonomous, long-lived agent conversations | Celery / raw queues + a state column; ad-hoc async | **Temporal** durable workflows | Operational weight of a Temporal cluster, and workflow-determinism discipline — bought durability, retries, and replayable state |
| Letting an agent take a money-committing action (`send_D`) | Trust the model + a confidence threshold | **Deterministic fail-closed gate** — every `send_D` is force-routed to human approval; the switch changes only via reviewed code, never an env var | The agent is never fully autonomous on irreversible actions (by design — this is the point) |
| Defending against prompt injection from scraped reviews/replies | Prompt-only "ignore instructions"; a classifier | **Spotlighting + datamarking** (Microsoft) + a **deterministic output leak-guard** | Extra pre/post-processing per turn; some benign inputs get datamarked — worth it for a hard DATA/instruction boundary |
| Model selection across ~8 operations | One strong model everywhere | **Per-operation routing** via an `llm_factory` + a hard cost guardrail | More config surface and more models to evaluate — but far lower cost and better fit per task |
| Prompt storage | Hardcode prompts in the repo | **Langfuse** (versioned, evaluated, fetched at runtime) | A runtime dependency — but prompts become versioned, A/B-testable assets, and stay out of source control (which is *why* this showcase can exist) |
| Showing scraping capability responsibly | Publish the real scraper | **Generic resilient-fetcher engine + neutral demo target**, engineering described in prose | No flashy site-specific demo — but no ToS violation and no "evasion tooling" optics |

Each row links to a full record under **[docs/adr/](docs/adr)**.

## Results / Evaluation

Two very different kinds of number, kept strictly separate for honesty:

### A. Deterministic security properties — reproduce offline, right now

`make eval` runs on a deterministic **mock** model (no keys, no network) and **gates its exit code** on the security guarantees — these are true regardless of the model:

| Property | Result |
|---|---|
| Money-gate blocks autonomous `send_D` (→ human) | **100%** (`_AUTO_SEND_D_ENABLED = False`, enforced in the real workflow path) |
| Input-guard catches known injection tripwires | **10/10**, with **0/4** false positives on benign replies |
| Output-guard catches outbound leaks (datamark / role-marker / scaffolding) | **5/5**, clean drafts pass |
| Opener supply-chain guard (markup / injection / PII from scraped reviews) | **4/4** |
| `ReviewHook` model-layer validators null poisoned staff names | ✔ |
| Structured outputs validate; all 4 harnesses run to completion; graph compiles | ✔ · `171 passed, 1 skipped` |

### B. Production model-quality figures — measured at Effective Tours

Measured in production against a real 70B model (`llama-3.3-70b-instruct`, temp 0.1). These are **not** reproduced by the offline mock (a mock cannot stand in for model quality); reproduce them by pointing the same harness at a live model (`MODEL_BACKEND=openrouter`):

| Metric | Production value |
|---|---|
| Router (Agent E) exact-match accuracy | **92.2%** (59/64 adversarial cases), next-action 93.8%, 0 LLM errors |
| Red-team suite (injection / jailbreak / funnel-abuse / STOP / messy / edge) | **42/42 defended** |
| Agent B honesty gate | **100%** |

> The offline mock deliberately prints an *illustrative* ~28% router accuracy, clearly labeled — it is **not** rigged to reproduce the 92.2%. The reproducible artifact is the **methodology + the deterministic guarantees**; the accuracy figure is a production measurement you can re-run live.

## Quick start

```bash
git clone https://github.com/baranlanka/et-hotel-ai-showcase && cd et-hotel-ai-showcase
make install         # creates .venv, installs pinned deps
make eval            # deterministic security eval — offline, no keys, gates on the guarantees
make demo-graph      # run the content graph on a synthetic hotel (mock model)
make demo-scrape     # run the resilient fetcher against a neutral demo backend
make test            # 171 passed, 1 skipped
```

No API keys are required — the default `MODEL_BACKEND=mock` is deterministic and offline. To see live generation, set `MODEL_BACKEND=ollama` (local) or `MODEL_BACKEND=openrouter` (a key) in `.env` (copy from `.env.example`).

## What's shown vs. withheld

This is a real production system, so it is shown *responsibly*:

- **Curated prompts are withheld.** The production prompts are curated, versioned, and evaluated in **Langfuse** and are not in this repo. The pipeline ships **generic baseline prompts** (`prompts/baseline/`) that exercise the same code paths — the separation is deliberate and is itself the intended engineering signal.
- **No real data, no secrets.** Every fixture under `data/` is synthetic and fictional; there are no credentials, real endpoints, or customer data anywhere in the tree (verified with `gitleaks` + `trufflehog`).
- **The fetcher ships generic.** Scraping a specific site would breach that site's terms; so the reusable **engineering** (proxy failover, circuit breaking, rate limiting, backoff) is shown as a generic engine demonstrated against a **neutral** target — never a site-specific scraper, and framed as reliability engineering, not evasion.
- **Honest numbers.** The deterministic guarantees reproduce offline; the production model-accuracy figures are labeled as production measurements, and the mock is not rigged to fake them.

## Roadmap / Limitations

- [x] Reproducible, offline, key-free security eval (`make eval`)
- [x] Runnable content-graph and fetcher demos on synthetic data
- [ ] Hosted live demo (mock-mode) + a recorded `make eval` GIF
- [ ] `MODEL_BACKEND=ollama` walkthrough reproducing live model metrics

**Known limitations (honest):** this is a *slice*, not the whole platform — the vision/image tail, the CMS/publishing integration, and the real orchestration wiring are out of scope here. The offline demo uses a mock model, so it proves the *architecture and the deterministic safety properties*, not model quality. The full production system (durable orchestration, distributed SQL, observability triad, CD) is described in [METHODS.md](METHODS.md) and the ADRs rather than shipped.

## Acknowledgments

Built at **Effective Tours**. OWASP-LLM guardrail design draws on the OWASP **Top-10 for LLM Applications** and Microsoft's **spotlighting/datamarking** defense (arXiv:2403.14720).

## Contact

**Ivans Novikovs** — [github.com/baranlanka](https://github.com/baranlanka) · ivan@novikov.lv

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).
