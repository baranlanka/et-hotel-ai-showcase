# Architecture

A deeper view of the platform than the README. Diagrams are native Mermaid (they render on GitHub and diff cleanly). This document describes the **production** system built at Effective Tours; the public repo ships a sanitized, runnable slice of it (see [what's withheld](../../README.md#whats-shown-vs-withheld)).

## 1. System context

The platform sits between raw hotel data and two outputs: **published listings** and **outreach conversations**.

```mermaid
flowchart TB
    ops[Ops / staff]
    hotels[Hotels<br/>inbound replies]
    web[(OTA / web sources)]
    models[(LLM providers<br/>OpenRouter · local)]
    langfuse[(Langfuse<br/>prompts + tracing)]
    cms[(CMS + CDN)]

    subgraph platform[et-hotel-ai platform]
      content[Content pipeline]
      outreach[Outreach engine]
    end

    web --> content
    content --> cms
    hotels --> outreach
    outreach -->|money actions| ops
    models -. inference .-> platform
    langfuse -. prompts/traces .-> platform
```

## 2. Pipeline runs

**Every subsystem runs as durable Temporal workflows/activities across ~10 worker services** — one retry/recovery/tracing spine for the whole platform. The headline is the **content pipeline run**: a `MasterHotelPipelineWorkflow` executes child workflows sequentially across task queues (stages 1–2 required, 3–4 best-effort → `partial_success`), while maintenance **Temporal Schedules** (proxy / cookie / sitemap) keep the shared CockroachDB pools fresh and a **bounded-WIP dispatcher** fans the pipeline out over thousands of hotels.

### Content pipeline run

```mermaid
flowchart TB
    API[FastAPI · /api/v2/orchestration] --> MASTER
    DISP[bounded-WIP dispatcher<br/>drains extraction_queue] --> S1
    SCHED[Temporal Schedules<br/>proxy · cookie · sitemap] -. refresh .-> DB[(CockroachDB<br/>proxies · cookies · queues · state)]

    subgraph temporal[Temporal — durable orchestration · retries · OTel tracing]
      direction TB
      MASTER([MasterHotelPipelineWorkflow]) --> S1
      S1[Stage 1 · Extraction workflow<br/>scrape via proxy pool] --> S2
      S2[Stage 2 · Content generation<br/>LangGraph graph] --> S3
      S3[Stage 3 · Static export] --> S4
      S4[Stage 4 · CMS publish]
      subgraph lg[Stage 2 · LangGraph content graph]
        direction LR
        A[aspect extraction<br/>ABSA + LLM] --> AGG[aggregate] --> V[Qwen-VL vision<br/>photo analysis · image select] --> TAX[image taxonomy] --> GEN[generate<br/>overviews · rooms · reviews] --> TR[translate]
      end
      S2 --> lg
    end

    S1 -->|rotating proxies + cookies| SRC[(OTA / hotel data)]
    S1 --> B2[(Backblaze B2 · data lake)]
    V -->|vision| LLM
    GEN -->|cheap open models| LLM[OpenRouter / LM Studio<br/>Langfuse: prompts + traces]
    S2 --> B2
    S4 -->|upsert| CMS[(Directus CMS)]
    temporal --- DB
    temporal --> OBS[OTel → Grafana / Tempo / Loki / Prometheus]
```

Stages: **1 Extraction** (scrape → raw JSON + images to B2; a fail-loud zero-record gate re-queues suspected soft-blocks) → **2 Content generation** (the LangGraph graph above; a CP0 idempotency probe short-circuits if a bundle already exists, saving LLM cost) → **3 Static export** (per-language SEO/meta JSON) → **4 CMS publish** (health → validate dry-run → upsert → verify). Every LLM/vision call goes through the `llm_factory` to **cheap open models** on OpenRouter (or local LM Studio), prompts + traces in Langfuse; all artifacts land in Backblaze B2; the scrape stage leases rotating proxies + cookies from the CockroachDB pool.

### Outreach pipeline run

```mermaid
flowchart LR
    DISC[discovery / ICP] --> AUD
    subgraph temporal2[Temporal · leadgen queues]
      direction LR
      AUD([LeadGenAuditWorkflow<br/>website audit + enrich]) --> PUSH([LeadGenPushWorkflow<br/>CRM push])
      PUSH --> CONV([OutreachConversationWorkflow<br/>5 agents · money-gate])
      CONV --> REVEAL([LeadGenETRegistrationWorkflow<br/>site reveal])
    end
    AUD -->|contacts| OUTS[(contact enrichment)]
    CONV -->|cheap open models| LLM2[OpenRouter · Langfuse]
    CONV <-->|messages| CRM[(CRM / email)]
    HOOK[POST /webhooks/reply<br/>+ poller] -. inbound signal .-> CONV
    REVEAL -->|register| ETAPI[(platform API)]
    REVEAL -->|link ids| CMS2[(Directus CMS)]
```

Lead discovery feeds an audit queue; `LeadGenAuditWorkflow` scores the hotel's website (deterministic, no LLM) and enriches contacts; the lead is pushed to the CRM; `OutreachConversationWorkflow` then runs the durable 5-agent conversation (30-day timeout, `continue_as_new` past ~500 history events). Inbound replies arrive by webhook or poller and are delivered as **Temporal signals** after input-guard sanitizing; the money action (site reveal) is human-gated in the pilot.

## 3. Runtime view — the 5-agent outreach turn

The security-critical path. Every inbound hotel reply is untrusted input; the router fires first on every turn, and the money action can never be taken autonomously.

```mermaid
sequenceDiagram
    autonumber
    participant H as Hotel (inbound reply)
    participant IG as Input guard
    participant E as Agent E · router
    participant C as Agent C · qualifier
    participant OG as Output guard
    participant MG as Money-gate
    participant Ops as Human (ops)

    H->>IG: raw reply (untrusted)
    IG->>IG: NFKC normalize · strip control chars · cap length
    IG->>IG: injection tripwire · datamark untrusted text (spotlighting)
    alt tripwire fires
        IG->>Ops: fail-closed → human
    else clean
        IG->>E: sanitized, datamarked text
        E->>E: classify intent + next action (structured Pydantic)
        alt low confidence / hostile / STOP
            E->>Ops: escalate / close
        else next_action = send_D (money action)
            E->>MG: intent send_D
            MG->>Ops: forced escalate — never auto (_AUTO_SEND_D_ENABLED=False)
        else next_action = send_C
            E->>C: draft qualifier reply
            C->>OG: candidate draft
            OG->>OG: leak / role-marker / scaffolding / PII checks
            alt leak detected
                OG-->>H: discard → safe generic fallback
            else clean
                OG-->>H: send
            end
        end
    end
```

**Defense layers on that path**

1. **Input guard** (`app/leadgen/outreach/input_guard.py`, stdlib-only): Unicode normalization, control-char stripping, a 4 KB cap (LLM10 unbounded-consumption), a narrow high-precision injection **tripwire**, role/structure-marker neutralization, and **datamarking** — interleaving a marker through untrusted text so the model can separate DATA from instructions ([Microsoft spotlighting, arXiv:2403.14720](https://arxiv.org/abs/2403.14720)).
2. **Output guard** (`output_guard.py`, stdlib-only): a deterministic leak check between draft and send — catches datamark echoes, role markers, internal scaffolding phrases, and the review supply-chain vector (markup / injection vocab / PII smuggled via scraped reviews). On any hit: discard and fall back to a safe generic phrase.
3. **Money-gate** (`outreach_conversation_workflow.py`): `_AUTO_SEND_D_ENABLED = False` — a hard constant, changeable only via reviewed code, that force-routes every money action to human approval. Plus a router-confidence floor and a per-conversation flood cap.
4. **Model-layer validators** (`ReviewHook`): poisoned staff names and unsafe highlights are nulled at the Pydantic boundary, before any prompt sees them.

## 4. Deployment view (production)

```mermaid
flowchart LR
    GH[GitHub Actions] -->|build| GHCR[(GHCR image)]
    GHCR -->|auto-deploy| KM[Komodo]
    KM --> HZ[Hetzner host<br/>~14 services via docker-compose]
    HZ --- TS[Tailscale / Cloudflare Tunnel]
    HZ --> OTEL[OTel collector → Tempo/Loki/Prometheus/Grafana]
```

A single image serves the whole stack; ~10 worker services (extraction, content, outreach, dispatcher, proxy, cookie, poller…) run against CockroachDB + Redis, with Temporal for durable orchestration and a full self-hosted observability triad. CI gates on ruff, a forward-only mock-ratchet, and pre-commit.

A **runnable, sanitized slice** of that observability triad ships in [`infra/observability/`](../../infra/observability/) (`docker compose up` → Grafana with the dashboards + datasources pre-wired), and a sanitized sketch of the full service graph is in [`docs/deployment/topology.docker-compose.yml`](../deployment/topology.docker-compose.yml).

## 5. Key module map (showcase)

| Path | Role |
|---|---|
| `llm_content_generation/et_langgraph/graph.py` | Content `StateGraph` factory (trimmed aspect→content slice here) |
| `llm_content_generation/services/llm_factory.py` | Per-operation multi-model routing + cost guardrail + `MODEL_BACKEND` switch |
| `llm_content_generation/core/langfuse_prompts.py` | Runtime prompt fetch with a baseline fallback when Langfuse is off |
| `app/leadgen/outreach/{input,output}_guard.py` | OWASP-LLM guards (stdlib-only) |
| `app/temporal/workflows/leadgen/outreach_conversation_workflow.py` | The durable 5-agent state machine + money-gate |
| `app/temporal/activities/leadgen/` | The five agents (router, opener, qualifier, miner, …) |
| `scripts/eval/run_outreach_*.py` | Red-team + eval harnesses |
| `app/shared/graphql_processing/`, `app/shared/scraping/` | Generic resilient fetcher engine (proxy rotation, circuit breaker, rate limiter, fingerprinting) |
| `scripts/demo/demo_resilience.py` + `hostile_endpoint.py` | Anti-bot resilience PoC — drives the real engine through a self-hosted hostile endpoint (`make demo-resilience`) |
| `infra/observability/` | Runnable, sanitized telemetry stack — OTel Collector → Tempo/Loki/Prometheus → Grafana (`docker compose up`); see its [README](../../infra/observability/README.md) |

See [METHODS.md](../../METHODS.md) for the evaluation methodology and [docs/adr/](../adr) for the decision records.
