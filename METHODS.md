# Methods & Case Study

A design-and-evaluation write-up for the reader who wants the reasoning, not just the code: the problem, the architectural bets, and — the centerpiece — **how the agentic system's safety is evaluated and made reproducible.** This describes production work built at Effective Tours; the public repo is a sanitized, runnable slice.

---

## 1. Context & goals

**Problem.** Operate two AI product lines for a hotel/travel platform at a scale no small team can hand-craft: (1) generate accurate, multilingual listing content for tens of thousands of hotels, and (2) recruit new hotels through personalized, largely-autonomous cold outreach.

**Quality goals (in priority order).**
1. **Safety** — an autonomous outreach agent must never self-authorize an irreversible/costly action, and must be robust to prompt injection arriving through scraped reviews and hotel replies.
2. **Cost** — per-operation model economics at scale; one frontier model everywhere is not viable.
3. **Reliability** — long-lived agent conversations must survive restarts, retries, and partial failure.
4. **Provenance discipline** — clean separation of untrusted data from instructions, and of proprietary prompts from code.

**Constraints.** Production scale and budget; a mixed model fleet (hosted + local); a small team; and a hard rule that the "money action" is human-gated.

## 2. Solution strategy (the architectural bets)

| Bet | Why |
|---|---|
| **Temporal** durable workflows for the agent conversations | Long-lived, resumable, replayable state with retries and circuit breakers — the conversation *is* the workflow |
| **LangGraph** `StateGraph` + an **`llm_factory`** routing each operation to a cost-appropriate model | Different tasks (extraction vs. vision vs. long-form generation) have different quality/cost sweet spots |
| **Defense-in-depth guardrails** (spotlighting/datamarking input guard, deterministic output leak-guard, model-layer validators) | Untrusted text enters from reviews and replies; a single prompt-level "ignore instructions" is not a control |
| **A deterministic, fail-closed money-gate** | Safety of an irreversible action must not depend on model judgment or a mutable config flag |
| **Langfuse** for prompt management | Prompts become versioned, evaluated, A/B-testable assets — and stay out of source control |

See [docs/adr/](docs/adr) for each decision as a record with the alternatives and consequences.

## 3. Building-block view

Three subsystems, detailed in [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md):

- **Content pipeline** — `aspect extraction (ABSA + LLM) → hotel-type aggregation → multi-model generation → translation`, wired as composable LangGraph subgraphs.
- **Outreach engine** — a 5-agent (`router / opener / qualifier / site-reveal / miner`) durable Temporal state machine over an inbound-signal loop, wrapped by the guards and the money-gate.
- **Resilient fetcher** — a generic fault-tolerant distributed HTTP/GraphQL engine (proxy failover, circuit breaker, token-bucket rate limiting, backoff-with-jitter).

## 4. Runtime view

The security-critical outreach turn is diagrammed as a sequence in [ARCHITECTURE.md §3](docs/architecture/ARCHITECTURE.md#3-runtime-view--the-5-agent-outreach-turn): input guard → router → (escalate | money-gate→human | qualifier→output-guard→send).

## 5. Evaluation methodology  ★

The core research contribution is **treating agent safety as a measurable, reproducible property**, not a claim. The harness (`scripts/eval/`) separates two fundamentally different things.

### 5.1 Two classes of metric

- **Deterministic properties** — hold regardless of the model (guards catch a fixed attack corpus; the money-gate constant blocks `send_D`; structured outputs validate). These are **reproducible offline** with a mock model and **gate CI**.
- **Model-quality metrics** — router accuracy, judged red-team defense — are properties of the *model*. A mock cannot reproduce them; they are measured against a live model and reported separately, never gating CI.

Conflating the two is the usual way LLM eval numbers become unfalsifiable. Keeping them apart is the point.

### 5.2 The suites

| Suite | Question | Method |
|---|---|---|
| `run_outreach_redteam` | Does the pipeline resist adversarial replies? | A labeled corpus across injection / embedded-injection / jailbreak / funnel-abuse / STOP / messy / edge / benign; run through the real guard→router→qualifier path |
| `run_outreach_review_poisoning` | Does indirect injection via *scraped reviews* leak or misbehave? | 15 adversarial review sets fed through the miner + guards |
| `run_outreach_multiturn` | Do funnel invariants hold across a whole conversation? | Multi-persona multi-turn transcripts (crescendo jailbreak, exfiltration-over-turns, funnel rusher…) asserting the money-gate never auto-fires |
| `run_outreach_live_eval` | How accurate is the router, and are drafts honest? | Exact-match over a synthetic golden set + an LLM judge for opener/qualifier honesty |

### 5.3 OWASP-LLM Top-10 mapping

| OWASP-LLM risk | Control in this system |
|---|---|
| LLM01 Prompt Injection | Input guard: tripwire + **datamarking/spotlighting**; structured system/data separation |
| LLM02 Insecure Output Handling | Output guard: deterministic leak / role-marker / scaffolding / PII checks before send |
| LLM06 Sensitive-Information Disclosure | Output guard leak detection; prompts withheld in Langfuse |
| LLM08 Excessive Agency | **Fail-closed money-gate** — human-in-the-loop for the irreversible action |
| LLM10 Unbounded Consumption | Input length cap + per-conversation flood cap |

### 5.4 Results

**Deterministic (reproduced offline in this repo — `make eval`, exit-code-gated):** money-gate blocks autonomous `send_D` 100%; input guard 10/10 injections caught, 0/4 benign false-positives; output guard 5/5 leaks caught; opener guard 4/4; `ReviewHook` nulls poisoned staff names; structured outputs validate; `171 passed, 1 skipped`.

**Production model-quality (measured at Effective Tours, `llama-3.3-70b-instruct` @ temp 0.1; reproduce live with `MODEL_BACKEND=openrouter`):** router exact-match **92.2%** (59/64), next-action 93.8%, 0 LLM errors; red-team **42/42** defended; Agent B honesty gate **100%**.

### 5.5 Reproducibility

The mock backend is deterministic and **not rigged**: it returns a fixed neutral routing class, so the offline router accuracy prints an obviously-low, clearly-labeled illustrative figure (~28%), never the production 92.2%. What reproduces offline is the *methodology and the deterministic guarantees*; the accuracy is a production measurement re-runnable against any live model. `pinned deps + Faker(seed=0) fixtures + mock model` ⇒ byte-identical results on any machine.

## 6. Trade-offs & lessons

- **Determinism where it counts.** The most valuable safety property (never auto-commit a money action) is enforced by a *constant*, not a model decision — the boring choice is the correct one.
- **Datamarking has a false-positive cost** (some benign inputs get marked); accepted, because a hard DATA/instruction boundary is worth more than marginal cleanliness.
- **Multi-model routing pays for its config complexity** in cost at scale, but demands an eval per operation to avoid silent quality regressions.
- **Prompts-as-assets (Langfuse)** turned out to be the enabling decision for *this very showcase*: because prompts were never in the repo, a safe public slice was feasible.

## 7. What's withheld & why

Curated production prompts (Langfuse), all real data (replaced by synthetic fixtures), all secrets, and any site-specific scraper. The withholding is deliberate and, for a research reviewer, is itself a signal: IP hygiene, prompt-versioning maturity, and data-provenance discipline. See [README §What's shown vs. withheld](README.md#whats-shown-vs-withheld).

## 8. Future work / open questions

- Automated **regression eval in CI against a live model** (nightly), tracking router-accuracy drift per prompt version.
- A **learned** injection detector to complement the deterministic tripwire (measuring the precision/recall trade-off vs. the current high-precision rule).
- Extending the fail-closed pattern to a general **"irreversible-action" policy layer** across agents.

## References

- OWASP **Top-10 for LLM Applications** — <https://genai.owasp.org/>
- Hines et al., **Defending Against Indirect Prompt Injection Attacks With Spotlighting**, arXiv:2403.14720 — <https://arxiv.org/abs/2403.14720>
- Temporal (durable execution) · LangGraph · Langfuse (prompt management + tracing)
