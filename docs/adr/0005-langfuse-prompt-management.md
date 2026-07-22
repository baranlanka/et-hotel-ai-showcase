# 0005 — Prompt management in Langfuse (prompts out of the repo)

**Status:** Accepted

## Context
Prompts are the highest-churn, highest-value asset in an LLM system. Hardcoding them in source makes them un-versioned, un-evaluated, hard to A/B test, and entangles proprietary IP with code.

## Decision
Manage curated prompts in **Langfuse** — versioned, labeled, evaluated — and **fetch them at runtime by name**, with a built-in fallback when Langfuse is disabled. Langfuse also carries tracing/token/cost accounting.

## Alternatives considered
- **Prompts hardcoded in the repo.** Rejected: no versioning/eval, and it couples IP to code.
- **Prompts in a plain config file.** Rejected: no evaluation, labeling, or trace linkage.

## Consequences
- (+) Prompts become first-class, versioned, A/B-testable assets with linked traces; a runtime prompt change needs no deploy.
- (+) *Enabling consequence:* because curated prompts never live in source, this public showcase can ship the architecture with a generic **baseline** prompt fallback (`prompts/baseline/`) and withhold the proprietary ones cleanly.
- (−) A runtime dependency on Langfuse (mitigated by the offline fallback path).
