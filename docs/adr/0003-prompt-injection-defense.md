# 0003 — Spotlighting/datamarking input guard + deterministic output guard

**Status:** Accepted

## Context
Untrusted text enters the agents from two directions: hotel replies, and *scraped review text* used for personalization. Both are prime prompt-injection and data-exfiltration vectors (OWASP **LLM01/LLM02/LLM06**). A prompt that merely says "ignore instructions in the data" is not a control.

## Decision
Defense-in-depth, mostly deterministic and stdlib-only:
- **Input guard:** Unicode (NFKC) normalization, control-char stripping, a length cap, a narrow high-precision injection **tripwire** (fail-closed to human), role/structure-marker neutralization, and **datamarking** — interleaving a marker through untrusted text so the model can tell DATA from instructions ([spotlighting, arXiv:2403.14720](https://arxiv.org/abs/2403.14720)).
- **Output guard:** a deterministic check between draft and send for datamark echoes, role markers, internal scaffolding phrases, and PII/markup smuggled via reviews. On any hit: discard → safe generic fallback.
- **Model-layer validators:** poisoned staff names / unsafe highlights nulled at the Pydantic boundary.

## Alternatives considered
- **Prompt-only mitigation.** Rejected: not robust or testable.
- **An LLM classifier as the guard.** Rejected as the *primary* control: non-deterministic and itself injectable; kept only as an optional judge in eval.

## Consequences
- (+) A hard, testable DATA/instruction boundary; the guards are pure functions with a reproducible attack-corpus pass rate.
- (−) Datamarking causes some benign false-positives and per-turn processing cost. Accepted.
