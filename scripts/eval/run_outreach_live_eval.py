"""Live (no-mock) eval harness for F3 outreach agents B / C / E.

Drives the REAL prompts against a REAL OpenRouter model (no mocks) by
replicating each activity's LLM chain OUTSIDE a Temporal context (the
@activity.defn functions use activity.logger and fail off-worker).

What it evaluates:
  - Agent E (router): objective classification accuracy + intent / next_action
    confusion matrices over the 41-case router golden set, with the STOP
    keyword gate + post-LLM STOP guard mirrored from the activity.
  - Agent B (opener-fill): generates {{2}} phrases from the 9 miner seed
    ReviewHooks, scored by the LLM-judge (honesty / personalization /
    non-spam).
  - Agent C (qualifier): generates qualifier drafts from the interested /
    question router cases (paired with a synthetic review hook), scored by
    the LLM-judge.

NOT evaluated: Agent D (no LLM — triggers ET-reg/Stitch/CMS, Tailscale-blocked)
and the full conversation loop (F2 seam activities not landed; CRM F4 absent).

Outputs (NEVER repo root):
  results/outreach_eval/router_results.json     — per-case E predictions
  results/outreach_eval/agent_b_judge.json       — B drafts + judge verdicts
  results/outreach_eval/agent_c_judge.json       — C drafts + judge verdicts
  results/outreach_eval/report.md                — human-readable summary

Run (model MUST be a live OpenRouter slug that supports structured output;
the configured MODEL_EXTRACTION default 404s, so set it explicitly):

  MODEL_EXTRACTION=meta-llama/llama-3.3-70b-instruct \
    PYTHONPATH=. .venv/bin/python scripts/eval/run_outreach_live_eval.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

ROUTER_SET = _ROOT / "data/eval/router_golden_set.json"
MINER_SET = _ROOT / "data/eval/synthetic_miner_set.json"
OUT_DIR = _ROOT / "results/outreach_eval"

_SEM = asyncio.Semaphore(5)  # bound OpenRouter concurrency


# ---------------------------------------------------------------------------
# Agent E — replicate interpret_response_activity chain (no Temporal context)
# ---------------------------------------------------------------------------


async def classify_router_case(case: dict[str, Any], llm_structured) -> dict[str, Any]:
    """Run one router case through the Agent E chain (STOP gate + LLM + guard)."""
    from app.temporal.activities.leadgen._langfuse_prompt import fetch_outreach_prompt
    from app.temporal.activities.leadgen.interpret_response_activity import (
        _is_stop_signal,
    )
    from app.temporal.workflows.leadgen.models import (
        IntentEnum,
        InterpretedResponse,
        NextActionEnum,
        SentimentEnum,
    )

    msg = case["inbound_message"]
    # funnel_state: use the case hint if present, else a neutral default that
    # mirrors what the workflow would pass mid-funnel.
    funnel_state = case.get("funnel_stage_hint") or "opener_sent"

    # ── STOP keyword gate (pre-LLM, unconditional) ──
    if _is_stop_signal(msg):
        return {
            "predicted_intent": "stop",
            "predicted_next_action": "close",
            "predicted_sentiment": "neutral",
            "path": "stop_gate",
            "error": None,
        }

    async with _SEM:
        try:
            # Langfuse is the single source of truth for the router prompt (mirrors
            # interpret_response_activity's template_vars). The passed structured LLM
            # is invoked directly with the fetched messages.
            messages, _model = fetch_outreach_prompt(
                "langchain/outreach_router",
                {"funnel_state": funnel_state, "inbound_message": msg},
            )
            if messages is None:
                return {
                    "predicted_intent": None,
                    "predicted_next_action": None,
                    "predicted_sentiment": None,
                    "path": "prompt_unavailable",
                    "error": "langchain/outreach_router unavailable",
                }
            result: InterpretedResponse = await llm_structured.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            return {
                "predicted_intent": None,
                "predicted_next_action": None,
                "predicted_sentiment": None,
                "path": "llm_error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ── Post-LLM STOP guard ──
    next_action = result.next_action
    if result.intent == IntentEnum.stop and next_action != NextActionEnum.close:
        next_action = NextActionEnum.close

    return {
        "predicted_intent": result.intent.value,
        "predicted_next_action": next_action.value,
        "predicted_sentiment": result.sentiment.value,
        "path": "llm",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Agent B — replicate fill_opener_template_activity chain
# ---------------------------------------------------------------------------


async def gen_agent_b_draft(hook: dict[str, Any], llm) -> dict[str, Any]:
    """Generate the {{2}} inline phrase for one ReviewHook (Agent B)."""
    from app.temporal.activities.leadgen._langfuse_prompt import fetch_outreach_prompt
    from app.temporal.activities.leadgen.fill_opener_template_activity import (
        M1_VAR2_MAX_CHARS,
        _truncate_var2,
    )
    from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (
        invoke_llm_with_validation,
    )

    name_matches_brand = bool(hook.get("expected_name_matches_brand"))
    # Langfuse is the single source of truth for the opener prompt (mirrors
    # fill_opener_template_activity's template_vars). The brand guard is a Langfuse
    # template variable; the deterministic staff_name=N/A forcing (mirrored below)
    # is the primary brand-echo defense, so the eval passes an empty guard string.
    template_vars = {
        "var2_max": str(M1_VAR2_MAX_CHARS),
        "brand_guard_instruction": "",
        "hotel_name": hook["hotel_name"],
        "tier": f"Tier {hook['expected_tier']}",
        "hook_text": hook.get("expected_evidence_quote", ""),  # seed has no hook_text; use evidence
        "staff_name": "N/A" if name_matches_brand else (hook.get("expected_staff_name") or "N/A"),
        "role": "N/A" if name_matches_brand else (hook.get("expected_role") or "N/A"),
        "evidence_quote": hook.get("expected_evidence_quote", ""),
        "highlights": hook.get("expected_highlights", "(none)"),  # miner seeds pre-date highlights field
        "is_reengagement": False,
    }
    messages, _model = fetch_outreach_prompt("langchain/outreach_opener_fill", template_vars)
    if messages is None:
        return {"draft": None, "error": "langchain/outreach_opener_fill unavailable"}
    async with _SEM:
        try:
            raw = await invoke_llm_with_validation(
                llm=llm, prompt=messages, config={}, timeout_seconds=60,
                min_response_length=5, operation_name=f"agentB:{hook['hotel_name']}",
            )
        except Exception as exc:  # noqa: BLE001
            return {"draft": None, "error": f"{type(exc).__name__}: {exc}"}
    var_2 = _truncate_var2(raw)
    return {"draft": var_2, "error": None}


# ---------------------------------------------------------------------------
# Agent C — replicate handle_reply_qualifier_activity chain
# ---------------------------------------------------------------------------


async def gen_agent_c_draft(
    inbound: str, review_hook: dict[str, Any], llm,
    funnel_state: str = "opener_sent", intent: str = "interested",
) -> dict[str, Any]:
    """Generate a qualifier draft for one inbound + review hook (Agent C).

    Langfuse is the single source of truth for the qualifier prompt; this mirrors
    handle_reply_qualifier_activity's template_vars so the eval exercises the real prompt.
    """
    from app.temporal.activities.leadgen._langfuse_prompt import fetch_outreach_prompt
    from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (
        invoke_llm_with_validation,
    )

    _channel = "chat" if "chat" in inbound.lower() else "email"
    template_vars = {
        "hotel_name": review_hook.get("hotel_name", "the property"),
        "ota_hotel_id": review_hook.get("ota_hotel_id", "unknown"),
        "inbound_message": inbound,
        "evidence_quote": review_hook.get("evidence_quote", ""),
        "hook_text": review_hook.get("hook_text", ""),
        "funnel_state": funnel_state,
        "intent": intent,
        "channel": _channel,
        "conversation_so_far": "(no prior messages yet)",
    }
    messages, _model = fetch_outreach_prompt("langchain/outreach_qualifier", template_vars)
    if messages is None:
        return {"draft": None, "error": "langchain/outreach_qualifier unavailable"}
    async with _SEM:
        try:
            raw = await invoke_llm_with_validation(llm, messages)
        except Exception as exc:  # noqa: BLE001
            return {"draft": None, "error": f"{type(exc).__name__}: {exc}"}
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")
        ).strip()
    try:
        draft = json.loads(cleaned)["draft"]
    except Exception as exc:  # noqa: BLE001
        return {"draft": None, "error": f"parse_error: {exc}", "raw": raw[:300]}
    return {"draft": draft, "error": None}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion(rows: list[dict], gold_key: str, pred_key: str) -> dict[str, dict[str, int]]:
    m: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        g = r[gold_key]
        p = r[pred_key] if r[pred_key] is not None else "ERROR"
        m[g][p] += 1
    return {k: dict(v) for k, v in m.items()}


def render_matrix(title: str, matrix: dict[str, dict[str, int]], labels: list[str]) -> str:
    cols = labels + ["ERROR"]
    head = "| gold ↓ / pred → | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [f"**{title}**", "", head, sep]
    for g in labels:
        row = matrix.get(g, {})
        cells = [str(row.get(c, 0)) for c in cols]
        lines.append(f"| **{g}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    from llm_content_generation.services.llm_factory import LLMFactory, LLMConfig
    from app.temporal.workflows.leadgen.models import InterpretedResponse
    from scripts.leadgen.outreach_llm_judge import (
        judge_draft_async,
        compute_aggregate_metrics,
    )

    cfg = LLMConfig.for_extraction()
    model_used = cfg.model_name
    print(f"[eval] provider={cfg.provider} model={model_used} temp={cfg.temperature}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    router = json.loads(ROUTER_SET.read_text())["cases"]
    miner = json.loads(MINER_SET.read_text())["hotels"]

    # ── Agent E: 41 router cases ──────────────────────────────────────────────
    print(f"[eval] Agent E — {len(router)} router cases ...")
    llm_struct = LLMFactory().create_for_extraction().with_structured_output(
        InterpretedResponse
    )
    e_preds = await asyncio.gather(
        *(classify_router_case(c, llm_struct) for c in router)
    )
    router_rows = []
    for case, pred in zip(router, e_preds):
        router_rows.append(
            {
                "id": case["id"],
                "inbound_message": case["inbound_message"],
                "gold_intent": case["expected_intent"],
                "gold_next_action": case["expected_next_action"],
                "tier": case.get("tier"),
                "sentiment": case.get("sentiment"),
                "notes": case.get("notes", ""),
                **pred,
                "intent_ok": pred["predicted_intent"] == case["expected_intent"],
                "next_action_ok": pred["predicted_next_action"] == case["expected_next_action"],
                "exact_ok": (
                    pred["predicted_intent"] == case["expected_intent"]
                    and pred["predicted_next_action"] == case["expected_next_action"]
                ),
            }
        )
    (OUT_DIR / "router_results.json").write_text(
        json.dumps({"model": model_used, "rows": router_rows}, indent=2, ensure_ascii=False)
    )

    # ── Agent B: 9 miner seeds ────────────────────────────────────────────────
    print(f"[eval] Agent B — {len(miner)} opener-fills + judge ...")
    llm_plain = LLMFactory().create_for_extraction()
    b_drafts = await asyncio.gather(*(gen_agent_b_draft(h, llm_plain) for h in miner))
    b_judge_rows = []
    b_judge_tasks = []
    for hook, d in zip(miner, b_drafts):
        rh = {
            "evidence_quote": hook.get("expected_evidence_quote", ""),
            "hook_text": hook.get("expected_evidence_quote", ""),
        }
        if d["draft"]:
            async def _j(draft=d["draft"], rh=rh, hid=hook["hotel_name"]):
                async with _SEM:
                    return await judge_draft_async(draft, "C", llm_plain, review_hook=rh, hotel_id=hid)
            b_judge_tasks.append(_j())
        else:
            b_judge_tasks.append(None)
    b_verdicts = await asyncio.gather(
        *[t for t in b_judge_tasks if t is not None]
    )
    vit = iter(b_verdicts)
    for hook, d in zip(miner, b_drafts):
        v = next(vit) if d["draft"] else {}
        b_judge_rows.append(
            {
                "hotel_name": hook["hotel_name"],
                "tier": hook["expected_tier"],
                "name_matches_brand": hook.get("expected_name_matches_brand"),
                "evidence_quote": hook.get("expected_evidence_quote", ""),
                "draft": d["draft"],
                "draft_len": len(d["draft"]) if d["draft"] else 0,
                "gen_error": d["error"],
                **{k: v.get(k) for k in (
                    "honesty", "honesty_reason", "personalization",
                    "non_spamminess", "non_spamminess_reason", "judge_error",
                )},
            }
        )
    b_metrics = compute_aggregate_metrics(b_judge_rows)
    (OUT_DIR / "agent_b_judge.json").write_text(
        json.dumps({"model": model_used, "metrics": b_metrics, "rows": b_judge_rows},
                   indent=2, ensure_ascii=False)
    )

    # ── Agent C: interested/question router cases ─────────────────────────────
    c_cases = [c for c in router if c["expected_next_action"] in ("send_C",)]
    print(f"[eval] Agent C — {len(c_cases)} qualifier drafts + judge ...")
    # Pair each C case with a rotating miner seed as the place-sourced hook.
    c_hooks = []
    for i, c in enumerate(c_cases):
        seed = miner[i % len(miner)]
        c_hooks.append(
            {
                "hotel_name": seed["hotel_name"],
                "ota_hotel_id": c["id"],
                "evidence_quote": seed.get("expected_evidence_quote", ""),
                "hook_text": seed.get("expected_evidence_quote", ""),
            }
        )
    c_drafts = await asyncio.gather(
        *(gen_agent_c_draft(c["inbound_message"], h, llm_plain)
          for c, h in zip(c_cases, c_hooks))
    )
    c_judge_tasks = []
    for c, h, d in zip(c_cases, c_hooks, c_drafts):
        if d["draft"]:
            async def _j(draft=d["draft"], rh=h, hid=c["id"]):
                async with _SEM:
                    return await judge_draft_async(draft, "C", llm_plain, review_hook=rh, hotel_id=hid)
            c_judge_tasks.append(_j())
        else:
            c_judge_tasks.append(None)
    c_verdicts = await asyncio.gather(*[t for t in c_judge_tasks if t is not None])
    vit = iter(c_verdicts)
    c_judge_rows = []
    for c, h, d in zip(c_cases, c_hooks, c_drafts):
        v = next(vit) if d["draft"] else {}
        c_judge_rows.append(
            {
                "case_id": c["id"],
                "inbound_message": c["inbound_message"],
                "evidence_quote": h["evidence_quote"],
                "draft": d["draft"],
                "draft_len": len(d["draft"]) if d["draft"] else 0,
                "gen_error": d["error"],
                **{k: v.get(k) for k in (
                    "honesty", "honesty_reason", "personalization",
                    "non_spamminess", "non_spamminess_reason", "judge_error",
                )},
            }
        )
    c_metrics = compute_aggregate_metrics(c_judge_rows)
    (OUT_DIR / "agent_c_judge.json").write_text(
        json.dumps({"model": model_used, "metrics": c_metrics, "rows": c_judge_rows},
                   indent=2, ensure_ascii=False)
    )

    # ── Aggregate router metrics ──────────────────────────────────────────────
    n = len(router_rows)
    errs = [r for r in router_rows if r["error"]]
    intent_acc = sum(r["intent_ok"] for r in router_rows) / n
    na_acc = sum(r["next_action_ok"] for r in router_rows) / n
    exact_acc = sum(r["exact_ok"] for r in router_rows) / n
    intent_labels = ["interested", "question", "has_site", "not_interested", "stop", "unclear"]
    na_labels = ["send_C", "send_D", "escalate", "close", "wait"]
    intent_cm = confusion(router_rows, "gold_intent", "predicted_intent")
    na_cm = confusion(router_rows, "gold_next_action", "predicted_next_action")
    disagreements = [r for r in router_rows if not r["exact_ok"]]

    # ── Render report.md ──────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# F3 Outreach Agents — Live Local Eval\n")
    lines.append(f"- **Model**: `{model_used}` (provider openrouter, temp {cfg.temperature})")
    lines.append(f"- **Router cases**: {n}  |  **Agent B drafts**: {len(b_judge_rows)}  |  **Agent C drafts**: {len(c_judge_rows)}")
    lines.append("- Agents B/C/E only. Agent D + full loop not evaluable locally (see header).\n")

    lines.append("## Agent E — Router accuracy\n")
    lines.append(f"- **Exact (intent+next_action)**: {exact_acc:.1%} ({sum(r['exact_ok'] for r in router_rows)}/{n})")
    lines.append(f"- **Intent accuracy**: {intent_acc:.1%}")
    lines.append(f"- **Next-action accuracy**: {na_acc:.1%}")
    lines.append(f"- **LLM errors**: {len(errs)}\n")
    lines.append(render_matrix("Intent confusion", intent_cm, intent_labels) + "\n")
    lines.append(render_matrix("Next-action confusion", na_cm, na_labels) + "\n")

    lines.append(f"## Agent E — Disagreements ({len(disagreements)})\n")
    lines.append("| id | inbound | gold (intent/action) | pred (intent/action) | note |")
    lines.append("|---|---|---|---|---|")
    for r in disagreements:
        inb = r["inbound_message"].replace("|", "\\|")[:60]
        note = (r["notes"] or "").replace("|", "\\|").replace("\n", " ")[:90]
        lines.append(
            f"| {r['id']} | {inb} | {r['gold_intent']}/{r['gold_next_action']} "
            f"| {r['predicted_intent']}/{r['predicted_next_action']} | {note} |"
        )
    lines.append("")

    lines.append("## Agent B — opener-fill judge\n")
    lines.append(_fmt_judge(b_metrics))
    lines.append("\n| hotel | tier | brand? | honest | pers | non-spam | draft |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in b_judge_rows:
        draft = (r["draft"] or f"ERROR: {r['gen_error']}").replace("|", "\\|")[:70]
        lines.append(
            f"| {r['hotel_name']} | {r['tier']} | {r['name_matches_brand']} | "
            f"{r['honesty']} | {r['personalization']} | {r['non_spamminess']} | {draft} |"
        )
    lines.append("")

    lines.append("## Agent C — qualifier judge\n")
    lines.append(_fmt_judge(c_metrics))
    lines.append("\n| case | honest | pers | non-spam | draft |")
    lines.append("|---|---|---|---|---|")
    for r in c_judge_rows:
        draft = (r["draft"] or f"ERROR: {r['gen_error']}").replace("|", "\\|").replace("\n", " ")[:80]
        lines.append(
            f"| {r['case_id']} | {r['honesty']} | {r['personalization']} | "
            f"{r['non_spamminess']} | {draft} |"
        )
    lines.append("")

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # ── Console summary ──
    print("\n========== SUMMARY ==========")
    print(f"Agent E exact: {exact_acc:.1%} | intent: {intent_acc:.1%} | next_action: {na_acc:.1%} | errors: {len(errs)}")
    print(f"Agent B judge: honesty={b_metrics['honesty_pass_rate']:.0%} non_spam={b_metrics['non_spam_pass_rate']:.0%} passed={b_metrics['passed']}")
    print(f"Agent C judge: honesty={c_metrics['honesty_pass_rate']:.0%} non_spam={c_metrics['non_spam_pass_rate']:.0%} passed={c_metrics['passed']}")
    print(f"Disagreements: {len(disagreements)}/{n}")
    print(f"Reports → {OUT_DIR}")


def _fmt_judge(m: dict[str, Any]) -> str:
    return (
        f"- honesty_pass_rate **{m['honesty_pass_rate']:.0%}** (gate 100%) | "
        f"non_spam_pass_rate **{m['non_spam_pass_rate']:.0%}** (gate 80%) | "
        f"avg_non_spam {m['avg_non_spamminess']:.2f} | "
        f"avg_personalization {m['avg_personalization']:.2f} | "
        f"judge_errors {m['judge_error_count']} | "
        f"**gate: {'PASS' if m['passed'] else 'FAIL'}**"
    )


if __name__ == "__main__":
    asyncio.run(main())
