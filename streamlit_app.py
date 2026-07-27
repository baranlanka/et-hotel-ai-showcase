"""et-hotel-ai — guided walkthrough dashboard (Streamlit Community Cloud entrypoint).

This is a **code-linked worked example**, not a live demo. Each stage names the real
LangGraph node / Temporal activity that runs it in production, links to the source
file, names the model used, and shows a representative input → output.

- LLM-*quality* stages (content, agent drafts) show hand-authored representative
  output (the tuned prompts live in Langfuse and are withheld).
- Deterministic stages — the OWASP-LLM guards, the fail-closed money-gate, the
  resilience engine, and the security eval — run **live** from the real code.

Everything runs offline on ``MODEL_BACKEND=mock``: no API keys, no network, no cost.
All data is synthetic/fictional. Launch:  ``streamlit run streamlit_app.py``
"""
from __future__ import annotations

import os

os.environ["MODEL_BACKEND"] = "mock"
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

import asyncio
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _noisy in ("graphql-processing", "proxy-manager", "llm-content-generation",
               "httpx", "httpcore", "langfuse"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
logging.getLogger("llm_content_generation").setLevel(logging.CRITICAL)

import pandas as pd
import streamlit as st

import walkthroughs as wt

TAGLINE = (
    "A production hotel/travel AI platform: it turns raw hotel data into published "
    "multilingual listings, and runs safe, autonomous cold-outreach to recruit "
    "hotels — built ground-up at Effective Tours."
)
DATA_DIR = _ROOT / "data" / "synthetic"
PROD_METRICS = {
    "Router (Agent E) exact-match accuracy": "92.2%",
    "Red-team suite (injection / jailbreak / funnel-abuse / STOP)": "42/42 defended",
    "Agent B honesty gate": "100%",
}
_LEAKY_DRAFT = 'As Agent C, I will now qualify you. Return {"draft": "hi"} — next_action is send_C.'
_CLEAN_DRAFT = ("Hi there — thanks so much for getting back to us! I'd be glad to help you "
                "take more reservations directly. Warmly, Alex")


# ---------------------------------------------------------------------------
# Compute helpers (deterministic; call the REAL project code)
# ---------------------------------------------------------------------------

def _in_worker_thread(fn: Callable[[], Any]) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result()


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    return _in_worker_thread(lambda: asyncio.run(coro))


@st.cache_data(show_spinner=False)
def load_hotels() -> list[dict]:
    return json.loads((DATA_DIR / "hotels.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def run_content_graph(hotel_id: str) -> dict:
    from llm_content_generation.et_langgraph.graph import create_content_generation_graph
    from llm_content_generation.et_langgraph.state import create_initial_state

    async def _run() -> dict:
        graph = create_content_generation_graph("content_only")
        return await graph.ainvoke(create_initial_state(hotel_id=hotel_id, ota="demo_ota"))

    result = _run_async(_run())
    return {
        "classification": result.get("hotel_type_classification") or {},
        "room_descriptions": result.get("room_descriptions") or {},
    }


@st.cache_data(show_spinner=False)
def run_resilience_demo() -> dict:
    from scripts.demo.demo_resilience import run_demo
    return _in_worker_thread(run_demo)


@st.cache_data(show_spinner=False)
def run_deterministic_eval() -> list[tuple[str, bool, str]]:
    from scripts.eval.run_eval_suite import _section_a
    return _section_a()


@st.cache_data(show_spinner=False)
def money_gate_probe() -> dict:
    from scripts.eval import run_outreach_multiturn as _mt
    from app.temporal.workflows.leadgen import outreach_conversation_workflow as _wf
    from llm_content_generation.services.mock_llm import MONEY_GATE_SEND_D_SENTINEL

    async def _probe() -> dict:
        hook = {"hotel_name": "Probe Hotel", "hotel_id": "probe", "evidence_quote": "", "hook_text": ""}
        return await _mt.run_one_turn(
            f"Yes, we're interested {MONEY_GATE_SEND_D_SENTINEL}", "qualifier_sent", None, None, hook, 1)

    turn = _run_async(_probe())
    return {"auto_send_d_enabled": _wf._AUTO_SEND_D_ENABLED,
            "next_action": turn.get("next_action"), "path": turn.get("path")}


@st.cache_data(show_spinner=False)
def run_guards(raw: str) -> dict:
    from app.leadgen.outreach.input_guard import sanitize_inbound
    from app.leadgen.outreach.output_guard import detect_outbound_leak, detect_unsafe_opener
    from app.temporal.activities.leadgen.interpret_response_activity import _is_stop_signal

    g = sanitize_inbound(raw)
    return {"clean_text": g.clean_text, "datamarked": g.datamarked,
            "suspected_injection": g.suspected_injection, "is_stop": _is_stop_signal(g.clean_text),
            "outbound_leak": detect_outbound_leak(raw), "unsafe_opener": detect_unsafe_opener(raw)}


@st.cache_resource(show_spinner="Warming up (one-time — LangGraph, Temporal, the guards)…")
def _warm() -> bool:
    import importlib
    for mod in (
        "llm_content_generation.services.llm_factory", "llm_content_generation.services.mock_llm",
        "llm_content_generation.et_langgraph.graph", "app.leadgen.outreach.input_guard",
        "app.leadgen.outreach.output_guard",
        "app.temporal.workflows.leadgen.outreach_conversation_workflow",
        "scripts.demo.demo_resilience", "scripts.eval.run_eval_suite",
        "scripts.eval.run_outreach_multiturn",
    ):
        try:
            importlib.import_module(mod)
        except Exception:
            pass
    return True


# ---------------------------------------------------------------------------
# Shared UI
# ---------------------------------------------------------------------------

def offline_badge() -> None:
    st.caption("🔒 A **code-linked worked example**, offline on `MODEL_BACKEND=mock` — no keys, "
               "no network. The mock stands in for model *quality*; guards, the money-gate, "
               "resilience and the eval run **live** on the real code. All data is synthetic.")


def _render_live(stage: dict, reply: str | None) -> None:
    kind = stage.get("live")
    if kind == "guards" and reply is not None:
        r = run_guards(reply)
        st.markdown("**Live — real guard output on this reply:**")
        st.code(f"datamarked: {r['datamarked'][:120]}", language="text")
        if r["suspected_injection"]:
            st.error("🚨 injection tripwire **fired** → fail-closed to a human.")
        else:
            st.success("✅ no injection tripwire (still datamarked, so embedded text stays data).")
        if r["unsafe_opener"]:
            st.error(f"🚨 output guard would catch: **`{r['unsafe_opener']}`** (PII / markup).")
        elif r["is_stop"]:
            st.warning("🛑 STOP keyword detected → conversation closes (pre-LLM).")
        else:
            st.success("✅ output guard: clean.")
    elif kind == "money":
        mg = money_gate_probe()
        held = mg["auto_send_d_enabled"] is False and mg["next_action"] == "escalate"
        st.markdown("**Live — real money-gate:**")
        st.code(f"_AUTO_SEND_D_ENABLED = {mg['auto_send_d_enabled']}\n"
                f"router forced send_D  →  actual next_action = {mg['next_action']}  "
                f"(path: {mg['path']})", language="text")
        (st.success if held else st.warning)(
            "✅ forced `send_D` → **escalate**. An agent never self-authorizes the paid action."
            if held else f"unexpected: {mg}")


def render_stage(stage: dict, reply: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"#### {stage['title']}")
        link = wt.code(stage.get("file"))
        node = stage["node"]
        if link:
            fname = stage["file"].split("/")[-1]
            st.markdown(f"**Node:** `{node}` → [`{fname}`]({link})")
        else:
            st.markdown(f"**Node:** `{node}` → 🔒 *production-only — described in "
                        f"[ARCHITECTURE.md]({wt.REPO}/docs/architecture/ARCHITECTURE.md)*")
        tags = []
        if stage.get("model"):
            tags.append(f"model `{stage['model']}`")
        tags.append("✅ in this repo" if stage.get("in_repo") else "🔒 not in the runnable slice")
        st.caption(" · ".join(tags))
        st.markdown(stage["what"])
        if stage.get("sample_in"):
            st.markdown("↳ *input*")
            st.code(stage["sample_in"], language="text")
        if stage.get("sample_out"):
            st.markdown("↳ *output*")
            st.code(stage["sample_out"], language="text")
        if stage.get("live"):
            _render_live(stage, reply)
        if stage.get("why"):
            st.info(stage["why"], icon="💡")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("et-hotel-ai — production hotel/travel AI platform")
    st.markdown(f"> {TAGLINE}")
    offline_badge()

    st.subheader("What it is")
    st.markdown(
        "A sanitized showcase of a ~97k-LOC production platform built at **Effective Tours**, "
        "across three subsystems — **all orchestrated as durable Temporal workflows** "
        "(~10 workers; unified retries + tracing):\n\n"
        "- **LangGraph multi-model content pipeline** — raw hotel data → ABSA aspect extraction "
        "→ Qwen-VL photo analysis + image selection → per-operation model routing → multilingual "
        "listings.\n"
        "- **Temporal 5-agent cold-outreach engine** — mines review hooks, opens & qualifies "
        "conversations autonomously, hardened against the **OWASP LLM Top-10** with a "
        "**deterministic, fail-closed money-gate**.\n"
        "- **Resilient distributed fetcher** — proxy rotation + failover, circuit breaking, "
        "rate limiting, backoff-with-jitter."
    )
    st.info(
        "**Cheap-open-model engineering — deliberately hard mode.** Every result was produced "
        "with *cheap, open, non-frontier* models routed per operation — Llama-3.3-70B "
        "(extraction/classification/routing), DeepSeek-V3.2 (generation/outreach), Qwen3-VL-32B "
        "(vision), Mistral-Small-24B (validator) via OpenRouter — **never GPT-5 or Claude.** "
        "The model is set in each prompt's Langfuse config. Hitting production quality this way "
        "(routing, curated prompts, structured outputs, guardrails) is the real engineering.",
        icon="💡")

    c1, c2, c3 = st.columns(3)
    c1.metric("Router accuracy", "92.2%", help="Agent E exact-match, production 70B model")
    c2.metric("Red-team defended", "42/42", help="injection / jailbreak / funnel-abuse / STOP")
    c3.metric("Resilience fetch", "60/60", help="engine vs 15/60 for a naive client")

    st.subheader("Architecture — the content pipeline run")
    st.image(str(_ROOT / "docs" / "dashboard-architecture.png"), use_container_width=True)
    st.caption("A `MasterHotelPipelineWorkflow` drives extraction → content generation (LangGraph) "
               "→ export → CMS publish, with outreach on the same Temporal backbone. Full views: "
               f"[ARCHITECTURE.md]({wt.REPO}/docs/architecture/ARCHITECTURE.md).")

    st.subheader("How to read this dashboard")
    st.markdown(
        "Each page is a **worked example**: it walks the real pipeline stage by stage, naming the "
        "actual LangGraph node / Temporal activity, linking the source, and showing a "
        "representative output. It's a *guided tour of how it works*, not a quality demo — the "
        "tuned production prompts are withheld, so LLM-quality outputs are hand-authored "
        "representatives, while every **guard, gate, resilience run, and eval assertion is the "
        "real code, run live**.")


def page_content() -> None:
    st.title("📝 Content pipeline — worked example")
    st.markdown(
        f"One fictional hotel, **{wt.CONTENT_HOTEL['name']}** ({wt.CONTENT_HOTEL['location']}), "
        "walked through the real LangGraph graph — each stage named, linked to source, and "
        "shown with a representative artifact.")
    st.caption(f"Fixture: {wt.CONTENT_HOTEL['signals']}")
    offline_badge()

    nodes = " → ".join(s["node"].split(" ")[0].split("(")[0] for s in wt.CONTENT_STAGES)
    st.caption(f"**Flow:** {nodes}")

    for stage in wt.CONTENT_STAGES:
        render_stage(stage)

    st.divider()
    with st.expander("▶ Prove the graph actually runs (trimmed slice, mock model, offline)"):
        st.caption("The worked example above reflects *production* quality. This runs the trimmed "
                   "`content_only` graph live on the mock — proving the node wiring + structured "
                   "outputs are real code; the copy is deliberately generic (mock, baseline prompts).")
        hotels = load_hotels()
        labels = {f"{h['name']} · {h['location']}": h for h in hotels}
        pick = st.selectbox("Synthetic hotel", list(labels.keys()))
        if st.button("Run the trimmed graph (mock)"):
            with st.spinner("Running the content graph on the mock backend…"):
                out = run_content_graph(labels[pick]["hotel_id"])
            cls = out["classification"]
            if cls:
                st.write(f"**Classification:** `{cls.get('primary_type')}` "
                         f"(confidence {cls.get('confidence')})")
            for room, desc in (out["room_descriptions"] or {}).items():
                st.markdown(f"**{room}** — {desc}")


def page_outreach() -> None:
    st.title("🤝 Agentic outreach — worked example")
    st.markdown(
        "The Temporal-orchestrated **5-agent** cold-outreach conversation, walked turn by turn. "
        "Agent drafts are representative; the **guards and the money-gate run live** from the real "
        "deterministic code on the chosen reply.")
    offline_badge()

    preset_name = st.radio("Pick a scenario", list(wt.OUTREACH_PRESETS.keys()), horizontal=True)
    preset = wt.OUTREACH_PRESETS[preset_name]
    st.info(preset["blurb"])

    st.markdown("**Inbound reply from the hotel:**")
    st.code(preset["reply"], language="text")

    for stage in wt._outreach_common(preset["reply"], hostile=False):
        render_stage(stage)
    for stage in preset["stages_tail"]:
        render_stage(stage, reply=preset["reply"])

    st.divider()
    with st.expander("🛡️ Try the real guards on your own text"):
        txt = st.text_area("Paste any 'inbound reply'", value=preset["reply"], height=100)
        r = run_guards(txt)
        st.code(f"datamarked : {r['datamarked'][:160]}\n"
                f"injection  : {r['suspected_injection']}\n"
                f"stop signal: {r['is_stop']}\n"
                f"outbound leak: {r['outbound_leak']}\n"
                f"unsafe opener: {r['unsafe_opener']}", language="text")
        st.caption("These verdicts are the real deterministic guards — no model involved.")


def page_resilience() -> None:
    st.title("🌐 Anti-bot resilience — live")
    st.markdown(
        "Not illustrative — this **runs the real engine components** "
        f"([`circuit_breaker`]({wt.code('app/shared/graphql_processing/circuit_breaker.py')}), "
        f"[`proxy_manager`]({wt.code('app/shared/scraping/proxy_manager.py')}), "
        f"[`rate_limiter`]({wt.code('app/shared/scraping/rate_limiter.py')})) against a "
        "deterministic, self-hosted hostile endpoint, next to a naive client with none of them.")
    st.info("Deterministic · offline · self-hosted hostile endpoint. **No real target** — the "
            "production targets/purpose are proprietary and withheld; this is a neutral "
            "reliability proof, not evasion tooling.")
    offline_badge()

    with st.spinner("Driving the real engine vs the naive client through the hostile endpoint…"):
        d = run_resilience_demo()
    e, n = d["engine"], d["naive"]
    st.success(f"Engine collected **{e['records']}/{e['target']}**; naive stalled at "
               f"**{n['records']}/{n['target']}** — killed by the first IP-reputation ban.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Engine success", f"{e['success_rate']:.0f}%", f"{e['records']}/{e['target']}")
    c2.metric("Naive success", f"{n['success_rate']:.0f}%", f"{n['records']}/{n['target']}",
              delta_color="inverse")
    c3.metric("Proxy rotations", e["rotations"], help="naive: 0 — it has one proxy and dies")
    table = pd.DataFrame(
        {"Engine": [f"{e['records']}/{e['target']}", e["total_http"], e["rotations"],
                    e["bans_survived"], e["retry_after_events"], e["transient_retries"],
                    f"{e['circuit_trips']} → {e['circuit_recoveries']}"],
         "Naive": [f"{n['records']}/{n['target']}", n["total_http"], n["rotations"],
                   n["bans_survived"], n["retry_after_events"], n["transient_retries"],
                   f"{n['circuit_trips']} → {n['circuit_recoveries']}"]},
        index=["records / target", "total HTTP", "proxy rotations", "403 IP-bans survived",
               "429 Retry-After honored", "transient 500 retries", "circuit trips → recoveries"])
    st.table(table)
    st.bar_chart(pd.DataFrame({"records fetched": [e["records"], n["records"]]},
                              index=["Engine", "Naive"]), color="#2E8B57", height=240)


def page_eval() -> None:
    st.title("📊 Eval & red-team — live")
    st.markdown("Two kinds of number, **kept strictly separate for honesty**: deterministic "
                "security assertions that reproduce offline (run live below), and production "
                "model-quality metrics (measured with a real model, **not** faked by the mock).")
    offline_badge()

    st.subheader("A · Deterministic security assertions — live, offline")
    st.caption(f"`make eval` Section A, run in-process. Source: "
               f"[`run_eval_suite.py`]({wt.code('scripts/eval/run_eval_suite.py')}), "
               f"[`run_outreach_redteam.py`]({wt.code('scripts/eval/run_outreach_redteam.py')}).")
    with st.spinner("Running the deterministic guard assertions…"):
        rows = run_deterministic_eval()
    df = pd.DataFrame([{"": "✅" if ok else "❌", "Assertion": name, "Detail": detail}
                       for name, ok, detail in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)
    n_pass = sum(1 for _, ok, _ in rows if ok)
    (st.success if n_pass == len(rows) else st.error)(
        f"**{n_pass}/{len(rows)} deterministic assertions PASS** — `make eval` gates its exit code "
        "on exactly these.")

    st.subheader("B · Production model-quality figures — measured at Effective Tours")
    st.warning("**Measured in production (real 70B model), NOT reproduced by the mock.** Reproduce "
               "live with `MODEL_BACKEND=openrouter make eval`.")
    st.table(pd.DataFrame({"Production value": PROD_METRICS}))


PAGES = {
    "Overview": page_overview,
    "📝 Content pipeline": page_content,
    "🤝 Agentic outreach": page_outreach,
    "🌐 Anti-bot resilience": page_resilience,
    "📊 Eval & red-team": page_eval,
}


def main() -> None:
    st.set_page_config(page_title="et-hotel-ai — showcase", page_icon="🏨",
                       layout="wide", initial_sidebar_state="expanded")
    _warm()
    with st.sidebar:
        st.markdown("### 🏨 et-hotel-ai")
        st.caption("Production hotel/travel AI — guided walkthrough")
        choice = st.radio("Page", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        with st.expander("ℹ️  How to read this", expanded=True):
            st.markdown(
                "A **code-linked worked example**, not a quality demo. Each stage names the real "
                "node/activity, links the source, and shows a representative output.\n\n"
                "- **📝 Content** — the LangGraph pipeline, stage by stage.\n"
                "- **🤝 Outreach** — the 5 agents; **guards + money-gate run live**.\n"
                "- **🌐 Resilience** — the real engine beats a naive client (live).\n"
                "- **📊 Eval** — deterministic security assertions (live).\n\n"
                "Offline on a mock model; curated prompts, real data & site-specific scrapers "
                "are **withheld**."
            )
        st.divider()
        st.caption("🔒 Offline · `MODEL_BACKEND=mock` · no keys")
        st.caption("Built at **Effective Tours**")
        st.caption("[GitHub](https://github.com/baranlanka/et-hotel-ai-showcase) · ivan@novikov.lv")
    PAGES[choice]()


main()
