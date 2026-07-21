"""Demo: run the trimmed LangGraph content pipeline on a synthetic hotel (mock).

Runs the ``content_only`` graph variant end-to-end, fully offline, on the
fictional ``data/synthetic`` fixtures with MODEL_BACKEND=mock. No DB, no S3, no
CMS, no API keys — it exercises the real node wiring (synthetic loaders ->
hotel-type classifier -> room-type loader -> room-description generator) against
the deterministic mock model.

Run:
    MODEL_BACKEND=mock PYTHONPATH=. .venv/bin/python scripts/demo/demo_graph.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

HOTEL_ID = "hotels.demo.coast.azure-lagoon"


async def main() -> None:
    from llm_content_generation.et_langgraph.graph import (
        create_content_generation_graph,
    )
    from llm_content_generation.et_langgraph.state import create_initial_state

    print(f"[demo-graph] MODEL_BACKEND={os.environ['MODEL_BACKEND']} "
          f"hotel={HOTEL_ID}")
    print("[demo-graph] running content-generation graph "
          "(synthetic loaders -> classify -> rooms) ...\n")

    graph = create_content_generation_graph("content_only")
    state = create_initial_state(hotel_id=HOTEL_ID, ota="demo_ota")
    result = await graph.ainvoke(state)

    classification = result.get("hotel_type_classification") or {}
    room_descriptions = result.get("room_descriptions") or {}

    print(f"hotel_type_classification: "
          f"{json.dumps(classification, ensure_ascii=False)}\n")
    print(f"room_descriptions ({len(room_descriptions)}):")
    for room, desc in room_descriptions.items():
        print(f"\n  ## {room}")
        print(f"  {desc}")

    print("\n[demo-graph] stats:",
          json.dumps(result.get("stats", {}), ensure_ascii=False))
    print("[demo-graph] OK — graph ran end-to-end offline on the mock backend.")


if __name__ == "__main__":
    asyncio.run(main())
