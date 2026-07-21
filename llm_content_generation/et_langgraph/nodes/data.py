"""Synthetic data-loading nodes for the trimmed showcase graph.

In the production platform the data nodes discover review files in object
storage, stream batches out of a bucket, and read a pre-aggregated parquet out
of a database-backed pipeline. All of that is replaced here by a set of
**synthetic loaders** that read a handful of clearly-fictional fixtures from
``data/synthetic/{hotels,rooms,reviews}.json`` — so the "aspect -> content"
graph runs end-to-end with no database, no object store, no credentials, and no
network.

The public function surface (the node callables the subgraphs wire) is what
matters; the bodies are deliberately simple. Every node keeps the original
state contract:

* ``synthetic_review_loader_node``     -> ``current_review_batch`` / ``review_keys``
* ``synthetic_room_type_loader_node``  -> ``room_types`` / ``room_metadata``
* ``synthetic_aggregation_node``       -> ``aggregated_data_key`` (from extraction output)
* ``synthetic_preprocessed_loader_node`` -> ``aggregated_data_key`` (content-only path)
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from llm_content_generation.et_langgraph.state import ContentGenerationState
from llm_content_generation.et_langgraph.utils.data_loader import (
    generate_aggregated_data_key,
    store_aggregated_dataframe,
)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _synthetic_dir() -> Path:
    """Locate the ``data/synthetic`` fixture directory.

    Resolves to ``<repo-root>/data/synthetic`` (the showcase layout) by walking
    up from this file, overridable via the ``SYNTHETIC_DATA_DIR`` env var for
    tests / alternate checkouts.
    """
    override = os.getenv("SYNTHETIC_DATA_DIR")
    if override:
        return Path(override)
    # data.py -> nodes -> et_langgraph -> llm_content_generation -> repo root
    return Path(__file__).resolve().parents[3] / "data" / "synthetic"


@functools.lru_cache(maxsize=8)
def _load_fixture(name: str) -> List[Dict[str, Any]]:
    """Load and cache a synthetic fixture list; empty list when absent/broken."""
    path = _synthetic_dir() / name
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("data", [])
    return [row for row in data if isinstance(row, dict)]


def _rows_for_hotel(name: str, hotel_id: str) -> List[Dict[str, Any]]:
    """Return fixture rows whose ``hotel_id`` matches (all rows if unlabeled)."""
    rows = _load_fixture(name)
    matched = [r for r in rows if r.get("hotel_id") == hotel_id]
    # Fall back to every row when the fixture rows carry no hotel_id label, so a
    # single-hotel demo fixture still works without an explicit id match.
    return matched if matched else [r for r in rows if "hotel_id" not in r]


# ---------------------------------------------------------------------------
# Review / room loaders
# ---------------------------------------------------------------------------

async def synthetic_review_loader_node(
    state: ContentGenerationState,
) -> Dict[str, Any]:
    """Load synthetic reviews for the hotel into ``current_review_batch``.

    Stand-in for the storage discovery + batch-load nodes. Honors a
    pre-provided ``current_review_batch`` (e.g. injected by a caller) and
    otherwise reads ``reviews.json``.
    """
    if state.get("current_review_batch"):
        return {"current_review_batch": state["current_review_batch"]}

    reviews = _rows_for_hotel("reviews.json", state["hotel_id"])
    review_keys = [str(r.get("id", i)) for i, r in enumerate(reviews)]
    return {
        "current_review_batch": reviews,
        "review_keys": review_keys,
        "stats": {
            **state.get("stats", {}),
            "total": len(reviews),
            "current_batch_size": len(reviews),
            "source": "synthetic",
        },
    }


async def synthetic_room_type_loader_node(
    state: ContentGenerationState,
) -> Dict[str, Any]:
    """Load synthetic room types + per-room metadata for the hotel.

    Sets ``room_types`` (comma-separated slot names, the shape
    ``room_description_node`` splits on) and ``room_metadata`` (per-room OTA
    facts the room prompt grounds on).
    """
    rooms = _rows_for_hotel("rooms.json", state["hotel_id"])
    room_types = [str(r["roomType"]) for r in rooms if r.get("roomType")]

    room_metadata: Dict[str, Dict[str, Any]] = {}
    for room in rooms:
        rt = room.get("roomType")
        if not rt:
            continue
        room_metadata[str(rt)] = {
            "description": room.get("description", ""),
            "roomSizeM2": room.get("roomSizeM2"),
            "bed_summary": room.get("bed_summary", ""),
            "bathroomCount": room.get("bathroomCount"),
            "apartmentRooms": room.get("apartmentRooms"),
        }

    return {
        "room_types": ", ".join(room_types),
        "room_metadata": room_metadata or None,
    }


# ---------------------------------------------------------------------------
# Aggregation (in-memory) — mirrors the production row contract
# ---------------------------------------------------------------------------

def _build_data_rows(
    extracted_aspects: List[Dict[str, Any]],
    hotel_context_signals: List[Dict[str, Any]],
    hotel_id: str,
) -> List[Dict[str, Any]]:
    """Normalise aspects + hotel signals into unified row dicts.

    Same row contract as the production aggregation node: every row carries a
    ``type`` discriminator (``aspect`` / ``room_feature`` / ``hotel_signal``)
    and a ``room_type`` key, so the pure downstream nodes
    (``hotel_type_aggregator``, ``room_description``) read it unchanged.
    """
    data_rows: List[Dict[str, Any]] = []

    for aspect in extracted_aspects:
        if aspect.get("aspect_name") == "room_feature":
            data_rows.append({
                "name": aspect.get("feature", "other"),
                "evidence": aspect.get("evidence", ""),
                "sentiment": aspect.get("sentiment", "neutral"),
                "review_id": aspect.get("review_id", "unknown"),
                "hotel_id": aspect.get("hotel_id", hotel_id),
                "type": "room_feature",
                "room_type": aspect.get("room_type"),
                "attribution_confidence": aspect.get("attribution_confidence"),
                "detail": aspect.get("detail"),
                "feature": aspect.get("feature"),
            })
            continue

        data_rows.append({
            "name": aspect.get("aspect_name", "general_experience"),
            "evidence": aspect.get("evidence", ""),
            "sentiment": aspect.get("sentiment", "neutral"),
            "review_id": aspect.get("review_id", "unknown"),
            "hotel_id": aspect.get("hotel_id", hotel_id),
            "type": "aspect",
            "room_type": aspect.get("room_type", None),
        })

    for signal in hotel_context_signals:
        data_rows.append({
            "name": signal.get("signal_type", "unknown"),
            "evidence": signal.get("evidence", ""),
            "sentiment": "neutral",
            "review_id": signal.get("review_id", "unknown"),
            "hotel_id": signal.get("hotel_id", hotel_id),
            "type": "hotel_signal",
            "room_type": None,
        })

    return data_rows


def _empty_aggregated_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["name", "evidence", "sentiment", "review_id",
                 "hotel_id", "type", "room_type"]
    )


async def synthetic_aggregation_node(
    state: ContentGenerationState,
) -> Dict[str, Any]:
    """Aggregate extraction output into an in-memory DataFrame + reference key.

    Consumes ``extracted_aspects`` / ``hotel_context_signals`` produced by the
    (pure) ``aspect_extraction_node`` and stores the resulting frame under a
    generated key so downstream nodes can load it on demand.
    """
    extracted_aspects = state.get("extracted_aspects", [])
    hotel_context_signals = state.get("hotel_context_signals", [])
    hotel_id = state.get("hotel_id", "unknown")

    if not extracted_aspects and not hotel_context_signals:
        result_df = _empty_aggregated_df()
    else:
        result_df = pd.DataFrame(
            _build_data_rows(extracted_aspects, hotel_context_signals, hotel_id)
        )

    data_key = generate_aggregated_data_key(hotel_id, state.get("ota", "demo_ota"))
    store_aggregated_dataframe(data_key, result_df)

    return {
        "aggregated_data_key": data_key,
        "stats": {
            **state.get("stats", {}),
            "aggregated_records": len(result_df),
            "aggregation_successful": True,
            "data_stored_to": data_key,
        },
    }


async def synthetic_preprocessed_loader_node(
    state: ContentGenerationState,
) -> Dict[str, Any]:
    """Content-only entry point: synthesise aggregated data from raw reviews.

    The content-generation subgraph normally consumes a pre-aggregated frame
    produced by the extraction pipeline. For a standalone content demo we build
    a lightweight aggregated frame directly from ``reviews.json`` (one aspect
    row per review plus a couple of hotel-level signal rows), so the pure
    ``hotel_type_aggregator`` and ``room_description`` nodes have real rows to
    read without first running the LLM extraction pass.

    If ``aggregated_data_key`` is already set upstream (extraction ran), reuse
    it untouched.
    """
    existing_key = state.get("aggregated_data_key")
    if existing_key:
        return {"aggregated_data_key": existing_key}

    hotel_id = state["hotel_id"]
    reviews = _rows_for_hotel("reviews.json", hotel_id)

    rows: List[Dict[str, Any]] = []
    for review in reviews:
        details = review.get("textDetails", {}) or {}
        positive = details.get("positiveText") or review.get("positiveText") or ""
        if positive:
            rows.append({
                "name": "general_experience",
                "evidence": positive,
                "sentiment": "positive",
                "review_id": str(review.get("id", "unknown")),
                "hotel_id": hotel_id,
                "type": "aspect",
                "room_type": review.get("roomType"),
            })

    # Derive a couple of hotel-level signal rows from the hotel fixture so the
    # classifier node has signals to work with (falls back gracefully if none).
    hotels = _rows_for_hotel("hotels.json", hotel_id)
    for hotel in hotels:
        for signal in hotel.get("type_signals", []) or []:
            rows.append({
                "name": str(signal),
                "evidence": f"Property positioned as {signal}.",
                "sentiment": "neutral",
                "review_id": "synthetic",
                "hotel_id": hotel_id,
                "type": "hotel_signal",
                "room_type": None,
            })

    result_df = pd.DataFrame(rows) if rows else _empty_aggregated_df()

    data_key = generate_aggregated_data_key(hotel_id, state.get("ota", "demo_ota"))
    store_aggregated_dataframe(data_key, result_df)

    return {
        "aggregated_data_key": data_key,
        "stats": {
            **state.get("stats", {}),
            "aggregated_records": len(result_df),
            "mode": "content_only_synthetic",
            "data_stored_to": data_key,
        },
    }
