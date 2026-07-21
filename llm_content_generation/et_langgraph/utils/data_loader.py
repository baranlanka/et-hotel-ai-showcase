"""On-demand aggregated-data loader.

Performance optimization: the pipeline passes a small *reference key* through
LangGraph state instead of a large DataFrame, then loads the DataFrame
on-demand where it is actually needed (hotel-type classification, room
descriptions).

In the production platform the DataFrame is persisted to / read from object
storage. For this public showcase the same key/value contract is backed by a
process-local in-memory registry, so the graph runs standalone with no object
store, no credentials, and no network — while keeping the exact
``generate_aggregated_data_key`` / ``load_aggregated_data`` surface the nodes
expect.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Process-local registry standing in for the object-store bucket. Keyed by the
# same string ``generate_aggregated_data_key`` produces.
_AGGREGATED_STORE: Dict[str, pd.DataFrame] = {}


def store_aggregated_dataframe(data_key: str, df: pd.DataFrame) -> None:
    """Persist an aggregated DataFrame under ``data_key``.

    Showcase stand-in for the object-store ``put_object`` write. A defensive
    copy is stored so downstream mutation of the returned frame cannot corrupt
    the registry.
    """
    if not data_key:
        return
    _AGGREGATED_STORE[data_key] = df.copy(deep=True)
    logger.debug("Stored aggregated data under %s (shape=%s)", data_key, df.shape)


async def load_aggregated_data(data_key: str) -> Optional[pd.DataFrame]:
    """Load an aggregated DataFrame previously stored under ``data_key``.

    Returns a defensive copy, or ``None`` when the key is falsy / unknown —
    matching the production loader's "not found → None" contract.
    """
    if not data_key:
        return None

    df = _AGGREGATED_STORE.get(data_key)
    if df is None:
        logger.debug("No aggregated data found for key: %s", data_key)
        return None

    logger.debug("Loaded aggregated data from %s (shape=%s)", data_key, df.shape)
    return df.copy(deep=True)


def generate_aggregated_data_key(
    hotel_id: str, ota: str = "demo_ota"
) -> str:
    """Generate the standardized storage key for aggregated data.

    Kept byte-for-byte identical to the production key scheme so any code /
    test asserting on the key shape is unaffected.

    Args:
        hotel_id: Hotel identifier
        ota: OTA identifier

    Returns:
        Standardized storage key
    """
    return f"aggregated_data/{ota}/{hotel_id}/processed_data.parquet"


def clear_aggregated_data_store() -> None:
    """Reset the in-memory registry (useful between runs / in tests)."""
    _AGGREGATED_STORE.clear()
