"""Aspect-extraction subgraph (trimmed showcase slice).

Wires the review-driven aspect-extraction path using only pure nodes plus the
synthetic data loader:

    review_loader -> aspect_extraction -> aggregation -> END

The production graph fans through storage-discovery / preloaded-filter /
storage-loader batch routing before extraction and a storage-backed
aggregation node afterward. Those nodes are DB/S3-bound and excluded from the
showcase; here a single synthetic loader supplies the review batch and an
in-memory aggregation node persists the result under a reference key.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from llm_content_generation.et_langgraph.state import (
    ContentGenerationState,
    ContentGenerationConfig,
)
from llm_content_generation.et_langgraph.nodes.data import (
    synthetic_review_loader_node,
    synthetic_aggregation_node,
)
from llm_content_generation.et_langgraph.nodes.extraction import (
    aspect_extraction_node,
)


def create_aspect_extraction_subgraph():
    """Compile the trimmed aspect-extraction subgraph."""
    graph = StateGraph(
        ContentGenerationState,
        context_schema=ContentGenerationConfig,
    )

    graph.add_node("review_loader", synthetic_review_loader_node)
    graph.add_node("aspect_extraction", aspect_extraction_node)
    graph.add_node("aggregation", synthetic_aggregation_node)

    graph.set_entry_point("review_loader")
    graph.add_edge("review_loader", "aspect_extraction")
    graph.add_edge("aspect_extraction", "aggregation")
    graph.add_edge("aggregation", END)

    return graph.compile()
