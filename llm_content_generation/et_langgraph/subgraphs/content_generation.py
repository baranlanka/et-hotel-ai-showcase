"""Content-generation subgraph (trimmed showcase slice).

Wires the content-generation path using only pure nodes plus the synthetic data
loader:

    preprocessed_data_loader -> hotel_type_aggregation ->
    room_type_loader -> room_descriptions -> END

The production subgraph additionally fans into a hotel-description +
validation-retry loop (``descriptions_hotel``) and a ``load_extraction_metadata``
node that pull the CMS / DB / vision layers. Those are excluded from the
showcase; this slice keeps the room-description path — which is fully pure —
grounded on synthetic aggregated review data and synthetic room metadata.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from llm_content_generation.et_langgraph.state import (
    ContentGenerationState,
    ContentGenerationConfig,
)
from llm_content_generation.et_langgraph.nodes.data import (
    synthetic_preprocessed_loader_node,
    synthetic_room_type_loader_node,
)
from llm_content_generation.et_langgraph.nodes.hotel_type_aggregator import (
    hotel_type_aggregator_node,
)
from llm_content_generation.et_langgraph.nodes.descriptions_rooms import (
    room_description_node,
)


def create_validated_content_generation_subgraph():
    """Compile the trimmed content-generation subgraph.

    Name retained for factory API compatibility; the hotel-description
    validation loop present in the production graph is omitted here (it lives in
    the excluded, DB/CMS-bound ``descriptions_hotel`` node).
    """
    graph = StateGraph(
        ContentGenerationState, context_schema=ContentGenerationConfig
    )

    graph.add_node("preprocessed_data_loader", synthetic_preprocessed_loader_node)
    graph.add_node("hotel_type_aggregation", hotel_type_aggregator_node)
    graph.add_node("room_type_loader", synthetic_room_type_loader_node)
    graph.add_node("room_descriptions", room_description_node)

    graph.set_entry_point("preprocessed_data_loader")
    graph.add_edge("preprocessed_data_loader", "hotel_type_aggregation")
    graph.add_edge("hotel_type_aggregation", "room_type_loader")
    graph.add_edge("room_type_loader", "room_descriptions")
    graph.add_edge("room_descriptions", END)

    return graph.compile()
