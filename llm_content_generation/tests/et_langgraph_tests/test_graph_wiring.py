from __future__ import annotations

"""Graph-wiring tests for the trimmed showcase factory.

Prove that the two exposed graph variants compile and are wired with the
expected nodes / edges using only pure nodes plus the synthetic data loader.
The production vision + taxonomy tail (and its DB/S3/CMS nodes) is excluded from
this build, so these tests assert the trimmed topology.
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_edges(compiled_graph) -> list[tuple[str, str]]:
    """Return a flat list of (source, target) string tuples via get_graph()."""
    return [
        (e.source, e.target)
        for e in compiled_graph.get_graph().edges
    ]


def _get_node_names(compiled_graph) -> set[str]:
    """Return the set of node names from the compiled graph."""
    return set(compiled_graph.nodes.keys())


# ---------------------------------------------------------------------------
# Aspect-extraction subgraph
# ---------------------------------------------------------------------------

class TestAspectExtractionWiring:
    """review_loader -> aspect_extraction -> aggregation."""

    def test_subgraph_compiles(self) -> None:
        from llm_content_generation.et_langgraph.subgraphs.aspect_extraction_graph import (
            create_aspect_extraction_subgraph,
        )
        assert create_aspect_extraction_subgraph() is not None

    def test_expected_nodes_present(self) -> None:
        from llm_content_generation.et_langgraph.subgraphs.aspect_extraction_graph import (
            create_aspect_extraction_subgraph,
        )
        nodes = _get_node_names(create_aspect_extraction_subgraph())
        assert {"review_loader", "aspect_extraction", "aggregation"} <= nodes

    def test_edge_ordering(self) -> None:
        from llm_content_generation.et_langgraph.subgraphs.aspect_extraction_graph import (
            create_aspect_extraction_subgraph,
        )
        edges = _get_edges(create_aspect_extraction_subgraph())
        assert ("review_loader", "aspect_extraction") in edges
        assert ("aspect_extraction", "aggregation") in edges


# ---------------------------------------------------------------------------
# Content-generation subgraph
# ---------------------------------------------------------------------------

class TestContentGenerationWiring:
    """preprocessed_data_loader -> hotel_type_aggregation -> room_type_loader
    -> room_descriptions."""

    def test_subgraph_compiles(self) -> None:
        from llm_content_generation.et_langgraph.subgraphs.content_generation import (
            create_validated_content_generation_subgraph,
        )
        assert create_validated_content_generation_subgraph() is not None

    def test_expected_nodes_present(self) -> None:
        from llm_content_generation.et_langgraph.subgraphs.content_generation import (
            create_validated_content_generation_subgraph,
        )
        nodes = _get_node_names(create_validated_content_generation_subgraph())
        assert {
            "preprocessed_data_loader",
            "hotel_type_aggregation",
            "room_type_loader",
            "room_descriptions",
        } <= nodes

    def test_aggregation_does_not_feed_excluded_hotel_description(self) -> None:
        """The DB/CMS-bound hotel_description node is excluded — no such edge."""
        from llm_content_generation.et_langgraph.subgraphs.content_generation import (
            create_validated_content_generation_subgraph,
        )
        edges = _get_edges(create_validated_content_generation_subgraph())
        assert ("hotel_type_aggregation", "hotel_description") not in edges

    def test_edge_ordering(self) -> None:
        from llm_content_generation.et_langgraph.subgraphs.content_generation import (
            create_validated_content_generation_subgraph,
        )
        edges = _get_edges(create_validated_content_generation_subgraph())
        assert ("preprocessed_data_loader", "hotel_type_aggregation") in edges
        assert ("hotel_type_aggregation", "room_type_loader") in edges
        assert ("room_type_loader", "room_descriptions") in edges


# ---------------------------------------------------------------------------
# Factory surface
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_aspect_extraction_only_compiles(self) -> None:
        from llm_content_generation.et_langgraph.graph import ContentGenerationGraphs
        assert ContentGenerationGraphs.create_aspect_extraction_only() is not None

    def test_create_content_generation_only_compiles(self) -> None:
        from llm_content_generation.et_langgraph.graph import ContentGenerationGraphs
        assert ContentGenerationGraphs.create_content_generation_only() is not None

    @pytest.mark.parametrize("workflow_type", ["aspect_only", "content_only"])
    def test_create_content_generation_graph_dispatch(self, workflow_type) -> None:
        from llm_content_generation.et_langgraph.graph import (
            create_content_generation_graph,
        )
        assert create_content_generation_graph(workflow_type) is not None
