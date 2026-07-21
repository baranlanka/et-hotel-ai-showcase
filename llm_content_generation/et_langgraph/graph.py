"""LangGraph workflow construction (trimmed showcase factory).

Factory for the two runnable graph variants in the showcase slice. Both are
built from pure nodes plus the synthetic data loader — no DB / S3 / CMS / vision
nodes are imported at module scope (the production factory's vision + taxonomy
tail is excluded from this public build).
"""

from __future__ import annotations

from typing import Literal

from .subgraphs.aspect_extraction_graph import create_aspect_extraction_subgraph
from .subgraphs.content_generation import (
    create_validated_content_generation_subgraph,
)


class ContentGenerationGraphs:
    """Factory class for creating the showcase workflow graphs."""

    @staticmethod
    def create_aspect_extraction_only():
        """Create the aspect-extraction-only workflow.

        review_loader -> aspect_extraction -> aggregation
        """
        return create_aspect_extraction_subgraph()

    @staticmethod
    def create_content_generation_only():
        """Create the content-generation-only workflow.

        preprocessed_data_loader -> hotel_type_aggregation ->
        room_type_loader -> room_descriptions
        """
        return create_validated_content_generation_subgraph()


# ---------------------------------------------------------------------------
# Compatibility shim — backwards-compatible entry point
# ---------------------------------------------------------------------------


def create_content_generation_graph(
    workflow_type: Literal["aspect_only", "content_only"] = "content_only",
):
    """Create a workflow graph based on ``workflow_type``.

    Args:
        workflow_type: ``"aspect_only"`` for the extraction slice or
            ``"content_only"`` (default) for the content-generation slice.

    Returns:
        Compiled LangGraph workflow.
    """
    factory = ContentGenerationGraphs()

    if workflow_type == "aspect_only":
        return factory.create_aspect_extraction_only()
    return factory.create_content_generation_only()
