"""Tests for LangGraph state schema basic functionality.

This test module validates the basic TypedDict state schema functionality
that is actually implemented.
"""

import pytest

from llm_content_generation.et_langgraph.state import (
    ContentGenerationState,
    create_initial_state,
)


class TestContentGenerationState:
    """Test suite for ContentGenerationState schema."""

    def test_create_initial_state_basic(self):
        """Test basic initial state creation."""
        state = create_initial_state(
            hotel_id="test_hotel",
            review_keys=["key1", "key2"],
            output_key="output_key",
        )

        # Verify required fields
        assert state["hotel_id"] == "test_hotel"
        assert state["review_keys"] == ["key1", "key2"]
        assert state["output_key"] == "output_key"
        assert state["ota"] == "demo_ota"

        # Verify default values
        assert state["batch_size"] == 10
        assert isinstance(state["processed_review_ids"], set)
        assert len(state["processed_review_ids"]) == 0
        assert isinstance(state["extracted_aspects"], list)
        assert len(state["extracted_aspects"]) == 0
        assert isinstance(state["room_descriptions"], dict)
        assert len(state["room_descriptions"]) == 0
        assert isinstance(state["hotel_descriptions"], dict)
        assert len(state["hotel_descriptions"]) == 0

    def test_create_initial_state_custom_params(self):
        """Test initial state creation with custom parameters."""
        state = create_initial_state(
            hotel_id="custom_hotel",
            review_keys=["a", "b", "c"],
            output_key="custom_output",
            ota="tripadvisor",
            batch_size=5,
        )

        assert state["hotel_id"] == "custom_hotel"
        assert state["ota"] == "tripadvisor"
        assert state["batch_size"] == 5
        assert len(state["review_keys"]) == 3

    def test_initial_state_has_validation_fields(self):
        """Test that initial state includes validation fields."""
        state = create_initial_state(
            hotel_id="test_hotel",
            review_keys=["key1"],
            output_key="output_key",
        )

        assert "validation_status" in state
        assert "validation_retry_count" in state
        assert "max_validation_retries" in state
        assert "validation_errors" in state
        assert "validation_feedback" in state
        assert "description_valid" in state

        # Verify validation field defaults
        assert state["validation_status"] is None
        assert state["validation_retry_count"] == 0
        assert state["max_validation_retries"] == 3
        assert isinstance(state["validation_errors"], list)
        assert state["validation_feedback"] is None
        assert state["description_valid"] is False

    def test_initial_state_has_stats(self):
        """Test that initial state includes stats tracking."""
        state = create_initial_state(
            hotel_id="test_hotel",
            review_keys=["key1"],
            output_key="output_key",
        )

        assert "stats" in state
        assert isinstance(state["stats"], dict)

        # Verify stats structure
        stats = state["stats"]
        assert stats["total"] == 0
        assert stats["completed"] == 0
        assert stats["skipped"] == 0
        assert stats["failed"] == 0


class TestStateReducers:
    """Test suite for state reducer functionality."""

    def test_set_reducer_simulation(self):
        """Test set reducer logic using operator.or_."""
        # Simulate how processed_review_ids would merge
        current_ids = {"id1", "id2"}
        new_ids = {"id2", "id3", "id4"}

        # operator.or_ should merge sets
        merged = current_ids | new_ids
        expected = {"id1", "id2", "id3", "id4"}

        assert merged == expected
        assert len(merged) == 4

    def test_list_reducer_simulation(self):
        """Test list reducer logic using operator.add."""
        # Simulate how extracted_aspects would merge
        current_aspects = [
            {"review_id": "r1", "aspects": {"service": "good"}},
            {"review_id": "r2", "aspects": {"room": "clean"}}
        ]
        new_aspects = [
            {"review_id": "r3", "aspects": {"location": "central"}},
        ]

        # operator.add should concatenate lists
        merged = current_aspects + new_aspects

        assert len(merged) == 3
        assert merged[0]["review_id"] == "r1"
        assert merged[2]["review_id"] == "r3"

    def test_dict_reducer_simulation(self):
        """Test dict reducer logic using operator.or_."""
        # Simulate how room_descriptions would merge
        current_rooms = {
            "Standard Room": "A comfortable standard room",
            "Deluxe Room": "Spacious deluxe accommodation"
        }
        new_rooms = {
            "Deluxe Room": "Updated deluxe room description",  # Override
            "Suite": "Luxurious suite with ocean view"  # New
        }

        # operator.or_ should merge dicts with new values overriding
        merged = current_rooms | new_rooms

        assert len(merged) == 3
        assert "Standard Room" in merged
        assert "Suite" in merged
        assert merged["Deluxe Room"] == "Updated deluxe room description"  # Overridden
        assert merged["Suite"] == "Luxurious suite with ocean view"