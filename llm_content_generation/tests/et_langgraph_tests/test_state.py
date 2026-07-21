import pytest
import os
import operator
from unittest.mock import patch
import importlib

from llm_content_generation.et_langgraph import state
from llm_content_generation.et_langgraph.state import (
    ContentGenerationConfig,
    create_initial_state,
)


class TestContentGenerationConfig:
    """Tests for the ContentGenerationConfig dataclass."""

    def test_config_defaults(self):
        """Test that the config initializes with default values."""
        # Reload to ensure we get the default value
        importlib.reload(state)
        config = state.ContentGenerationConfig()
        assert config.batch_size == 10

    @patch.dict(os.environ, {"REVIEW_BATCH_SIZE": "20"})
    def test_config_from_env(self):
        """Test that the config reads from environment variables."""
        importlib.reload(state)
        config = state.ContentGenerationConfig()
        assert config.batch_size == 20


class TestCreateInitialState:
    """Tests for the create_initial_state function."""

    def test_create_initial_state_defaults(self):
        """Test creating an initial state with default values."""
        initial_state = create_initial_state(hotel_id="h1")
        assert initial_state["hotel_id"] == "h1"
        assert initial_state["review_keys"] == []
        assert initial_state["batch_size"] == 10
        assert initial_state["processed_review_ids"] == set()
        assert initial_state["extracted_aspects"] == []
        assert initial_state["validation_retry_count"] == 0

    def test_ota_metadata_fields_declared_and_initialized(self):
        """New OTA-metadata fields present and None by default.

        ContentGenerationState declares image_alt_texts, room_metadata,
        property_facilities, property_highlights, and create_initial_state()
        initializes all four to None.
        """
        initial_state = create_initial_state(hotel_id="hotels.lk.test.hotel")
        assert initial_state["image_alt_texts"] is None
        assert initial_state["room_metadata"] is None
        assert initial_state["property_facilities"] is None
        assert initial_state["property_highlights"] is None

    def test_create_initial_state_with_args(self):
        """Test creating an initial state with provided arguments."""
        initial_state = create_initial_state(
            hotel_id="h2",
            review_keys=["r1", "r2"],
            output_key="out.json",
            ota="expedia",
            batch_size=5,
            content_types=["brief"],
        )
        assert initial_state["hotel_id"] == "h2"
        assert initial_state["review_keys"] == ["r1", "r2"]
        assert initial_state["output_key"] == "out.json"
        assert initial_state["ota"] == "expedia"
        assert initial_state["batch_size"] == 5
        assert initial_state["content_types"] == ["brief"]


def test_generation_activities_initial_state_includes_ota_metadata_fields():
    """generation_activities.py initial-state literal must
    include the 4 OTA-metadata fields set to None.

    Why: ContentGenerationState is a TypedDict — a missing key at activity
    construction time would surface as KeyError downstream rather than at
    schema validation. We therefore verify the literal source itself.
    """
    import inspect

    # The Temporal content-generation activity lives in the excluded CMS/DB
    # tail and is not shipped in this showcase build — skip when absent.
    _mod = pytest.importorskip("app.temporal.activities.content_generation")
    generation_activities = _mod.generation_activities

    source = inspect.getsource(generation_activities)
    for field in (
        "image_alt_texts",
        "room_metadata",
        "property_facilities",
        "property_highlights",
    ):
        assert f'"{field}": None' in source, (
            f"'{field}': None missing from generation_activities.py "
            "initial-state literal"
        )


class TestStateReducers:
    """Test suite for state reducer functionality."""

    def test_set_reducer_operator_or(self):
        """Test set reducer using operator.or_ for processed_review_ids."""
        current_ids = {"id1", "id2"}
        new_ids = {"id2", "id3", "id4"}
        merged = operator.or_(current_ids, new_ids)
        expected = {"id1", "id2", "id3", "id4"}
        assert merged == expected

    def test_list_reducer_operator_add(self):
        """Test list reducer using operator.add for extracted_aspects and errors."""
        current_aspects = [{"review_id": "r1"}]
        new_aspects = [{"review_id": "r2"}]
        merged = operator.add(current_aspects, new_aspects)
        assert len(merged) == 2
        assert merged[1]["review_id"] == "r2"

    def test_dict_reducer_operator_or(self):
        """Test dict reducer using operator.or_ for room_descriptions."""
        current_rooms = {"Standard Room": "A"}
        new_rooms = {"Deluxe Room": "B"}
        merged = operator.or_(current_rooms, new_rooms)
        assert len(merged) == 2
        assert "Deluxe Room" in merged
