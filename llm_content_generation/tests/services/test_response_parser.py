import pytest
import json
from unittest.mock import patch

from llm_content_generation.services.response_parser import ResponseParser


@pytest.fixture
def parser():
    """Fixture for a ResponseParser instance."""
    with patch("llm_content_generation.services.response_parser.get_logger"):
        yield ResponseParser()


class TestResponseParser:
    """Tests for the ResponseParser class."""

    def test_parse_json_response_success(self, parser):
        """Test successful parsing of a JSON response."""
        json_string = '{"key": "value"}'
        data = parser.parse_json_response(json_string)
        assert data == {"key": "value"}

    def test_parse_json_response_failure(self, parser):
        """Test failure cases for JSON parsing."""
        with pytest.raises(ValueError):
            parser.parse_json_response("not a json string")

        with pytest.raises(ValueError, match="not a JSON object"):
            parser.parse_json_response('[1, 2, 3]')

    def test_clean_response_text(self, parser):
        """Test the _clean_response_text helper method."""
        assert parser._clean_response_text('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert parser._clean_response_text('Some text {"a": 1} more text') == '{"a": 1}'
        assert parser._clean_response_text(lambda: '{"a": 1}') == '{"a": 1}'
        assert parser._clean_response_text(['{"a": 1}', '{"b": 2}']) == '{"a": 1}\n{"b": 2}'


    def test_parse_validation_response(self, parser):
        """Test parsing of validation responses."""
        valid_response = '{"status": "PASS", "issues": []}'
        parsed = parser.parse_validation_response(valid_response)
        assert parsed["status"] == "PASS"

        invalid_json = 'not json'
        parsed_invalid = parser.parse_validation_response(invalid_json)
        assert parsed_invalid["status"] == "FAIL"

    def test_parse_extraction_response(self, parser):
        """Test parsing of extraction responses."""
        valid_response = '{"amenities": [{"name": "wifi", "sentiment": "positive"}]}'
        data, errors = parser.parse_extraction_response(valid_response)
        assert data is not None
        assert not errors
        assert "wifi" in [a["name"] for a in data["amenities"]]

        invalid_json = 'not json'
        data_invalid, errors_invalid = parser.parse_extraction_response(invalid_json)
        assert data_invalid is None
        assert errors_invalid

    def test_normalize_extraction_data(self, parser):
        """Test the _normalize_extraction_data helper method."""
        raw_data = {
            "amenities": [{"name": "pool", "sentiment": "good"}],
            "room_type": "suite",
            "room_sentiment": "great",
        }
        normalized = parser._normalize_extraction_data(raw_data)
        assert "room" in normalized
        assert normalized["room"]["name"] == "suite"
        assert "facts" in normalized
        assert len(normalized["facts"]) > 0

    def test_normalize_room_data(self, parser):
        """Test the _normalize_room_data helper method."""
        data_legacy = {"room_type": "deluxe", "room_sentiment": "clean"}
        room_legacy = parser._normalize_room_data(data_legacy)
        assert room_legacy["name"] == "deluxe"

        data_list = {"room_types": [{"name": "standard", "sentiment": "ok"}]}
        room_list = parser._normalize_room_data(data_list)
        assert room_list["name"] == "standard"

        assert parser._normalize_room_data({})["name"] == ""

    def test_build_facts_from_data(self, parser):
        """Test the _build_facts_from_data helper method."""
        data = {
            "room": {"name": "suite", "sentiment": "excellent"},
            "amenities": [{"name": "wifi", "opinion": "fast"}],
            "service": [{"name": "staff", "polarity": "friendly"}],
            "location": [{"name": "view", "value": "amazing"}]
        }
        facts = parser._build_facts_from_data(data)
        assert len(facts) == 4
        assert facts[0]["category"] == "room"
        assert facts[1]["category"] == "amenities"
        assert facts[1]["sentiment"] == "fast"
