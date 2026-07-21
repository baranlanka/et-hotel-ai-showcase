import pytest
from llm_content_generation.shared.models import (
    NormalizedReview,
    ExtractionRequest,
    ExtractionResponse,
    SummarizationRequest,
    SummarizationResponse,
    DescriptionRequest,
    DescriptionResponse,
    RoomDescriptionRequest,
    RoomDescriptionResponse,
)


class TestSharedModels:
    """Tests for the shared data models."""

    def test_normalized_review(self):
        """Test NormalizedReview initialization."""
        review = NormalizedReview(
            review_id="r1",
            text="Great hotel",
            date="2024-01-01",
            rating=5.0,
            source="demo_ota",
            lang="en",
            meta={"ota": "demo_ota"},
        )
        assert review.review_id == "r1"
        assert review.meta["ota"] == "demo_ota"

    def test_extraction_request(self):
        """Test ExtractionRequest initialization."""
        request = ExtractionRequest(
            review_id="r1",
            review_text="Great hotel",
            hotel_id="h1",
        )
        assert request.review_id == "r1"
        assert request.hotel_id == "h1"
        assert request.extraction_type == "unified_extraction"

    def test_extraction_response(self):
        """Test ExtractionResponse initialization."""
        response = ExtractionResponse(
            success=True,
            review_id="r1",
            raw_response="some response",
            parsed_data={"sentiment": "positive"},
        )
        assert response.success is True
        assert response.parsed_data["sentiment"] == "positive"
        assert response.provider is not None # From settings

    def test_summarization_request(self):
        """Test SummarizationRequest initialization and post_init."""
        request = SummarizationRequest(hotel_id="h1", data_source="path/to/data")
        assert request.hotel_id == "h1"
        assert request.formats == ["brief", "detailed", "ota_style"]

        request_with_formats = SummarizationRequest(
            hotel_id="h1", data_source="path/to/data", formats=["custom"]
        )
        assert request_with_formats.formats == ["custom"]

    def test_summarization_response(self):
        """Test SummarizationResponse initialization."""
        response = SummarizationResponse(
            success=True,
            hotel_id="h1",
            summaries={"brief": "A good hotel."},
            metadata={},
            coverage={},
        )
        assert response.success is True
        assert response.summaries["brief"] == "A good hotel."

    def test_description_request(self):
        """Test DescriptionRequest initialization and post_init."""
        request = DescriptionRequest(hotel_id="h1", data_source="path/to/data")
        assert request.hotel_id == "h1"
        assert request.formats == ["brief", "detailed", "marketing"]

        request_with_formats = DescriptionRequest(
            hotel_id="h1", data_source="path/to/data", formats=["custom"]
        )
        assert request_with_formats.formats == ["custom"]

    def test_description_response(self):
        """Test DescriptionResponse initialization."""
        response = DescriptionResponse(
            success=True,
            hotel_id="h1",
            descriptions={"brief": "A great place to stay."},
            metadata={},
        )
        assert response.success is True
        assert response.descriptions["brief"] == "A great place to stay."

    def test_room_description_request(self):
        """Test RoomDescriptionRequest initialization."""
        request = RoomDescriptionRequest(
            hotel_id="h1",
            room_type="suite",
            facilities=["wifi", "tv"],
            evidence=["Great room!"],
        )
        assert request.hotel_id == "h1"
        assert request.room_type == "suite"

    def test_room_description_response(self):
        """Test RoomDescriptionResponse initialization."""
        response = RoomDescriptionResponse(
            success=True,
            hotel_id="h1",
            room_type="suite",
            description="A wonderful suite.",
            metadata={},
        )
        assert response.success is True
        assert response.description == "A wonderful suite."
