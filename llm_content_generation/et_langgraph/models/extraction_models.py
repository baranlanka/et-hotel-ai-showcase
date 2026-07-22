"""Pydantic models for structured LLM outputs in extraction workflows."""

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


class RoomAspect(BaseModel):
    """Structured model for room-specific sentiment analysis."""
    
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment about the room"
    )
    evidence: str = Field(
        description="Specific text from review supporting this sentiment"
    )
    type: Optional[str] = Field(
        default=None,
        description="Room type mentioned (if any)"
    )


class CategoryAspect(BaseModel):
    """Structured model for categorized aspects (amenities, service, location, other)."""

    name: str = Field(
        description="Name of the specific aspect (e.g., 'wifi', 'front desk', 'beach access')"
    )
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Sentiment about this specific aspect"
    )
    evidence: str = Field(
        description="Specific text from review supporting this sentiment"
    )


class HotelSignal(BaseModel):
    """Hotel property type signal detected in review - hotel-type integration."""

    signal_type: str = Field(
        description="Type of hotel property signal detected"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for this signal (0.0-1.0)"
    )
    evidence: str = Field(
        description="Specific text from review supporting this hotel type signal"
    )

    @field_validator('signal_type')
    @classmethod
    def normalize_signal_type(cls, v: str) -> str:
        """Normalize signal type to lowercase standard format."""
        # Normalize to lowercase and handle common variations
        normalized = v.lower().replace("-", "_").replace("&", "").replace(" ", "_")

        # Map common variations to standard values
        signal_map = {
            "luxury": "luxury",
            "budget": "budget",
            "business": "business",
            "boutique": "boutique",
            "resort": "resort",
            "extended_stay": "extended_stay",
            "hostel": "hostel",
            "bnb": "bnb",
            "bb": "bnb"
        }

        return signal_map.get(normalized, normalized)


RoomFeatureName = Literal[
    "bathroom",
    "bed",
    "view",
    "size",
    "noise",
    "balcony",
    "kitchenette",
    "temperature",
    "cleanliness",
    "layout",
    "other",
]


class RoomFeature(BaseModel):
    """Single attested feature observation tied to a specific room mention (v10+)."""

    feature: RoomFeatureName = Field(
        description="Canonical feature name; reviews are bucketed into this fixed vocabulary."
    )
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Sentiment of the feature observation"
    )
    evidence: str = Field(
        description="Verbatim contiguous span from the review (≤30 words) supporting this feature"
    )
    detail: Optional[str] = Field(
        default=None,
        description=(
            "Concrete physical detail (e.g. 'thin walls', 'walk-in shower'). "
            "MUST be null unless the noun appears verbatim in evidence. "
            "Enforced by post-validation in extraction.py."
        ),
    )


class RoomMention(BaseModel):
    """One room referenced in a review, with its attributed feature observations (v10+)."""

    room_type: str = Field(
        description="Exact roomtypes string from the canonical list, or 'Unknown'"
    )
    attribution_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0-1.0 confidence the features here truly belong to this room",
    )
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Aggregate sentiment for this room mention"
    )
    features: List[RoomFeature] = Field(
        default_factory=list,
        description="Per-feature observations attested in the review",
    )


class AspectExtractionResponse(BaseModel):
    """Complete structured response for aspect extraction from hotel reviews (v10+ schema)."""

    rooms: List[RoomMention] = Field(
        default_factory=list,
        description="One entry per room referenced in the review. Empty when no room is identifiable.",
    )
    amenities: List[CategoryAspect] = Field(
        default_factory=list,
        description="Hotel amenities mentioned (pool, gym, wifi, etc.)"
    )
    service: List[CategoryAspect] = Field(
        default_factory=list,
        description="Service-related aspects (staff, front desk, housekeeping, etc.)"
    )
    location: List[CategoryAspect] = Field(
        default_factory=list,
        description="Location-related aspects (proximity to attractions, transportation, etc.)"
    )
    other: List[CategoryAspect] = Field(
        default_factory=list,
        description="Other aspects not fitting the above categories"
    )
    hotel_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hotel property type context from review analysis"
    )
    hotel_signals: Optional[List[HotelSignal]] = Field(
        default=None,
        description="Direct hotel property signals detected in review"
    )


class HotelDescription(BaseModel):
    """Structured model for hotel description generation."""
    
    brief_description: str = Field(
        description="Concise 1-2 sentence overview of the hotel"
    )
    detailed_description: str = Field(
        description="Comprehensive 3-4 paragraph description including key features"
    )
    marketing_description: str = Field(
        description="Engaging, promotional description for marketing use"
    )


class RoomDescription(BaseModel):
    """Structured model for room description generation."""

    room_type: str = Field(
        description="Type of room being described"
    )
    brief_description: str = Field(
        description="Concise overview of the room features"
    )
    detailed_description: str = Field(
        description="Detailed description including amenities and layout"
    )
    key_features: List[str] = Field(
        description="List of key room features and amenities"
    )


class HotelTypeClassification(BaseModel):
    """Hotel type classification result from aggregated signals - hotel-type integration."""

    primary_type: Literal["luxury", "boutique", "budget", "business", "resort", "extended_stay", "hostel", "bnb"] = Field(
        description="Primary hotel property type classification"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for primary classification (0.0-1.0)"
    )
    secondary_types: List[str] = Field(
        default_factory=list,
        description="Secondary property types for mixed properties (e.g., 'luxury resort')"
    )
    supporting_evidence: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Evidence by signal type supporting the classification"
    )
    signal_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis of signal frequency and confidence weighting"
    )