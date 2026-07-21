"""Shared data models for the LLM content generation system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.config import settings


@dataclass
class NormalizedReview:
    """Normalized review data structure."""
    
    review_id: str
    text: str
    date: Optional[str]
    rating: Optional[float]
    source: Optional[str]
    lang: Optional[str]
    meta: Dict[str, Any]


@dataclass
class ExtractionRequest:
    """Request for LLM aspect/sentiment extraction."""
    
    review_id: str
    review_text: str
    room_types: Optional[str] = None
    traveller_type: Optional[str] = None
    room_type: Optional[str] = None
    extraction_type: str = "unified_extraction"
    hotel_id: Optional[str] = None  # For dynamic room extraction
    ota: Optional[str] = "demo_ota"  # OTA identifier for room extraction


@dataclass
class ExtractionResponse:
    """Response from LLM extraction."""
    
    success: bool
    review_id: str
    raw_response: str
    parsed_data: Optional[Dict[str, Any]]
    tokens_used: int = 0
    finish_reason: str = "unknown"
    provider: str = settings.llm.provider
    is_mock: bool = False
    error_message: Optional[str] = None


@dataclass
class SummarizationRequest:
    """Request for hotel summarization."""
    
    hotel_id: str
    data_source: Any  # DataFrame or storage path
    source_type: str = "dataframe"  # "dataframe" or "storage"
    country_hint: Optional[str] = None
    formats: List[str] = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ["brief", "detailed", "ota_style"]


@dataclass 
class SummarizationResponse:
    """Response from summarization chain."""
    
    success: bool
    hotel_id: str
    summaries: Dict[str, str]
    metadata: Dict[str, Any]
    coverage: Dict[str, float]
    error_message: Optional[str] = None


@dataclass
class DescriptionRequest:
    """Request for hotel description generation."""
    
    hotel_id: str
    data_source: Any  # DataFrame or storage path
    source_type: str = "dataframe"  # "dataframe" or "storage"
    country_hint: Optional[str] = None
    ota_data: Optional[Dict[str, Any]] = None
    formats: List[str] = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ["brief", "detailed", "marketing"]


@dataclass 
class DescriptionResponse:
    """Response from description generation chain."""
    
    success: bool
    hotel_id: str
    descriptions: Dict[str, str]
    metadata: Dict[str, Any]
    storage_key: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class RoomDescriptionRequest:
    """Request for room-level description generation.

    Args:
        hotel_id: Hotel identifier
        room_type: Room type name to generate description for
        facilities: List of facilities for the room (from final_result.json)
        evidence: Positive room-specific evidence snippets from reviews
        country_hint: Optional country hint
        data_source: Optional reference for traceability (e.g., parquet path)
    """

    hotel_id: str
    room_type: str
    facilities: List[str]
    evidence: List[str]
    country_hint: Optional[str] = None
    data_source: Any = None


@dataclass
class RoomDescriptionResponse:
    """Response from room-level description generation."""

    success: bool
    hotel_id: str
    room_type: str
    description: str
    metadata: Dict[str, Any]
    error_message: Optional[str] = None