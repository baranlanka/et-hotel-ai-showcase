"""LangGraph state schema for content generation workflows.

This module defines the TypedDict-based state schema for LangGraph workflows,
replacing the @dataclass-based approach used in LangChain LCEL orchestration.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Set
from typing_extensions import Annotated, TypedDict
from dataclasses import dataclass, field


def _default_batch_size() -> int:
    """Read REVIEW_BATCH_SIZE from settings — uncached so monkeypatched tests work.

    PipelineExtraConfig is instantiated fresh each call (no lru_cache wrapper
    used here) so ``patch.dict(os.environ, {"REVIEW_BATCH_SIZE": "20"})`` is
    respected without requiring an explicit cache-reset call from the test.
    """
    from llm_content_generation.core.config import PipelineExtraConfig
    return PipelineExtraConfig().review_batch_size


@dataclass
class ContentGenerationConfig:
    """Configuration schema for LangGraph Static Runtime Context."""
    batch_size: int = field(default_factory=_default_batch_size)


class ContentGenerationState(TypedDict):
    """Minimal state schema for LangGraph content generation workflow.

    This state schema contains only actually used fields, supporting parallel
    processing through proper reducers while eliminating over-engineering.
    """

    # Core Input
    hotel_id: str
    ota_hotel_id: Optional[str]  # numeric OTA id for HotelImage/vision queries
    review_keys: List[str]
    output_key: Optional[str]
    ota: str

    # Processing Control
    processed_review_ids: Annotated[Set[str], operator.or_]
    room_types: Optional[str]
    current_review_batch: List[Dict[str, Any]]
    batch_size: int
    content_types: Optional[List[str]]  # For content-only generation

    # Processing Data
    extracted_aspects: Annotated[List[Dict[str, Any]], operator.add]
    aggregated_data_key: Optional[str]  # B2 reference key - PERFORMANCE OPT

    # Hotel Type Classification Data - Epic 9 Integration
    hotel_context_signals: Annotated[List[Dict[str, Any]], operator.add]  # Per-review hotel signals
    hotel_type_classification: Optional[Dict[str, Any]]  # Aggregated hotel type result

    # Generated Content
    room_descriptions: Annotated[Dict[str, str], operator.or_]  # {room_type: full paragraph}
    # Additive short card blurbs {room_type: ~30-word brief}. Populated only when
    # the room prompt label in use emits {"brief","full"} (v46+); empty under v45.
    room_descriptions_brief: Annotated[Dict[str, str], operator.or_]
    hotel_descriptions: Annotated[Dict[str, str], operator.or_]

    # Validation State Management
    validation_status: Optional[str]  # "pending", "passed", "failed"
    validation_retry_count: int  # Current retry attempt (0-based)
    max_validation_retries: int  # Configurable retry limit (default: 3)
    validation_errors: Annotated[List[str], operator.add]
    validation_feedback: Optional[str]  # For providing feedback to regen

    # Content Validation Flags
    description_valid: bool

    # Essential Tracking
    stats: Dict[str, Any]

    # OTA Metadata
    image_alt_texts: Optional[Dict[str, List[str]]]
    room_metadata: Optional[Dict[str, Dict[str, Any]]]
    property_facilities: Optional[Any]
    property_highlights: Optional[Any]

    # Hotel-level CMS facts (name/location/star/accommodation from the
    # Stage-1 details.json scrape; synthesis carries none of these).
    hotel_details: Optional[Dict[str, Any]]

    # Real OTA display name (canonical_name from hotel_details) — surfaced as
    # a top-level field so description prompts use it instead of the slug-bearing
    # hotel_id. Populated in load_extraction_metadata; None until then.
    hotel_name: Optional[str]

    # hotel-level vision-grounded synthesis
    property_visual: Optional[Dict[str, Any]]

    # stitch taxonomy JSON + property visual bundle keys
    taxonomy_json: Optional[Dict[str, Any]]
    property_visual_json: Optional[Dict[str, Any]]

    # deterministic display-image selection (HERO/ABOUT/gallery +
    # per-room featured image), built by select_display_images from the
    # retained Qwen sketches (no new LLM calls). Shape:
    # {"images": {b2_storage_path: {"sort_order": int, "active": bool}},
    #  "rooms": {room_name: b2_storage_path}}
    image_selection: Optional[Dict[str, Any]]

    # enriched positive reviews (LLM-rewritten by the
    # generate_enriched_reviews node, which runs after build_stitch_taxonomy so it
    # can read the visual register). Each entry: {review_body, sort_order}.
    enriched_reviews: Optional[List[Dict[str, Any]]]

    # target languages for the translate_content node (excludes "en";
    # None/empty = no-op, byte-identical English-only path). Resolved upstream
    # of the graph (trigger/orchestration); the node only reads it.
    languages: Optional[List[str]]
    # MT output, produced by translate_content (runs after
    # content_generation so it can translate the FINAL English text).
    # {lang: {hotel_descriptions: {...}, room_descriptions: {...}, room_descriptions_brief: {...}}}
    translations: Annotated[Dict[str, Dict[str, Any]], operator.or_]


def create_initial_state(
    hotel_id: str,
    review_keys: Optional[List[str]] = None,
    output_key: Optional[str] = None,
    ota: str = "demo_ota",
    batch_size: int = 10,
    content_types: Optional[List[str]] = None,
    languages: Optional[List[str]] = None
) -> ContentGenerationState:
    """Create initial state for content generation workflow.

    Args:
        hotel_id: Hotel identifier
        review_keys: List of review file keys to process
        output_key: Storage key for final output
        ota: OTA identifier (default: demo_ota)
        batch_size: Number of reviews to process in each batch
        content_types: Types of content to generate (for content-only mode)
        languages: target language codes to translate into (excl. "en");
            None/empty = translate_content no-ops (byte-identical English-only path)

    Returns:
        Initial state dictionary for the workflow
    """
    return ContentGenerationState(
        # Core Input
        hotel_id=hotel_id,
        ota_hotel_id=None,  # resolved in load_extraction_metadata
        review_keys=review_keys or [],
        output_key=output_key,
        ota=ota,

        # Processing Control
        processed_review_ids=set(),
        room_types=None,
        current_review_batch=[],
        batch_size=batch_size,
        content_types=content_types,

        # Processing Data
        extracted_aspects=[],
        aggregated_data_key=None,

        # Hotel Type Classification Data - Epic 9 Integration
        hotel_context_signals=[],
        hotel_type_classification=None,

        # Generated Content
        room_descriptions={},
        room_descriptions_brief={},
        hotel_descriptions={},

        # Validation State Management
        validation_status=None,
        validation_retry_count=0,
        max_validation_retries=3,
        validation_errors=[],
        validation_feedback=None,

        # Content Validation Flags
        description_valid=False,

        # Essential Tracking
        stats={
            "total": 0,
            "completed": 0,
            "skipped": 0,
            "failed": 0
        },

        # OTA Metadata
        image_alt_texts=None,
        room_metadata=None,
        property_facilities=None,
        property_highlights=None,

        # Hotel-level CMS facts
        hotel_details=None,
        hotel_name=None,  # populated by load_extraction_metadata from canonical_name

        # stitch taxonomy (keep in sync with generation_activities.py manual dict)
        taxonomy_json=None,
        property_visual_json=None,

        # display-image selection (keep in sync with
        # generation_activities.py manual dict)
        image_selection=None,

        # enriched reviews (populated by generate_enriched_reviews node)
        enriched_reviews=None,

        # target languages (input) + translations (populated by
        # translate_content node; keep in sync with generation_activities.py manual dict)
        languages=languages,
        translations={},
    )
