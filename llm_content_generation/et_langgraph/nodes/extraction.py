from __future__ import annotations
import asyncio
import os
from typing import Any, Dict, List
from llm_content_generation.et_langgraph.state import ContentGenerationState
from llm_content_generation.et_langgraph.models.extraction_models import (
    AspectExtractionResponse,
)

# Temporary override knob for extraction-prompt A/B testing — set ET87_EXTRACTION_PROMPT_LABEL
# (or fall back to the broader ET76_PROMPT_LABEL) to pull a non-production
# label such as `latest`. Sourced from PipelineExtraConfig (ADR-5); resolved
# lazily so tests that monkeypatch env vars and reset the cache see overrides.
def _get_extraction_prompt_label() -> str:
    from llm_content_generation.core.config import get_pipeline_extra_settings

    return get_pipeline_extra_settings().extraction_prompt_label


_EXTRACTION_PROMPT_LABEL = _get_extraction_prompt_label()
from llm_content_generation.shared.singletons import (
    get_shared_prompt_manager, get_shared_llm_factory
)
from llm_content_generation.core.observability.langfuse_config import LangGraphObservability
from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (
    invoke_llm_with_validation,
    LLMCallError,
    DEFAULT_TIMEOUT_EXTRACTION,
)

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException


_MIN_ATTRIBUTION_CONFIDENCE = 0.5

# Common short words that should not by themselves count as a "physical noun"
# anchoring a RoomFeature.detail. Detail post-validation requires at least one
# non-stopword token from detail to appear verbatim in evidence.
_DETAIL_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "is", "was", "were", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "very", "really", "quite",
    "so", "too", "as", "than", "then", "our", "their", "his", "her", "my",
})


def _tokenize_detail(text: str) -> List[str]:
    return [
        t.strip(".,;:!?'\"()[]{}").lower()
        for t in text.split()
        if t.strip(".,;:!?'\"()[]{}")
    ]


def _detail_grounded_in_evidence(detail: str, evidence: str) -> bool:
    """Return True iff detail has at least one non-stopword token also in evidence."""
    detail_tokens = {t for t in _tokenize_detail(detail) if t and t not in _DETAIL_STOPWORDS}
    if not detail_tokens:
        return False
    evidence_tokens = set(_tokenize_detail(evidence))
    return bool(detail_tokens & evidence_tokens)


async def _get_review_text(review_item: Dict[str, Any]) -> str | None:
    """Extract review text from review item. Returns None if empty.

    Handles multiple review data formats:
    - OTA GraphQL: positiveText/negativeText nested under textDetails
    - Legacy formats: text, content, review_text, etc.
    """
    if isinstance(review_item, str):
        return review_item

    review_text = ""

    # OTA GraphQL format: textDetails.positiveText / textDetails.negativeText
    text_details = review_item.get("textDetails", {})
    if text_details:
        positive = text_details.get("positiveText", "")
        negative = text_details.get("negativeText", "")
        if positive:
            review_text += f"Positive: {positive}. "
        if negative:
            review_text += f"Negative: {negative}. "

    # Also check for direct positiveText/negativeText (flattened format)
    if not review_text:
        if review_item.get("positiveText"):
            review_text += f"Positive: {review_item['positiveText']}. "
        if review_item.get("negativeText"):
            review_text += f"Negative: {review_item['negativeText']}. "

    # Legacy/fallback fields
    if not review_text:
        text_fields = [
            "review",
            "text",
            "content",
            "review_text",
            "reviewTitle",
            "likedText",
            "dislikedText",
        ]
        for field in text_fields:
            if field in review_item and review_item[field]:
                review_text += str(review_item[field]) + ". "

    return review_text.strip() if review_text.strip() else None


def _review_combined_text(review: Any) -> str:
    """A review's positive+negative text — used for BOTH the length ranking and the
    English-language check in the aspect-extraction cap. Aspect extraction needs
    both sentiments (negativeText carries the complaints that become room-feature
    negatives)."""
    if isinstance(review, str):
        return review
    if not isinstance(review, dict):
        return ""
    td = review.get("textDetails") if isinstance(review.get("textDetails"), dict) else {}
    pos = td.get("positiveText") or review.get("positiveText") or ""
    neg = td.get("negativeText") or review.get("negativeText") or ""
    pos = pos if isinstance(pos, str) else ""
    neg = neg if isinstance(neg, str) else ""
    return f"{pos} {neg}".strip()


def _review_combined_len(review: Any) -> int:
    """Length of a review's combined text — the ranking key for the cap (longest
    first), ranked by combined length not positive-only (cf. the enriched node)."""
    return len(_review_combined_text(review))


def _select_capped_reviews(reviews: List[Any], cap: int) -> "tuple[List[Any], int]":
    """Cap the reviews fed to aspect extraction to the ``cap`` longest ENGLISH ones.

    Prefer English (ranking by length alone biases toward long non-English reviews
    — the longest reviews are often Russian/German — which would feed non-English
    evidence quotes downstream); fall back to all reviews only if none are English.
    ``cap <= 0`` or ``len(reviews) <= cap`` returns the input unchanged.

    Returns ``(selected, english_count)``; pure (no I/O) so it is unit-tested.
    """
    if not cap or len(reviews) <= cap:
        return reviews, 0
    from llm_content_generation.et_langgraph.nodes.generate_enriched_reviews import (
        _is_english_text,
    )
    english = [r for r in reviews if _is_english_text(_review_combined_text(r))]
    pool = english if english else reviews
    return sorted(pool, key=_review_combined_len, reverse=True)[:cap], len(english)


async def _convert_structured_to_aspects(
    response: AspectExtractionResponse,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert structured response to legacy aspect format and hotel signals.

    Returns:
        tuple: (aspects, hotel_signals) for backward compatibility and Epic 9 integration
    """
    aspects: List[Dict[str, Any]] = []

    # v10+ schema: per-room mentions with per-feature attributions.
    # Each RoomFeature becomes a row with type discriminator 'room_feature'
    # downstream in aggregation. Confidence threshold drops weak attributions
    # rather than poisoning per-room insights.
    for room_mention in response.rooms:
        if room_mention.attribution_confidence < _MIN_ATTRIBUTION_CONFIDENCE:
            continue
        for feat in room_mention.features:
            # Detail post-validation: must be grounded in the verbatim evidence span.
            detail = feat.detail
            if detail and not _detail_grounded_in_evidence(detail, feat.evidence):
                detail = None
            aspects.append(
                {
                    "aspect_name": "room_feature",
                    "room_type": room_mention.room_type,
                    "feature": feat.feature,
                    "sentiment": feat.sentiment,
                    "evidence": feat.evidence,
                    "detail": detail,
                    "attribution_confidence": room_mention.attribution_confidence,
                }
            )

    # Add all categorized aspects
    for category in ["amenities", "service", "location", "other"]:
        category_aspects = getattr(response, category, [])
        for item in category_aspects:
            aspects.append(
                {
                    "aspect_name": item.name,
                    "sentiment": item.sentiment,
                    "evidence": item.evidence,
                    "category": category,
                }
            )

    # Epic 9 Integration: Extract hotel signals from both hotel_context and direct hotel_signals
    hotel_signals: List[Dict[str, Any]] = []

    # Method 1: Extract from hotel_context (legacy format)
    if response.hotel_context:
        property_signals = response.hotel_context.get("property_signals", [])
        confidence = response.hotel_context.get("confidence", 0.5)
        signal_sources = response.hotel_context.get("signal_sources", [])

        get_shared_llm_factory().logger.info(
            f"Hotel context found: property_signals={property_signals}, confidence={confidence}"
        )

        for signal_str in property_signals:
            signal_type = signal_str.lower().replace("-", "_").replace("&", "").strip()
            if signal_type and signal_type in ["luxury", "budget", "business", "boutique", "resort", "extended_stay", "hostel", "bnb"]:
                evidence = signal_sources[0] if signal_sources else "Signal detected in review"
                hotel_signals.append(
                    {
                        "signal_type": signal_type,
                        "confidence": confidence,
                        "evidence": evidence,
                    }
                )
                get_shared_llm_factory().logger.info(f"Added hotel signal from context: {signal_type}")

    # Method 2: Extract from direct hotel_signals (new format)
    elif response.hotel_signals:
        get_shared_llm_factory().logger.info(f"Direct hotel signals found: {len(response.hotel_signals)} signals")

        for signal in response.hotel_signals:
            # Convert HotelSignal object to dict format
            hotel_signals.append(
                {
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "evidence": signal.evidence,
                }
            )
            get_shared_llm_factory().logger.info(f"Added hotel signal from direct format: {signal.signal_type}")

    else:
        get_shared_llm_factory().logger.warning("No hotel signals found in LLM response (neither hotel_context nor hotel_signals)")

    return aspects, hotel_signals


async def _extract_aspects_from_review(
    review_item: Dict[str, Any],
    ctx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract aspects from a single review.

    Uses a Pydantic parser for structured output. Returns empty list on
    short input or parse errors.
    """
    review_text = await _get_review_text(review_item)

    # @observe decorator handles observability automatically

    # Skip empty reviews
    if not review_text or len(review_text.strip()) < 10:
        # Review too short, returning empty list
        return []

    # Compile prompt with provided format instructions
    formatted_prompt = ctx["prompt_manager"].compile_prompt(
        ctx["prompt_obj"],
        {
            "reviewtext": review_text,
            "roomtypes": ctx["room_types"],
            "format_instructions": ctx["format_instructions"],
        },
        ctx["metadata"],
    )

    # Use node-provided config (callbacks/metadata)
    config = ctx["config"]

    try:
        # Invoke LLM with timeout and validation wrapper
        raw_text = await invoke_llm_with_validation(
            llm=ctx["llm"],
            prompt=formatted_prompt,
            config=config,
            timeout_seconds=DEFAULT_TIMEOUT_EXTRACTION,  # 3 min for extraction
            min_response_length=20,  # JSON should have some content
            operation_name=f"extraction:{ctx['hotel_id']}",
        )
        structured_response = ctx["parser"].parse(raw_text)

        # Convert to legacy format for compatibility and extract hotel signals
        aspects, hotel_signals = await _convert_structured_to_aspects(structured_response)

        # Add hotel_id to all aspects and signals
        for aspect in aspects:
            aspect["hotel_id"] = ctx["hotel_id"]

        for signal in hotel_signals:
            signal["hotel_id"] = ctx["hotel_id"]
            signal["review_id"] = review_item.get("id", "unknown")

        # Extraction successful

        return aspects, hotel_signals

    except LLMCallError as err:
        # LLM call failed (timeout, empty response, etc.) - skip review
        get_shared_llm_factory().logger.warning(
            "LLM call failed for review extraction: %s",
            str(err),
            extra={"hotel_id": ctx["hotel_id"], "error": str(err), "retryable": err.retryable},
        )
        return [], []
    except OutputParserException as err:
        # Return empty lists on parsing error
        get_shared_llm_factory().logger.warning(
            "Structured output parse failed, skipping review: %s",
            str(err),
            extra={"hotel_id": ctx["hotel_id"], "error": str(err)},
        )
        # Parsing failed, returning empty lists
        return [], []
    except (ValueError, TypeError, AttributeError) as err:
        get_shared_llm_factory().logger.warning(
            "Structured output validation failed, skipping review: %s",
            str(err),
            extra={"hotel_id": ctx["hotel_id"], "error": str(err)},
        )
        # Validation failed, returning empty lists
        return [], []


async def aspect_extraction_node(state: ContentGenerationState) -> Dict[str, Any]:
    """Extract aspects from review batch - parallel processing with fail-fast."""
    current_batch = state.get("current_review_batch", [])
    
    # CallbackHandler handles node observability automatically
    
    if not current_batch:
        # Return empty results if no batch to process
        # No batch to process
        return {
            "extracted_aspects": [],
            "skipped_reviews_count": 0,
            "processed_review_ids": set(),
        }

    # Cost cap: the deep-reviews backfill grew the pool to ~1,000 reviews/hotel and
    # this node makes one LLM call per review. ABSA aspect signal saturates well
    # before hundreds, so extract from the N LONGEST (combined positive+negative)
    # English reviews — full per-review depth is preserved on the ones kept (no
    # batching, no model change). ASPECT_MAX_REVIEWS=0 disables the cap.
    from llm_content_generation.core.config import get_pipeline_extra_settings
    cap = get_pipeline_extra_settings().aspect_max_reviews
    loaded_count = len(current_batch)
    current_batch, english_count = _select_capped_reviews(current_batch, cap)
    if len(current_batch) < loaded_count:
        get_shared_llm_factory().logger.info(
            "aspect_extraction: capped to longest English reviews",
            extra={"hotel_id": state.get("hotel_id"), "loaded": loaded_count,
                   "english": english_count, "extracted": len(current_batch), "cap": cap},
        )

    # Shared resources - fetch once
    prompt_manager = get_shared_prompt_manager()
    llm_factory = get_shared_llm_factory()

    # Get prompt config first to extract model
    prompt_obj, metadata = prompt_manager.get_prompt(
        "langchain/hotel_review_analyzer",
        _EXTRACTION_PROMPT_LABEL,
    )

    # Extract model from prompt config and create LLM accordingly
    model_from_prompt = prompt_manager.extract_model_from_config(metadata)
    if model_from_prompt:
        base_llm = llm_factory.create_from_prompt_config(model_from_prompt)
    else:
        base_llm = llm_factory.create_for_extraction()

    # Create a Pydantic parser and shared context
    parser = PydanticOutputParser(pydantic_object=AspectExtractionResponse)
    format_instructions = parser.get_format_instructions()
    room_types = state.get("room_types", "")

    # Use standardized observability configuration
    config = LangGraphObservability.get_config_with_metadata(
        tags=["aspect-extraction", "hotel-reviews"],
        session_id=f"extraction_{state['hotel_id']}",
        user_id=state["hotel_id"]
    )
    ctx = {
        "hotel_id": state["hotel_id"],
        "room_types": room_types,
        "llm": base_llm,
        "parser": parser,
        "prompt_obj": prompt_obj,
        "prompt_manager": prompt_manager,
        "metadata": metadata,
        "format_instructions": format_instructions,
        "config": config,
    }

    tasks = [
        _extract_aspects_from_review(review_item, ctx)
        for review_item in current_batch
    ]

    # Execute all review processing concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten all aspects and hotel signals from all reviews and count skipped
    extracted_aspects: List[Dict[str, Any]] = []
    hotel_context_signals: List[Dict[str, Any]] = []
    skipped_count = 0

    for result in results:
        if isinstance(result, tuple) and len(result) == 2:
            aspects, signals = result
            if aspects or signals:  # Non-empty results
                extracted_aspects.extend(aspects)
                hotel_context_signals.extend(signals)
            else:  # Empty results (skipped review)
                skipped_count += 1
        # Note: exceptions would be handled by LangGraph error handling

    # Track processed review IDs to prevent infinite loops
    processed_ids: set[str] = set()
    for review_item in current_batch:
        if isinstance(review_item, dict):
            # Use the same ID extraction logic as the filtering
            key_str = str(review_item.get("key", ""))
            base_name = os.path.basename(key_str)
            derived_id = os.path.splitext(base_name)[0]
            rid = (
                review_item.get("id")
                or review_item.get("review_id")
                or derived_id
            )
            if rid:
                processed_ids.add(rid)

    # Batch processing complete

    return {
        "extracted_aspects": extracted_aspects,
        "hotel_context_signals": hotel_context_signals,  # Epic 9 Integration
        "skipped_reviews_count": skipped_count,
        "processed_review_ids": processed_ids,
    }

