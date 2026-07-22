"""Hotel type aggregator node for hotel-type integration.

Clean implementation that loads DataFrame from the aggregated store, filters hotel signals,
and makes a single LLM call for classification. Fails fast on issues.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict

from langfuse import observe

from llm_content_generation.et_langgraph.state import ContentGenerationState
from llm_content_generation.et_langgraph.utils.data_loader import (
    load_aggregated_data
)
from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (
    invoke_llm_with_validation,
    DEFAULT_TIMEOUT_EXTRACTION,
)
from llm_content_generation.shared.singletons import (
    get_shared_prompt_manager,
    get_shared_llm_factory
)


def _extract_json_from_response(raw_content: str) -> str:
    """Extract JSON from LLM response, handling reasoning model output.

    Handles:
    - <think>...</think> tags from DeepSeek R1
    - ```json code blocks
    - Raw JSON
    """
    content = raw_content.strip()

    # Remove <think>...</think> blocks (DeepSeek R1 reasoning)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    # Remove markdown code fences
    if content.startswith("```"):
        content = content[3:]
        if content.startswith("json"):
            content = content[4:]
        if content.endswith("```"):
            content = content[:-3]

    content = content.strip()

    # Extract JSON object by finding first { and last }
    if "{" in content and "}" in content:
        first = content.find("{")
        last = content.rfind("}")
        if first != -1 and last != -1 and last > first:
            content = content[first:last+1]

    return content


@observe(
    name="hotel_type_classification",
    as_type="generation",
    capture_input=False
)
async def _classify_hotel_type_from_signals(
    hotel_id: str,
    signals_data: str,
    logger: Any = None
) -> Dict[str, Any]:
    """LLM helper function for hotel type classification."""

    prompt_manager = get_shared_prompt_manager()
    llm_factory = get_shared_llm_factory()

    prompt_obj, metadata = prompt_manager.get_prompt(
        "langchain/hotel_classify"
    )

    model_from_prompt = prompt_manager.extract_model_from_config(metadata)
    if model_from_prompt:
        llm = llm_factory.create_from_prompt_config(model_from_prompt)
        if logger:
            logger.info(f"Hotel classification using model: {model_from_prompt}")
    else:
        llm = llm_factory.create_for_extraction()
        if logger:
            logger.info("Hotel classification using default extraction model")

    variables = {
        "hotel_id": hotel_id,
        "signals_data": signals_data
    }

    formatted_prompt = prompt_manager.compile_prompt(
        prompt_obj, variables, metadata
    )

    config = {
        "metadata": {
            "operation": "hotel_type_classification",
            "hotel_id": hotel_id
        }
    }

    # Use wrapper with timeout for slow models
    raw_content = await invoke_llm_with_validation(
        llm=llm,
        prompt=formatted_prompt,
        config=config,
        timeout_seconds=DEFAULT_TIMEOUT_EXTRACTION,
        min_response_length=20,
        operation_name=f"hotel_classification:{hotel_id}",
    )

    # Extract JSON from response (handles reasoning model output)
    json_content = _extract_json_from_response(raw_content)

    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        if logger:
            logger.error(f"JSON parse error. Raw content: {raw_content[:500]}")
            logger.error(f"Extracted JSON: {json_content[:500]}")
        raise ValueError(f"Failed to parse classification response: {e}")


async def hotel_type_aggregator_node(
    state: ContentGenerationState
) -> Dict[str, Any]:
    """Aggregate hotel signals into type classification."""

    from llm_content_generation.shared.singletons import get_shared_llm_factory
    logger = get_shared_llm_factory().logger

    # Diagnostic logging
    hotel_id = state["hotel_id"]
    data_key = state.get("aggregated_data_key")

    if not data_key:
        logger.error(f"No aggregated_data_key found for hotel {hotel_id}")
        return {"hotel_type_classification": None}

    logger.info(f"Loading aggregated data for hotel {hotel_id} from key: {data_key}")

    try:
        aggregated = await load_aggregated_data(data_key)
        logger.info(f"Loaded DataFrame with shape: {aggregated.shape}")

        # Log data breakdown
        if 'type' in aggregated.columns:
            type_counts = aggregated['type'].value_counts().to_dict()
            logger.info(f"Data type breakdown: {type_counts}")
        else:
            logger.warning(f"No 'type' column found in DataFrame. Columns: {list(aggregated.columns)}")
            return {"hotel_type_classification": None}

        hotel_signals = aggregated[aggregated["type"] == "hotel_signal"]
        logger.info(f"Found {len(hotel_signals)} hotel signals for classification")

        if hotel_signals.empty:
            logger.warning(f"No hotel signals found for hotel {hotel_id}, returning default classification")
            # Fallback classification for legacy data without hotel signals
            return {
                "hotel_type_classification": {
                    "primary_type": "unknown",
                    "confidence": 0.0,
                    "secondary_types": [],
                    "supporting_evidence": {},
                    "signal_analysis": {"reason": "No hotel signals found in processed data"}
                }
            }

        signals_json = hotel_signals[["name", "evidence"]].to_json(orient="records")
        logger.info(f"Generated signals JSON for classification: {signals_json[:200]}...")

        classification = await _classify_hotel_type_from_signals(
            hotel_id, signals_json, logger=logger
        )
        logger.info(f"Hotel type classification completed for {hotel_id}")

        return {"hotel_type_classification": classification}

    except Exception as e:
        logger.error(f"Hotel type aggregation failed for {hotel_id}: {str(e)}")
        return {"hotel_type_classification": None}