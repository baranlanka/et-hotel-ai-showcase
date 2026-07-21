"""Centralized response parsing for LLM outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("llm-content-generation", context="activity")


def get_logger():
    """Return module-level logger via unified manager (backward-compat shim).

    Why: test_response_parser.py patches this symbol; preserving the name
    avoids test-side changes.
    """
    return _obs.logger


class ResponseParser:
    """Handles parsing and validation of LLM responses."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with cleanup and validation.
        
        Args:
            raw_text: Raw text response from LLM
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        cleaned = self._clean_response_text(raw_text)
        
        try:
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(
                "Failed to parse JSON response",
                extra={"raw_text": raw_text[:200], "cleaned": cleaned[:200], "error": str(e)}
            )
            raise ValueError(f"Invalid JSON response: {e}")
    
    def parse_validation_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse validation-specific response format.
        
        Expected format: {"status": "PASS|FAIL", "issues": [...], "meta": {...}}
        """
        try:
            data = self.parse_json_response(raw_text)
            return {
                "status": str(data.get("status", "FAIL")).upper(),
                "issues": data.get("issues", []),
                "meta": data.get("meta", {}),
            }
        except ValueError:
            # Fallback for non-JSON responses
            return {"status": "FAIL", "issues": [], "meta": {}}
    
    def parse_extraction_response(
        self, 
        raw_text: str
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Parse extraction response with error collection.
        
        Returns:
            Tuple of (parsed_data, errors)
        """
        try:
            data = self.parse_json_response(raw_text)
            normalized_data = self._normalize_extraction_data(data)
            return normalized_data, []
        except ValueError as e:
            return None, [str(e)]
    
    def _clean_response_text(self, raw_text: str) -> str:
        """Clean and extract JSON from raw LLM response."""
        # Handle callable responses
        if callable(raw_text):
            try:
                raw_text = raw_text()
            except Exception:
                pass
        
        # Convert to string
        if isinstance(raw_text, list):
            raw_text = "\n".join(str(part) for part in raw_text)
        
        cleaned = str(raw_text).strip()
        
        # Remove code fences
        if cleaned.startswith("```"):
            cleaned = cleaned[3:-3].strip()
        if cleaned.lower().startswith("json\n"):
            cleaned = cleaned[5:]
        
        # Extract JSON object
        if "{" in cleaned and "}" in cleaned:
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first != -1 and last != -1 and last > first:
                cleaned = cleaned[first:last+1]
        
        return cleaned
    
    def _normalize_extraction_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extraction data to consistent format.

        Accepts both v9 (singular `room` dict) and v10+ (plural `rooms` list)
        schemas. Transition shim: when only the legacy `room`/`room_type`
        fields are present, synthesises a single-entry `rooms` list so
        downstream consumers can branch on shape. Remove the legacy branch
        in a follow-up cleanup ticket once v10 is stable in production.
        """
        # Ensure required keys exist
        for key in ("amenities", "service", "location"):
            if key not in data or not isinstance(data[key], list):
                data[key] = []

        # Normalize room structure (legacy singular dict)
        room_entry = self._normalize_room_data(data)
        data["room"] = room_entry

        # v10+ rooms list — accept if already present and well-typed,
        # otherwise synthesise from the legacy singular shape so downstream
        # code paths can iterate uniformly.
        rooms_field = data.get("rooms")
        if isinstance(rooms_field, list):
            data["rooms"] = [r for r in rooms_field if isinstance(r, dict)]
        else:
            if room_entry.get("name"):
                data["rooms"] = [{
                    "room_type": room_entry["name"],
                    "attribution_confidence": 1.0,
                    "overall_sentiment": room_entry.get("sentiment", "neutral"),
                    "features": [],
                }]
            else:
                data["rooms"] = []

        # Clean up legacy fields
        for legacy_key in ("room_types", "room_type", "room_sentiment", "room_evidence"):
            data.pop(legacy_key, None)

        # Ensure facts array exists
        if not isinstance(data.get("facts"), list):
            data["facts"] = self._build_facts_from_data(data)

        return data
    
    def _normalize_room_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract and normalize room data from various formats."""
        # Try legacy top-level fields first
        if isinstance(data.get("room_type"), str):
            return {
                "name": data.get("room_type", ""),
                "sentiment": data.get("room_sentiment", ""),
                "evidence": data.get("room_evidence", ""),
            }
        
        # Try room_types list
        room_types = data.get("room_types", [])
        if isinstance(room_types, list) and room_types:
            first_room = room_types[0]
            if isinstance(first_room, dict):
                return {
                    k: str(v) for k, v in first_room.items() 
                    if k in {"name", "sentiment", "evidence"}
                }
        
        # Default empty room
        return {"name": "", "sentiment": "", "evidence": ""}
    
    def _build_facts_from_data(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build facts array from extracted data."""
        facts = []
        
        # Add room fact
        room = data.get("room", {})
        if isinstance(room, dict) and room.get("sentiment"):
            facts.append({
                "category": "room",
                "name": str(room.get("name", "")),
                "sentiment": str(room.get("sentiment", "")),
                "evidence": str(room.get("evidence", "")),
            })
        
        # Add aspect facts
        for category in ("amenities", "service", "location", "other"):
            items = data.get(category, [])
            if not isinstance(items, list):
                continue
                
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                sentiment = (
                    item.get("sentiment") or 
                    item.get("polarity") or 
                    item.get("opinion") or 
                    item.get("value")
                )
                
                if sentiment:
                    facts.append({
                        "category": category,
                        "name": str(item.get("name", "")),
                        "sentiment": str(sentiment),
                        "evidence": str(item.get("evidence", "")),
                    })
        
        return facts