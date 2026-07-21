from __future__ import annotations

"""Prompt-input formatters for DeepSeek V3.2 narrative generation.

DeepSeek V3.2 (and the V3 lineage generally) processes external context
better as labeled prose than as raw JSON for narrative-generation tasks.
DeepSeek's own injection templates use ``[file content begin] ... [end]``
delimiters and labeled blocks rather than JSON dumps; raw JSON forces
the model to spend tokens de-schematizing keys before it can write.

This module renders three structured prompt variables —
``image_alt_texts`` (Dict), ``property_facilities`` (raw OTA
GraphQL JSON), ``property_highlights`` (raw OTA GraphQL JSON) — into
short labeled strings that fit Langfuse's flat-string variable model
and DeepSeek's preferred input style.

Why: Langfuse ``compile()`` only does flat ``{{var}}`` substitution
(no nested-object access), so any Dict/list ultimately becomes a
string at the template boundary. Pre-formatting in Python keeps the
serialization choice in source control and out of prompt-template
authoring.
"""

from typing import Any, Dict, Iterable, List, Optional


_FACILITY_CAP: int = 16
_HIGHLIGHT_CAP: int = 12
_IMAGE_PER_TYPE_CAP: int = 8


def format_image_alt_texts(
    image_alt_texts: Optional[Dict[str, List[str]]],
) -> str:
    """Render image alt-text dict as labeled-block prose.

    Input shape (from ``image_metadata_loader``)::

        {"room": ["spacious king bed", "city view"],
         "bathroom": ["walk-in shower"]}

    Output::

        Room images: spacious king bed, city view
        Bathroom images: walk-in shower

    None or empty dict returns ``""`` so the Langfuse template can use
    a section block to skip the variable cleanly.

    Args:
        image_alt_texts: Dict[image_type, List[str]] from
            ``load_hotel_image_alt_texts``, or ``None``.

    Returns:
        Multi-line labeled-prose string, or ``""`` when no data.
    """
    if not image_alt_texts:
        return ""

    lines: List[str] = []
    for image_type, alts in image_alt_texts.items():
        if not alts:
            continue
        capped = list(alts)[:_IMAGE_PER_TYPE_CAP]
        label = str(image_type or "image").replace("_", " ").strip().capitalize()
        lines.append(f"{label} images: {', '.join(capped)}")

    return "\n".join(lines)


def format_property_facilities(raw: Any) -> str:
    """Render OTA ``categorizedFacilitiesForAllRooms`` as a flat list.

    The raw structure varies but is most commonly::

        [
            {"roomId": "...",
             "categorizedFacilities": [
                {"category": "Bathroom",
                 "facilities": [{"id": 1, "name": "Bathtub"}, ...]},
                ...]},
            ...
        ]

    A simpler test-fixture variant is ``[{"id": 1, "name": "Wi-Fi"}]``.
    The walker handles both: it descends into any ``categorizedFacilities``
    / ``facilities`` keys it finds and collects every ``name`` string.
    Order is first-seen, deduped case-insensitively, capped at
    ``_FACILITY_CAP`` (16) to keep prompt tokens bounded.

    Args:
        raw: Whatever shape arrived in state (List, Dict, None).

    Returns:
        Comma-separated facility names, or ``""`` when no usable data.
    """
    names = _collect_names(raw)
    if not names:
        return ""
    return ", ".join(names[:_FACILITY_CAP])


def format_property_highlights(raw: Any) -> str:
    """Render OTA ``highlightsForAllRooms`` as a flat list.

    The raw structure is a union of GraphQL types (RDSRoomPrivacyHighlight,
    RDSRoomSizeHighlight, RDSRoomFacilityHighlight, RDSHotelFacilityHighlight)
    each with their own field shape. The walker collects the ``name``
    attribute wherever it appears, deduped case-insensitively and capped at
    ``_HIGHLIGHT_CAP`` (12).

    Args:
        raw: Whatever shape arrived in state (List, Dict, None).

    Returns:
        Comma-separated highlight names, or ``""`` when no usable data.
    """
    names = _collect_names(raw)
    if not names:
        return ""
    return ", ".join(names[:_HIGHLIGHT_CAP])


def _collect_names(obj: Any) -> List[str]:
    """Walk a nested OTA GraphQL structure and collect unique ``name`` values.

    Handles Dict, List, and scalars defensively. Skips internal fields
    (``id``, ``__typename``, ``roomId``, ``category``). Dedups on
    casefolded value, preserves first-seen casing and order.
    """
    seen: set[str] = set()
    out: List[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            if isinstance(name, str):
                stripped = name.strip()
                key = stripped.casefold()
                if stripped and key not in seen:
                    seen.add(key)
                    out.append(stripped)
            for k, v in node.items():
                if k in ("name", "id", "__typename", "roomId"):
                    continue
                visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)
        # scalars: ignored

    visit(obj)
    return out
