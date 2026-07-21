from __future__ import annotations

"""Strip markdown code fences from LLM responses before JSON parsing.

DeepSeek V3.2 — and most chat models — emit JSON wrapped in
``` ```json ... ``` ``` fences even when ``response_format: json_object`` is
set on the API call.  Naive ``json.loads(raw)`` blows up on the leading
backticks, so we strip the fence before parsing.

This is a hot path called from every description generation and every
validation cycle; keep it allocation-light.
"""

import json
import re
from typing import Any, Optional


# Match optional opening fence ``` or ```<lang> on its own line, plus the
# closing ``` fence at end-of-string.  We strip both ends and any
# surrounding whitespace, then hand off to json.loads.
_FENCE_OPEN_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?\s*```\s*$")


def strip_code_fence(raw: str) -> str:
    """Return ``raw`` with surrounding markdown ``` fences removed.

    No-op when the string isn't fenced.  Handles ``` ```json ... ``` ``` and
    plain ``` ``` ... ``` ``` variants.  Preserves inner content as-is.

    Examples:
        >>> strip_code_fence('```json\\n{"a": 1}\\n```')
        '{"a": 1}'
        >>> strip_code_fence('{"a": 1}')
        '{"a": 1}'
        >>> strip_code_fence('  ```\\n{"a":1}\\n```\\n')
        '{"a":1}'
    """
    if not raw:
        return raw
    stripped = raw.strip()
    # Cheap precheck — only run regex when a fence is plausibly present.
    if not stripped.startswith("```"):
        return stripped
    out = _FENCE_OPEN_RE.sub("", stripped)
    out = _FENCE_CLOSE_RE.sub("", out)
    return out.strip()


def parse_llm_json(raw: str) -> Optional[Any]:
    """Parse a possibly-fenced JSON string; return ``None`` on failure.

    Convenience wrapper for the common pattern of ``json.loads`` after
    fence-stripping, used by hotel-description and validator parsers.

    Returns the parsed object on success, ``None`` if the cleaned string
    isn't valid JSON.  Callers that need the exception detail should use
    ``strip_code_fence`` + ``json.loads`` directly.
    """
    cleaned = strip_code_fence(raw)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
