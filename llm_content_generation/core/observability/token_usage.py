from __future__ import annotations

from typing import Any, Dict, Optional


def _as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return None


def _map_usage_dict(d: Dict[str, Any]) -> Optional[Dict[str, int]]:
    input_tokens = (
        d.get("input_tokens") or d.get("prompt_tokens") or d.get("input")
    )
    output_tokens = (
        d.get("output_tokens") or d.get("completion_tokens") or d.get("output")
    )
    total_tokens = d.get("total_tokens") or d.get("total")

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    result: Dict[str, int] = {
        "input": int(input_tokens or 0),
        "output": int(output_tokens or 0),
    }
    if total_tokens is not None:
        result["total"] = int(total_tokens or 0)
    return result


def extract_token_usage(
    result: Any, model_name: Optional[str] = None
) -> Optional[Dict[str, int]]:
    """General-purpose token extraction for Llama, DeepSeek, Qwen, etc.

    One-pass, no exceptions, best-effort mapping to
    {"input": int, "output": int, "total": int?}.
    """
    candidates: list[Dict[str, Any]] = []

    # usage
    usage = _as_dict(getattr(result, "usage", None))
    if usage:
        candidates.append(usage)

    # usage_metadata
    usage_meta = _as_dict(getattr(result, "usage_metadata", None))
    if usage_meta:
        candidates.append(usage_meta)

    # response_metadata.{usage|token_usage}
    resp_meta = getattr(result, "response_metadata", None)
    if isinstance(resp_meta, dict):
        u = resp_meta.get("usage")
        if isinstance(u, dict):
            candidates.append(u)
        tu = resp_meta.get("token_usage")
        if isinstance(tu, dict):
            candidates.append(tu)

    # additional_kwargs.usage
    add = getattr(result, "additional_kwargs", None)
    if isinstance(add, dict):
        au = add.get("usage")
        if isinstance(au, dict):
            candidates.append(au)

    # direct attributes on result
    direct = {
        "input_tokens": getattr(result, "input_tokens", None),
        "output_tokens": getattr(result, "output_tokens", None),
        "total_tokens": getattr(result, "total_tokens", None),
        "prompt_tokens": getattr(result, "prompt_tokens", None),
        "completion_tokens": getattr(result, "completion_tokens", None),
    }
    if any(v is not None for v in direct.values()):
        candidates.append(direct)

    for d in candidates:
        mapped = _map_usage_dict(d)
        if mapped:
            return mapped

    return None


