from __future__ import annotations

import json as _json
import re as _re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("llm-content-generation", context="activity")

# ---------------------------------------------------------------------------
# Baseline prompts (manifest S1)
# ---------------------------------------------------------------------------
# Curated production prompts live in Langfuse and are intentionally WITHHELD.
# When Langfuse is disabled/unreachable we load a generic, obviously-untuned
# stand-in from ``prompts/baseline/<sanitized-name>.(json|txt)`` so the "no
# Langfuse" path produces real, runnable output instead of a raw kwargs dump.

# langfuse_prompts.py -> core -> llm_content_generation -> repo root
_BASELINE_DIR = Path(__file__).resolve().parents[2] / "prompts" / "baseline"

# {var} / {{var}} placeholder — substituted with the caller's compile() value.
_PLACEHOLDER = _re.compile(r"\{\{?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}?\}")


def _sanitize_name(name: str) -> str:
    """``langchain/outreach_router`` -> ``outreach_router`` (basename only)."""
    return name.split("/")[-1]


def _subst(text: str, variables: Dict[str, Any]) -> str:
    """Substitute {var}/{{var}} placeholders from ``variables``.

    Unknown placeholders are left intact; inserted VALUES are literal (never
    re-parsed), so a value containing braces cannot inject further templating.
    """
    def _repl(m: "_re.Match[str]") -> str:
        key = m.group(1)
        if key in variables:
            val = variables[key]
            return "" if val is None else str(val)
        return m.group(0)

    return _PLACEHOLDER.sub(_repl, text)


class _BaselinePrompt:
    """A Langfuse-compatible baseline prompt.

    Exposes ``.compile(**vars)`` returning either a chat message list
    (``[{"role", "content"}]``) or a single string — both accepted by
    ``PromptManager._convert_to_langchain_messages``.
    """

    def __init__(self, messages: Optional[List[Dict[str, str]]] = None,
                 text: Optional[str] = None) -> None:
        self._messages = messages
        self._text = text

    def compile(self, **variables: Any) -> Any:
        if self._messages is not None:
            return [
                {"role": m.get("role", "user"),
                 "content": _subst(str(m.get("content", "")), variables)}
                for m in self._messages
            ]
        return _subst(self._text or "", variables)

    # Some Langfuse call-sites probe this; keep it harmless for baselines.
    def get_langchain_prompt(self) -> Any:  # pragma: no cover - compat shim
        return self


def _load_baseline_prompt(name: str, label: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Load ``prompts/baseline/<name>.(json|txt)`` or return None if absent."""
    stem = _sanitize_name(name)
    json_path = _BASELINE_DIR / f"{stem}.json"
    txt_path = _BASELINE_DIR / f"{stem}.txt"

    prompt_obj: Optional[_BaselinePrompt] = None
    if json_path.exists():
        try:
            data = _json.loads(json_path.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if isinstance(data, list):
            prompt_obj = _BaselinePrompt(messages=data)
        elif isinstance(data, dict) and isinstance(data.get("messages"), list):
            prompt_obj = _BaselinePrompt(messages=data["messages"])
    elif txt_path.exists():
        prompt_obj = _BaselinePrompt(text=txt_path.read_text(encoding="utf-8"))

    if prompt_obj is None:
        return None

    metadata: Dict[str, Any] = {
        "langfuse_prompt_name": name,
        "langfuse_prompt_label": label,
        # NOT "fallback": fetch_outreach_prompt treats "fallback" as "prompt
        # unavailable" and returns (None, None). "baseline" means "use me".
        "langfuse_prompt_version": "baseline",
        # No model pinned → callers fall back to create_for_extraction() (the
        # MODEL_BACKEND-selected model, incl. the mock).
        "config": {},
    }
    return prompt_obj, metadata


def get_logger():
    """Return module-level logger via unified manager (backward-compat shim).

    Why: test_langfuse_prompts.py patches this symbol; keeping the name
    avoids changing the test mock path.
    """
    return _obs.logger


@dataclass
class PromptSpec:
    name: str
    label: str = "production"


def _get_langfuse_client():
    # Use the v3-recommended singleton
    from langfuse import get_client  # type: ignore
    return get_client()


@lru_cache(maxsize=64)
def fetch_prompt_from_langfuse(
    name: str,
    label: str = "production",
) -> Tuple[Any, Dict[str, Any]]:
    """Fetch a prompt and return (prompt_obj, metadata).

    Metadata includes name/label/version and may include prompt config
    (e.g., model, variables) for downstream components.
    """
    logger = get_logger()

    def _make_dummy() -> Tuple[Any, Dict[str, Any]]:
        from langchain_core.messages import (
            HumanMessage as _HumanMessage,
        )  # type: ignore

        class _SimpleTemplate:
            def format_messages(self, **kwargs: Any):  # noqa: D401
                import json as _json
                content = _json.dumps(kwargs, ensure_ascii=False)
                return [_HumanMessage(content=content)]

        class _DummyPrompt:
            def get_langchain_prompt(self) -> Any:  # noqa: D401
                return _SimpleTemplate()

        return _DummyPrompt(), {
            "langfuse_prompt_name": name,
            "langfuse_prompt_label": label,
            "langfuse_prompt_version": "fallback",
            "config": {},
        }

    # If LangFuse is disabled or missing keys, prefer a generic baseline prompt
    # (manifest S1) so the "no Langfuse" path produces real output; only when no
    # baseline stand-in exists for this name do we fall back to the dummy.
    if (
        not settings.langfuse.enable
        or not settings.langfuse.public_key
        or not settings.langfuse.secret_key
    ):
        baseline = _load_baseline_prompt(name, label)
        return baseline if baseline is not None else _make_dummy()

    try:
        client = _get_langfuse_client()
        # Prefer chat-type prompt via modern SDK
        prompt: Any
        if not hasattr(client, "get_prompt"):
            raise RuntimeError(
                "LangFuse SDK too old; upgrade to a version with get_prompt"
            )
        try:  # nosec - SDK differences handled via broad except
            prompt = client.get_prompt(  # type: ignore[arg-type]
                name,
                type="chat",
                label=label,
            )
        except Exception:  # nosec - fall back path is safe
            prompt = client.get_prompt(  # type: ignore[attr-defined]
                name=name,
                label=label,
            )

        # Build rich metadata
        metadata: Dict[str, Any] = {
            "langfuse_prompt_name": name,
            "langfuse_prompt_label": label,
            "langfuse_prompt_version": str(
                getattr(prompt, "version", "unknown")
            ),
        }

        # Attach config/variables if available on the prompt object
        try:
            cfg = getattr(prompt, "config", None)
            if cfg is not None:
                metadata["config"] = cfg
        except Exception:
            pass

        # Attempt to surface input variables for convenience
        try:
            if hasattr(prompt, "get_langchain_prompt"):
                lc = prompt.get_langchain_prompt()
                vars_candidate = getattr(lc, "input_variables", None)
                if vars_candidate:
                    metadata.setdefault("config", {})
                    # Store as list under config.variables if not present
                    if (
                        isinstance(metadata["config"], dict)
                        and "variables" not in metadata["config"]
                    ):
                        metadata["config"]["variables"] = list(vars_candidate)
        except Exception:
            pass

        return prompt, metadata
    except Exception as exc:  # pragma: no cover - network path
        logger.error(
            "LangFuse prompt fetch failed",
            extra={
                "error": str(exc),
                "prompt_name": name,
                "prompt_label": label,
            },
        )
        # Fall back instead of raising to allow local runs without LangFuse
        return _make_dummy()
