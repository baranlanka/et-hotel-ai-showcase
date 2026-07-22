from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional

# Temporary override knob for room-prompt A/B testing — set ROOM_PROMPT_LABEL
# (or fall back to PROMPT_LABEL) to pull a non-production label.
# Sourced from PipelineExtraConfig — resolved lazily so tests that
# monkeypatch env vars and reset the settings cache see the override.
def _get_room_prompt_label() -> str:
    from llm_content_generation.core.config import get_pipeline_extra_settings

    return get_pipeline_extra_settings().room_prompt_label


def _get_room_brief_prompt_label() -> str:
    """Label for the SEPARATE room card-blurb prompt
    (``langchain/room_description_brief``). Empty string → brief disabled."""
    from llm_content_generation.core.config import get_pipeline_extra_settings

    return get_pipeline_extra_settings().room_brief_prompt_label


_DESCRIPTION_PROMPT_LABEL = _get_room_prompt_label()
# Dedicated prompt that distils a ~22-word card hook from the full paragraph.
# Two-prompt design: the full paragraph stays on the tuned production room
# prompt (v45), so adding briefs can never regress it.
_BRIEF_PROMPT_NAME = "langchain/room_description_brief"
from llm_content_generation.et_langgraph.state import ContentGenerationState
from llm_content_generation.shared.singletons import (
    get_shared_prompt_manager, get_shared_llm_factory
)
from llm_content_generation.et_langgraph.utils.data_loader import load_aggregated_data
from llm_content_generation.et_langgraph.utils.llm_call_wrapper import (
    invoke_llm_with_validation,
    LLMCallError,
    DEFAULT_TIMEOUT_SLOW,
)
from llm_content_generation.et_langgraph.utils.prompt_formatters import (
    format_property_facilities,
    format_property_highlights,
)
from llm_content_generation.et_langgraph.utils.json_response_parser import (
    strip_code_fence,
)
import json


# Aspect ``room_type`` values that the aspect-extraction LLM uses when it
# cannot classify a review to one of the known room slot names.  These get
# treated as hotel-wide (same as NaN) so per-room prompts still receive
# them.  Compared case-insensitively.
_UNCLASSIFIED_ROOM_TYPE_VALUES = frozenset({"unknown", "n/a", "none", ""})

# Room descriptions are marketing copy. We filter negative-sentiment
# aspects out at the formatter rather than asking the description prompt to
# selectively suppress them — the data layer keeps everything, the marketing
# pipeline gets a positive-skewed view. "mixed" stays in (guest gave both
# sides; the positive quote is still usable). "neutral" stays. Only pure
# "negative" gets dropped.
_DESCRIPTION_DROPPED_SENTIMENTS = frozenset({"negative"})


def _build_per_room_insights(
    aggregated: Any,
    room: str,
    known_room_types: Optional[set[str]] = None,
) -> str:
    """Slice aggregated DataFrame to per-room insights.

    Why: Each room prompt should receive only its own aspects, hotel-wide
    aspects (room_type null/missing/unclassified), and all hotel_signal
    rows.  Passing the full DataFrame pollutes per-room prompts with other
    rooms' signals.  The aspect-extraction LLM frequently emits
    ``"Unknown"`` (literal string, not NaN) when it can't tie an aspect to
    a specific room — those are treated as hotel-wide here.  Aspects with a
    ``room_type`` that isn't among ``known_room_types`` are also treated as
    hotel-wide, since they're effectively unattributable.

    Args:
        aggregated: Pandas DataFrame produced by the aggregation node, or
            any object — non-DataFrame values return an empty string.
        room: Room slot name used as the per-room filter value.
        known_room_types: Set of room slot names the room loader returned.
            Used to identify aspects with an unrecognised ``room_type`` so
            they can still be surfaced as hotel-wide rather than dropped.

    Returns:
        Pretty-printed string representation of the sliced DataFrame, or
        ``str(aggregated)`` as fallback when slicing is impossible.
    """
    try:
        import pandas as pd  # noqa: PLC0415 — lazy import, heavy dep
        if not isinstance(aggregated, pd.DataFrame):
            return str(aggregated) if aggregated is not None else ""

        if "room_type" not in aggregated.columns:
            return aggregated.to_string(index=False)

        rt_series = aggregated["room_type"]
        rt_str_lower = rt_series.astype(str).str.strip().str.lower()
        known_lower = {r.lower() for r in (known_room_types or set())}
        type_series = aggregated["type"] if "type" in aggregated.columns else None

        if type_series is None:
            return aggregated.to_string(index=False)

        mask_aspect_room = (type_series == "aspect") & (rt_series == room)
        mask_aspect_hotel_wide = (
            (type_series == "aspect")
            & (
                rt_series.isna()
                | rt_str_lower.isin(_UNCLASSIFIED_ROOM_TYPE_VALUES)
                | (~rt_str_lower.isin(known_lower) if known_lower else False)
            )
        )
        mask_hotel_signal = type_series == "hotel_signal"

        # room_feature rows are surfaced through the dedicated
        # {{room_features}} prompt variable instead of leaking into the
        # broader insights blob. Keep them out of `insights` so the model
        # sees attested per-room evidence with structure, not as raw rows.
        sliced = aggregated[mask_aspect_room | mask_aspect_hotel_wide | mask_hotel_signal]

        # Sales-tone: strip negative-sentiment aspects from the
        # marketing-prompt view. hotel_signal rows have no aspect sentiment
        # (they're property-type detections), so they pass through. The
        # underlying parquet still carries the full data for non-marketing
        # consumers.
        if "sentiment" in sliced.columns:
            sentiment_lower = sliced["sentiment"].astype(str).str.lower()
            sliced = sliced[
                (type_series.loc[sliced.index] == "hotel_signal")
                | (~sentiment_lower.isin(_DESCRIPTION_DROPPED_SENTIMENTS))
            ]
        return sliced.to_string(index=False)
    except Exception:  # noqa: BLE001 — never crash the pipeline over formatting
        return str(aggregated) if aggregated is not None else ""


_ROOM_FEATURE_MIN_CONFIDENCE = 0.5
_ROOM_FEATURE_MAX_FEATURES = 8
_ROOM_FEATURE_MAX_QUOTES = 3


def _build_visual_block(visual: dict[str, Any] | None) -> str:
    """Render the vision-synthesis bundle as labeled prose for the prompt.

    The vision node writes a structured token bundle to
    ``room_metadata[room_name]["visual"]`` (see
    ``vision/room_synthesis`` LangFuse prompt for the schema). This helper
    inlines those tokens into the description prompt as a single
    ``{{visual_block}}`` variable.

    Returns "" when ``visual`` is falsy, so the template's Mustache section
    block skips cleanly without leaving an empty header.
    """
    if not visual or not isinstance(visual, dict):
        return ""

    def _csv(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v)
        if isinstance(value, str):
            return value
        return ""

    lines: list[str] = []
    register = visual.get("style_register")
    scale = visual.get("scale")
    complexity = visual.get("complexity")
    light_warmth = visual.get("light_warmth")
    register_parts: list[str] = []
    if register:
        register_parts.append(f"style {register}")
    if scale:
        register_parts.append(f"scale {scale}")
    if complexity:
        register_parts.append(f"complexity {complexity}")
    if light_warmth:
        register_parts.append(f"light warmth {light_warmth}")
    if register_parts:
        lines.append(f"register: {', '.join(register_parts)}")

    # comparable_to phrase — register-targeting cue.
    comparable_to = visual.get("comparable_to")
    if comparable_to:
        lines.append(f"comparable to: {comparable_to}")

    # numeric tone vectors.
    tone_vectors = visual.get("tone_vectors")
    if isinstance(tone_vectors, dict) and tone_vectors:
        tv_parts = [f"{k}={v}" for k, v in tone_vectors.items() if v is not None]
        if tv_parts:
            lines.append(f"tone vectors (1-5 scale): {', '.join(tv_parts)}")

    vibe = visual.get("vibe_phrase")
    if vibe:
        lines.append(f"vibe: {vibe}")

    mood_tokens = _csv(visual.get("mood_tokens"))
    if mood_tokens:
        lines.append(f"mood: {mood_tokens}")

    palette_dom = _csv(visual.get("palette_dominant"))
    palette_acc = _csv(visual.get("palette_accent"))
    if palette_dom:
        lines.append(f"palette (dominant): {palette_dom}")
    if palette_acc:
        lines.append(f"palette (accent): {palette_acc}")

    materials = _csv(visual.get("materials"))
    if materials:
        lines.append(f"materials: {materials}")

    light_quality = visual.get("light_quality")
    if light_quality:
        lines.append(f"light quality: {light_quality}")

    design_motifs = _csv(visual.get("design_motifs"))
    if design_motifs:
        lines.append(f"design motifs (recurring): {design_motifs}")

    quirks = _csv(visual.get("quirks"))
    if quirks:
        lines.append(f"quirks (single-photo distinguishing features): {quirks}")

    # room evidence rollup (concrete objects).
    evidence = _csv(visual.get("room_evidence_rollup"))
    if evidence:
        lines.append(f"visible objects (photo-attested inventory): {evidence}")

    vista = visual.get("vista_or_view")
    if vista:
        lines.append(f"vista: {vista}")

    hooks = visual.get("narrative_hooks") or []
    if isinstance(hooks, list) and hooks:
        joined = "; ".join(str(h) for h in hooks if h)
        if joined:
            lines.append(f"photo-grounded hooks: {joined}")

    avoid = visual.get("writing_avoid_list") or []
    if isinstance(avoid, list) and avoid:
        joined = ", ".join(str(a) for a in avoid if a)
        if joined:
            lines.append(
                "DO NOT MENTION (features absent from photos — extends NULL-FACT rule): "
                + joined
            )

    return "\n".join(lines)


def _build_hotel_visual_block(property_visual: dict[str, Any] | None) -> str:
    """Render the hotel-level property_synthesis bundle as labelled prose.

    Mirrors ``_build_visual_block`` but for the hotel-scope schema
    (``common_area_signals``, ``amenity_inventory``, no
    ``room_evidence_rollup``). Empty string when ``property_visual.synthesis``
    is falsy so the hotel-description template's ``{{#property_visual_block}}``
    section gate skips cleanly.

    Accepts EITHER the full state shape (``{"synthesis": {...}, "rooms_rollup": ..., ...}``)
    or a bare synthesis dict.
    """
    if not property_visual or not isinstance(property_visual, dict):
        return ""

    # Unwrap if caller passed the state shape
    synthesis = property_visual.get("synthesis", property_visual)
    if not isinstance(synthesis, dict) or not synthesis:
        return ""

    def _csv(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v)
        if isinstance(value, str):
            return value
        return ""

    lines: list[str] = []

    register = synthesis.get("property_register")
    scale = synthesis.get("scale")
    complexity = synthesis.get("complexity")
    light_warmth = synthesis.get("light_warmth")
    register_parts: list[str] = []
    if register:
        register_parts.append(f"style {register}")
    if scale:
        register_parts.append(f"scale {scale}")
    if complexity:
        register_parts.append(f"complexity {complexity}")
    if light_warmth:
        register_parts.append(f"light warmth {light_warmth}")
    if register_parts:
        lines.append(f"property register: {', '.join(register_parts)}")

    comparable = synthesis.get("comparable_to")
    if comparable:
        lines.append(f"comparable to: {comparable}")

    vibe = synthesis.get("property_vibe_phrase") or synthesis.get("vibe_phrase")
    if vibe:
        lines.append(f"vibe: {vibe}")

    mood = _csv(synthesis.get("mood_tokens"))
    if mood:
        lines.append(f"mood: {mood}")

    palette_dom = _csv(synthesis.get("palette_dominant"))
    palette_acc = _csv(synthesis.get("palette_accent"))
    if palette_dom:
        lines.append(f"palette (dominant): {palette_dom}")
    if palette_acc:
        lines.append(f"palette (accent): {palette_acc}")

    materials = _csv(synthesis.get("materials"))
    if materials:
        lines.append(f"materials: {materials}")

    light_quality = synthesis.get("light_quality")
    if light_quality:
        lines.append(f"light quality: {light_quality}")

    design_motifs = _csv(synthesis.get("design_motifs"))
    if design_motifs:
        lines.append(f"design motifs (recurring): {design_motifs}")

    cas = synthesis.get("common_area_signals") or {}
    if isinstance(cas, dict):
        ext = cas.get("exterior_signature")
        if ext:
            lines.append(f"exterior signature: {ext}")
        lobby = cas.get("lobby_vibe")
        if lobby:
            lines.append(f"lobby vibe: {lobby}")
        dining = cas.get("dining_vibe")
        if dining:
            lines.append(f"dining vibe: {dining}")

    amenity = _csv(synthesis.get("amenity_inventory"))
    if amenity:
        lines.append(
            f"photo-attested amenities (claim only THESE as property amenities): {amenity}"
        )

    quirks = _csv(synthesis.get("property_quirks"))
    if quirks:
        lines.append(f"property quirks (signature elements): {quirks}")

    vista = synthesis.get("property_vista") or synthesis.get("vista_or_view")
    if vista:
        lines.append(f"property vista (the building's setting): {vista}")

    hooks = synthesis.get("narrative_hooks") or []
    if isinstance(hooks, list) and hooks:
        joined = "; ".join(str(h) for h in hooks if h)
        if joined:
            lines.append(f"photo-grounded hooks for marketing copy: {joined}")

    avoid = synthesis.get("writing_avoid_list") or []
    if isinstance(avoid, list) and avoid:
        joined = ", ".join(str(a) for a in avoid if a)
        if joined:
            lines.append(
                "DO NOT MENTION (hotel-scope absences — features the property "
                "does NOT show in photos): " + joined
            )

    return "\n".join(lines)


def _build_per_room_features(
    aggregated: Any,
    room: str,
) -> str:
    """Render attested per-room features as labeled prose for the prompt.

    Why: Room description prompts need experiential per-room evidence
    (which features guests mentioned, what they said, how often) so the
    model can ground "color" in real guest quotes instead of inventing
    atmospheric specifics. The labeled-prose shape mirrors the pattern
    introduced in ``prompt_formatters.py`` (commit 28ada6f) — DeepSeek
    V3.2 grounds significantly better in labeled prose than raw rows.

    Filters: ``type=='room_feature'`` ∧ ``room_type==room`` ∧
    ``attribution_confidence >= 0.5``. Groups by ``feature``, sorts by
    mention count desc, caps at top 8 features and 3 quotes per feature.

    Returns the empty string when there are no attested features for
    this room — prompt template can ``{{#room_features}}`` gate on it.
    """
    if room is None or room == "":
        return ""
    try:
        import pandas as pd  # noqa: PLC0415

        if not isinstance(aggregated, pd.DataFrame):
            return ""
        if aggregated.empty:
            return ""
        for required in ("type", "room_type"):
            if required not in aggregated.columns:
                return ""

        type_series = aggregated["type"].astype(str)
        rt_series = aggregated["room_type"].astype(str)
        mask = (type_series == "room_feature") & (rt_series == room)

        if "attribution_confidence" in aggregated.columns:
            conf = pd.to_numeric(
                aggregated["attribution_confidence"], errors="coerce"
            )
            mask = mask & (conf >= _ROOM_FEATURE_MIN_CONFIDENCE)

        # Sales-tone: room descriptions are marketing copy; we mine
        # only positive/mixed/neutral guest evidence. Negative-only features
        # are dropped here so the prompt never has to "decide" to suppress
        # them. The full data still lives in the parquet for other consumers.
        if "sentiment" in aggregated.columns:
            sent_lower = aggregated["sentiment"].astype(str).str.lower()
            mask = mask & (~sent_lower.isin(_DESCRIPTION_DROPPED_SENTIMENTS))

        sliced = aggregated[mask]
        if sliced.empty:
            return ""

        feature_col = "feature" if "feature" in sliced.columns else "name"

        # Order features by mention count, then alphabetically for stability.
        counts = sliced[feature_col].value_counts()
        top_features = list(counts.head(_ROOM_FEATURE_MAX_FEATURES).index)

        lines: list[str] = ["Room-specific guest observations:"]
        for feat in top_features:
            feat_rows = sliced[sliced[feature_col] == feat]
            count = len(feat_rows)
            # Dominant sentiment is the most-attested one for that feature.
            sentiment_counts = feat_rows["sentiment"].value_counts()
            sentiment = (
                str(sentiment_counts.index[0]) if not sentiment_counts.empty else "neutral"
            )
            quotes_raw = feat_rows["evidence"].dropna().astype(str).tolist()
            quotes: list[str] = []
            for q in quotes_raw:
                q_clean = q.strip().strip('"').strip()
                if q_clean and q_clean not in quotes:
                    quotes.append(q_clean)
                if len(quotes) >= _ROOM_FEATURE_MAX_QUOTES:
                    break
            if not quotes:
                continue
            quote_str = ", ".join(f'"{q}"' for q in quotes)
            mention_word = "mention" if count == 1 else "mentions"
            lines.append(
                f"- {feat} ({sentiment}, ~{count} {mention_word}): {quote_str}"
            )

        if len(lines) == 1:
            return ""
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — never crash the pipeline over formatting
        return ""


# Patterns that DeepSeek V3.2 anchors to in OTA editorial descriptions.
# Stripping them at injection time gives the model less to copy verbatim.
import re as _re  # noqa: E402 — co-located with the helper that uses it

_OTA_LISTY_PATTERNS: tuple[tuple[str, str], ...] = (
    # 1. Drop "The unit/room has N beds" sentences (bed source = bed_summary).
    (r"\b(?:The|This)\s+(?:unit|room|suite|studio)\s+has\s+\d+\s+beds?\.\s*", ""),
    # 2. "Boasting"/"boasting" → "With"/"with"  — V3.2's most-echoed verb.
    (r"\bBoasting\b", "With"),
    (r"\bboasting\b", "with"),
    # 3. "Featuring" sentence-opener → "With" (V3.2 echoes this opener).
    (r"^Featuring\b", "With"),
    (r"(?<=[.!?]\s)Featuring\b", "With"),
    # 4. Bathroom alternation "X and a Y or a Z" → just X.
    (
        r"(\b(?:walk-in shower|shower|bath))\s+and\s+a\s+(?:bath|shower)(?:\s+or\s+a\s+(?:bath|shower))?",
        r"\1",
    ),
    # 5. Surgical: strip 4-item appliance list after "features"/"includes"/
    # "will find".  Replaces with "is equipped" so the sentence stays
    # grammatical.  Pattern matches: "features a X, a Y, Z and a W".
    (
        r"\b(?:features|equipped\s+with|including|includes|will\s+find|featuring)\s+(?:a|an|\d+)?\s*[\w-]+(?:\s+[\w-]+)?,\s+(?:a|an|\d+)?\s*[\w-]+(?:\s+[\w-]+)?,\s+(?:a\s+|an\s+|\d+\s+)?[\w-]+(?:\s+[\w-]+)?\s+and\s+(?:a|an|\d+)?\s*[\w-]+(?:\s+[\w-]+)?",
        "is equipped",
    ),
    # 6. Same pattern but for 3-item lists ending "X, Y, and a Z".
    (
        r"\b(?:features|equipped\s+with|including|includes|will\s+find|featuring)\s+(?:a|an|\d+)?\s*[\w-]+(?:\s+[\w-]+)?,\s+(?:a|an|\d+)?\s*[\w-]+(?:\s+[\w-]+)?\s+and\s+(?:a|an|\d+)?\s*[\w-]+(?:\s+[\w-]+)?",
        "is equipped",
    ),
    # 7. Whitespace cleanup.
    (r"\s{2,}", " "),
    (r"\s+\.", "."),
)


def _strip_appliance_enumeration(text: str) -> str:
    """Strip appliance/fixture enumerations from LLM-generated room copy.

    DeepSeek V3.2 has a deeply baked training prior: any mention of
    ``kitchenette`` triggers a list of typical contents (stovetop,
    refrigerator, microwave, kitchenware...).  Eight iterations of
    increasingly explicit prompt rules — including consequence-framed
    \"output rejected\" rules — failed to suppress it.

    Post-processing is the only reliable fix: detect the enumeration and
    remove it while preserving the ``kitchenette`` word itself.  We match
    the connector words the model uses (``with``, ``that includes``,
    ``featuring``, etc.) plus an appliance keyword, then strip up to the
    sentence end.

    Returns the cleaned text; never crashes on bad input.
    """
    if not text:
        return text
    _APPLIANCE_TRIGGERS = (
        r"(?:stovetop|stove|cooktop|hob|refrigerator|fridge|microwave|"
        r"kitchenware|toaster|kettle|coffee\s+(?:machine|maker)|"
        r"dishwasher|dishes|utensils|oven)"
    )
    # Match: "kitchenette/kitchen + connector + <stuff with appliance>"
    # Replace with just "kitchenette" (preserves the feature mention).
    pattern = (
        r"\b(kitchen(?:ette)?)\b"
        r"(?:\s+(?:that\s+includes|that\s+features|featuring|including|includes|"
        r"with|equipped\s+with|comes\s+with|is\s+equipped\s+with|having|has))"
        r"[^.]*?" + _APPLIANCE_TRIGGERS + r"[^.]*?(?=\.\s|\.$|$)"
    )
    try:
        out = _re.sub(pattern, r"\1", text, flags=_re.IGNORECASE)
        # Clean up doubled spaces / orphaned punctuation.
        out = _re.sub(r"\s{2,}", " ", out)
        out = _re.sub(r"\s+([,.])", r"\1", out)
        return out.strip()
    except Exception:  # noqa: BLE001
        return text


def _normalize_description_for_prompt(description: str) -> str:
    """Strip OTA-template patterns that DeepSeek V3.2 anchors to.

    Conservative transformations applied in order:
    1. Drop "The unit has N beds." sentences (bed source = bed_summary).
    2. Case-preserving boasting → with (V3.2's most-echoed verb).
    3. Capital-F "Featuring" sentence-opener → "With".
    4. Collapse "shower and a bath or a shower" alternations to just X.
    5. Strip 3-4 item appliance/fixture lists after enumerative verbs.

    Each pattern is applied with explicit case sensitivity so "Boasting"
    and "boasting" map to "With" and "with" respectively.  Returns the
    cleaned description; never crashes on bad input.
    """
    if not description:
        return ""
    out = description
    try:
        # 1. Bed-count sentences (case-insensitive — "The"/"This" both fine).
        out = _re.sub(
            r"\b(?:The|This)\s+(?:unit|room|suite|studio)\s+has\s+\d+\s+beds?\.\s*",
            "",
            out,
            flags=_re.IGNORECASE,
        )
        # 2. Case-preserving boasting → with.
        out = _re.sub(r"\bBoasting\b", "With", out)
        out = _re.sub(r"\bboasting\b", "with", out)
        # 3. Capital "Featuring" at sentence-start → "With".
        out = _re.sub(r"\bFeaturing\b", "With", out)
        # 4. Bathroom-alternation collapse.
        out = _re.sub(
            r"(\b(?:walk-in shower|shower|bath))\s+and\s+a\s+(?:bath|shower)(?:\s+or\s+a\s+(?:bath|shower))?",
            r"\1",
            out,
            flags=_re.IGNORECASE,
        )
        # 5. Strip ENTIRE sentences containing specific OTA-template
        # appliance/fixture enumeration keywords.  Surgical and predictable —
        # any sentence mentioning "stovetop", "refrigerator", "microwave",
        # "kitchenware", "toaster", "wardrobe" etc. is part of the templated
        # equipment list that V3.2 reproduces verbatim.  These keywords
        # almost never appear except inside such lists.
        out = _re.sub(
            r"[^.]*?\b(?:stovetop|fridge|refrigerator|microwave|kitchenware|toaster|kettle|wardrobe|coffee\s+machine|coffee\s+maker|flat-screen|cable\s+channels)\b[^.]*?\.\s*",
            "",
            out,
            flags=_re.IGNORECASE,
        )
        # 6. Strip sentences listing rooms ("features 1 living room, 1
        # separate bedroom and 1 bathroom") — keeps the salient feature
        # words in apartmentRooms metadata, not in copy.
        out = _re.sub(
            r"[^.]*?\b\d+\s+(?:living\s+room|separate\s+bedroom|bathroom)\b[^.]*?\.\s*",
            "",
            out,
            flags=_re.IGNORECASE,
        )
        # 7. Whitespace + punctuation cleanup.
        out = _re.sub(r"\s{2,}", " ", out)
        out = _re.sub(r"\s+\.", ".", out)
        out = _re.sub(r"\s+,", ",", out)
    except Exception:  # noqa: BLE001
        return description
    return out.strip()


def _parse_room_output(raw_content: str) -> tuple[str, str]:
    """Parse a room LLM response into ``(full, brief)``.

    Tolerates BOTH prompt shapes so the node is never deploy-coupled to a
    specific prompt version:
      * v46+ (dev): a JSON object ``{"brief": "...", "full": "..."}``.
      * v45 (production): a bare paragraph → ``(paragraph, "")``.

    The appliance-enumeration strip is applied to the FULL paragraph
    only; the brief is a distilled one-liner that never enumerates appliances.
    A legacy/error JSON object (e.g. ``{"error": ...}``) is passed through as
    ``full`` to preserve the pre-existing behaviour.
    """
    cleaned = strip_code_fence(raw_content)
    parsed = None
    if cleaned.startswith('{') and cleaned.endswith('}'):
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = None

    if isinstance(parsed, dict) and ("full" in parsed or "brief" in parsed):
        full = _strip_appliance_enumeration(str(parsed.get("full") or "").strip())
        brief = str(parsed.get("brief") or "").strip()
        return full, brief
    if isinstance(parsed, dict):
        # Legacy/error JSON object — preserve prior pass-through behaviour.
        return cleaned, ""
    # Plain-text output (production v45 single paragraph).
    return _strip_appliance_enumeration(cleaned), ""


async def _generate_room_brief(
    room_type: str,
    full_text: str,
    hotel_id: str,
    brief_llm,
    brief_prompt_obj,
    brief_metadata,
) -> str:
    """Distil a short card blurb from the already-written full paragraph via the
    dedicated ``langchain/room_description_brief`` prompt.

    Two-prompt design (chosen after an A/B showed a single {brief,full} prompt
    measurably flattens the tuned full paragraph): the full paragraph is produced
    by the untouched production room prompt, then this pass summarises it. The
    brief is SUPPLEMENTARY — any failure soft-fails to ``""`` and the renderer
    falls back to the full paragraph, so a brief error never fails the room.
    """
    if brief_llm is None or brief_prompt_obj is None or not full_text:
        return ""
    try:
        prompt_manager = get_shared_prompt_manager()
        formatted_prompt = prompt_manager.compile_prompt(
            brief_prompt_obj,
            {"room_type": room_type, "full_description": full_text},
            brief_metadata,
        )
        config = {
            "metadata": {
                "operation": "room_brief_generation",
                "hotel_id": hotel_id,
                "room_type": room_type,
                "langfuse_tags": ["room-brief"],
                "langfuse_prompt": brief_prompt_obj,
            }
        }
        raw = await invoke_llm_with_validation(
            llm=brief_llm,
            prompt=formatted_prompt,
            config=config,
            timeout_seconds=DEFAULT_TIMEOUT_SLOW,
            min_response_length=5,
            operation_name=f"room_brief:{hotel_id}:{room_type}",
        )
        # The brief prompt returns a bare sentence; strip fences/quotes defensively.
        return strip_code_fence(raw).strip().strip('"').strip()
    except Exception:  # noqa: BLE001 - brief is supplementary; never fail the room
        return ""


async def _process_single_room(
    room_type: str,
    hotel_id: str,
    aggregated_data: str,
    llm,
    prompt_obj,
    metadata,
    room_meta: dict[str, Any] | None = None,
    property_facilities: Any = None,
    property_highlights: Any = None,
    sibling_room_types: str = "",
    room_features: str = "",
    hotel_type: str = "",
    hotel_type_confidence: float = 0.0,
    brief_llm=None,
    brief_prompt_obj=None,
    brief_metadata=None,
) -> tuple[str, str, str]:
    """Process a single room type with full observability.

    Returns ``(room_type, full, brief)``. ``full`` is the listing paragraph from
    the tuned production room prompt (post-processed). ``brief`` is a short card
    blurb distilled from ``full`` by a SEPARATE prompt; it is ``""`` when the
    brief prompt is disabled or soft-fails (renderer falls back to ``full``).
    """

    # @observe decorator handles observability automatically

    # Extract per-room metadata with safe defaults
    # NOTE: `ota_description` is the editorial copy from the OTA; it feeds
    # the prompt as the `description` variable. The local `description` later
    # in this function holds the LLM output and is what we return.
    meta = room_meta or {}
    # Pre-process OTA description to strip the listy patterns
    # DeepSeek V3.2 keeps copying verbatim despite explicit prompt rules.
    ota_description: str = _normalize_description_for_prompt(
        meta.get("description", "") or ""
    )
    room_size_m2: Any = meta.get("roomSizeM2", None)
    bed_summary: str = meta.get("bed_summary", "") or ""
    bathroom_count: Any = meta.get("bathroomCount", None)
    apartment_rooms: Any = meta.get("apartmentRooms", None)
    # Pre-format the vision-synthesis bundle (if present) as a single
    # labeled prose block. Empty when the vision node found no photos or
    # failed — template's Mustache section gate skips the block cleanly.
    visual_block: str = _build_visual_block(meta.get("visual"))

    prompt_manager = get_shared_prompt_manager()
    formatted_prompt = prompt_manager.compile_prompt(
        prompt_obj,
        {
            "hotel_id": hotel_id,
            "room_type": room_type,
            "insights": aggregated_data,
            # Per-room OTA metadata. occupancy_max
            # intentionally omitted — guest counts belong on the structured
            # listing page, not in editorial prose.
            "description": ota_description,
            "roomSizeM2": room_size_m2,
            "bed_summary": bed_summary,
            "bathroomCount": bathroom_count,
            "apartmentRooms": apartment_rooms,
            # Nullable property-level pass-throughs
            "property_facilities": property_facilities,
            "property_highlights": property_highlights,
            # Sibling rooms in this property — lets the model
            # actively avoid producing copy interchangeable with peers.
            "sibling_room_types": sibling_room_types,
            # Attested per-room guest observations (feature + sentiment
            # + verbatim quotes), pre-formatted as labeled prose. Empty
            # string when the review corpus has no attributable observations
            # for this room — template gates with {{#room_features}}.
            "room_features": room_features,
            # Vision-grounded aesthetic synthesis. Empty when absent.
            "visual_block": visual_block,
            # Hotel-type voice shaping. Empty when classifier
            # didn't fire — prompt's confidence-tier rules treat 0.0 / "" as
            # "ignore type, evidence-led only".
            "hotel_type": hotel_type,
            "confidence": hotel_type_confidence,
        },
        metadata,
    )
    
    # Use metadata-only approach - PERFORMANCE OPTIMIZATION
    # Context manager handles observability automatically
    config = {
        "metadata": {
            "operation": "room_description_generation",
            "hotel_id": hotel_id,
            "room_type": room_type,
            "parallel_batch": True,
            "langfuse_tags": ["room-description"],
            "langfuse_prompt": prompt_obj
        }
    }

    # Use robust wrapper with timeout and validation
    try:
        raw_content = await invoke_llm_with_validation(
            llm=llm,
            prompt=formatted_prompt,
            config=config,
            timeout_seconds=DEFAULT_TIMEOUT_SLOW,  # 5 min for slow models
            min_response_length=20,  # Room descriptions should have content
            operation_name=f"room_description:{hotel_id}:{room_type}",
        )
    except LLMCallError as e:
        raise ValueError(f"Room description generation failed for {room_type}: {e}")

    # The production room prompt returns a bare paragraph → (paragraph, "").
    # (_parse_room_output also tolerates a {brief,full} JSON prompt, so the full
    # call is never deploy-coupled to a specific room-prompt version.)
    full, brief = _parse_room_output(raw_content)

    if not full:
        raise ValueError(f"Failed to generate description for room type: {room_type}")

    # Two-prompt design: distil the short card blurb FROM the full paragraph via
    # the dedicated brief prompt. Only when the full call didn't already yield a
    # brief (it won't under the production plain-text prompt). Soft-fails to "".
    if not brief:
        brief = await _generate_room_brief(
            room_type, full, hotel_id, brief_llm, brief_prompt_obj, brief_metadata
        )

    # Room description generated successfully

    return room_type, full, brief


async def room_description_node(state: ContentGenerationState) -> Dict[str, Any]:
    """Generate room descriptions for each room type - parallel processing."""
    # Handle None or empty room_types gracefully
    room_types_str = state.get("room_types")
    if not room_types_str:
        # No room types available - skip room description generation
        return {
            "room_descriptions": {},
            "room_descriptions_brief": {},
            "stats": {
                **state.get("stats", {}),
                "room_descriptions_skipped": True,
                "reason": "no_room_types_available",
            },
        }

    room_types = [rt.strip() for rt in room_types_str.split(",") if rt.strip()]

    if not room_types:
        # Room types string was empty or whitespace-only
        return {
            "room_descriptions": {},
            "room_descriptions_brief": {},
            "stats": {
                **state.get("stats", {}),
                "room_descriptions_skipped": True,
                "reason": "empty_room_types_list",
            },
        }
    
    # Shared resources - fetch once
    prompt_manager = get_shared_prompt_manager()
    llm_factory = get_shared_llm_factory()
    
    # Get prompt config first to extract model
    prompt_obj, metadata = prompt_manager.get_prompt("langchain/room_description", _DESCRIPTION_PROMPT_LABEL)
    
    # Extract model from prompt config and create LLM accordingly
    model_from_prompt = prompt_manager.extract_model_from_config(metadata)
    if model_from_prompt:
        llm = llm_factory.create_from_prompt_config(model_from_prompt)
    else:
        llm = llm_factory.create_for_descriptions()

    # Apply Langfuse prompt-config sampling overrides (temperature, top_p,
    # max_tokens).  Without this, LLMFactory uses global settings defaults
    # regardless of what's set on the prompt in Langfuse — making prompt-side
    # tuning a no-op.  Binds them so the cached LLM instance isn't mutated.
    prompt_cfg = metadata.get("config") if isinstance(metadata, dict) else None
    if isinstance(prompt_cfg, str):
        try:
            prompt_cfg = json.loads(prompt_cfg)
        except Exception:  # noqa: BLE001
            prompt_cfg = None
    if isinstance(prompt_cfg, dict):
        overrides = {
            k: prompt_cfg[k]
            for k in ("temperature", "top_p", "max_tokens")
            if k in prompt_cfg and prompt_cfg[k] is not None
        }
        if overrides:
            llm = llm.bind(**overrides)

    # Brief prompt/LLM (two-prompt design) — fetched once, reused across rooms.
    # Guarded by the label: "" disables the brief. Any fetch/build error leaves
    # brief_llm=None so every room soft-fails its brief to "" without failing.
    brief_llm = brief_prompt_obj = brief_metadata = None
    brief_label = _get_room_brief_prompt_label()
    if brief_label:
        try:
            brief_prompt_obj, brief_metadata = prompt_manager.get_prompt(
                _BRIEF_PROMPT_NAME, brief_label
            )
            brief_model = prompt_manager.extract_model_from_config(brief_metadata)
            brief_llm = (
                llm_factory.create_from_prompt_config(brief_model)
                if brief_model else llm_factory.create_for_descriptions()
            )
            brief_cfg = brief_metadata.get("config") if isinstance(brief_metadata, dict) else None
            if isinstance(brief_cfg, str):
                try:
                    brief_cfg = json.loads(brief_cfg)
                except Exception:  # noqa: BLE001
                    brief_cfg = None
            if isinstance(brief_cfg, dict):
                b_over = {
                    k: brief_cfg[k]
                    for k in ("temperature", "top_p", "max_tokens")
                    if k in brief_cfg and brief_cfg[k] is not None
                }
                if b_over:
                    brief_llm = brief_llm.bind(**b_over)
        except Exception:  # noqa: BLE001 - brief is supplementary; degrade to no-brief
            brief_llm = brief_prompt_obj = brief_metadata = None

    # Load aggregated data on-demand - PERFORMANCE OPTIMIZATION
    data_key = state.get("aggregated_data_key")
    aggregated = await load_aggregated_data(data_key) if data_key else None

    # Nullable property-level pass-throughs
    # Pre-serialize raw OTA JSON to labeled prose for
    # DeepSeek V3.2. Empty/None → "" so Mustache section blocks skip cleanly.
    property_facilities = format_property_facilities(state.get("property_facilities"))
    property_highlights = format_property_highlights(state.get("property_highlights"))

    # Per-room OTA metadata (safe — missing key → empty dict)
    room_metadata: dict[str, Any] = state.get("room_metadata") or {}

    # hotel_type classification (from hotel_type_aggregator_node)
    # feeds the room prompt's TYPE-SENSITIVE EMPHASIS rules. Same gate logic as
    # in descriptions_hotel.py — both fields always present, empty when absent.
    hotel_type_data = state.get("hotel_type_classification", {}) or {}
    hotel_type_primary: str = hotel_type_data.get("primary_type", "") or ""
    hotel_type_confidence: float = float(hotel_type_data.get("confidence", 0.0) or 0.0)

    # Per-room filter needs the full set of known room slots so it can
    # treat unrecognised aspect ``room_type`` values as hotel-wide.
    known_room_types = set(room_types)

    # Execute all room processing concurrently with shared resources.
    # sibling_room_types is computed (not used by v41+) so the
    # prompt variable stays bound for backward-compat with older prompt labels.
    tasks = [
        _process_single_room(
            room_type,
            state["hotel_id"],
            _build_per_room_insights(aggregated, room_type, known_room_types),
            llm,
            prompt_obj,
            metadata,
            room_meta=room_metadata.get(room_type),
            property_facilities=property_facilities,
            property_highlights=property_highlights,
            sibling_room_types=", ".join(rt for rt in room_types if rt != room_type),
            room_features=_build_per_room_features(aggregated, room_type),
            hotel_type=hotel_type_primary,
            hotel_type_confidence=hotel_type_confidence,
            brief_llm=brief_llm,
            brief_prompt_obj=brief_prompt_obj,
            brief_metadata=brief_metadata,
        )
        for room_type in room_types
    ]
    
    results = await asyncio.gather(*tasks)
    # Split the (room_type, full, brief) tuples into two parallel maps.
    # room_descriptions keeps its historical shape {room_type: full_paragraph};
    # room_descriptions_brief is additive and only carries non-empty briefs
    # (empty under production v45, which returns a single paragraph).
    room_descriptions: Dict[str, str] = {}
    room_descriptions_brief: Dict[str, str] = {}
    for room_type, full, brief in results:
        room_descriptions[room_type] = full
        if brief:
            room_descriptions_brief[room_type] = brief

    # Room descriptions batch completed

    return {
        "room_descriptions": room_descriptions,
        "room_descriptions_brief": room_descriptions_brief,
    }