# LLM Content Generation — LangGraph pipeline

A LangGraph pipeline that turns raw hotel guest reviews into structured,
grounded marketing content. This is the public **showcase slice**: it runs
end-to-end on synthetic fixtures with a mock/local/hosted model and never
requires a database, object store, or curated prompts.

Two composable graphs are exposed:

- **Aspect extraction** — `review_loader → aspect_extraction → aggregation`.
  Runs one structured-output LLM call per review to pull room features,
  amenities, service/location aspects, and hotel-type signals, then aggregates
  them into a single tabular reference.
- **Content generation** — `preprocessed_data_loader → hotel_type_aggregation →
  room_type_loader → room_descriptions`. Classifies the property type from the
  aggregated signals and writes per-room descriptions grounded strictly in the
  attested guest evidence.

## Key modules

- `core/config.py` — Pydantic-settings model map (provider/model routing,
  Langfuse, prompt-label resolution). Importable with zero required env vars.
- `core/langfuse_prompts.py` — runtime prompt fetch with a built-in **dummy
  fallback** when Langfuse keys are absent, so the pipeline runs offline.
- `core/observability/` — Langfuse tracing config + token-usage helpers.
- `services/llm_factory.py` — multi-model routing factory (Langfuse-wrapped
  OpenAI + LangChain `ChatOpenAI`, per-operation config, OpenRouter / LM Studio
  branches, an expensive-model cost guard).
- `services/prompt_manager.py`, `services/response_parser.py` — prompt
  compile + response parsing.
- `et_langgraph/graph.py` — the `ContentGenerationGraphs` factory.
- `et_langgraph/nodes/` — the pure nodes (`extraction`, `descriptions_rooms`,
  `hotel_type_aggregator`) plus a synthetic `data.py` loader that reads
  `data/synthetic/{hotels,rooms,reviews}.json` instead of a DB / object store.
- `et_langgraph/utils/` — stdlib-only helpers (LLM call wrapper, prompt
  formatters, JSON parsing, in-memory aggregated-data store).

## Observability

The extracted code calls a small observability facade at import time. In this
build that facade is a logging-only shim, so no OpenTelemetry / metrics stack is
required to import or run the package.

## Prompts

Prompt **content** is not part of this repository — the code fetches prompts by
name from Langfuse at runtime, and falls back to a neutral built-in template
when Langfuse credentials are absent. Only the prompt *names* (e.g.
`langchain/hotel_review_analyzer`) appear in the code.

## Model configuration

Provider and per-operation models are env-driven; the defaults are public model
slugs:

- `LLM_MODEL_PROVIDER` = `lm_studio` | `openrouter` | default (OpenAI-style)
- `MODEL_EXTRACTION`, `MODEL_DESCRIPTIONS`, `MODEL_VALIDATION`, … (see
  `core/config.py` `ModelMappingConfig`)
- `LLM_OPENROUTER_API_KEY`, `LLM_OPENROUTER_BASE_URL`,
  `LLM_OPENROUTER_DEFAULT_MODEL` (public slug default)
- `OPENROUTER_REFERER`, `OPENROUTER_APP_TITLE` — neutral attribution headers for
  OpenRouter calls (both default to placeholders)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` — optional; when
  unset the pipeline uses the offline dummy-prompt path.

## Running the graphs

```python
from llm_content_generation.et_langgraph.graph import ContentGenerationGraphs

aspect_graph = ContentGenerationGraphs.create_aspect_extraction_only()
content_graph = ContentGenerationGraphs.create_content_generation_only()
```

Both variants read the synthetic hotels/rooms/reviews fixtures under
`data/synthetic/`. Point `SYNTHETIC_DATA_DIR` at another directory to swap in
your own fictional data.

## Testing

Tests use mock LLM / mock Langfuse and run offline. Run `pytest` from the
package root.
