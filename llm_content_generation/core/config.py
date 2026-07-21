from __future__ import annotations

import os
import functools
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is loaded BEFORE reading any env vars in this module.
try:
    from dotenv import load_dotenv  # type: ignore

    def _load_env_from_repo_root() -> None:
        # Resolve repository root path
        base_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
        candidates = [
            os.getenv("ENV_FILE"),
            os.path.join(repo_root, ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                load_dotenv(path, override=False)
                break

    _load_env_from_repo_root()
except Exception:
    # Safe fallback if python-dotenv isn't available; rely on process env
    pass


class LLMConfig(BaseSettings):
    """Pydantic-settings model for LLM provider configuration.

    Env var names are preserved exactly from the original os.getenv dataclass
    so deployments require no changes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: str = Field(default="lm_studio", alias="LLM_MODEL_PROVIDER")
    model_name: str = Field(
        default="mistralai/mistral-7b-instruct-v0.3", alias="LLM_MODEL_NAME"
    )
    max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    request_timeout: int = Field(default=120, alias="PIPELINE_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    seed: int = Field(default=42, alias="LLM_SEED")

    # LM Studio
    lm_studio_base_url: Optional[str] = Field(default=None, alias="LLM_LM_STUDIO_BASE_URL")
    lm_studio_api_key: Optional[str] = Field(default=None, alias="LLM_LM_STUDIO_API_KEY")

    # OpenRouter
    openrouter_api_key: Optional[str] = Field(default=None, alias="LLM_OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="LLM_OPENROUTER_BASE_URL"
    )
    openrouter_default_model: str = Field(
        default="anthropic/claude-3-haiku", alias="LLM_OPENROUTER_DEFAULT_MODEL"
    )

    # Stray-call-site fields (migrated from llm_factory.py)
    descriptions_max_tokens: int = Field(default=4096, alias="DESCRIPTIONS_MAX_TOKENS")
    vision_max_tokens: int = Field(default=1100, alias="VISION_MAX_TOKENS")
    allow_expensive_models: bool = Field(default=False, alias="LLM_ALLOW_EXPENSIVE_MODELS")


class LangFuseConfig(BaseSettings):
    """Pydantic-settings model for LangFuse configuration.

    Enhanced LangFuse configuration with v3 SDK features and debugging
    capabilities. Env var names are preserved exactly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Core configuration
    enable: bool = Field(default=True, alias="LANGFUSE_TRACING_ENABLED")
    secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # Enhanced debugging and performance settings
    debug: bool = Field(default=False, alias="LANGFUSE_DEBUG")
    flush_interval: int = Field(default=1, alias="LANGFUSE_FLUSH_INTERVAL")
    sample_rate: float = Field(default=1.0, alias="LANGFUSE_SAMPLE_RATE")

    # Session and correlation settings
    session_timeout_minutes: int = Field(default=60, alias="LANGFUSE_SESSION_TIMEOUT_MINUTES")
    enable_hierarchical_traces: bool = Field(
        default=True, alias="LANGFUSE_ENABLE_HIERARCHICAL_TRACES"
    )
    enable_business_context: bool = Field(
        default=True, alias="LANGFUSE_ENABLE_BUSINESS_CONTEXT"
    )

    # Performance and reliability settings
    max_retries: int = Field(default=3, alias="LANGFUSE_MAX_RETRIES")
    timeout_seconds: int = Field(default=30, alias="LANGFUSE_TIMEOUT_SECONDS")
    batch_size: int = Field(default=100, alias="LANGFUSE_BATCH_SIZE")

    # Trace filtering and categorization
    enable_epic1_tracing: bool = Field(default=True, alias="LANGFUSE_ENABLE_EPIC1_TRACING")
    enable_epic3_tracing: bool = Field(default=True, alias="LANGFUSE_ENABLE_EPIC3_TRACING")

    @field_validator("enable", "debug", "enable_hierarchical_traces",
                     "enable_business_context", "enable_epic1_tracing",
                     "enable_epic3_tracing", mode="before")
    @classmethod
    def parse_bool_string(cls, v: Any) -> Any:
        """Accept 'true'/'false'/'1'/'0'/'' with same semantics as the original
        os.getenv(...).lower() == 'true' pattern.

        - 'true' / 'True' / '1' → True
        - 'false' / 'False' / '0' / '' (empty string) → False
        - bool / int pass through as-is for direct construction compatibility.
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            return v.lower() == "true"
        return v

    def is_trace_enabled_for_operation(self, operation: str) -> bool:
        """Check if tracing is enabled for a specific operation type."""
        if not self.enable:
            return False

        operation_mappings = {
            "epic1": self.enable_epic1_tracing,
            "epic3": self.enable_epic3_tracing,
            "extraction": self.enable_epic1_tracing,
            "summarization": self.enable_epic3_tracing,
        }

        return operation_mappings.get(operation, True)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get configuration debug information."""
        return {
            "enabled": self.enable,
            "debug_mode": self.debug,
            "host": self.host,
            "has_credentials": bool(self.secret_key and self.public_key),
            "sample_rate": self.sample_rate,
            "flush_interval": self.flush_interval,
            "hierarchical_traces": self.enable_hierarchical_traces,
            "business_context": self.enable_business_context,
            "epic1_tracing": self.enable_epic1_tracing,
            "epic3_tracing": self.enable_epic3_tracing,
        }


class S3Config(BaseSettings):
    """Pydantic-settings model for S3 storage configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    aws_access_key_id: Optional[str] = Field(default=None, alias="S3_AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        default=None, alias="S3_AWS_SECRET_ACCESS_KEY"
    )
    region_name: Optional[str] = Field(default="us-east-1", alias="S3_REGION")


class PipelineConfig(BaseSettings):
    """Pydantic-settings model for pipeline configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    max_concurrency: int = Field(default=4, alias="PIPELINE_MAX_CONCURRENCY")
    export_format: str = Field(default="json", alias="PIPELINE_EXPORT_FORMAT")


class ModelMappingConfig(BaseSettings):
    """Pydantic-settings model for operation→model mapping.

    Maps each operation type to its preferred model. This allows you to
    configure all model assignments in one place via environment variables.

    Environment Variables:
        MODEL_EXTRACTION: Model for Epic 1 extraction/processing
        MODEL_VALIDATION: Model for Epic 3 validation
        MODEL_DESCRIPTIONS: Model for Epic 3 hotel descriptions
        MODEL_SUMMARIES: Model for Epic 3 legacy summaries
        MODEL_VISION_EXTRACTION: Model for vision sketch + room synthesis
        MODEL_REVIEWS: Model for enriched-review rewriting (DeepSeek V3.2)
        MODEL_TRANSLATION: Model for multi-language content translation (DeepSeek V3.2)
        MODEL_DEFAULT: Fallback model for any unmapped operations
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Core operation models
    extraction: str = Field(
        default="mistralai/mistral-7b-instruct-v0.3", alias="MODEL_EXTRACTION"
    )
    validation: str = Field(
        default="mistralai/mistral-7b-instruct-v0.3", alias="MODEL_VALIDATION"
    )
    descriptions: str = Field(
        default="google/gemma-2-12b-it", alias="MODEL_DESCRIPTIONS"
    )
    summaries: str = Field(
        default="mistralai/mistral-7b-instruct-v0.3", alias="MODEL_SUMMARIES"
    )
    vision_extraction: str = Field(
        default="qwen/qwen3-vl-32b-instruct", alias="MODEL_VISION_EXTRACTION"
    )
    # Enriched-review rewriting. Pinned to DeepSeek V3.2 (the same prose
    # model the hotel-description prompt uses); NOT the `deepseek-chat` alias,
    # which now points at V4-Flash. The Langfuse prompt config overrides this at
    # runtime when present; this is the operation-mapped fallback.
    reviews: str = Field(
        default="deepseek/deepseek-v3.2", alias="MODEL_REVIEWS"
    )
    # Multi-language content translation. Same DeepSeek V3.2 pin as
    # `reviews` (mechanical, best-effort text transformation), kept as its own
    # operation so cost/latency/timeout tuning stays independent of the reviews
    # rewrite. The Langfuse prompt config overrides this at runtime when present;
    # this is the operation-mapped fallback.
    translation: str = Field(
        default="deepseek/deepseek-v3.2", alias="MODEL_TRANSLATION"
    )
    default: str = Field(
        default="mistralai/mistral-7b-instruct-v0.3", alias="MODEL_DEFAULT"
    )

    def get_model_for_operation(self, operation: str) -> str:
        """Get the configured model for a specific operation type."""
        model_map = {
            "extraction": self.extraction,
            "validation": self.validation,
            "descriptions": self.descriptions,
            "summaries": self.summaries,
            "vision_extraction": self.vision_extraction,
            "reviews": self.reviews,
            "translation": self.translation,
            "epic1_processing": self.extraction,
            "epic3_validation": self.validation,
            "epic3_descriptions": self.descriptions,
            "epic3_summaries": self.summaries,
        }
        return model_map.get(operation, self.default)


class ValidationConfig(BaseSettings):
    """Pydantic-settings model for Epic 3 validation/refinement.

    Controlled via environment variables as documented in the
    validation integration guide.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    enabled: bool = Field(default=True, alias="EPIC3_VALIDATION_ENABLED")
    threshold: float = Field(default=0.8, alias="EPIC3_VALIDATION_THRESHOLD")
    max_refines: int = Field(default=1, alias="EPIC3_VALIDATION_MAX_REFINES")
    # New iterative generator+validator loop attempts (refine via generator feedback)
    max_attempts: int = Field(default=3, alias="EPIC3_VALIDATION_MAX_ATTEMPTS")
    validator_prompt: str = Field(
        default="descriptor_validator ", alias="EPIC3_VALIDATION_PROMPT"
    )
    # Optional enhanced generator prompt name override
    generator_prompt: Optional[str] = Field(default=None, alias="EPIC3_GENERATOR_PROMPT")
    provider_override: Optional[str] = Field(
        default=None, alias="EPIC3_VALIDATION_MODEL_PROVIDER"
    )
    model_override: Optional[str] = Field(
        default=None, alias="EPIC3_VALIDATION_MODEL_NAME"
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def parse_enabled_bool(cls, v: Any) -> Any:
        """Same bool-string semantics as LangFuseConfig."""
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            return v.lower() == "true"
        return v


class SecurityConfig(BaseSettings):
    """Pydantic-settings model for security/CORS configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        alias="CORS_ORIGINS",
    )
    allowed_hosts_raw: str = Field(default="*", alias="ALLOWED_HOSTS")

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",")]

    @property
    def allowed_hosts(self) -> List[str]:
        return [h.strip() for h in self.allowed_hosts_raw.split(",")]


class PipelineExtraConfig(BaseSettings):
    """Additional pipeline/prompt configuration fields migrated from stray call sites.

    Env var names:
        REVIEW_BATCH_SIZE       — used in et_langgraph/state.py
        ET76_HOTEL_PROMPT_LABEL — used in et_langgraph/nodes/descriptions_hotel.py
        ET76_PROMPT_LABEL       — fallback for ET76_HOTEL_PROMPT_LABEL
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    review_batch_size: int = Field(default=10, alias="REVIEW_BATCH_SIZE")
    # Cost cap on aspect extraction (one LLM call per review). The deep-reviews
    # backfill grew the pool to ~1,000 reviews/hotel; ABSA aspect signal saturates
    # well before hundreds, so the node extracts from the N LONGEST (combined
    # positive+negative) reviews rather than all of them. Full per-review depth is
    # preserved on the ones kept (no batching, no model change). 0 disables the cap.
    # Default 150 = the coverage-vs-N knee (≈99% of aspect coverage, marginal gain
    # in single digits beyond it); tunable. See aspect_extraction_node.
    aspect_max_reviews: int = Field(default=150, alias="ASPECT_MAX_REVIEWS")
    # Both ET76_* fields are read; description_prompt_label resolves the
    # ET76_HOTEL_PROMPT_LABEL → ET76_PROMPT_LABEL → "production" fallback chain.
    et76_hotel_prompt_label: Optional[str] = Field(
        default=None, alias="ET76_HOTEL_PROMPT_LABEL"
    )
    et76_prompt_label: Optional[str] = Field(
        default=None, alias="ET76_PROMPT_LABEL"
    )
    # A/B override knobs for the room-description and extraction
    # prompts; each falls back to ET76_PROMPT_LABEL → "production".
    et76_room_prompt_label: Optional[str] = Field(
        default=None, alias="ET76_ROOM_PROMPT_LABEL"
    )
    et87_extraction_prompt_label: Optional[str] = Field(
        default=None, alias="ET87_EXTRACTION_PROMPT_LABEL"
    )
    # Room card-blurb ("brief") prompt label. The brief is generated by a
    # SEPARATE prompt (langchain/room_description_brief) that distils a short
    # hook from the already-written full paragraph — so the tuned full-paragraph
    # prompt (langchain/room_description) stays untouched. Default "dev" (the
    # only label that currently exists); set "" to disable the brief entirely.
    room_brief_prompt_label_env: Optional[str] = Field(
        default=None, alias="ROOM_BRIEF_PROMPT_LABEL"
    )
    # Computed in model_validator; not direct env vars.
    description_prompt_label: str = "production"
    room_prompt_label: str = "production"
    extraction_prompt_label: str = "production"
    room_brief_prompt_label: str = "production"

    @model_validator(mode="after")
    def _resolve_prompt_label(self) -> "PipelineExtraConfig":
        """Resolve the specific-label → ET76_PROMPT_LABEL → "production" chains."""
        self.description_prompt_label = (
            self.et76_hotel_prompt_label
            or self.et76_prompt_label
            or "production"
        )
        self.room_prompt_label = (
            self.et76_room_prompt_label
            or self.et76_prompt_label
            or "production"
        )
        self.extraction_prompt_label = (
            self.et87_extraction_prompt_label
            or self.et76_prompt_label
            or "production"
        )
        # Brief label is independent of the ET76 chain (it's a net-new prompt);
        # an explicit env value wins, else default to "dev". Empty string ("")
        # is honoured as "brief disabled".
        if self.room_brief_prompt_label_env is not None:
            self.room_brief_prompt_label = self.room_brief_prompt_label_env
        return self


class Settings(BaseSettings):
    """Composite settings for the llm_content_generation package.

    All sub-settings are nested instances. The package remains importable
    without the 15 env vars required by app/core/config.py Settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    models: ModelMappingConfig = Field(default_factory=ModelMappingConfig)
    langfuse: LangFuseConfig = Field(default_factory=LangFuseConfig)
    s3: S3Config = Field(default_factory=S3Config)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    pipeline_extra: PipelineExtraConfig = Field(default_factory=PipelineExtraConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.model_dump(),
            "models": self.models.model_dump(),
            "langfuse": self.langfuse.model_dump(),
            "s3": self.s3.model_dump(),
            "pipeline": self.pipeline.model_dump(),
            "validation": self.validation.model_dump(),
            "security": {
                "cors_origins": self.security.cors_origins,
                "allowed_hosts": self.security.allowed_hosts,
            },
        }


# ---------------------------------------------------------------------------
# Cached accessor — mirrors app/core/config.py get_settings() pattern.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_llm_settings() -> LLMConfig:
    """Return cached LLMConfig singleton.

    Note on monkeypatching: pydantic-settings reads env vars at construction
    time. Tests that use ``patch.dict(os.environ, ...)`` around a specific
    call must call ``reset_llm_settings_cache()`` afterward (or before) to
    invalidate the cache so the next call re-reads the updated environment.
    Tests that patch at module import time (conftest.py os.environ.update)
    are unaffected because the cache is populated after the env is set.
    """
    return LLMConfig()


@functools.lru_cache(maxsize=1)
def get_langfuse_settings() -> LangFuseConfig:
    """Return cached LangFuseConfig singleton."""
    return LangFuseConfig()


@functools.lru_cache(maxsize=1)
def get_pipeline_extra_settings() -> PipelineExtraConfig:
    """Return cached PipelineExtraConfig singleton.

    REVIEW_BATCH_SIZE, ET76_HOTEL_PROMPT_LABEL and ET76_PROMPT_LABEL are
    read here. Call ``reset_llm_settings_cache()`` to invalidate if tests
    monkeypatch these vars.
    """
    return PipelineExtraConfig()


def reset_llm_settings_cache() -> None:
    """Clear all lru_cache singletons for this module.

    Use in tests that monkeypatch env vars for settings covered by the
    cached accessors:
        - DESCRIPTIONS_MAX_TOKENS, VISION_MAX_TOKENS, LLM_ALLOW_EXPENSIVE_MODELS
          → get_llm_settings()
        - REVIEW_BATCH_SIZE, ET76_*_PROMPT_LABEL → get_pipeline_extra_settings()
        - LANGFUSE_* → get_langfuse_settings()

    Example::

        with patch.dict(os.environ, {"REVIEW_BATCH_SIZE": "20"}):
            reset_llm_settings_cache()
            cfg = get_pipeline_extra_settings()
            assert cfg.review_batch_size == 20
            reset_llm_settings_cache()  # clean up for next test
    """
    get_llm_settings.cache_clear()
    get_langfuse_settings.cache_clear()
    get_pipeline_extra_settings.cache_clear()


# ---------------------------------------------------------------------------
# Backward-compatibility: module-level singleton (existing code imports
# `from llm_content_generation.core.config import settings`).
# The singleton is constructed once at import time (same as the old dataclass
# approach) so the module surface is unchanged.
# ---------------------------------------------------------------------------

settings = Settings()
