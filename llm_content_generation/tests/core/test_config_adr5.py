"""Comprehensive ADR-5 tests for the pydantic-settings config conversion.

Covers:
- Every settings model: defaults when env unset; env override per field
- Type coercion: ints, floats, bools (including "true"/"True"/"1"/"" edge cases)
- Cached accessor returns same instance; cache reset works
- Stray-call-site fields (DESCRIPTIONS_MAX_TOKENS, VISION_MAX_TOKENS,
  LLM_ALLOW_EXPENSIVE_MODELS, REVIEW_BATCH_SIZE, ET76_* labels)
- Backward-compat: module-level `settings` singleton still importable
- SecurityConfig list properties
- PipelineExtraConfig fallback chain
- Settings.to_dict() structure
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfigDefaults:
    """LLMConfig defaults when no env vars are set."""

    def test_provider_default(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {}, clear=False):
            # Restore the env var to default if it was set
            env = {k: v for k, v in os.environ.items() if k != "LLM_MODEL_PROVIDER"}
            with patch.dict(os.environ, env, clear=True):
                c = LLMConfig()
                assert c.provider == "lm_studio"

    def test_model_name_default(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {}, clear=False):
            env = {k: v for k, v in os.environ.items() if k != "LLM_MODEL_NAME"}
            with patch.dict(os.environ, env, clear=True):
                c = LLMConfig()
                assert c.model_name == "mistralai/mistral-7b-instruct-v0.3"

    def test_max_tokens_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "LLM_MAX_TOKENS"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.max_tokens == 2048

    def test_temperature_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "LLM_TEMPERATURE"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.temperature == 0.1

    def test_request_timeout_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "PIPELINE_REQUEST_TIMEOUT"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.request_timeout == 120

    def test_max_retries_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "LLM_MAX_RETRIES"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.max_retries == 3

    def test_seed_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "LLM_SEED"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.seed == 42

    def test_optional_fields_default_none(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items()
               if k not in ("LLM_LM_STUDIO_BASE_URL", "LLM_LM_STUDIO_API_KEY",
                            "LLM_OPENROUTER_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.lm_studio_base_url is None
            assert c.lm_studio_api_key is None
            assert c.openrouter_api_key is None

    def test_openrouter_url_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "LLM_OPENROUTER_BASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.openrouter_base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_default_model(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "LLM_OPENROUTER_DEFAULT_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.openrouter_default_model == "anthropic/claude-3-haiku"

    def test_descriptions_max_tokens_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "DESCRIPTIONS_MAX_TOKENS"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.descriptions_max_tokens == 4096

    def test_vision_max_tokens_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "VISION_MAX_TOKENS"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.vision_max_tokens == 1100

    def test_allow_expensive_models_default(self):
        from llm_content_generation.core.config import LLMConfig
        env = {k: v for k, v in os.environ.items() if k != "LLM_ALLOW_EXPENSIVE_MODELS"}
        with patch.dict(os.environ, env, clear=True):
            c = LLMConfig()
            assert c.allow_expensive_models is False


class TestLLMConfigEnvOverride:
    """LLMConfig reads env vars (by alias) correctly."""

    def test_provider_from_env(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"LLM_MODEL_PROVIDER": "openrouter"}):
            c = LLMConfig()
            assert c.provider == "openrouter"

    def test_max_tokens_from_env(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"LLM_MAX_TOKENS": "8192"}):
            c = LLMConfig()
            assert c.max_tokens == 8192

    def test_temperature_from_env(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.7"}):
            c = LLMConfig()
            assert c.temperature == 0.7

    def test_descriptions_max_tokens_from_env(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"DESCRIPTIONS_MAX_TOKENS": "3000"}):
            c = LLMConfig()
            assert c.descriptions_max_tokens == 3000

    def test_vision_max_tokens_from_env(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"VISION_MAX_TOKENS": "2048"}):
            c = LLMConfig()
            assert c.vision_max_tokens == 2048

    def test_allow_expensive_models_true_from_env(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"LLM_ALLOW_EXPENSIVE_MODELS": "true"}):
            c = LLMConfig()
            assert c.allow_expensive_models is True

    def test_allow_expensive_models_false_from_env(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"LLM_ALLOW_EXPENSIVE_MODELS": "false"}):
            c = LLMConfig()
            assert c.allow_expensive_models is False

    def test_type_coercion_int_from_string(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"LLM_SEED": "99"}):
            c = LLMConfig()
            assert isinstance(c.seed, int)
            assert c.seed == 99

    def test_type_coercion_float_from_string(self):
        from llm_content_generation.core.config import LLMConfig
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.55"}):
            c = LLMConfig()
            assert isinstance(c.temperature, float)
            assert abs(c.temperature - 0.55) < 1e-9


# ---------------------------------------------------------------------------
# LangFuseConfig — bool parsing edge cases
# ---------------------------------------------------------------------------


class TestLangFuseConfigBoolParsing:
    """Verify bool semantics match original os.getenv(...).lower() == 'true'."""

    def test_enable_true_lowercase(self):
        from llm_content_generation.core.config import LangFuseConfig
        c = LangFuseConfig(enable="true")
        assert c.enable is True

    def test_enable_true_titlecase(self):
        from llm_content_generation.core.config import LangFuseConfig
        c = LangFuseConfig(enable="True")
        assert c.enable is True

    def test_enable_false_string(self):
        from llm_content_generation.core.config import LangFuseConfig
        c = LangFuseConfig(enable="false")
        assert c.enable is False

    def test_enable_empty_string_is_false(self):
        """Original: os.getenv(..., 'true').lower() == 'true'; empty string → False."""
        from llm_content_generation.core.config import LangFuseConfig
        c = LangFuseConfig(enable="")
        assert c.enable is False

    def test_enable_one_string_is_false(self):
        """Original only treated 'true' as True, not '1'."""
        from llm_content_generation.core.config import LangFuseConfig
        c = LangFuseConfig(enable="1")
        assert c.enable is False

    def test_enable_bool_true(self):
        from llm_content_generation.core.config import LangFuseConfig
        c = LangFuseConfig(enable=True)
        assert c.enable is True

    def test_enable_bool_false(self):
        from llm_content_generation.core.config import LangFuseConfig
        c = LangFuseConfig(enable=False)
        assert c.enable is False

    def test_debug_from_env(self):
        from llm_content_generation.core.config import LangFuseConfig
        with patch.dict(os.environ, {"LANGFUSE_DEBUG": "true"}):
            c = LangFuseConfig()
            assert c.debug is True

    def test_debug_default_false(self):
        from llm_content_generation.core.config import LangFuseConfig
        env = {k: v for k, v in os.environ.items() if k != "LANGFUSE_DEBUG"}
        with patch.dict(os.environ, env, clear=True):
            c = LangFuseConfig()
            assert c.debug is False

    def test_sample_rate_coercion(self):
        from llm_content_generation.core.config import LangFuseConfig
        env = {k: v for k, v in os.environ.items() if k != "LANGFUSE_SAMPLE_RATE"}
        with patch.dict(os.environ, env, clear=True):
            c = LangFuseConfig()
            assert isinstance(c.sample_rate, float)
            assert c.sample_rate == 1.0

    def test_flush_interval_int_coercion(self):
        from llm_content_generation.core.config import LangFuseConfig
        with patch.dict(os.environ, {"LANGFUSE_FLUSH_INTERVAL": "5"}):
            c = LangFuseConfig()
            assert isinstance(c.flush_interval, int)
            assert c.flush_interval == 5

    def test_host_default(self):
        from llm_content_generation.core.config import LangFuseConfig
        env = {k: v for k, v in os.environ.items() if k != "LANGFUSE_HOST"}
        with patch.dict(os.environ, env, clear=True):
            c = LangFuseConfig()
            assert c.host == "https://cloud.langfuse.com"

    def test_batch_size_from_env(self):
        from llm_content_generation.core.config import LangFuseConfig
        with patch.dict(os.environ, {"LANGFUSE_BATCH_SIZE": "50"}):
            c = LangFuseConfig()
            assert c.batch_size == 50


# ---------------------------------------------------------------------------
# S3Config
# ---------------------------------------------------------------------------


class TestS3ConfigEnvOverride:
    def test_access_key_from_env(self):
        from llm_content_generation.core.config import S3Config
        with patch.dict(os.environ, {"S3_AWS_ACCESS_KEY_ID": "AKIATEST"}):
            c = S3Config()
            assert c.aws_access_key_id == "AKIATEST"

    def test_region_from_env(self):
        from llm_content_generation.core.config import S3Config
        with patch.dict(os.environ, {"S3_REGION": "eu-west-1"}):
            c = S3Config()
            assert c.region_name == "eu-west-1"

    def test_region_default(self):
        from llm_content_generation.core.config import S3Config
        env = {k: v for k, v in os.environ.items() if k != "S3_REGION"}
        with patch.dict(os.environ, env, clear=True):
            c = S3Config()
            assert c.region_name == "us-east-1"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfigEnvOverride:
    def test_max_concurrency_from_env(self):
        from llm_content_generation.core.config import PipelineConfig
        with patch.dict(os.environ, {"PIPELINE_MAX_CONCURRENCY": "8"}):
            c = PipelineConfig()
            assert c.max_concurrency == 8

    def test_export_format_from_env(self):
        from llm_content_generation.core.config import PipelineConfig
        with patch.dict(os.environ, {"PIPELINE_EXPORT_FORMAT": "csv"}):
            c = PipelineConfig()
            assert c.export_format == "csv"


# ---------------------------------------------------------------------------
# ModelMappingConfig
# ---------------------------------------------------------------------------


class TestModelMappingConfigEnvOverride:
    def test_extraction_from_env(self):
        from llm_content_generation.core.config import ModelMappingConfig
        with patch.dict(os.environ, {"MODEL_EXTRACTION": "test/model"}):
            c = ModelMappingConfig()
            assert c.extraction == "test/model"

    def test_descriptions_from_env(self):
        from llm_content_generation.core.config import ModelMappingConfig
        with patch.dict(os.environ, {"MODEL_DESCRIPTIONS": "anthropic/claude-3"}):
            c = ModelMappingConfig()
            assert c.descriptions == "anthropic/claude-3"

    def test_default_model_fallback(self):
        from llm_content_generation.core.config import ModelMappingConfig
        c = ModelMappingConfig(default="my/fallback")
        assert c.get_model_for_operation("nonexistent") == "my/fallback"

    def test_vision_extraction_default(self):
        from llm_content_generation.core.config import ModelMappingConfig
        env = {k: v for k, v in os.environ.items() if k != "MODEL_VISION_EXTRACTION"}
        with patch.dict(os.environ, env, clear=True):
            c = ModelMappingConfig()
            assert c.vision_extraction == "qwen/qwen3-vl-32b-instruct"


# ---------------------------------------------------------------------------
# ValidationConfig
# ---------------------------------------------------------------------------


class TestValidationConfigEnvOverride:
    def test_enabled_from_env_false(self):
        from llm_content_generation.core.config import ValidationConfig
        with patch.dict(os.environ, {"EPIC3_VALIDATION_ENABLED": "false"}):
            c = ValidationConfig()
            assert c.enabled is False

    def test_threshold_from_env(self):
        from llm_content_generation.core.config import ValidationConfig
        with patch.dict(os.environ, {"EPIC3_VALIDATION_THRESHOLD": "0.95"}):
            c = ValidationConfig()
            assert abs(c.threshold - 0.95) < 1e-9

    def test_validator_prompt_default(self):
        from llm_content_generation.core.config import ValidationConfig
        env = {k: v for k, v in os.environ.items() if k != "EPIC3_VALIDATION_PROMPT"}
        with patch.dict(os.environ, env, clear=True):
            c = ValidationConfig()
            assert c.validator_prompt == "descriptor_validator "

    def test_max_attempts_from_env(self):
        from llm_content_generation.core.config import ValidationConfig
        with patch.dict(os.environ, {"EPIC3_VALIDATION_MAX_ATTEMPTS": "5"}):
            c = ValidationConfig()
            assert c.max_attempts == 5

    def test_bool_true_string(self):
        from llm_content_generation.core.config import ValidationConfig
        with patch.dict(os.environ, {"EPIC3_VALIDATION_ENABLED": "true"}):
            c = ValidationConfig()
            assert c.enabled is True

    def test_bool_empty_is_false(self):
        """Matches original .lower() == 'true' semantics."""
        from llm_content_generation.core.config import ValidationConfig
        with patch.dict(os.environ, {"EPIC3_VALIDATION_ENABLED": ""}):
            c = ValidationConfig()
            assert c.enabled is False


# ---------------------------------------------------------------------------
# SecurityConfig list properties
# ---------------------------------------------------------------------------


class TestSecurityConfigListProperties:
    def test_cors_origins_default(self):
        from llm_content_generation.core.config import SecurityConfig
        env = {k: v for k, v in os.environ.items() if k != "CORS_ORIGINS"}
        with patch.dict(os.environ, env, clear=True):
            c = SecurityConfig()
            assert c.cors_origins == ["http://localhost:3000", "http://localhost:8000"]

    def test_cors_origins_from_env(self):
        from llm_content_generation.core.config import SecurityConfig
        with patch.dict(os.environ, {"CORS_ORIGINS": "http://a.com,http://b.com"}):
            c = SecurityConfig()
            assert c.cors_origins == ["http://a.com", "http://b.com"]

    def test_allowed_hosts_default(self):
        from llm_content_generation.core.config import SecurityConfig
        env = {k: v for k, v in os.environ.items() if k != "ALLOWED_HOSTS"}
        with patch.dict(os.environ, env, clear=True):
            c = SecurityConfig()
            assert c.allowed_hosts == ["*"]

    def test_allowed_hosts_from_env(self):
        from llm_content_generation.core.config import SecurityConfig
        with patch.dict(os.environ, {"ALLOWED_HOSTS": "example.com,api.example.com"}):
            c = SecurityConfig()
            assert c.allowed_hosts == ["example.com", "api.example.com"]

    def test_cors_origins_strips_whitespace(self):
        from llm_content_generation.core.config import SecurityConfig
        with patch.dict(os.environ, {"CORS_ORIGINS": " http://a.com , http://b.com "}):
            c = SecurityConfig()
            assert c.cors_origins == ["http://a.com", "http://b.com"]


# ---------------------------------------------------------------------------
# PipelineExtraConfig — REVIEW_BATCH_SIZE + ET76_* fallback chain
# ---------------------------------------------------------------------------


class TestPipelineExtraConfig:
    def test_review_batch_size_default(self):
        from llm_content_generation.core.config import PipelineExtraConfig
        env = {k: v for k, v in os.environ.items() if k != "REVIEW_BATCH_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            c = PipelineExtraConfig()
            assert c.review_batch_size == 10

    def test_review_batch_size_from_env(self):
        from llm_content_generation.core.config import PipelineExtraConfig
        with patch.dict(os.environ, {"REVIEW_BATCH_SIZE": "25"}):
            c = PipelineExtraConfig()
            assert c.review_batch_size == 25

    def test_description_prompt_label_default(self):
        from llm_content_generation.core.config import PipelineExtraConfig
        env = {k: v for k, v in os.environ.items()
               if k not in ("ET76_HOTEL_PROMPT_LABEL", "ET76_PROMPT_LABEL")}
        with patch.dict(os.environ, env, clear=True):
            c = PipelineExtraConfig()
            assert c.description_prompt_label == "production"

    def test_description_prompt_label_from_et76_prompt_label(self):
        from llm_content_generation.core.config import PipelineExtraConfig
        env = {k: v for k, v in os.environ.items()
               if k not in ("ET76_HOTEL_PROMPT_LABEL", "ET76_PROMPT_LABEL")}
        env["ET76_PROMPT_LABEL"] = "staging"
        with patch.dict(os.environ, env, clear=True):
            c = PipelineExtraConfig()
            assert c.description_prompt_label == "staging"

    def test_description_prompt_label_hotel_overrides_base(self):
        """ET76_HOTEL_PROMPT_LABEL takes precedence over ET76_PROMPT_LABEL."""
        from llm_content_generation.core.config import PipelineExtraConfig
        env = {k: v for k, v in os.environ.items()
               if k not in ("ET76_HOTEL_PROMPT_LABEL", "ET76_PROMPT_LABEL")}
        env["ET76_HOTEL_PROMPT_LABEL"] = "hotel-v2"
        env["ET76_PROMPT_LABEL"] = "old-label"
        with patch.dict(os.environ, env, clear=True):
            c = PipelineExtraConfig()
            assert c.description_prompt_label == "hotel-v2"

    def test_int_type_coercion(self):
        from llm_content_generation.core.config import PipelineExtraConfig
        with patch.dict(os.environ, {"REVIEW_BATCH_SIZE": "7"}):
            c = PipelineExtraConfig()
            assert isinstance(c.review_batch_size, int)
            assert c.review_batch_size == 7


# ---------------------------------------------------------------------------
# Cached accessors
# ---------------------------------------------------------------------------


class TestCachedAccessors:
    """get_llm_settings / get_langfuse_settings / get_pipeline_extra_settings."""

    def setup_method(self):
        from llm_content_generation.core.config import reset_llm_settings_cache
        reset_llm_settings_cache()

    def teardown_method(self):
        from llm_content_generation.core.config import reset_llm_settings_cache
        reset_llm_settings_cache()

    def test_get_llm_settings_returns_same_instance(self):
        from llm_content_generation.core.config import get_llm_settings
        a = get_llm_settings()
        b = get_llm_settings()
        assert a is b

    def test_get_langfuse_settings_returns_same_instance(self):
        from llm_content_generation.core.config import get_langfuse_settings
        a = get_langfuse_settings()
        b = get_langfuse_settings()
        assert a is b

    def test_get_pipeline_extra_settings_returns_same_instance(self):
        from llm_content_generation.core.config import get_pipeline_extra_settings
        a = get_pipeline_extra_settings()
        b = get_pipeline_extra_settings()
        assert a is b

    def test_reset_cache_allows_fresh_read(self):
        """After reset, the accessor re-reads env vars."""
        from llm_content_generation.core.config import get_llm_settings, reset_llm_settings_cache
        a = get_llm_settings()
        reset_llm_settings_cache()
        b = get_llm_settings()
        # Different object instances after reset
        assert a is not b

    def test_get_llm_settings_returns_llm_config_instance(self):
        from llm_content_generation.core.config import get_llm_settings, LLMConfig
        s = get_llm_settings()
        assert isinstance(s, LLMConfig)

    def test_get_langfuse_settings_returns_langfuse_config_instance(self):
        from llm_content_generation.core.config import get_langfuse_settings, LangFuseConfig
        s = get_langfuse_settings()
        assert isinstance(s, LangFuseConfig)

    def test_get_pipeline_extra_returns_pipeline_extra_config_instance(self):
        from llm_content_generation.core.config import get_pipeline_extra_settings, PipelineExtraConfig
        s = get_pipeline_extra_settings()
        assert isinstance(s, PipelineExtraConfig)


# ---------------------------------------------------------------------------
# Module-level `settings` singleton backward compatibility
# ---------------------------------------------------------------------------


class TestSettingsSingleton:
    def test_settings_singleton_importable(self):
        from llm_content_generation.core.config import settings, Settings
        assert isinstance(settings, Settings)

    def test_settings_has_all_sub_configs(self):
        from llm_content_generation.core.config import settings
        assert hasattr(settings, "llm")
        assert hasattr(settings, "langfuse")
        assert hasattr(settings, "models")
        assert hasattr(settings, "s3")
        assert hasattr(settings, "pipeline")
        assert hasattr(settings, "validation")
        assert hasattr(settings, "security")
        assert hasattr(settings, "pipeline_extra")

    def test_settings_to_dict_structure(self):
        from llm_content_generation.core.config import settings
        d = settings.to_dict()
        for key in ("llm", "models", "langfuse", "s3", "pipeline", "validation", "security"):
            assert key in d, f"Expected key '{key}' in to_dict()"

    def test_settings_llm_provider_is_string(self):
        from llm_content_generation.core.config import settings
        assert isinstance(settings.llm.provider, str)

    def test_settings_pipeline_extra_accessible(self):
        from llm_content_generation.core.config import settings
        assert isinstance(settings.pipeline_extra.review_batch_size, int)


# ---------------------------------------------------------------------------
# Stray call-site field integration — verify fields read from env
# ---------------------------------------------------------------------------


class TestStrayCallSiteFields:
    """Verify the three stray call-site env vars are now in LLMConfig."""

    def test_descriptions_max_tokens_field_exists(self):
        from llm_content_generation.core.config import LLMConfig
        c = LLMConfig()
        assert hasattr(c, "descriptions_max_tokens")
        assert c.descriptions_max_tokens == 4096  # default

    def test_vision_max_tokens_field_exists(self):
        from llm_content_generation.core.config import LLMConfig
        c = LLMConfig()
        assert hasattr(c, "vision_max_tokens")
        assert c.vision_max_tokens == 1100  # default

    def test_allow_expensive_models_field_exists(self):
        from llm_content_generation.core.config import LLMConfig
        c = LLMConfig()
        assert hasattr(c, "allow_expensive_models")
        assert c.allow_expensive_models is False  # default

    def test_review_batch_size_field_in_pipeline_extra(self):
        from llm_content_generation.core.config import PipelineExtraConfig
        c = PipelineExtraConfig()
        assert hasattr(c, "review_batch_size")
        assert c.review_batch_size == 10

    def test_description_prompt_label_in_pipeline_extra(self):
        from llm_content_generation.core.config import PipelineExtraConfig
        c = PipelineExtraConfig()
        assert hasattr(c, "description_prompt_label")
        assert c.description_prompt_label == "production"


# ---------------------------------------------------------------------------
# ContentGenerationConfig — REVIEW_BATCH_SIZE respected in state.py
# ---------------------------------------------------------------------------


class TestContentGenerationConfigBatchSize:
    """Verify REVIEW_BATCH_SIZE is read from env via PipelineExtraConfig."""

    def test_default_batch_size(self):
        from llm_content_generation.et_langgraph.state import ContentGenerationConfig
        env = {k: v for k, v in os.environ.items() if k != "REVIEW_BATCH_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ContentGenerationConfig()
            assert cfg.batch_size == 10

    def test_batch_size_from_env(self):
        from llm_content_generation.et_langgraph.state import ContentGenerationConfig
        with patch.dict(os.environ, {"REVIEW_BATCH_SIZE": "30"}):
            cfg = ContentGenerationConfig()
            assert cfg.batch_size == 30

    def test_batch_size_type_is_int(self):
        from llm_content_generation.et_langgraph.state import ContentGenerationConfig
        with patch.dict(os.environ, {"REVIEW_BATCH_SIZE": "15"}):
            cfg = ContentGenerationConfig()
            assert isinstance(cfg.batch_size, int)


# ---------------------------------------------------------------------------
# No os.getenv calls in config.py (except the .env loader infra)
# ---------------------------------------------------------------------------


class TestNoOsGetenvInConfigModule:
    """ADR-5 acceptance criterion: zero os.getenv data-class defaults."""

    def test_no_getenv_in_field_defaults(self):
        """Read config.py source and verify no os.getenv in field default lines.

        The only permitted os.getenv call is in the _load_env_from_repo_root
        helper (infrastructure, not a field default).
        """
        import ast
        import inspect
        import llm_content_generation.core.config as cfg_module

        source = inspect.getsource(cfg_module)
        tree = ast.parse(source)

        # Find all Call nodes where func is os.getenv
        getenv_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # os.getenv(...)
                if (isinstance(func, ast.Attribute) and func.attr == "getenv"
                        and isinstance(func.value, ast.Name) and func.value.id == "os"):
                    getenv_calls.append(node.lineno)

        # Only allow getenv inside _load_env_from_repo_root (line < 30 in the
        # original file structure; check by walking FunctionDef nodes instead)
        infra_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_load_env_from_repo_root":
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        infra_lines.add(child.lineno)

        # Every getenv call must be inside the infra helper
        non_infra = [ln for ln in getenv_calls if ln not in infra_lines]
        assert non_infra == [], (
            f"Found os.getenv() calls at lines {non_infra} outside the "
            f"_load_env_from_repo_root infrastructure helper. These must be "
            f"replaced with pydantic-settings Field() declarations per ADR-5."
        )
