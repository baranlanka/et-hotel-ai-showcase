import pytest


class TestLLMConfig:
    """Tests for the LLMConfig dataclass."""

    def test_llm_config_reads_environment(self):
        """Test that LLMConfig reads from current environment variables."""
        from llm_content_generation.core.config import LLMConfig
        config_obj = LLMConfig()
        # These test the actual environment values, not defaults
        assert config_obj.provider in ["lm_studio", "openrouter"]  # Accept either value
        assert isinstance(config_obj.model_name, str)
        assert isinstance(config_obj.max_tokens, int)
        assert isinstance(config_obj.temperature, float)
        assert isinstance(config_obj.request_timeout, int)
        assert isinstance(config_obj.max_retries, int)
        assert isinstance(config_obj.seed, int)
        assert isinstance(config_obj.openrouter_base_url, str)
        assert isinstance(config_obj.openrouter_default_model, str)

    def test_llm_config_with_custom_values(self):
        """Test that LLMConfig can be created with custom values."""
        from llm_content_generation.core.config import LLMConfig

        config_obj = LLMConfig(
            provider="openrouter",
            model_name="test_model",
            max_tokens=1024,
            temperature=0.5,
            request_timeout=60,
            max_retries=5,
            seed=123,
            lm_studio_base_url="http://localhost:1234",
            lm_studio_api_key="test_key",
            openrouter_api_key="openrouter_key",
            openrouter_base_url="https://example.com",
            openrouter_default_model="test/model"
        )

        assert config_obj.provider == "openrouter"
        assert config_obj.model_name == "test_model"
        assert config_obj.max_tokens == 1024
        assert config_obj.temperature == 0.5
        assert config_obj.request_timeout == 60
        assert config_obj.max_retries == 5
        assert config_obj.seed == 123
        assert config_obj.lm_studio_base_url == "http://localhost:1234"
        assert config_obj.lm_studio_api_key == "test_key"
        assert config_obj.openrouter_api_key == "openrouter_key"
        assert config_obj.openrouter_base_url == "https://example.com"
        assert config_obj.openrouter_default_model == "test/model"


class TestLangFuseConfig:
    """Tests for the LangFuseConfig dataclass."""

    def test_langfuse_config_reads_environment(self):
        """Test that LangFuseConfig reads from current environment variables."""
        from llm_content_generation.core.config import LangFuseConfig
        config_obj = LangFuseConfig()
        # These test the actual environment values, not defaults
        assert isinstance(config_obj.enable, bool)
        assert isinstance(config_obj.host, str)
        assert isinstance(config_obj.debug, bool)
        assert isinstance(config_obj.flush_interval, int)
        assert isinstance(config_obj.sample_rate, float)
        assert isinstance(config_obj.session_timeout_minutes, int)
        assert isinstance(config_obj.enable_hierarchical_traces, bool)
        assert isinstance(config_obj.enable_business_context, bool)
        assert isinstance(config_obj.max_retries, int)
        assert isinstance(config_obj.timeout_seconds, int)
        assert isinstance(config_obj.batch_size, int)
        assert isinstance(config_obj.enable_epic1_tracing, bool)
        assert isinstance(config_obj.enable_epic3_tracing, bool)

    def test_langfuse_config_with_custom_values(self):
        """Test that LangFuseConfig can be created with custom values."""
        from llm_content_generation.core.config import LangFuseConfig
        config_obj = LangFuseConfig(
            enable=False,
            secret_key="secret",
            public_key="public",
            host="http://localhost:3000",
            debug=True,
            flush_interval=10,
            sample_rate=0.5,
            session_timeout_minutes=30,
            enable_hierarchical_traces=False,
            enable_business_context=False,
            max_retries=5,
            timeout_seconds=60,
            batch_size=50,
            enable_epic1_tracing=False,
            enable_epic3_tracing=False
        )
        assert config_obj.enable is False
        assert config_obj.secret_key == "secret"
        assert config_obj.public_key == "public"
        assert config_obj.host == "http://localhost:3000"
        assert config_obj.debug is True
        assert config_obj.flush_interval == 10
        assert config_obj.sample_rate == 0.5
        assert config_obj.session_timeout_minutes == 30
        assert config_obj.enable_hierarchical_traces is False
        assert config_obj.enable_business_context is False
        assert config_obj.max_retries == 5
        assert config_obj.timeout_seconds == 60
        assert config_obj.batch_size == 50
        assert config_obj.enable_epic1_tracing is False
        assert config_obj.enable_epic3_tracing is False

    def test_is_trace_enabled_for_operation(self):
        """Test the is_trace_enabled_for_operation method."""
        from llm_content_generation.core.config import LangFuseConfig

        # Test with tracing enabled
        config_obj = LangFuseConfig(enable=True, enable_epic1_tracing=True, enable_epic3_tracing=False)
        assert config_obj.is_trace_enabled_for_operation("epic1") is True
        assert config_obj.is_trace_enabled_for_operation("epic3") is False
        assert config_obj.is_trace_enabled_for_operation("extraction") is True  # maps to epic1

        # Test with tracing disabled
        config_obj = LangFuseConfig(enable=False)
        assert config_obj.is_trace_enabled_for_operation("epic1") is False

    def test_get_debug_info(self):
        """Test the get_debug_info method."""
        from llm_content_generation.core.config import LangFuseConfig

        # Test with explicit credentials (set enable explicitly so the assertion does
        # not depend on the ambient LANGFUSE_TRACING_ENABLED env var, e.g. in CI).
        config_obj = LangFuseConfig(enable=True, secret_key="secret", public_key="public")
        debug_info = config_obj.get_debug_info()
        assert debug_info["enabled"] is True
        assert debug_info["has_credentials"] is True

        # Test without explicit credentials
        config_obj = LangFuseConfig(secret_key=None, public_key=None)
        debug_info = config_obj.get_debug_info()
        assert debug_info["has_credentials"] is False


class TestS3Config:
    """Tests for the S3Config dataclass."""

    def test_s3_config_defaults(self):
        """Test that S3Config uses default values."""
        from llm_content_generation.core.config import S3Config
        config_obj = S3Config()
        assert config_obj.aws_access_key_id is None
        assert config_obj.aws_secret_access_key is None
        assert config_obj.region_name == "us-east-1"

    def test_s3_config_with_custom_values(self):
        """Test that S3Config can be created with custom values."""
        from llm_content_generation.core.config import S3Config
        config_obj = S3Config(
            aws_access_key_id="access_key",
            aws_secret_access_key="secret_key",
            region_name="us-west-2"
        )
        assert config_obj.aws_access_key_id == "access_key"
        assert config_obj.aws_secret_access_key == "secret_key"
        assert config_obj.region_name == "us-west-2"


class TestPipelineConfig:
    """Tests for the PipelineConfig dataclass."""

    def test_pipeline_config_defaults(self):
        """Test that PipelineConfig uses default values."""
        from llm_content_generation.core.config import PipelineConfig
        config_obj = PipelineConfig()
        assert config_obj.max_concurrency == 4
        assert config_obj.export_format == "json"

    def test_pipeline_config_with_custom_values(self):
        """Test that PipelineConfig can be created with custom values."""
        from llm_content_generation.core.config import PipelineConfig
        config_obj = PipelineConfig(max_concurrency=8, export_format="csv")
        assert config_obj.max_concurrency == 8
        assert config_obj.export_format == "csv"


class TestModelMappingConfig:
    """Tests for the ModelMappingConfig dataclass."""

    def test_model_mapping_config_defaults(self):
        """Test that ModelMappingConfig uses default values."""
        from llm_content_generation.core.config import ModelMappingConfig
        config_obj = ModelMappingConfig()
        assert config_obj.extraction == "mistralai/mistral-7b-instruct-v0.3"
        assert config_obj.validation == "mistralai/mistral-7b-instruct-v0.3"
        assert config_obj.descriptions == "google/gemma-2-12b-it"
        assert config_obj.summaries == "mistralai/mistral-7b-instruct-v0.3"
        assert config_obj.default == "mistralai/mistral-7b-instruct-v0.3"

    def test_model_mapping_config_with_custom_values(self):
        """Test that ModelMappingConfig can be created with custom values."""
        from llm_content_generation.core.config import ModelMappingConfig
        config_obj = ModelMappingConfig(
            extraction="model1",
            validation="model2",
            descriptions="model3",
            summaries="model4",
            default="model5"
        )
        assert config_obj.extraction == "model1"
        assert config_obj.validation == "model2"
        assert config_obj.descriptions == "model3"
        assert config_obj.summaries == "model4"
        assert config_obj.default == "model5"

    def test_get_model_for_operation(self):
        """Test the get_model_for_operation method."""
        from llm_content_generation.core.config import ModelMappingConfig
        config_obj = ModelMappingConfig(
            extraction="model1",
            validation="model2",
            descriptions="model3",
            summaries="model4",
            default="model5"
        )
        assert config_obj.get_model_for_operation("extraction") == "model1"
        assert config_obj.get_model_for_operation("validation") == "model2"
        assert config_obj.get_model_for_operation("descriptions") == "model3"
        assert config_obj.get_model_for_operation("summaries") == "model4"
        assert config_obj.get_model_for_operation("unknown_operation") == "model5"


class TestValidationConfig:
    """Tests for the ValidationConfig dataclass."""

    def test_validation_config_defaults(self):
        """Test that ValidationConfig uses default values."""
        from llm_content_generation.core.config import ValidationConfig
        config_obj = ValidationConfig()
        assert config_obj.enabled is True
        assert config_obj.threshold == 0.8
        assert config_obj.max_refines == 1
        assert config_obj.max_attempts == 3
        assert config_obj.validator_prompt == "descriptor_validator "
        assert config_obj.generator_prompt is None
        assert config_obj.provider_override is None
        assert config_obj.model_override is None

    def test_validation_config_with_custom_values(self):
        """Test that ValidationConfig can be created with custom values."""
        from llm_content_generation.core.config import ValidationConfig
        config_obj = ValidationConfig(
            enabled=False,
            threshold=0.9,
            max_refines=2,
            max_attempts=5,
            validator_prompt="test_prompt",
            generator_prompt="gen_prompt",
            provider_override="test_provider",
            model_override="test_model"
        )
        assert config_obj.enabled is False
        assert config_obj.threshold == 0.9
        assert config_obj.max_refines == 2
        assert config_obj.max_attempts == 5
        assert config_obj.validator_prompt == "test_prompt"
        assert config_obj.generator_prompt == "gen_prompt"
        assert config_obj.provider_override == "test_provider"
        assert config_obj.model_override == "test_model"


class TestSettings:
    """Tests for the Settings class."""

    def test_settings_initialization(self):
        """Test that the Settings class initializes correctly."""
        from llm_content_generation.core.config import Settings
        settings = Settings()
        assert hasattr(settings, 'llm')
        assert hasattr(settings, 'models')
        assert hasattr(settings, 'langfuse')
        assert hasattr(settings, 's3')
        assert hasattr(settings, 'pipeline')
        assert hasattr(settings, 'validation')

    def test_settings_to_dict(self):
        """Test the to_dict method of the Settings class."""
        from llm_content_generation.core.config import Settings
        settings = Settings()
        settings_dict = settings.to_dict()
        assert "llm" in settings_dict
        assert "models" in settings_dict
        assert "langfuse" in settings_dict
        assert "s3" in settings_dict
        assert "pipeline" in settings_dict
        assert "validation" in settings_dict
        assert settings_dict["llm"]["provider"] in ["lm_studio", "openrouter"]