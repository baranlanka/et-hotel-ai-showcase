import pytest
from unittest.mock import patch

from llm_content_generation.services.llm_factory import LLMConfig, LLMFactory


@pytest.fixture
def mock_settings():
    """Fixture to mock the settings object."""
    with patch("llm_content_generation.services.llm_factory.settings") as mock:
        mock.llm.provider = "test_provider"
        mock.llm.model_name = "test_model"
        mock.llm.temperature = 0.5
        mock.llm.max_tokens = 1024
        mock.llm.request_timeout = 60
        mock.llm.max_retries = 5
        mock.llm.openrouter_base_url = "http://openrouter"
        mock.llm.openrouter_api_key = "openrouter_key"
        mock.llm.lm_studio_base_url = "http://lmstudio"
        mock.llm.lm_studio_api_key = "lmstudio_key"
        mock.models.get_model_for_operation.return_value = "operation_model"
        mock.validation.provider_override = None
        mock.validation.model_override = None
        yield mock


class TestLLMConfig:
    """Tests for the LLMConfig dataclass."""

    def test_from_settings(self, mock_settings):
        """Test creating config from settings."""
        config = LLMConfig.from_settings()
        assert config.provider == "test_provider"
        assert config.model_name == "test_model"

    def test_from_settings_with_operation(self, mock_settings):
        """Test creating config for a specific operation."""
        config = LLMConfig.from_settings(operation="extraction")
        assert config.model_name == "operation_model"

    def test_for_validation(self, mock_settings):
        """Test creating config for validation."""
        mock_settings.validation.provider_override = "validation_provider"
        mock_settings.validation.model_override = "validation_model"
        config = LLMConfig.for_validation()
        assert config.provider == "validation_provider"
        assert config.model_name == "validation_model"

    def test_for_descriptions(self, mock_settings):
        """Test creating config for descriptions."""
        with patch.dict("os.environ", {"DESCRIPTIONS_MAX_TOKENS": "3000"}):
            config = LLMConfig.for_descriptions()
        assert config.max_tokens == 3000


@patch("llm_content_generation.services.llm_factory.get_logger")
@patch("langfuse.openai.OpenAI")
@patch("langchain_openai.ChatOpenAI")
class TestLLMFactory:
    """Tests for the LLMFactory class."""

    @pytest.fixture(autouse=True)
    def _real_backend(self, monkeypatch):
        """Exercise the REAL provider-construction path in these tests.

        The showcase default MODEL_BACKEND=mock returns a deterministic
        MockChatModel (no ChatOpenAI / langfuse.openai construction), which is
        what these tests assert on. Select a passthrough backend so the switch
        does not short-circuit them.
        """
        monkeypatch.setenv("MODEL_BACKEND", "passthrough")

    def test_create_langfuse_client(self, mock_chat_openai, mock_langfuse_openai, mock_get_logger, mock_settings):
        """Test creation of LangFuse-wrapped OpenAI client."""
        factory = LLMFactory()
        client = factory.create_langfuse_client()
        assert client is not None
        mock_langfuse_openai.assert_called_once()

    def test_create_chat_llm(self, mock_chat_openai, mock_langfuse_openai, mock_get_logger, mock_settings):
        """Test creation of LangChain ChatOpenAI instance."""
        factory = LLMFactory()
        llm = factory.create_chat_llm()
        assert llm is not None
        mock_chat_openai.assert_called_once()

    def test_client_caching(self, mock_chat_openai, mock_langfuse_openai, mock_get_logger, mock_settings):
        """Test that LLM clients are cached."""
        factory = LLMFactory()
        client1 = factory.create_langfuse_client()
        client2 = factory.create_langfuse_client()
        assert client1 is client2
        mock_langfuse_openai.assert_called_once()

        llm1 = factory.create_chat_llm()
        llm2 = factory.create_chat_llm()
        assert llm1 is llm2
        mock_chat_openai.assert_called_once()

    def test_create_for_operation(self, mock_chat_openai, mock_langfuse_openai, mock_get_logger, mock_settings):
        """Test creating a client for a specific operation."""
        factory = LLMFactory()
        client = factory.create_for_operation("descriptions")
        # create_for_operation calls create_langfuse_client
        mock_langfuse_openai.assert_called_once()

    def test_create_from_prompt_config(self, mock_chat_openai, mock_langfuse_openai, mock_get_logger, mock_settings):
        """Test creating a client from prompt config."""
        factory = LLMFactory()
        llm = factory.create_from_prompt_config("prompt_model")
        mock_chat_openai.assert_called_once()
        assert mock_chat_openai.call_args[1]["model"] == "prompt_model"

    def test_safe_openrouter_model(self, mock_chat_openai, mock_langfuse_openai, mock_get_logger, mock_settings):
        """Test the safety fallback for expensive OpenRouter models."""
        factory = LLMFactory()

        # Patch the name the production code actually reads: llm_factory
        # imported `settings` at module level, so patching core.config's
        # attribute never reaches the fallback-return path.
        with patch(
            "llm_content_generation.services.llm_factory.settings"
        ) as mock_core_settings:
            mock_core_settings.llm.openrouter_default_model = "default/safe-model"

            # Test blocking an expensive model
            model = factory._get_safe_openrouter_model("anthropic/claude-3-opus")
            assert model == "default/safe-model"

        # Test allowing an expensive model
        with patch.dict("os.environ", {"LLM_ALLOW_EXPENSIVE_MODELS": "true"}):
            model = factory._get_safe_openrouter_model("anthropic/claude-3-opus")
            assert model == "anthropic/claude-3-opus"

