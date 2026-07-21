import pytest
from unittest.mock import patch, MagicMock, call

from llm_content_generation.core import langfuse_prompts
from llm_content_generation.core.langfuse_prompts import (
    PromptSpec,
)


class TestPromptSpec:
    """Tests for the PromptSpec dataclass."""

    def test_prompt_spec_initialization(self):
        """Test that PromptSpec initializes correctly."""
        spec = PromptSpec(name="test_prompt", label="test_label")
        assert spec.name == "test_prompt"
        assert spec.label == "test_label"

    def test_prompt_spec_defaults(self):
        """Test that PromptSpec uses default values."""
        spec = PromptSpec(name="test_prompt")
        assert spec.label == "production"


@patch("langfuse.get_client")
def test_get_langfuse_client(mock_get_client):
    """Test the _get_langfuse_client function."""
    langfuse_prompts._get_langfuse_client()
    mock_get_client.assert_called_once()


class TestFetchPromptFromLangfuse:
    """Tests for the fetch_prompt_from_langfuse function."""

    @pytest.fixture(autouse=True)
    def clear_lru_cache(self):
        """Clear the LRU cache before each test."""
        langfuse_prompts.fetch_prompt_from_langfuse.cache_clear()

    @patch("llm_content_generation.core.langfuse_prompts.settings")
    def test_fetch_prompt_langfuse_disabled(self, mock_settings):
        """Test that a dummy prompt is returned when LangFuse is disabled."""
        mock_settings.langfuse.enable = False
        prompt, metadata = langfuse_prompts.fetch_prompt_from_langfuse("test_prompt")
        assert metadata["langfuse_prompt_version"] == "fallback"
        assert "get_langchain_prompt" in dir(prompt)

    @patch("llm_content_generation.core.langfuse_prompts.settings")
    def test_fetch_prompt_no_keys(self, mock_settings):
        """Test that a dummy prompt is returned when LangFuse keys are missing."""
        mock_settings.langfuse.enable = True
        mock_settings.langfuse.public_key = None
        prompt, metadata = langfuse_prompts.fetch_prompt_from_langfuse("test_prompt")
        assert metadata["langfuse_prompt_version"] == "fallback"

    @patch("llm_content_generation.core.langfuse_prompts._get_langfuse_client")
    @patch("llm_content_generation.core.langfuse_prompts.settings")
    def test_fetch_prompt_success_modern_sdk(self, mock_settings, mock_get_client):
        """Test successful prompt fetching with a modern LangFuse SDK."""
        mock_settings.langfuse.enable = True
        mock_settings.langfuse.public_key = "public"
        mock_settings.langfuse.secret_key = "secret"

        mock_prompt = MagicMock()
        mock_prompt.version = "1.0"
        mock_prompt.config = {"model": "gpt-4"}
        mock_prompt.get_langchain_prompt.return_value.input_variables = ["var1"]

        mock_client = MagicMock()
        mock_client.get_prompt.return_value = mock_prompt
        mock_get_client.return_value = mock_client

        prompt, metadata = langfuse_prompts.fetch_prompt_from_langfuse("test_prompt")

        assert prompt == mock_prompt
        assert metadata["langfuse_prompt_name"] == "test_prompt"
        assert metadata["langfuse_prompt_version"] == "1.0"
        assert metadata["config"]["model"] == "gpt-4"
        assert metadata["config"]["variables"] == ["var1"]
        mock_client.get_prompt.assert_called_with(
            "test_prompt", type="chat", label="production"
        )

    @patch("llm_content_generation.core.langfuse_prompts._get_langfuse_client")
    @patch("llm_content_generation.core.langfuse_prompts.settings")
    def test_fetch_prompt_success_old_sdk(self, mock_settings, mock_get_client):
        """Test successful prompt fetching with an older LangFuse SDK."""
        mock_settings.langfuse.enable = True
        mock_settings.langfuse.public_key = "public"
        mock_settings.langfuse.secret_key = "secret"

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.version = "2.0"
        mock_prompt_obj.config = None
        # The old prompt object does not have get_langchain_prompt
        del mock_prompt_obj.get_langchain_prompt

        mock_client = MagicMock()
        mock_client.get_prompt.side_effect = [
            TypeError("type not supported"),
            mock_prompt_obj,
        ]
        mock_get_client.return_value = mock_client

        prompt, metadata = langfuse_prompts.fetch_prompt_from_langfuse("test_prompt_old")

        assert prompt == mock_prompt_obj
        assert metadata["langfuse_prompt_version"] == "2.0"
        # Ensure the calls were made as expected
        expected_calls = [
            call("test_prompt_old", type="chat", label="production"),
            call(name="test_prompt_old", label="production"),
        ]
        mock_client.get_prompt.assert_has_calls(expected_calls)


    @patch("llm_content_generation.core.langfuse_prompts.get_logger")
    @patch("llm_content_generation.core.langfuse_prompts._get_langfuse_client")
    @patch("llm_content_generation.core.langfuse_prompts.settings")
    def test_fetch_prompt_failure(self, mock_settings, mock_get_client, mock_get_logger):
        """Test that a dummy prompt is returned on fetch failure."""
        mock_settings.langfuse.enable = True
        mock_settings.langfuse.public_key = "public"
        mock_settings.langfuse.secret_key = "secret"

        mock_client = MagicMock()
        mock_client.get_prompt.side_effect = Exception("Fetch failed")
        mock_get_client.return_value = mock_client

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        prompt, metadata = langfuse_prompts.fetch_prompt_from_langfuse("test_prompt")

        assert metadata["langfuse_prompt_version"] == "fallback"
        mock_logger.error.assert_called_once_with(
            "LangFuse prompt fetch failed",
            extra={
                "error": "Fetch failed",
                "prompt_name": "test_prompt",
                "prompt_label": "production",
            },
        )

    @patch("llm_content_generation.core.langfuse_prompts._get_langfuse_client")
    @patch("llm_content_generation.core.langfuse_prompts.settings")
    def test_fetch_prompt_caching(self, mock_settings, mock_get_client):
        """Test that the LRU cache is working."""
        mock_settings.langfuse.enable = True
        mock_settings.langfuse.public_key = "public"
        mock_settings.langfuse.secret_key = "secret"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        langfuse_prompts.fetch_prompt_from_langfuse("cached_prompt")
        langfuse_prompts.fetch_prompt_from_langfuse("cached_prompt")

        mock_client.get_prompt.assert_called_once_with(
            "cached_prompt", type="chat", label="production"
        )
