import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage

from llm_content_generation.services.prompt_manager import PromptManager


@pytest.fixture
def manager():
    """Fixture for a PromptManager instance."""
    with patch("llm_content_generation.services.prompt_manager.get_logger"):
        yield PromptManager()


class TestPromptManager:
    """Tests for the PromptManager class."""

    @patch("llm_content_generation.services.prompt_manager.lf_prompts.fetch_prompt_from_langfuse")
    def test_get_prompt(self, mock_fetch, manager):
        """Test that get_prompt calls the fetch function."""
        manager.get_prompt("test_prompt", label="test_label")
        mock_fetch.assert_called_once_with("test_prompt", "test_label")

    def test_extract_model_from_config(self, manager):
        """Test extracting model name from prompt metadata."""
        metadata = {"config": {"model": "test_model"}}
        model = manager.extract_model_from_config(metadata)
        assert model == "test_model"

        metadata_json = {"config": '{"model": "json_model"}'}
        model_json = manager.extract_model_from_config(metadata_json)
        assert model_json == "json_model"

        metadata_no_model = {"config": {}}
        assert manager.extract_model_from_config(metadata_no_model) is None

        assert manager.extract_model_from_config({}) is None

    def test_format_json_template(self, manager):
        """Test formatting a JSON template."""
        mock_prompt = MagicMock()
        mock_prompt.compile.return_value = "compiled_string"

        messages = manager.format_json_template(
            mock_prompt, "Hotel California", {"brief": "A lovely place"}
        )

        mock_prompt.compile.assert_called_once_with(
            hotel_name="Hotel California",
            descriptions={"brief": "A lovely place"},
        )
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "compiled_string"

    def test_compile_prompt(self, manager):
        """Test compiling a prompt."""
        mock_prompt = MagicMock()
        mock_prompt.compile.return_value = [{"role": "system", "content": "You are a helpful assistant."}]

        messages = manager.compile_prompt(mock_prompt, {"var": "value"})

        mock_prompt.compile.assert_called_once_with(var="value")
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are a helpful assistant."

    def test_format_chat_prompt(self, manager):
        """Test that format_chat_prompt delegates to compile_prompt."""
        with patch.object(manager, "compile_prompt", return_value=[]) as mock_compile:
            manager.format_chat_prompt(MagicMock(), {})
            mock_compile.assert_called_once()

    def test_convert_to_langchain_messages(self, manager):
        """Test the _convert_to_langchain_messages helper."""
        # Test with string
        messages_str = manager._convert_to_langchain_messages("hello")
        assert isinstance(messages_str[0], HumanMessage)
        assert messages_str[0].content == "hello"

        # Test with list of dicts
        message_dicts = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]
        messages_list = manager._convert_to_langchain_messages(message_dicts)
        assert isinstance(messages_list[0], SystemMessage)
        assert isinstance(messages_list[1], HumanMessage)
        assert messages_list[0].content == "System message"
        assert messages_list[1].content == "User message"
