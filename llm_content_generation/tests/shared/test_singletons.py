import pytest
from unittest.mock import patch, MagicMock

from llm_content_generation.shared.singletons import (
    get_shared_response_parser,
    get_shared_llm_factory,
    get_shared_prompt_manager,
    reset_singletons,
)


@pytest.fixture(autouse=True)
def reset_singletons_fixture():
    """Reset singletons before and after each test."""
    reset_singletons()
    yield
    reset_singletons()


class TestSingletons:
    """Tests for the singleton provider functions."""

    @patch("llm_content_generation.services.response_parser.ResponseParser")
    def test_get_shared_response_parser_singleton(self, mock_parser):
        """Test that get_shared_response_parser returns a singleton."""
        parser1 = get_shared_response_parser()
        parser2 = get_shared_response_parser()
        assert parser1 is parser2
        mock_parser.assert_called_once()

    @patch("llm_content_generation.services.llm_factory.LLMFactory")
    def test_get_shared_llm_factory_singleton(self, mock_factory):
        """Test that get_shared_llm_factory returns a singleton."""
        factory1 = get_shared_llm_factory()
        factory2 = get_shared_llm_factory()
        assert factory1 is factory2
        mock_factory.assert_called_once()

    @patch("llm_content_generation.services.prompt_manager.PromptManager")
    def test_get_shared_prompt_manager_singleton(self, mock_manager):
        """Test that get_shared_prompt_manager returns a singleton."""
        manager1 = get_shared_prompt_manager()
        manager2 = get_shared_prompt_manager()
        assert manager1 is manager2
        mock_manager.assert_called_once()

    @patch("llm_content_generation.services.response_parser.ResponseParser")
    def test_reset_singletons(self, mock_parser):
        """Test that reset_singletons allows new instances to be created."""
        # Configure the mock to return a new MagicMock on each call
        mock_parser.side_effect = [MagicMock(), MagicMock()]

        parser1 = get_shared_response_parser()
        reset_singletons()
        parser2 = get_shared_response_parser()

        assert parser1 is not parser2
        assert mock_parser.call_count == 2
