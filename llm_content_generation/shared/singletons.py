"""Centralized singleton instances for memory optimization."""

from __future__ import annotations

from typing import Optional, Any, Dict

# Global singleton instances
# ARCHIVED: _shared_chain_factory removed in the chainless migration
_shared_response_parser: Optional[object] = None
_shared_validator: Optional[object] = None
_shared_llm_factory: Optional[object] = None
_shared_prompt_manager: Optional[object] = None


# ARCHIVED: Legacy chain factory removed in the chainless migration
# All chain functionality moved to direct component usage in LangGraph nodes


# ARCHIVED: Removed in the chainless migration
# def get_shared_chain_factory(): ...


def get_shared_response_parser():
	"""Get shared ResponseParser instance."""
	global _shared_response_parser
	if _shared_response_parser is None:
		from ..services.response_parser import ResponseParser
		_shared_response_parser = ResponseParser()
	return _shared_response_parser


# ARCHIVED: Legacy validation engine removed with pipeline cleanup


def get_shared_llm_factory():
	"""Get shared LLMFactory instance."""
	global _shared_llm_factory
	if _shared_llm_factory is None:
		from ..services.llm_factory import LLMFactory
		_shared_llm_factory = LLMFactory()
	return _shared_llm_factory


def get_shared_prompt_manager():
	"""Get shared PromptManager instance."""
	global _shared_prompt_manager
	if _shared_prompt_manager is None:
		from ..services.prompt_manager import PromptManager
		_shared_prompt_manager = PromptManager()
	return _shared_prompt_manager


# ARCHIVED: Removed in the chainless migration
# def clear_chain_cache(): ...


# ARCHIVED: Removed in the chainless migration
# def clear_chain_cache_for_operation(operation: str): ...


def reset_singletons():
	"""Reset all singleton instances. Useful for testing."""
	global _shared_response_parser, _shared_validator
	global _shared_llm_factory, _shared_prompt_manager
	_shared_response_parser = None
	_shared_validator = None
	_shared_llm_factory = None
	_shared_prompt_manager = None