"""Centralized prompt management for consistent LangFuse integration."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # type: ignore

from ..core import langfuse_prompts as lf_prompts
from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("llm-content-generation", context="activity")


def get_logger():
    """Return module-level logger via unified manager (backward-compat shim).

    Why: test_prompt_manager.py patches this symbol; preserving the name
    avoids test-side changes.
    """
    return _obs.logger


class PromptManager:
    """Manages prompt retrieval and formatting with LangFuse integration."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def get_prompt(self, name: str, label: str = "production") -> tuple[Any, Dict[str, Any]]:
        """Retrieve prompt from LangFuse.
        
        Args:
            name: Prompt name in LangFuse
            label: Prompt label (e.g., "production")
            
        Returns:
            Tuple of (prompt_object, metadata)
        """
        self.logger.info(
            "Retrieving prompt from LangFuse",
            extra={"prompt_name": name, "prompt_label": label}
        )
        
        return lf_prompts.fetch_prompt_from_langfuse(name, label)
    
    def extract_model_from_config(self, prompt_metadata: Dict[str, Any]) -> Optional[str]:
        """Extract model name from LangFuse prompt config."""
        config = prompt_metadata.get("config", {})
        if isinstance(config, str):
            config = json.loads(config)
        return config.get("model")
    
    def format_json_template(
        self,
        prompt_obj: Any,
        hotel_name: str,
        descriptions: Dict[str, str]
    ) -> List[HumanMessage]:
        """Format template for validation prompts using compile()."""
        variables = {
            "hotel_name": hotel_name,
            "descriptions": descriptions
        }
        
        compiled_messages = prompt_obj.compile(**variables)
        return self._convert_to_langchain_messages(compiled_messages)
    

    def compile_prompt(
        self, 
        prompt_obj: Any, 
        variables: Dict[str, Any],
        prompt_metadata: Optional[Dict[str, Any]] = None
    ) -> List[HumanMessage]:
        """Compile prompt using LangFuse prompt.compile().
        
        Args:
            prompt_obj: Prompt object returned from LangFuse
            variables: Variables for template compilation
            prompt_metadata: Optional metadata (unused)
            
        Returns:
            List of HumanMessage instances ready for LLM invocation
        """
        compiled_messages = prompt_obj.compile(**variables)
        return self._convert_to_langchain_messages(compiled_messages)

    def format_chat_prompt(
        self,
        prompt_obj: Any,
        variables: Dict[str, Any],
        prompt_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[HumanMessage]:
        """Legacy method - delegates to compile_prompt."""
        return self.compile_prompt(prompt_obj, variables, prompt_metadata)
    

    def _convert_to_langchain_messages(self, compiled_messages: Any) -> List[HumanMessage]:
        """Convert LangFuse compiled messages to LangChain format."""
        if isinstance(compiled_messages, str):
            return [HumanMessage(content=compiled_messages)]
        
        # Chat prompt result - convert message dicts to LangChain messages
        messages = []
        for msg in compiled_messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

