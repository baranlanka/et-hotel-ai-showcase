"""Centralized LLM factory with optimized LangFuse tracing integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Dict

from ..core.config import settings
from app.core.observability.factory import ObservabilityFactory


def _model_backend() -> str:
    """Resolve the MODEL_BACKEND switch (manifest S3).

    ``mock`` (the default, incl. when unset) → a deterministic, offline
    ``MockChatModel`` — no network, no API key, CI-safe. ``ollama`` → a local
    OpenAI-compatible endpoint. ``openrouter`` → the real hosted path.
    """
    return os.getenv("MODEL_BACKEND", "mock").strip().lower() or "mock"

# Lazy: creating the unified manager touches app.core.config (15 required env
# vars via RotatingLoggingManager settings). This package must stay importable
# without the app environment (ADR-5), so the manager is built on first use.
_obs = None


def _get_obs():
    global _obs
    if _obs is None:
        _obs = ObservabilityFactory.create_unified(
            "llm-content-generation", context="activity"
        )
    return _obs


def get_logger():
    """Return module-level logger via unified manager (backward-compat shim).

    Why: test_llm_factory.py patches this symbol at module path; keeping the
    name preserves existing mock targets without test changes.
    """
    return _get_obs().logger


@dataclass
class LLMConfig:
    """Configuration for LLM instances."""
    
    provider: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 120
    max_retries: int = 3
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    @classmethod
    def from_settings(cls, operation: Optional[str] = None) -> LLMConfig:
        """Create config from application settings.
        
        Args:
            operation: Optional operation type to get specific model for
                     (e.g., "extraction", "validation", "descriptions")
        """
        # Get model name based on operation or use default
        model_name = settings.llm.model_name  # fallback
        if operation:
            model_name = settings.models.get_model_for_operation(operation)
        
        # Select appropriate base_url and api_key based on provider
        if settings.llm.provider == "openrouter":
            base_url = settings.llm.openrouter_base_url
            api_key = settings.llm.openrouter_api_key
        else:  # lm_studio or default
            base_url = settings.llm.lm_studio_base_url
            api_key = settings.llm.lm_studio_api_key
        
        return cls(
            provider=settings.llm.provider,
            model_name=model_name,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.request_timeout,
            max_retries=settings.llm.max_retries,
            base_url=base_url,
            api_key=api_key,
        )
    
    @classmethod
    def for_validation(cls) -> LLMConfig:
        """Create config with validation-specific model."""
        config = cls.from_settings("validation")
        # Allow legacy override settings
        if settings.validation.provider_override:
            config.provider = settings.validation.provider_override
        if settings.validation.model_override:
            config.model_name = settings.validation.model_override
        return config
    
    @classmethod
    def for_extraction(cls) -> LLMConfig:
        """Create config specifically for Epic 1 extraction/processing."""
        return cls.from_settings("extraction")
    
    @classmethod
    def for_descriptions(cls) -> LLMConfig:
        """Create config specifically for Epic 3 hotel descriptions."""
        config = cls.from_settings("descriptions")
        # Increase max_tokens for descriptions to avoid truncation
        # Hotel descriptions can be quite long (brief, detailed, marketing formats)
        # Read uncached so test patch.dict(os.environ, ...) is respected.
        from ..core.config import LLMConfig as _LLMConfig
        config.max_tokens = _LLMConfig().descriptions_max_tokens
        return config
    
    @classmethod
    def for_summaries(cls) -> LLMConfig:
        """Create config specifically for Epic 3 legacy summaries."""
        return cls.from_settings("summaries")

    @classmethod
    def for_vision_extraction(cls) -> LLMConfig:
        """Create config for vision sketch + synthesis calls.

        Forces ``provider=openrouter`` (vision models live there) and
        ``temperature=0.0`` regardless of global default — vision extraction
        wants token-level determinism, not stylistic flavor. The model itself
        is sourced from ``MODEL_VISION_EXTRACTION`` env var via the standard
        operation mapping.
        """
        config = cls.from_settings("vision_extraction")
        config.provider = "openrouter"
        config.base_url = settings.llm.openrouter_base_url
        config.api_key = settings.llm.openrouter_api_key
        config.temperature = 0.0
        # Read uncached so test patch.dict(os.environ, ...) is respected.
        from ..core.config import LLMConfig as _LLMConfig
        config.max_tokens = _LLMConfig().vision_max_tokens
        return config



class LLMFactory:
    """Factory for creating LLM instances with optimized LangFuse tracing."""
    
    def __init__(self):
        self.logger = get_logger()
        self._client_cache: Dict[str, Any] = {}  # Cache for OpenAI clients
        
        # Initialize enhanced LangFuse CallbackHandler with minimal bloat configuration
        self._callback_handler = None  # Lazy initialization for clean tracing
    
    def create_langfuse_client(self, config: Optional[LLMConfig] = None) -> Any:
        """Create OpenAI client with LangFuse tracing - optimal pattern.
        
        This creates a LangFuse-wrapped OpenAI client that automatically provides:
        - Token usage tracking
        - Cost calculation 
        - Model metadata capture
        - Clean prompt/response format in traces
        
        Args:
            config: Optional configuration. If None, uses default settings.
            
        Returns:
            LangFuse-wrapped OpenAI client with automatic instrumentation.
        """
        if config is None:
            config = LLMConfig.from_settings()

        # MODEL_BACKEND=mock (default): deterministic offline stand-in — no
        # network, no key. Returned for every operation so the whole pipeline
        # runs offline (manifest S3).
        if _model_backend() == "mock":
            from .mock_llm import MockChatModel
            return MockChatModel(temperature=config.temperature)

        # Create cache key based on config parameters
        cache_key = f"lf:{config.provider}:{config.model_name}:{config.temperature}:{config.max_tokens}"
        
        # Return cached instance if available
        if cache_key in self._client_cache:
            self.logger.debug(f"Using cached LangFuse OpenAI client for {cache_key}")
            return self._client_cache[cache_key]
        
        self.logger.info(
            "Creating new LangFuse OpenAI client",
            extra={
                "provider": config.provider,
                "model": config.model_name,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "cache_key": cache_key,
            }
        )
        
        # Use LangFuse OpenAI wrapper for automatic instrumentation
        from langfuse.openai import OpenAI  # type: ignore
        
        client_kwargs = {
            "api_key": config.api_key,
            "base_url": config.base_url,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
        
        if config.provider == "openrouter":
            # OpenRouter-specific configuration. The brand headers OpenRouter
            # attributes calls to are env-driven so the showcase carries no
            # hardcoded site identity; both default to neutral placeholders.
            client_kwargs.update({
                "default_headers": {
                    "HTTP-Referer": os.getenv(
                        "OPENROUTER_REFERER", "https://example.com"
                    ),
                    "X-Title": os.getenv(
                        "OPENROUTER_APP_TITLE", "LLM Content Generation"
                    ),
                }
            })
        elif config.provider == "lm_studio":
            # LM Studio defaults
            client_kwargs.update({
                "api_key": config.api_key or "lm-studio",
                "base_url": config.base_url or "http://localhost:1234/v1",
            })
        
        # Create LangFuse-wrapped client
        client = OpenAI(**client_kwargs)
        
        # Cache the client for future use
        self._client_cache[cache_key] = client
        return client
    
    def create_chat_llm(self, config: Optional[LLMConfig] = None) -> Any:
        """Create a chat LLM instance with connection pooling.
        
        Creates LangChain ChatOpenAI instances for use with CallbackHandler tracing.
        
        Args:
            config: Optional configuration. If None, uses default settings.
            
        Returns:
            Configured ChatOpenAI instance (cached if available).
        """
        if config is None:
            config = LLMConfig.from_settings()

        # ── MODEL_BACKEND switch (manifest S3) ────────────────────────────────
        backend = _model_backend()
        if backend == "mock":
            # Deterministic, offline, CI-safe. Every node / activity / harness
            # picks this up unchanged because all model creation flows here.
            from .mock_llm import MockChatModel
            return MockChatModel(temperature=config.temperature)
        if backend == "ollama":
            # Local OpenAI-compatible endpoint (e.g. Ollama / LM Studio).
            from langchain_openai import ChatOpenAI  # type: ignore
            return ChatOpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        if backend == "openrouter":
            # Force the OpenRouter provider path regardless of LLM_MODEL_PROVIDER.
            config.provider = "openrouter"
            config.base_url = settings.llm.openrouter_base_url
            config.api_key = settings.llm.openrouter_api_key

        # Create cache key based on config parameters
        cache_key = f"lc:{config.provider}:{config.model_name}:{config.temperature}:{config.max_tokens}"
        
        # Return cached instance if available
        if cache_key in self._client_cache:
            self.logger.debug(f"Using cached LLM instance for {cache_key}")
            return self._client_cache[cache_key]
        
        self.logger.info(
            "Creating new LangChain ChatOpenAI instance",
            extra={
                "provider": config.provider,
                "model": config.model_name,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "cache_key": cache_key,
            }
        )
        
        from langchain_openai import ChatOpenAI  # type: ignore
        
        if config.provider == "lm_studio":
            llm = ChatOpenAI(
                api_key=config.api_key or "lm-studio",
                base_url=config.base_url or "http://localhost:1234/v1",
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                stream_usage=True,  # Enable usage tracking directly
            )
        elif config.provider == "openrouter":
            # DeepSeek V3.2 enables "thinking mode" by default, which silently
            # ignores temperature, top_p, frequency_penalty, and presence_penalty.
            # Disable it for all OpenRouter→DeepSeek calls so the sampling config
            # we set here is actually respected.  Sent as extra_body so it
            # reaches the underlying provider through OpenRouter's passthrough.
            llm = ChatOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                stream_usage=True,  # Enable usage tracking for OpenRouter directly
                extra_body={"thinking": {"type": "disabled"}},
            )
        else:
            # Default OpenAI-style configuration
            llm = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                stream_usage=True,  # Enable usage tracking directly
            )
        
        # Cache the instance for future use
        self._client_cache[cache_key] = llm
        return llm
    
    def create_for_operation(self, operation: str) -> Any:
        """Create LangFuse OpenAI client for a specific operation type.
        
        Args:
            operation: Operation type (e.g., "extraction", "validation", "descriptions")
        """
        # Use specialized config methods when available
        if operation == "descriptions":
            config = LLMConfig.for_descriptions()
        elif operation == "extraction":
            config = LLMConfig.for_extraction()
        elif operation == "validation":
            config = LLMConfig.for_validation()
        elif operation == "summaries":
            config = LLMConfig.for_summaries()
        elif operation == "vision_extraction":
            config = LLMConfig.for_vision_extraction()
        else:
            config = LLMConfig.from_settings(operation)

        return self.create_langfuse_client(config)

    def create_for_vision_extraction(self) -> Any:
        """Create LangFuse OpenAI client configured for vision calls."""
        return self.create_langfuse_client(LLMConfig.for_vision_extraction())
    
    def create_from_prompt_config(self, model_name: str) -> Any:
        """Create LangChain ChatOpenAI client using model name from LangFuse prompt config.
        
        Args:
            model_name: Model name extracted from LangFuse prompt metadata
            
        Returns:
            LangChain ChatOpenAI client configured for the specified model
        """
        # Create config with the specified model
        config = LLMConfig.from_settings()
        
        # Apply safety fallback for OpenRouter to prevent accidental expensive model usage
        if config.provider == "openrouter":
            # Use the safer default model if the prompt model is not explicitly trusted
            final_model = self._get_safe_openrouter_model(model_name)
            if final_model != model_name:
                self.logger.warning(
                    "Using OpenRouter default model fallback for safety",
                    extra={
                        "requested_model": model_name,
                        "fallback_model": final_model,
                        "reason": "untrusted_model_name"
                    }
                )
            config.model_name = final_model
        else:
            config.model_name = model_name
        
        self.logger.info(
            "Creating LangChain ChatOpenAI client from prompt config",
            extra={
                "requested_model": model_name,
                "final_model": config.model_name,
                "provider": config.provider
            }
        )
        
        return self.create_chat_llm(config)
    
    def create_for_validation(self) -> Any:
        """Create LangFuse OpenAI client specifically configured for validation tasks."""
        return self.create_langfuse_client(LLMConfig.for_validation())
    
    def create_for_extraction(self) -> Any:
        """Create LangChain ChatOpenAI client specifically configured for Epic 1 extraction."""
        return self.create_chat_llm(LLMConfig.for_extraction())
    
    def create_for_descriptions(self) -> Any:
        """Create LangChain ChatOpenAI client specifically configured for Epic 3 hotel descriptions."""
        return self.create_chat_llm(LLMConfig.for_descriptions())
    
    def create_for_summaries(self) -> Any:
        """Create LangChain ChatOpenAI client specifically configured for Epic 3 legacy summaries."""
        return self.create_chat_llm(LLMConfig.for_summaries())
    
    def create_for_generation(self) -> Any:
        """Create LangChain ChatOpenAI client specifically configured for content generation."""
        return self.create_chat_llm(LLMConfig.from_settings())

    def create_for_reviews(self) -> Any:
        """Create LangChain ChatOpenAI client for enriched-review rewriting.

        Resolves the ``reviews`` operation model (DeepSeek V3.2 by default; see
        ModelMappingConfig). Used as the fallback when the Langfuse prompt config
        carries no explicit model.
        """
        return self.create_chat_llm(LLMConfig.from_settings("reviews"))

    def create_for_translation(self) -> Any:
        """Create LangChain ChatOpenAI client for multi-language translation.

        Resolves the ``translation`` operation model (DeepSeek V3.2 by default;
        see ModelMappingConfig). Used as the fallback when the Langfuse prompt
        config carries no explicit model.
        """
        return self.create_chat_llm(LLMConfig.from_settings("translation"))



    
    def _get_safe_openrouter_model(self, requested_model: str) -> str:
        """Determine OpenRouter model to use with minimal restrictions.
        
        Args:
            requested_model: Model name requested from LangFuse prompt config
            
        Returns:
            Model name to use (requested model or fallback for expensive models only)
        """
        # Only restriction: block expensive models unless explicitly allowed (cost control)
        expensive_models = {
            "anthropic/claude-3-opus",
            "anthropic/claude-3-opus-20240229",
            "openai/gpt-4",
            "openai/gpt-4-turbo",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
        }
        
        # Override expensive models unless explicitly allowed.
        # Read directly from LLMConfig (no cache) so that test monkeypatches
        # via patch.dict(os.environ, ...) are respected without requiring a
        # cache reset call from the test.
        from ..core.config import LLMConfig as _LLMConfig
        allow_expensive = _LLMConfig().allow_expensive_models
        if requested_model in expensive_models and not allow_expensive:
            self.logger.warning(
                f"Blocking expensive model '{requested_model}' - set LLM_ALLOW_EXPENSIVE_MODELS=true to allow",
                extra={"blocked_model": requested_model}
            )
            return settings.llm.openrouter_default_model
        
        # Allow any non-expensive model - trust the user's choice
        self.logger.info(f"Using requested model: {requested_model}")
        return requested_model