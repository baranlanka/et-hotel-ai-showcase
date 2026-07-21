from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from llm_content_generation.core.config import settings

from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("llm-content-generation", context="activity")
logger = _obs.logger


def get_logger():
    """Return the module-level logger (backward-compat shim).

    Why: langfuse_tracing has 8 call sites using get_logger() inline.
    This shim avoids touching every call site while routing through
    the unified manager.
    """
    return logger

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object


@dataclass
class LangFuseState:
    enabled: bool
    client: Optional[Any]


def _init_client() -> LangFuseState:
    """Initialize LangFuse client with enhanced v3 configuration and debugging."""
    if not settings.langfuse.enable:
        return LangFuseState(enabled=False, client=None)

    try:
        # Lazy import to avoid hard dependency during tests
        from langfuse import Langfuse

        # Avoid initializing the SDK when keys are missing to prevent noisy warnings
        if not settings.langfuse.public_key or not settings.langfuse.secret_key:
            logger = get_logger()
            if settings.langfuse.debug:
                logger.warning(
                    "LangFuse credentials missing",
                    extra={
                        "has_public_key": bool(settings.langfuse.public_key),
                        "has_secret_key": bool(settings.langfuse.secret_key),
                        "host": settings.langfuse.host,
                    }
                )
            return LangFuseState(enabled=False, client=None)

        # Enhanced client configuration with v3 features
        client_kwargs = {
            "secret_key": settings.langfuse.secret_key,
            "public_key": settings.langfuse.public_key,
            "host": settings.langfuse.host,
        }
        
        # Add optional v3 configuration if available
        try:
            import inspect
            langfuse_sig = inspect.signature(Langfuse.__init__)
            
            # Add debug mode if supported
            if "debug" in langfuse_sig.parameters and settings.langfuse.debug:
                client_kwargs["debug"] = True
                
            # Add flush interval if supported
            if "flush_interval" in langfuse_sig.parameters:
                client_kwargs["flush_interval"] = settings.langfuse.flush_interval
                
            # Add sample rate if supported
            if "sample_rate" in langfuse_sig.parameters:
                client_kwargs["sample_rate"] = settings.langfuse.sample_rate
                
            # Add timeout if supported
            if "timeout" in langfuse_sig.parameters:
                client_kwargs["timeout"] = settings.langfuse.timeout_seconds
                
        except Exception:
            # Fallback to basic configuration if introspection fails
            pass

        client = Langfuse(**client_kwargs)
        
        # Log successful initialization with debug info
        logger = get_logger()
        if settings.langfuse.debug:
            logger.info(
                "LangFuse client initialized with enhanced configuration",
                extra={
                    "host": settings.langfuse.host,
                    "debug_mode": settings.langfuse.debug,
                    "sample_rate": settings.langfuse.sample_rate,
                    "flush_interval": settings.langfuse.flush_interval,
                    "hierarchical_traces": settings.langfuse.enable_hierarchical_traces,
                    "business_context": settings.langfuse.enable_business_context,
                    "client_kwargs": list(client_kwargs.keys()),
                }
            )
        
        return LangFuseState(enabled=True, client=client)
        
    except Exception as exc:  # pragma: no cover
        logger = get_logger()
        logger.warning(
            "LangFuse init failed; continuing without tracing",
            extra={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "config_debug": settings.langfuse.get_debug_info(),
            },
        )
        return LangFuseState(enabled=False, client=None)


_STATE = _init_client()


@contextlib.contextmanager
def traced_operation(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    input_data: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    operation_type: str = "general",
    business_context: Optional[Dict[str, Any]] = None,
) -> Generator[Optional[str], None, None]:
    """Context manager yielding a trace id if tracing is enabled.

    Provides comprehensive trace management following LangFuse v3 SDK best practices:
    - Proper trace metadata with session and user correlation
    - Business context tags for filtering and analysis
    - Comprehensive input/output capture
    - Deterministic trace ID support for linking
    - Enhanced error handling with trace context
    - Operation-specific naming conventions

    Args:
        name: Business-meaningful name for the trace (e.g., "hotel-review-aspect-extraction")
        metadata: Additional technical metadata for the trace
        session_id: Session identifier for correlation (e.g., "extraction-hotel123-20241201")
        user_id: User identifier for correlation (e.g., hotel_id, review_id, or "system")
        tags: List of business context tags (e.g., ["epic1", "extraction", "hotel-reviews"])
        input_data: Input data to log with the trace for debugging
        trace_id: Optional predefined trace ID for linking parent/child traces
        operation_type: Type of operation for enhanced categorization
        business_context: Additional business context (hotel_id, review_count, etc.)

    Yields:
        Trace ID if tracing is enabled, None otherwise

    Examples:
        # Epic 1 extraction
        with traced_operation(
            name="hotel-review-aspect-extraction",
            session_id=f"extraction-{hotel_id}-{date}",
            user_id=hotel_id,
            tags=["epic1", "extraction", "hotel-reviews", "en"],
            input_data={"review_preview": review_text[:200], "room_types": room_types},
            operation_type="extraction",
            business_context={"hotel_id": hotel_id, "review_count": 150}
        ) as trace_id:
            # Extraction logic here
            pass
            
        # Epic 3 summarization
        with traced_operation(
            name="hotel-description-generation",
            session_id=f"summarization-{hotel_id}-{date}",
            user_id=hotel_id,
            tags=["epic3", "summarization", "descriptions"],
            operation_type="summarization",
            business_context={"hotel_id": hotel_id, "aspects_count": 45}
        ) as trace_id:
            # Summarization logic here
            pass
    """
    logger = get_logger()
    actual_trace_id: Optional[str] = None
    
    if _STATE.enabled and _STATE.client is not None:
        client = _STATE.client
        
        try:
            # Use provided trace_id or generate new one
            if trace_id:
                actual_trace_id = trace_id
            elif hasattr(client, "create_trace_id"):
                actual_trace_id = client.create_trace_id()  # type: ignore[attr-defined]
            
            # Build comprehensive trace metadata following LangFuse v3 standards
            trace_metadata = metadata or {}
            
            # Core correlation fields
            if session_id:
                trace_metadata["session_id"] = session_id
            if user_id:
                trace_metadata["user_id"] = user_id
            if input_data:
                trace_metadata["input"] = input_data
                
            # Operation classification
            trace_metadata["operation_type"] = operation_type
            
            # Business context integration
            if business_context:
                trace_metadata["business_context"] = business_context
                # Extract key business metrics for top-level visibility
                if "hotel_id" in business_context:
                    trace_metadata["hotel_id"] = business_context["hotel_id"]
                if "review_count" in business_context:
                    trace_metadata["review_count"] = business_context["review_count"]
                if "aspects_count" in business_context:
                    trace_metadata["aspects_count"] = business_context["aspects_count"]
                    
            # Enhanced tags with operation context
            enhanced_tags = tags or []
            if operation_type != "general":
                enhanced_tags.append(f"operation-{operation_type}")
            if business_context and "hotel_id" in business_context:
                enhanced_tags.append(f"hotel-{business_context['hotel_id']}")
                
            # Technical metadata
            from datetime import datetime
            trace_metadata["langfuse_sdk_version"] = "v3"
            trace_metadata["trace_created_at"] = datetime.now().isoformat()
                
            # Create span with enhanced context
            if hasattr(client, "start_as_current_span"):
                span_kwargs = {
                    "name": name,
                    "metadata": trace_metadata,
                    "end_on_exit": True,
                }
                
                # Add trace context if deterministic ID provided
                if actual_trace_id:
                    span_kwargs["trace_context"] = {"trace_id": actual_trace_id}
                
                try:
                    span_cm = client.start_as_current_span(**span_kwargs)  # type: ignore[attr-defined]
                except Exception as span_error:
                    logger.debug(
                        "Failed to create span, falling back to basic tracing",
                        extra={
                            "span_error": str(span_error),
                            "operation_name": name,
                            "session_id": session_id,
                            "user_id": user_id,
                        }
                    )
                    # Yield None to indicate no span context is available
                    yield None
                    return
                
                with span_cm as span:
                    try:
                        # Set comprehensive trace attributes with enhanced business context
                        if hasattr(span, "update_trace"):
                            update_kwargs = {}
                            if session_id:
                                update_kwargs["session_id"] = session_id
                            if user_id:
                                update_kwargs["user_id"] = user_id
                            if enhanced_tags:
                                update_kwargs["tags"] = enhanced_tags
                            if input_data:
                                update_kwargs["input"] = input_data
                                
                            # Add business context to trace attributes
                            if business_context:
                                update_kwargs["metadata"] = {
                                    **(update_kwargs.get("metadata", {})),
                                    "business_context": business_context,
                                    "operation_type": operation_type
                                }
                            
                            if update_kwargs:
                                span.update_trace(**update_kwargs)
                        
                        yield actual_trace_id
                        
                    except Exception as exc:
                        # Enhanced error logging with comprehensive trace context
                        error_context = {
                            "trace_id": actual_trace_id,
                            "operation_name": name,
                            "operation_type": operation_type,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "session_id": session_id,
                            "user_id": user_id,
                        }
                        
                        # Add business context to error logging
                        if business_context:
                            error_context["business_context"] = business_context
                        
                        logger.error(
                            "Traced operation failed with business context",
                            extra=error_context,
                            exc_info=True
                        )
                        
                        # Update trace with comprehensive error information
                        if hasattr(span, "update_trace"):
                            try:
                                error_tags = (enhanced_tags or []) + ["error", f"error-{type(exc).__name__.lower()}"]
                                error_output = {
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                    "success": False,
                                    "failed_at": name,
                                    "operation_type": operation_type
                                }
                                if business_context:
                                    error_output["business_context"] = business_context
                                    
                                span.update_trace(
                                    output=error_output,
                                    tags=error_tags
                                )
                            except Exception:
                                pass
                        raise
                    finally:
                        try:
                            if hasattr(client, "flush"):
                                client.flush()
                        except Exception:
                            pass
                return
            else:
                yield None
                return
                
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "LangFuse trace creation failed", 
                extra={
                    "error": str(exc),
                    "operation_name": name,
                    "session_id": session_id,
                    "user_id": user_id,
                }
            )
            yield None
            return

    # Fallback when disabled or error
    logger.debug(
        "LangFuse tracing disabled or unavailable",
        extra={"operation_name": name}
    )
    yield None


def record_event(event_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    logger = get_logger()
    if not _STATE.enabled or _STATE.client is None:
        logger.debug(
            "LF event skipped", extra={"event": event_name, "meta": metadata or {}}
        )
        return
    try:
        client = _STATE.client
        if hasattr(client, "event"):
            client.event(  # type: ignore[attr-defined]
                name=event_name, metadata=metadata or {}
            )
        elif hasattr(client, "create_event"):
            client.create_event(  # type: ignore[attr-defined]
                name=event_name, metadata=metadata or {}
            )
    except Exception as exc:  # pragma: no cover
        logger.debug("LF event error", extra={"error": str(exc)})


# Global callback handler instance (singleton pattern)
_CALLBACK_HANDLER: Optional[Any] = None

def get_callback_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_name: Optional[str] = None,
    operation_type: str = "langchain",
    business_context: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """Return a Langfuse LangChain callback handler with enhanced tracing context.

    Creates context-aware handlers that eliminate null/NaN logs through proper
    trace correlation, business context integration, and comprehensive metadata.

    Args:
        session_id: Session identifier for correlation (e.g., "extraction-hotel123-20241201")
        user_id: User identifier for correlation (e.g., hotel_id, review_id, "system")
        tags: Business context tags (e.g., ["epic1", "extraction", "hotel-reviews"])
        metadata: Technical metadata for debugging and analysis
        trace_name: Business-meaningful trace name for the handler context
        operation_type: Operation classification for enhanced categorization
        business_context: Business metrics and context (hotel_id, review_count, etc.)

    Returns:
        Configured CallbackHandler instance with comprehensive context or None if disabled

    Examples:
        # Epic 1 extraction handler
        handler = get_callback_handler(
            session_id=f"extraction-{hotel_id}-{date}",
            user_id=hotel_id,
            tags=["epic1", "extraction", "langchain"],
            trace_name="hotel-review-aspect-extraction",
            operation_type="extraction",
            business_context={"hotel_id": hotel_id, "review_count": 150},
            metadata={"prompt_version": "v2.1", "model": "mistral-7b"}
        )
        
        # Epic 3 summarization handler
        handler = get_callback_handler(
            session_id=f"summarization-{hotel_id}-{date}",
            user_id=hotel_id,
            tags=["epic3", "summarization", "langchain"],
            trace_name="hotel-description-generation",
            operation_type="summarization",
            business_context={"hotel_id": hotel_id, "aspects_count": 45}
        )
    """
    global _CALLBACK_HANDLER
    logger = get_logger()
    
    if not _STATE.enabled or _STATE.client is None:
        logger.debug("LangFuse tracing disabled or client unavailable")
        return None
    
    try:
        # For context-aware tracing, create a new handler with comprehensive context
        if session_id or user_id or tags or metadata or trace_name or business_context:
            cb_module = importlib.import_module("langfuse.langchain")
            CallbackHandler = getattr(cb_module, "CallbackHandler")  # type: ignore[assignment]
            
            # Build comprehensive context metadata
            context_metadata = metadata or {}
            
            # Core correlation fields
            if session_id:
                context_metadata["session_id"] = session_id
            if user_id:
                context_metadata["user_id"] = user_id
            if tags:
                context_metadata["tags"] = tags
            if trace_name:
                context_metadata["trace_name"] = trace_name
                
            # Operation classification
            context_metadata["operation_type"] = operation_type
            
            # Business context integration for debugging
            if business_context:
                context_metadata["business_context"] = business_context
                # Extract key business identifiers for visibility
                if "hotel_id" in business_context:
                    context_metadata["hotel_id"] = business_context["hotel_id"]
                if "review_id" in business_context:
                    context_metadata["review_id"] = business_context["review_id"]
                    
            # Enhanced tags for categorization
            enhanced_tags = list(tags or [])
            enhanced_tags.append(f"operation-{operation_type}")
            if business_context and "hotel_id" in business_context:
                enhanced_tags.append(f"hotel-{business_context['hotel_id']}")
            
            try:
                # Create handler with comprehensive configuration
                handler_kwargs = {}
                
                # Try to pass context through constructor (v3 SDK pattern)
                if hasattr(CallbackHandler, "__init__"):
                    import inspect
                    sig = inspect.signature(CallbackHandler.__init__)
                    if "metadata" in sig.parameters:
                        handler_kwargs["metadata"] = context_metadata
                    if "tags" in sig.parameters:
                        handler_kwargs["tags"] = enhanced_tags
                    if "session_id" in sig.parameters:
                        handler_kwargs["session_id"] = session_id
                    if "user_id" in sig.parameters:
                        handler_kwargs["user_id"] = user_id
                
                handler = CallbackHandler(**handler_kwargs)
                
                logger.info(
                    "Created enhanced LangFuse callback handler",
                    extra={
                        "session_id": session_id,
                        "user_id": user_id,
                        "trace_name": trace_name,
                        "operation_type": operation_type,
                        "tags": enhanced_tags,
                        "handler_id": id(handler),
                        "business_context": business_context,
                        "has_context_metadata": bool(context_metadata)
                    }
                )
                return handler
                
            except (TypeError, AttributeError) as e:
                # Fallback to basic handler if enhanced configuration not supported
                handler = CallbackHandler()
                logger.debug(
                    "Created basic LangFuse callback handler - enhanced config not supported",
                    extra={
                        "handler_id": id(handler),
                        "error": str(e),
                        "fallback_reason": "constructor_signature_mismatch"
                    }
                )
                return handler
        
        # Return or create singleton handler for basic usage
        if _CALLBACK_HANDLER is not None:
            logger.debug(f"Reusing existing LangFuse callback handler: {id(_CALLBACK_HANDLER)}")
            return _CALLBACK_HANDLER
        
        # Create singleton handler instance
        cb_module = importlib.import_module("langfuse.langchain")
        CallbackHandler = getattr(cb_module, "CallbackHandler")  # type: ignore[assignment]
        _CALLBACK_HANDLER = CallbackHandler()
        
        logger.debug(f"Created singleton LangFuse callback handler: {id(_CALLBACK_HANDLER)}")
        return _CALLBACK_HANDLER
        
    except Exception as exc:  # pragma: no cover
        logger.debug(
            "LangFuse callback init failed", 
            extra={
                "error": str(exc),
                "session_id": session_id,
                "user_id": user_id,
            }
        )
        return None


def reset_callback_handler() -> None:
    """Reset the singleton callback handler.
    
    Useful for testing or when you want to start fresh tracing context.
    """
    global _CALLBACK_HANDLER
    _CALLBACK_HANDLER = None
    logger = get_logger()
    logger.debug("Reset singleton LangFuse callback handler")


class PromptLinkingCallback(BaseCallbackHandler):
    """Callback handler that links LangFuse prompts to generations.
    
    This ensures that when an LLM is invoked, the generation is properly
    linked to the LangFuse prompt that was used, enabling proper prompt
    tracking and versioning in the LangFuse dashboard.
    
    Based on the working implementation from Epic 3 pipeline.
    """
    
    def __init__(self, prompt_obj: Any):
        """Initialize with LangFuse prompt object.
        
        Args:
            prompt_obj: LangFuse prompt object returned from LangFuse API
        """
        super().__init__()
        self._prompt_obj = prompt_obj
        self.logger = get_logger()
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Called when LLM starts - link prompt to generation."""
        try:
            if _STATE.client is not None:
                _STATE.client.update_current_generation(prompt=self._prompt_obj)
                self.logger.debug(
                    "Linked prompt to generation via on_llm_start",
                    extra={"prompt_name": getattr(self._prompt_obj, 'name', 'unknown')}
                )
        except Exception as exc:
            self.logger.debug(
                "Failed to link prompt in on_llm_start",
                extra={"error": str(exc)}
            )
    
    def on_chat_model_start(
        self, 
        serialized: Dict[str, Any], 
        messages: List[List[Any]], 
        **kwargs: Any
    ) -> None:
        """Called when chat model starts - link prompt to generation."""
        try:
            if _STATE.client is not None:
                _STATE.client.update_current_generation(prompt=self._prompt_obj)
                self.logger.debug(
                    "Linked prompt to generation via on_chat_model_start",
                    extra={"prompt_name": getattr(self._prompt_obj, 'name', 'unknown')}
                )
        except Exception as exc:
            self.logger.debug(
                "Failed to link prompt in on_chat_model_start",
                extra={"error": str(exc)}
            )


def create_prompt_linking_callback(prompt_obj: Any) -> Optional[PromptLinkingCallback]:
    """Create a prompt linking callback if LangFuse is enabled.
    
    Args:
        prompt_obj: LangFuse prompt object to link to generations
        
    Returns:
        PromptLinkingCallback instance or None if LangFuse is disabled
    """
    if not _STATE.enabled or _STATE.client is None:
        return None
    
    return PromptLinkingCallback(prompt_obj)


