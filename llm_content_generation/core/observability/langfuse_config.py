"""Centralized LangFuse observability configuration for LangGraph workflows."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# The Langfuse LangChain CallbackHandler (v4) hard-requires the full ``langchain``
# meta-package. In the showcase, Langfuse tracing is DISABLED by design, so this
# is an OPTIONAL dependency: import it defensively and degrade to a no-op (empty
# callbacks) when it — or ``langchain`` — is unavailable. This keeps the content
# graph importable/runnable offline without pulling in ``langchain``.
try:  # pragma: no cover - import-environment dependent
    from langfuse.langchain import CallbackHandler  # type: ignore
    from langfuse import get_client  # type: ignore
    _LANGFUSE_LC_AVAILABLE = True
except Exception:  # noqa: BLE001
    CallbackHandler = None  # type: ignore
    _LANGFUSE_LC_AVAILABLE = False

    def get_client():  # type: ignore
        """No-op stand-in when the Langfuse LangChain integration is absent."""
        return None


class LangGraphObservability:
    """Centralized LangFuse observability configuration for LangGraph workflows."""

    _handler: "CallbackHandler | None" = None

    @classmethod
    def get_callback_handler(cls) -> "CallbackHandler | None":
        """Get CallbackHandler singleton, or None when Langfuse tracing is off.

        Returns None when the Langfuse LangChain integration is unavailable
        (``langchain`` not installed) or a handler cannot be constructed — the
        content graph then runs with no tracing callbacks (showcase default).
        """
        if not _LANGFUSE_LC_AVAILABLE:
            return None
        if cls._handler is None:
            try:
                cls._handler = CallbackHandler()
            except Exception:  # noqa: BLE001
                return None
        return cls._handler

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get LangGraph config with CallbackHandler (empty when tracing is off)."""
        handler = cls.get_callback_handler()
        return {"callbacks": [handler] if handler is not None else []}
    
    @classmethod
    def get_config_with_metadata(
        cls,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get LangGraph config with CallbackHandler and custom metadata.

        Args:
            tags: List of tags for LangFuse tracing
            metadata: Additional metadata for the trace
            session_id: Session ID for LangFuse tracing (passed via metadata in v3)
            user_id: User ID for LangFuse tracing (passed via metadata in v3)

        Returns:
            Configuration dict for LangGraph with callbacks and metadata
        """
        # In LangFuse v3, session_id/user_id are passed via metadata, not handler constructor
        handler = cls.get_callback_handler()
        config = {"callbacks": [handler] if handler is not None else []}

        # Build metadata with session/user info and custom data
        config_metadata = {}

        if tags:
            config_metadata["langfuse_tags"] = tags

        if session_id:
            config_metadata["langfuse_session_id"] = session_id

        if user_id:
            config_metadata["langfuse_user_id"] = user_id

        if metadata:
            config_metadata.update(metadata)

        if config_metadata:
            config["metadata"] = config_metadata

        return config
    
    @classmethod
    @asynccontextmanager
    async def trace_langgraph_execution(
        cls,
        trace_name: str,
        input_data: Dict[str, Any],
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Context manager for wrapping LangGraph execution with standardized trace naming.
        
        This ensures:
        1. Single main trace named after endpoint (e.g., "process-reviews")
        2. All LangGraph operations as spans within the main trace
        3. Proper nesting of subgraph operations like aspect_extraction
        4. Centralized metadata and tag management
        
        Args:
            trace_name: Name for the main trace (e.g., "process-reviews", "generate-content")
            input_data: Input data for the trace
            tags: Optional tags for the trace
            metadata: Optional metadata for the trace
            user_id: Optional user ID
            session_id: Optional session ID
            
        Usage:
            async with LangGraphObservability.trace_langgraph_execution(
                "process-reviews", 
                {"hotel_id": "123", "ota": "demo_ota"},
                tags=["api-endpoint"],
                metadata={"hotel_id": "123", "ota": "demo_ota"}
            ) as trace_context:
                config = LangGraphObservability.get_config()
                result = await graph.ainvoke(state, config=config)
                trace_context.end(output=result)
        """
        langfuse = get_client()
        
        # Create main trace with endpoint name using context manager
        with langfuse.start_as_current_span(
            name=trace_name,
            input=input_data
        ) as span:
            # Set trace metadata using update_trace method
            span.update_trace(
                user_id=user_id,
                session_id=session_id,
                tags=tags or [],
                metadata=metadata or {}
            )
            try:
                yield span
            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    level="ERROR"
                )
                raise
    
    @classmethod
    def reset_handler(cls) -> None:
        """Reset the singleton handler. Useful for testing."""
        cls._handler = None