# app/shared/graphql_processing/clients/manager.py (OPTIONAL)
from typing import Optional
from .httpx_client import HTTPXGraphQLClient

class GraphQLClientManager:
    """Optional: Simple client manager (no threading complexity)."""
    _client: Optional[HTTPXGraphQLClient] = None

    @classmethod
    def get_client(cls, endpoint: str, **kwargs) -> HTTPXGraphQLClient:
        """Get or create client."""
        if cls._client is None:
            cls._client = HTTPXGraphQLClient(endpoint, **kwargs)
        return cls._client
