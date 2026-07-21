# app/shared/graphql_processing/backends/base.py
"""
Abstract backend interface for multi-target GraphQL fetching.

This module defines the Strategy pattern interface for GraphQL backends, enabling
extensible support for multiple GraphQL targets behind a single, target-agnostic
orchestrator.

Architecture:
    - Strategy pattern: Each target implements the GraphQLBackend interface
    - Three-method contract: build_search_query() + build_variables() + parse_response()
    - Type-safe return values (Pydantic HotelDataContract)
    - Target-specific logic encapsulated in concrete implementations

Examples:
    >>> # Implementing a new backend
    >>> class MyBackend(GraphQLBackend):
    ...     async def build_search_query(self, region_id, offset, limit):
    ...         return build_my_query()  # target-specific
    ...
    ...     def build_variables(self, region_id, offset, limit):
    ...         return {"offset": offset, "limit": limit}
    ...
    ...     async def parse_response(self, response, min_quality=0.0, correlation_id=None):
    ...         return [parse_record(r) for r in response["data"]["results"]]
    >>>
    >>> # Using a backend
    >>> backend = MyBackend()
    >>> query = await backend.build_search_query("demo-region", 0, 25)
    >>> variables = backend.build_variables("demo-region", 0, 25)
    >>> response = await client.execute(query, variables)
    >>> records = await backend.parse_response(response)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..contracts import HotelDataContract


class GraphQLBackend(ABC):
    """
    Abstract backend interface for multi-target GraphQL fetching.

    Defines the contract that all GraphQL backend implementations must follow.
    Each target implements this interface with target-specific query building,
    variable building, and response parsing logic.

    Architecture:
        - Strategy pattern: Enables runtime backend selection without code changes
        - Three-method contract: query building + variable building + response parsing
        - Target agnostic: Orchestrator works with any backend implementation
        - Type safety: Returns standardized HotelDataContract regardless of source

    Design Rationale:
        - Small method surface: each backend only needs to know how to build a
          query, build its variables, and parse responses
        - Async query/parse methods: enable I/O-bound work (schema validation, etc.)
        - Returns List[HotelDataContract]: standardized output for the orchestrator

    Backend Implementation Checklist:
        - [ ] Implement build_search_query() - Generate target-specific GraphQL
        - [ ] Implement build_variables() - Supply the query's variable payload
        - [ ] Implement parse_response() - Map target response to HotelDataContract
        - [ ] Handle partial data gracefully (nullable fields)
        - [ ] Calculate quality scores for parsed records
        - [ ] Add target-specific error handling

    Examples:
        >>> # Backend usage in orchestrator
        >>> backend = DemoBackend()
        >>> query = await backend.build_search_query("demo-region", 0, 25)
        >>> variables = backend.build_variables("demo-region", 0, 25)
        >>> response = await client.execute(query, variables)
        >>> records = await backend.parse_response(response)
        >>>
        >>> # All backends return the same contract
        >>> assert all(isinstance(h, HotelDataContract) for h in records)
    """

    @abstractmethod
    async def build_search_query(
        self,
        region_id: str,
        offset: int,
        limit: int
    ) -> str:
        """
        Build a target-specific GraphQL search query.

        Generates a GraphQL query string for searching records in a region with
        pagination. The query structure is target-specific but the method
        signature is standardized.

        Args:
            region_id: Target-specific region identifier
            offset: Pagination offset (0 for first page)
            limit: Number of results per page

        Returns:
            str: Complete GraphQL query string ready for HTTP POST

        Implementation Requirements:
            - Return valid GraphQL query syntax
            - Include all fields needed for HotelDataContract
            - Support pagination (offset + limit)

        Examples:
            >>> query = await backend.build_search_query("demo-region", 0, 25)
            >>> assert "query" in query
        """
        pass

    @abstractmethod
    def build_variables(
        self,
        region_id: str,
        offset: int,
        limit: int
    ) -> Dict[str, Any]:
        """
        Build the GraphQL variables payload for the search query.

        The orchestrator delegates variable construction to the backend so the
        engine itself carries no knowledge of any target's variable schema. Each
        backend owns the exact shape its query expects.

        Args:
            region_id: Target-specific region identifier
            offset: Pagination offset (0 for first page)
            limit: Number of results per page

        Returns:
            dict: GraphQL variables matching this backend's query.

        Examples:
            >>> variables = backend.build_variables("demo-region", 0, 25)
            >>> variables["limit"]
            25
        """
        pass

    @abstractmethod
    async def parse_response(
        self,
        response: Dict[str, Any],
        min_quality: float = 0.0,
        correlation_id: Optional[str] = None,
    ) -> List[HotelDataContract]:
        """
        Parse a target-specific GraphQL response to standardized contracts.

        Extracts record data from the GraphQL response JSON and maps it to
        HotelDataContract instances. Handles target-specific response structure
        and field naming.

        Args:
            response: GraphQL response dictionary (parsed JSON)
                Expected format: {"data": {...}, "errors": [...]} (standard GraphQL)
            min_quality: Minimum quality score (0.0-1.0) below which records are
                filtered out.
            correlation_id: Optional correlation id for structured logging.

        Returns:
            List[HotelDataContract]: Parsed and validated records with quality scores

        Implementation Requirements:
            - Handle partial data gracefully (many fields are optional)
            - Calculate quality scores for each record
            - Map target field names to contract field names (aliases)
            - Return empty list (not None) if no results found
            - Log parsing errors but continue with other records

        Examples:
            >>> response = {"data": {"search": {"results": [ ... ]}}}
            >>> records = await backend.parse_response(response)
            >>> assert all(r.quality_score >= 0 for r in records)
        """
        pass
