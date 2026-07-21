# app/shared/graphql_processing/orchestrator.py
"""
GraphQL fetch orchestration layer.

This module coordinates GraphQL fetching business logic, integrating a pluggable
backend and HTTP clients to fetch and parse records. The orchestrator carries zero
workflow-engine dependencies, so it runs identically from a CLI script or from an
async task runner.

Architecture:
    - Pure business logic (no workflow-engine decorators or dependencies)
    - Delegates query building / variable building / parsing to an injected backend
      (Strategy pattern) so the engine is target-agnostic
    - Manages pagination loops for multi-page results
    - Circuit breaker + timeout recovery for resilience

Backend injection:
    The orchestrator does NOT know about any specific target. The concrete
    ``GraphQLBackend`` is injected at construction time (defaulting to the neutral
    :class:`DemoBackend`). To fetch from a different endpoint you supply a different
    backend implementing ``build_search_query`` / ``build_variables`` /
    ``parse_response`` — the loop below never changes.

Examples:
    >>> from app.shared.graphql_processing.orchestrator import GraphQLScraperOrchestrator
    >>> from app.shared.graphql_processing.contracts import ScraperConfig
    >>> from app.shared.graphql_processing.backends.demo_backend import DemoBackend
    >>>
    >>> config = ScraperConfig(endpoint="https://api.example.com/graphql")
    >>> orchestrator = GraphQLScraperOrchestrator(backend=DemoBackend())
    >>> records = await orchestrator.scrape_region("demo-region", config, max_pages=2)
    >>> print(f"Fetched {len(records)} records")
"""
from typing import Any, Callable, Dict, List, Optional
from .contracts import HotelDataContract, ScraperConfig
from .clients.httpx_client import HTTPXGraphQLClient
from .backends.base import GraphQLBackend
from .backends.demo_backend import DemoBackend
from .errors import NetworkError
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from common.observability import trace
from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("graphql-processing", context="activity")
logger = _obs.logger


class GraphQLScraperOrchestrator:
    """
    GraphQL fetch orchestrator coordinating a pluggable backend and HTTP client.

    Manages the end-to-end fetch workflow: query building via the injected backend,
    HTTP execution via clients, pagination loop management, and response parsing.
    The orchestrator is the central business logic component and is fully target
    agnostic — it works with any :class:`GraphQLBackend`.

    Architecture:
        - Separation of concerns: delegates query/variable/parse to the backend
        - Pagination loop with configurable max_pages limit
        - Async context manager usage for client lifecycle
        - Circuit breaker for fault tolerance
        - Timeout recovery with partial results

    Design Principles:
        - Pure business logic (no execution-context dependencies)
        - Configuration via parameters (not global state)
        - Type-safe contracts (Pydantic models)
        - Async-first for concurrent fetching
        - Graceful degradation under failure

    Examples:
        >>> orchestrator = GraphQLScraperOrchestrator(backend=DemoBackend())
        >>> config = ScraperConfig(
        ...     endpoint="https://api.example.com/graphql",
        ...     timeout=30,
        ... )
        >>> records = await orchestrator.scrape_region(
        ...     region_id="demo-region",
        ...     config=config,
        ...     max_pages=3,
        ... )
        >>> avg_quality = sum(h.quality_score for h in records) / len(records)
        >>> print(f"Fetched {len(records)} records, avg quality: {avg_quality:.1f}%")
    """

    def __init__(self, backend: Optional[GraphQLBackend] = None):
        """Initialize orchestrator with a circuit breaker and an injected backend.

        Args:
            backend: Concrete :class:`GraphQLBackend` used to build queries /
                variables and parse responses. Defaults to the neutral
                :class:`DemoBackend` so the engine is runnable out of the box.
        """
        self.backend: GraphQLBackend = backend or DemoBackend()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60,
            name="graphql_scraper"
        )

    # Break the page loop after this many consecutive pages yield ZERO new record
    # IDs. Some paginated endpoints silently repeat the final page once you request
    # an offset past the real result count — every higher offset returns the same
    # rows. Several consecutive "all duplicates" pages is strong evidence the real
    # cursor has been exhausted.
    _NO_NEW_HOTELS_BREAK = 3

    @trace(span_name="graphql.scrape_region", high_throughput=True)
    async def scrape_region(
        self,
        region_id: str,
        config: ScraperConfig,
        max_pages: int = None,
        min_quality: float = 0.3,
        correlation_id: Optional[str] = None,
        heartbeat: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[HotelDataContract]:
        """
        Fetch all records for a region with pagination, via the injected backend.

        Main entry point for region fetching. Executes a pagination loop to fetch
        all available records for the region, building queries and variables via the
        injected ``backend`` and executing them via ``HTTPXGraphQLClient``.

        Resilience Features:
            - Circuit breaker: Fails fast after 5 consecutive errors
            - Timeout recovery: Returns partial results on timeout with metadata
            - Quality filtering: Passed to backend for low-quality record filtering

        Workflow:
            1. Use the injected backend instance
            2. Initialize HTTPXGraphQLClient (async context manager)
            3. Loop: Build query + variables -> Execute (circuit breaker) -> Parse -> Accumulate
            4. Stop when: No more results OR max_pages reached OR timeout
            5. Return all accumulated contracts (partial if timeout)

        Args:
            region_id: Backend-specific region identifier as a string. The meaning is
                defined entirely by the backend (the demo backend treats it as an
                opaque label).
            config: ScraperConfig with endpoint URL, cookies, timeout
            max_pages: Optional limit on pages to fetch (default: None = unlimited)
            min_quality: Minimum quality score threshold (0.0-1.0, default: 0.3)

        Returns:
            List[HotelDataContract]: All records found for the region, with quality
                scores. May be partial results if a timeout occurred.

        Raises:
            CircuitBreakerOpenError: Circuit is open after repeated failures
            NetworkError: Network errors that aren't timeout-related
            RateLimitError: Rate limit exceeded (handled by retry decorator)
        """
        logger.info(
            "Starting region fetch",
            extra={
                "correlation_id": correlation_id,
                "region_id": region_id,
                "max_pages": max_pages,
                "min_quality": min_quality,
                "backend_name": type(self.backend).__name__,
                "span_name": "graphql.scrape_region"
            }
        )

        backend = self.backend
        all_hotels: List[HotelDataContract] = []
        seen_ids: set = set()
        consecutive_no_new = 0
        offset = 0
        limit = 25
        page_count = 0

        if config.proxy_url:
            from .clients.httpx_client import ProxiedHTTPXGraphQLClient
            _client_ctx = ProxiedHTTPXGraphQLClient(
                endpoint=str(config.endpoint),
                proxy_url=config.proxy_url,
                connect_timeout=10.0,
                read_timeout=float(config.timeout),
                cookies=config.cookies,
            )
        else:
            _client_ctx = HTTPXGraphQLClient(
                endpoint=str(config.endpoint),
                timeout=config.timeout,
                cookies=config.cookies,
            )

        async with _client_ctx as client:
            try:
                while True:
                    page_count += 1

                    logger.info(
                        "Fetching page",
                        extra={
                            "correlation_id": correlation_id,
                            "region_id": region_id,
                            "page": page_count,
                            "offset": offset,
                            "limit": limit
                        }
                    )

                    # Build query + variables via the injected backend
                    query = await backend.build_search_query(region_id, offset, limit)
                    variables = backend.build_variables(region_id, offset, limit)

                    # Execute with circuit breaker protection
                    try:
                        response = await self.circuit_breaker.call(
                            client.execute,
                            query,
                            variables
                        )
                    except CircuitBreakerOpenError as e:
                        logger.error(
                            "Circuit breaker open, stopping fetch",
                            extra={
                                "correlation_id": correlation_id,
                                "error_type": "circuit_breaker_open",
                                "region_id": region_id,
                                "page": page_count,
                                "hotels_scraped": len(all_hotels)
                            }
                        )
                        raise

                    # Parse with quality filtering
                    hotels = await backend.parse_response(
                        response,
                        min_quality=min_quality,
                        correlation_id=correlation_id
                    )
                    # Dedupe against IDs already seen on earlier pages. Some paginated
                    # endpoints repeat the final page of results past the real result
                    # count — the same rows returned for every higher offset. Without
                    # dedupe we would keep accepting duplicate rows and pagination
                    # would never terminate.
                    new_hotels = [h for h in hotels if str(h.id) not in seen_ids]
                    for h in new_hotels:
                        seen_ids.add(str(h.id))

                    logger.info(
                        "Parsed records",
                        extra={
                            "correlation_id": correlation_id,
                            "region_id": region_id,
                            "page": page_count,
                            "result_count": len(hotels),
                            "new_count": len(new_hotels),
                            "total_unique": len(seen_ids),
                            "min_quality": min_quality,
                        },
                    )

                    all_hotels.extend(new_hotels)

                    # Optional progress heartbeat — best-effort; a heartbeat failure
                    # must not halt the fetch.
                    if heartbeat is not None:
                        try:
                            heartbeat(
                                {
                                    "region_id": region_id,
                                    "page": page_count,
                                    "offset": offset,
                                    "total_unique": len(seen_ids),
                                }
                            )
                        except Exception as hb_exc:  # noqa: BLE001
                            logger.debug(
                                f"heartbeat failed (non-fatal): {hb_exc}"
                            )

                    # Termination #1 — the endpoint signalled "no more" by returning
                    # a short final page.
                    if len(hotels) < limit:
                        logger.info(
                            "No more results available",
                            extra={
                                "correlation_id": correlation_id,
                                "region_id": region_id,
                                "page": page_count,
                            },
                        )
                        break

                    # Termination #2 — local dupe-loop detector. We accepted a full
                    # page but none of it was new. If this happens
                    # ``_NO_NEW_HOTELS_BREAK`` pages in a row, we are paging into a
                    # repeated tail; stop.
                    if not new_hotels:
                        consecutive_no_new += 1
                        if consecutive_no_new >= self._NO_NEW_HOTELS_BREAK:
                            logger.info(
                                "Pagination loop detected — all IDs duplicates",
                                extra={
                                    "correlation_id": correlation_id,
                                    "region_id": region_id,
                                    "page": page_count,
                                    "consecutive_no_new": consecutive_no_new,
                                    "total_unique": len(seen_ids),
                                },
                            )
                            break
                    else:
                        consecutive_no_new = 0

                    # Termination #3 — hard cap on pages (defensive belt-and-suspenders
                    # in case both the server-break and dupe-detector somehow miss).
                    if max_pages and page_count >= max_pages:
                        logger.info(
                            "Reached maximum page limit",
                            extra={
                                "correlation_id": correlation_id,
                                "region_id": region_id,
                                "max_pages": max_pages,
                                "page": page_count,
                            },
                        )
                        break

                    offset += limit

            except NetworkError as e:
                # Handle timeout with partial results
                if "timeout" in str(e).lower():
                    logger.info(
                        "Timeout, returning partial results",
                        extra={
                            "correlation_id": correlation_id,
                            "error_type": "timeout",
                            "region_id": region_id,
                            "pages_scraped": page_count - 1,
                            "expected_pages": max_pages or "unlimited",
                            "partial_results": True,
                            "hotels_count": len(all_hotels)
                        }
                    )
                    # Return partial results instead of raising
                    return all_hotels

                # Re-raise non-timeout network errors
                raise

        avg_quality = sum(h.quality_score for h in all_hotels) / len(all_hotels) if all_hotels else 0
        logger.info(
            "Fetch complete",
            extra={
                "correlation_id": correlation_id,
                "region_id": region_id,
                "total_hotels": len(all_hotels),
                "pages_scraped": page_count,
                "quality_score_avg": avg_quality
            }
        )
        return all_hotels
