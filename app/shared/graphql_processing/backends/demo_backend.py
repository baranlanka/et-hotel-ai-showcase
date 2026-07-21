# app/shared/graphql_processing/backends/demo_backend.py
"""
Neutral demo backend for the generic GraphQL fetch engine.

This backend implements the :class:`GraphQLBackend` Strategy interface with a
trivial, target-agnostic query, its own simple variable payload, and a permissive
parser. It exists so the orchestrator / clients / circuit-breaker / pagination
machinery can be exercised end-to-end against a local fixture or any public demo
GraphQL endpoint — WITHOUT coupling the engine to any real target's schema.

It expects a generic response shape::

    {
        "data": {
            "search": {
                "results": [
                    {
                        "id": "1",
                        "name": "Example Property",
                        "pageName": "example-property",
                        "city": "Springfield",
                        "countryCode": "US",
                        "reviewScore": 8.4,
                        "reviewCount": 128,
                        "starRating": 4.0,
                        "photos": {"main": "/img/1.jpg"}
                    }
                ]
            }
        }
    }

Anything missing is tolerated (fields are optional in the contract). To point the
demo at a real public GraphQL API, subclass this and override the three methods to
match that API's schema.
"""
from typing import Any, Dict, List, Optional

from ..contracts import HotelDataContract, HotelLocation, HotelReview
from .base import GraphQLBackend


class DemoBackend(GraphQLBackend):
    """A minimal, neutral :class:`GraphQLBackend` implementation for demos/tests."""

    #: Number of results the demo query requests per page (mirrors the orchestrator).
    DEFAULT_PAGE_SIZE = 25

    async def build_search_query(self, region_id: str, offset: int, limit: int) -> str:
        """Return a simple, generic paginated search query.

        The query is intentionally plain GraphQL with no target-specific fragments
        or operation names — it maps 1:1 onto the generic response shape documented
        at the module level.
        """
        return (
            "query DemoSearch($region: String, $offset: Int!, $limit: Int!) {\n"
            "  search(region: $region, offset: $offset, limit: $limit) {\n"
            "    results {\n"
            "      id\n"
            "      name\n"
            "      pageName\n"
            "      city\n"
            "      countryCode\n"
            "      reviewScore\n"
            "      reviewCount\n"
            "      starRating\n"
            "      photos\n"
            "    }\n"
            "  }\n"
            "}"
        )

    def build_variables(self, region_id: str, offset: int, limit: int) -> Dict[str, Any]:
        """Supply the demo query's variables — a flat, obvious payload."""
        return {
            "region": region_id,
            "offset": offset,
            "limit": limit,
        }

    async def parse_response(
        self,
        response: Dict[str, Any],
        min_quality: float = 0.0,
        correlation_id: Optional[str] = None,
    ) -> List[HotelDataContract]:
        """Map the generic response shape to :class:`HotelDataContract` records.

        Applies the same quality-score + threshold-filter contract the orchestrator
        relies on. Malformed rows are skipped rather than raising, so a partial
        response still yields the rows that parsed cleanly.
        """
        data = response.get("data") or {}
        search = data.get("search") or {}
        results = search.get("results") or []

        records: List[HotelDataContract] = []
        for row in results:
            try:
                reviews = None
                if row.get("reviewScore") is not None or row.get("reviewCount") is not None:
                    reviews = HotelReview(
                        score=row.get("reviewScore"),
                        review_count=row.get("reviewCount"),
                    )

                record = HotelDataContract(
                    id=str(row.get("id")),
                    name=row.get("name") or "",
                    page_name=row.get("pageName") or str(row.get("id")),
                    location=HotelLocation(
                        city=row.get("city"),
                        country_code=row.get("countryCode"),
                    ),
                    reviews=reviews,
                    star_rating=row.get("starRating"),
                    photos=row.get("photos"),
                )
                record.quality_score = record.calculate_quality_score()

                if (record.quality_score / 100.0) >= min_quality:
                    records.append(record)
            except Exception:  # noqa: BLE001 — skip malformed rows, keep the rest
                continue

        return records
