"""Demo: exercise the generic GraphQL fetch engine's neutral demo backend.

Runs the Strategy-pattern ``DemoBackend`` end-to-end OFFLINE against a canned,
target-agnostic GraphQL response fixture — no network, no cookies, no real
target. It shows the three backend seams the orchestrator relies on
(build_search_query / build_variables / parse_response) and the quality-scoring
contract on the parsed records.

To point the engine at a real public GraphQL API you would subclass DemoBackend
and override those three methods; this repo ships NO site-specific scraper.

Run:
    PYTHONPATH=. .venv/bin/python scripts/demo/demo_scrape.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# A neutral, generic GraphQL response — the shape DemoBackend documents. Entirely
# fictional; mirrors what a public demo GraphQL endpoint might return.
_FIXTURE = {
    "data": {
        "search": {
            "results": [
                {
                    "id": "1", "name": "Example Harbour Hotel",
                    "pageName": "example-harbour-hotel", "city": "Springfield",
                    "countryCode": "US", "reviewScore": 8.6, "reviewCount": 214,
                    "starRating": 4.0, "photos": {"main": "/img/1.jpg"},
                },
                {
                    "id": "2", "name": "Example Garden Lodge",
                    "pageName": "example-garden-lodge", "city": "Rivertown",
                    "countryCode": "GB", "reviewScore": 9.1, "reviewCount": 87,
                    "starRating": 3.5, "photos": {"main": "/img/2.jpg"},
                },
                {
                    # A deliberately-partial row — parse_response skips/scores it.
                    "id": "3", "name": "Example Budget Rooms", "pageName": "",
                    "city": None, "countryCode": "FR",
                },
            ]
        }
    }
}


async def main() -> None:
    from app.shared.graphql_processing.backends.demo_backend import DemoBackend

    backend = DemoBackend()
    print("[demo-scrape] generic resilient-fetcher engine — neutral DemoBackend\n")

    query = await backend.build_search_query("demo-region", offset=0, limit=25)
    variables = backend.build_variables("demo-region", offset=0, limit=25)
    print("build_search_query() ->")
    print("    " + query.splitlines()[0] + " ...")
    print(f"build_variables()    -> {json.dumps(variables)}\n")

    records = await backend.parse_response(_FIXTURE, min_quality=0.0)
    print(f"parse_response() -> {len(records)} record(s):")
    for r in records:
        print(f"  - {r.name} ({r.location.city or 'n/a'}, "
              f"{r.location.country_code}) "
              f"score={r.reviews.score if r.reviews else 'n/a'} "
              f"quality={r.quality_score:.0f}/100")

    print("\n[demo-scrape] OK — engine parsed a neutral fixture offline "
          "(no network, no target).")


if __name__ == "__main__":
    asyncio.run(main())
