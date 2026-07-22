# 0006 — Ship a generic resilient fetcher, not a site-specific scraper

**Status:** Accepted

## Context
The production platform ingests hotel data from web sources, which required genuinely hard reliability engineering: a rotating proxy pool with failover, circuit breaking, token-bucket rate limiting, backoff-with-jitter, and browser-profile management. Publishing a scraper targeted at a specific commercial site would breach that site's terms of service and reads, to a reviewer, as "evasion tooling" — a reputational and legal liability regardless of who wrote it.

## Decision
Extract and publish only the **generic, reusable engine** as a fault-tolerant distributed HTTP/GraphQL fetcher, demonstrated against a **neutral demo backend** (`make demo-scrape`). Frame it as reliability engineering — proxy failover, circuit breaking, rate-limit compliance, backoff — not evasion. Keep all site-specific backends, queries, and anti-bot specifics out of the repo entirely.

## Alternatives considered
- **Publish the real, site-targeted scraper.** Rejected: ToS breach, DMCA/anti-circumvention gray area, and poor optics for an academic/industry audience.
- **Obfuscate the target.** Rejected: dishonest, and still ships the capability.

## Consequences
- (+) The distributed-systems engineering is demonstrated, ToS-safely and legally, against a target we control; the framing reads as good judgment.
- (−) No flashy site-specific demo. Accepted — the reliability metrics (recovery after a circuit trip, throughput under throttling) carry the story instead.
