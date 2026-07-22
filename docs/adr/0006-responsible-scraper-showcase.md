# 0006 — Ship a generic resilient fetcher, not a site-specific scraper

**Status:** Accepted

## Context
The production platform ingests hotel data from web sources, which required genuinely hard reliability engineering: a rotating proxy pool with failover, circuit breaking, token-bucket rate limiting, backoff-with-jitter, and browser-profile management. Publishing a scraper targeted at a specific commercial site would breach that site's terms of service and reads, to a reviewer, as "evasion tooling" — a reputational and legal liability regardless of who wrote it.

## Decision
Extract and publish only the **generic, reusable engine** as a fault-tolerant distributed HTTP/GraphQL fetcher, demonstrated against a **neutral demo backend** (`make demo-scrape`) and a **self-hosted hostile endpoint** (`make demo-resilience`) that exercises proxy rotation, circuit breaking, and rate limiting under simulated IP-bans, `429`s, and bot-challenges. Frame it as reliability engineering — proxy failover, circuit breaking, rate-limit compliance, backoff — not evasion. Keep all site-specific backends, queries, and anti-bot specifics out of the repo entirely.

## Alternatives considered
- **Publish the real, site-targeted scraper.** Rejected: ToS breach, DMCA/anti-circumvention gray area, and poor optics for an academic/industry audience.
- **Obfuscate the target.** Rejected: dishonest, and still ships the capability.

## Consequences
- (+) The distributed-systems engineering is **demonstrated runnably** — `make demo-resilience` drives the real engine (proxy rotation, circuit breaker, rate limiter, backoff) through a self-hosted hostile endpoint and fetches 60/60 records where a naïve client manages 15/60, ToS-safely and against a target we control.
- (−) No site-specific demo, and the neutral hostile endpoint is a simplification of any real anti-bot system. Accepted — the resilience is proven (rotation under IP-bans, `Retry-After` compliance, circuit trip→recover) without shipping evasion tooling.
