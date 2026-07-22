# scripts/demo/demo_resilience.py
"""Anti-bot resilience PoC: the REAL fetcher engine vs a NAIVE baseline.

This drives the extracted resilience components — the engine's own
:class:`CircuitBreaker`, :class:`SimpleRateLimiter`, :class:`ProxyManager`
rotation over ``data/synthetic/proxies.json``, and its typed error hierarchy
(:class:`RateLimitError`, :class:`BackendBlockedError`, :class:`NetworkError`,
:class:`CircuitBreakerOpenError`) — against the deterministic, offline
:mod:`hostile_endpoint` server, and prints an HONEST, reproducible metrics table.

Nothing here is site-specific and nothing touches the network: the "proxy" and
"browser fingerprint" are represented purely as request HEADERS (``X-Proxy-Id`` +
``User-Agent`` from each leased proxy profile), so no real proxying is needed.
Every number in the table comes from actually running the loop against the server.

The engine loop pages until all target records are collected or an attempt budget
is hit; on ``403``/``429``/``500``/challenge it rotates the proxy, honours
``Retry-After``, applies backoff-with-jitter, and trips/recovers the circuit — all
via the engine's real classes. The naive loop uses one fixed proxy with no
rotation, no client rate limiter, no backoff, and no circuit breaker, so it is
stranded by the first IP-reputation ban.

Determinism: the server is a pure function of the request sequence, and all client
randomness (backoff jitter) is seeded, so records / rotations / 429s / 500s /
challenges / circuit trips are identical across runs (only raw wall-clock varies).
Sleeps are SCALED (``TIME_FACTOR``) so the whole demo finishes in a few seconds.

Exit code: ``0`` iff the ENGINE collected ALL target records AND the NAIVE baseline
did NOT (i.e. resilience is genuinely demonstrated); non-zero otherwise.

Run:
    PYTHONPATH=. .venv/bin/python scripts/demo/demo_resilience.py
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Keep the demo output focused on the metrics table: silence httpx's per-request
# INFO spam; leave the engine's own loggers at ERROR so a circuit-OPEN event
# (which IS a resilience signal worth seeing) still surfaces.
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
for _engine_log in ("graphql-processing", "proxy-manager"):
    logging.getLogger(_engine_log).setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- REAL engine resilience components (imported, never reimplemented) -------
from app.shared.graphql_processing.circuit_breaker import (  # noqa: E402
    CircuitBreaker,
    CircuitBreakerOpenError,
)
from app.shared.graphql_processing.errors import (  # noqa: E402
    BackendBlockedError,
    NetworkError,
    RateLimitError,
)
from app.shared.scraping.proxy_manager import (  # noqa: E402
    InMemoryProxyRepository,
    ProxyManager,
    normalize_domain,
)
from app.shared.scraping.rate_limiter import SimpleRateLimiter  # noqa: E402

from hostile_endpoint import HostileConfig, HostileServer  # noqa: E402

# --- Demo-only tuning (glue lives in the demo, never in the engine) ----------
TIME_FACTOR = 0.02        # scale: 1 nominal second -> 20 ms of real sleep
CB_THRESHOLD = 4          # circuit opens after this many failures
CB_TIMEOUT_S = 0.10       # real seconds the circuit stays open before half-open
CB_RECOVER_MARGIN_S = 0.05
CLIENT_RPM = 12           # client-side SimpleRateLimiter budget...
CLIENT_WINDOW_S = 0.5     # ...over this SCALED window (real attr on the real limiter)
BACKOFF_BASE_S = 0.5      # nominal base for exponential backoff-with-jitter
BACKOFF_CAP_S = 4.0
ENGINE_MAX_ATTEMPTS = 400
NAIVE_MAX_ATTEMPTS = 45


@dataclass
class RunMetrics:
    """Everything reported for one run — all measured, none hardcoded."""

    label: str
    target: int = 0
    records: int = 0
    http_200: int = 0
    http_202: int = 0
    http_403: int = 0
    http_429: int = 0
    http_500: int = 0
    rotations: int = 0
    bans_survived: int = 0
    retry_after_events: int = 0
    retry_after_seconds: int = 0
    transient_retries: int = 0
    challenges_solved: int = 0
    circuit_trips: int = 0
    circuit_recoveries: int = 0
    circuit_blocked: int = 0
    client_throttle_s: float = 0.0
    wall_clock_s: float = 0.0

    @property
    def total_http(self) -> int:
        return self.http_200 + self.http_202 + self.http_403 + self.http_429 + self.http_500

    @property
    def success_rate(self) -> float:
        return (100.0 * self.records / self.target) if self.target else 0.0


# --- thin transport glue: map HTTP status -> the engine's real exceptions ----
# This mirrors exactly how the engine's own httpx/curl clients classify
# responses (429 -> RateLimitError, 403/405 -> BackendBlockedError, other 4xx/5xx
# -> NetworkError). It reuses the engine's exception CLASSES; it does not
# reimplement any resilience primitive.
async def _fetch_page(
    http: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    offset: int,
    limit: int,
) -> Dict[str, Any]:
    payload = {
        "operationName": "DemoSearch",
        "query": "query DemoSearch($offset: Int!, $limit: Int!) { search(offset: $offset, limit: $limit) { results { id } } }",
        "variables": {"region": "demo-region", "offset": offset, "limit": limit},
    }
    resp = await http.post(url, json=payload, headers=headers)
    status = resp.status_code

    if status == 200:
        data = resp.json()
        results = ((data.get("data") or {}).get("search") or {}).get("results") or []
        return {"kind": "ok", "records": results}

    if status == 202:
        token = resp.headers.get("X-Challenge-Token")
        if not token:
            try:
                token = resp.json().get("challenge_token")
            except ValueError:
                token = None
        return {"kind": "challenge", "token": token}

    if status == 429:
        try:
            retry_after = int(resp.headers.get("Retry-After", "1"))
        except (ValueError, TypeError):
            retry_after = 1
        raise RateLimitError(
            f"Rate limit exceeded, retry after {retry_after}s",
            context={"retry_after": retry_after, "status_code": 429},
        )

    if status in (403, 405):
        raise BackendBlockedError(
            f"Access blocked HTTP {status}",
            context={"status_code": status, "response_type": "AccessBlocked"},
        )

    raise NetworkError(f"HTTP {status}", context={"status_code": status})


def _identity_headers(profile: Any, challenge_token: Optional[str]) -> Dict[str, str]:
    """Represent a proxy identity + browser fingerprint as request headers."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # A real browser UA arrives ONLY from the leased proxy profile; the engine
        # default ("resilient-fetcher/1.0") would be rejected by the bot challenge.
        "User-Agent": profile.profile_headers.get("User-Agent", "resilient-fetcher/1.0"),
        "X-Proxy-Id": str(profile.proxy_id),
    }
    for key, value in profile.profile_headers.items():
        headers.setdefault(key, value)
    if challenge_token:
        headers["X-Challenge"] = challenge_token
    return headers


def _backoff_seconds(rng: random.Random, attempt: int) -> float:
    """Exponential backoff with ±20% jitter (nominal seconds; caller scales)."""
    raw = BACKOFF_BASE_S * (2 ** max(0, attempt - 1))
    return min(BACKOFF_CAP_S, raw) * rng.uniform(0.8, 1.2)


async def _lease_unbanned(
    pm: ProxyManager, repo: InMemoryProxyRepository, banned: set
) -> Any:
    """Lease the next LRU proxy that is not in the banned set (engine rotation)."""
    limit = max(1, len(repo._rows) * 3)
    for _ in range(limit):
        profile = await pm.get_next_proxy(repo)
        if profile is None:
            raise RuntimeError("proxy pool is empty")
        if str(profile.proxy_id) not in banned:
            return profile
    raise RuntimeError("all proxies banned — pool exhausted")


def _tripped(state_before: str, breaker: CircuitBreaker) -> bool:
    return state_before != "open" and breaker.state == "open"


def _recovered(state_before: str, breaker: CircuitBreaker) -> bool:
    return state_before != "closed" and breaker.state == "closed"


# ---------------------------------------------------------------------------
# ENGINE run: the real resilience components composed into a paging loop.
# ---------------------------------------------------------------------------
async def run_engine(server: HostileServer) -> RunMetrics:
    cfg = server.config
    url = server.url
    domain = normalize_domain(url)
    target = cfg.total_records

    repo = InMemoryProxyRepository()                 # 5 synthetic proxies from fixture
    pm = ProxyManager()                              # LRU rotation
    limiter = SimpleRateLimiter(requests_per_minute=CLIENT_RPM)
    limiter.window_seconds = CLIENT_WINDOW_S         # configure the real limiter (scaled)
    breaker = CircuitBreaker(
        failure_threshold=CB_THRESHOLD, timeout=CB_TIMEOUT_S, name="hostile-endpoint"
    )
    rng = random.Random(cfg.seed)

    m = RunMetrics(label="ENGINE", target=target)
    collected: Dict[str, Dict[str, Any]] = {}
    banned: set = set()
    challenge_by_id: Dict[str, str] = {}
    offset = 0
    iterations = 0

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=5.0) as http:
        profile = await _lease_unbanned(pm, repo, banned)
        while len(collected) < target and iterations < ENGINE_MAX_ATTEMPTS:
            iterations += 1
            pid = str(profile.proxy_id)

            # Real client-side throttle (paces our own request rate).
            _tw = time.monotonic()
            await limiter.acquire(domain)
            m.client_throttle_s += time.monotonic() - _tw

            headers = _identity_headers(profile, challenge_by_id.get(pid))
            state_before = breaker.state

            try:
                result = await breaker.call(
                    _fetch_page, http, url, headers, offset, cfg.page_size
                )
            except CircuitBreakerOpenError:
                # Circuit open + still cooling down: honour it, wait, retry.
                m.circuit_blocked += 1
                await asyncio.sleep(CB_TIMEOUT_S + CB_RECOVER_MARGIN_S)
                continue
            except RateLimitError as exc:
                m.http_429 += 1
                if _tripped(state_before, breaker):
                    m.circuit_trips += 1
                retry_after = int(exc.context.get("retry_after", cfg.retry_after))
                m.retry_after_events += 1
                m.retry_after_seconds += retry_after
                await asyncio.sleep(retry_after * TIME_FACTOR)   # honour Retry-After
                continue
            except BackendBlockedError:
                m.http_403 += 1
                if _tripped(state_before, breaker):
                    m.circuit_trips += 1
                # IP-reputation ban -> rotate to a fresh proxy identity.
                banned.add(pid)
                m.bans_survived += 1
                profile = await _lease_unbanned(pm, repo, banned)
                m.rotations += 1
                await asyncio.sleep(_backoff_seconds(rng, m.rotations) * TIME_FACTOR)
                continue
            except NetworkError:
                m.http_500 += 1
                if _tripped(state_before, breaker):
                    m.circuit_trips += 1
                m.transient_retries += 1
                await asyncio.sleep(_backoff_seconds(rng, m.transient_retries) * TIME_FACTOR)
                continue

            # ---- successful round-trip (200 or 202) ----
            if _recovered(state_before, breaker):
                m.circuit_recoveries += 1

            if result["kind"] == "challenge":
                m.http_202 += 1
                m.challenges_solved += 1
                if result["token"]:
                    challenge_by_id[pid] = result["token"]
                continue  # retry same offset, now echoing the token

            m.http_200 += 1
            records: List[Dict[str, Any]] = result["records"]
            for row in records:
                rid = str(row.get("id"))
                if rid not in collected:
                    collected[rid] = row
            if not records:
                break  # short/empty page -> catalog exhausted
            offset += len(records)

    m.wall_clock_s = time.monotonic() - t0
    m.records = len(collected)
    return m


# ---------------------------------------------------------------------------
# NAIVE run: one fixed proxy, no rotation / limiter / backoff / circuit breaker.
# ---------------------------------------------------------------------------
async def run_naive(server: HostileServer) -> RunMetrics:
    cfg = server.config
    url = server.url
    target = cfg.total_records

    repo = InMemoryProxyRepository()
    pm = ProxyManager()
    profile = await pm.get_next_proxy(repo)   # ONE identity, chosen once, never rotated
    pid = str(profile.proxy_id)

    m = RunMetrics(label="NAIVE", target=target)
    collected: Dict[str, Dict[str, Any]] = {}
    challenge_token: Optional[str] = None
    offset = 0
    iterations = 0

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=5.0) as http:
        while len(collected) < target and iterations < NAIVE_MAX_ATTEMPTS:
            iterations += 1
            headers = _identity_headers(profile, challenge_token)
            try:
                # No breaker, no limiter, no backoff, no rotation.
                result = await _fetch_page(http, url, headers, offset, cfg.page_size)
            except RateLimitError:
                m.http_429 += 1
                continue  # ignores Retry-After; retries immediately
            except BackendBlockedError:
                m.http_403 += 1
                continue  # cannot rotate; hammers the same banned identity
            except NetworkError:
                m.http_500 += 1
                continue  # no backoff; retries immediately

            if result["kind"] == "challenge":
                m.http_202 += 1
                challenge_token = result["token"]  # basic session echo (not a resilience feature)
                continue

            m.http_200 += 1
            records: List[Dict[str, Any]] = result["records"]
            for row in records:
                rid = str(row.get("id"))
                if rid not in collected:
                    collected[rid] = row
            if not records:
                break
            offset += len(records)

    m.wall_clock_s = time.monotonic() - t0
    m.records = len(collected)
    return m


# ---------------------------------------------------------------------------
# Defence self-check: prove the endpoint's anti-bot behaviours are real.
# ---------------------------------------------------------------------------
async def _defense_probe(server: HostileServer) -> bool:
    url = server.url
    ok = True
    async with httpx.AsyncClient(timeout=5.0) as http:
        # (a) default (non-browser) UA -> bot challenge 403
        r = await http.post(
            url,
            json={"variables": {"offset": 0, "limit": 5}},
            headers={"User-Agent": "resilient-fetcher/1.0", "X-Proxy-Id": "probe-a"},
        )
        ok &= r.status_code == 403
        print(f"  [probe] default UA rejected              -> HTTP {r.status_code} "
              f"({'PASS' if r.status_code == 403 else 'FAIL'})")

        # (b) missing proxy identity -> 403
        r = await http.post(
            url,
            json={"variables": {"offset": 0, "limit": 5}},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        ok &= r.status_code == 403
        print(f"  [probe] missing X-Proxy-Id rejected      -> HTTP {r.status_code} "
              f"({'PASS' if r.status_code == 403 else 'FAIL'})")

        # (c) good fingerprint -> 202 challenge, then echo -> 200
        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "X-Proxy-Id": "probe-c"}
        r = await http.post(url, json={"variables": {"offset": 0, "limit": 5}}, headers=hdrs)
        step1 = r.status_code == 202
        token = r.headers.get("X-Challenge-Token")
        r2 = await http.post(
            url,
            json={"variables": {"offset": 0, "limit": 5}},
            headers={**hdrs, "X-Challenge": token or ""},
        )
        step2 = r2.status_code == 200 and len(r2.json()["data"]["search"]["results"]) > 0
        ok &= step1 and step2
        print(f"  [probe] browser UA -> 202 challenge -> 200 "
              f"-> {'PASS' if step1 and step2 else 'FAIL'} "
              f"(challenge {r.status_code}, echoed {r2.status_code})")
    return ok


# ---------------------------------------------------------------------------
def _print_table(engine: RunMetrics, naive: RunMetrics) -> None:
    def row(label: str, a: str, b: str) -> str:
        return f"  {label:<34}{a:>16}{b:>16}"

    print("\n" + "=" * 66)
    print(row("METRIC", "ENGINE", "NAIVE"))
    print("  " + "-" * 64)
    print(row("records fetched / target",
              f"{engine.records}/{engine.target}", f"{naive.records}/{naive.target}"))
    print(row("success rate", f"{engine.success_rate:.1f}%", f"{naive.success_rate:.1f}%"))
    print(row("total HTTP requests", str(engine.total_http), str(naive.total_http)))
    print(row("  200 / 202 / 403 / 429 / 500",
              f"{engine.http_200}/{engine.http_202}/{engine.http_403}/{engine.http_429}/{engine.http_500}",
              f"{naive.http_200}/{naive.http_202}/{naive.http_403}/{naive.http_429}/{naive.http_500}"))
    print(row("proxy rotations", str(engine.rotations), str(naive.rotations)))
    print(row("403 IP-bans survived", str(engine.bans_survived), str(naive.bans_survived)))
    print(row("429 Retry-After honored (n)", str(engine.retry_after_events), str(naive.retry_after_events)))
    print(row("429 total wait honored (nom. s)",
              f"{engine.retry_after_seconds}", f"{naive.retry_after_seconds}"))
    print(row("transient 500 retries", str(engine.transient_retries), str(naive.transient_retries)))
    print(row("session challenges solved", str(engine.challenges_solved), str(naive.http_202)))
    print(row("circuit trips -> recoveries",
              f"{engine.circuit_trips} -> {engine.circuit_recoveries}",
              f"{naive.circuit_trips} -> {naive.circuit_recoveries}"))
    print(row("client-side throttle wait (s)",
              f"{engine.client_throttle_s:.3f}", f"{naive.client_throttle_s:.3f}"))
    print(row("wall-clock (s)", f"{engine.wall_clock_s:.2f}", f"{naive.wall_clock_s:.2f}"))
    print("=" * 66)


async def _amain() -> int:
    config = HostileConfig()
    server = HostileServer(config)
    url = server.start()
    print("[demo-resilience] anti-bot resilience PoC — REAL engine vs NAIVE baseline")
    print(f"[demo-resilience] hostile endpoint: {url}")
    print(f"[demo-resilience] target={config.total_records} records, page={config.page_size}, "
          f"ban-after={config.k_ban}/proxy, seed={config.seed}\n")

    try:
        print("[1/3] defence self-check (proving the endpoint is genuinely hostile):")
        server.reset()
        probe_ok = await _defense_probe(server)
        if not probe_ok:
            print("[demo-resilience] FAIL: hostile endpoint defences did not behave as specified")
            return 2

        print("\n[2/3] ENGINE run (circuit breaker + rate limiter + proxy rotation + backoff):")
        server.reset()
        engine = await run_engine(server)
        print(f"      collected {engine.records}/{engine.target} in {engine.total_http} HTTP requests, "
              f"{engine.rotations} rotations, circuit {engine.circuit_trips}->{engine.circuit_recoveries}")

        print("\n[3/3] NAIVE run (one fixed proxy, no rotation/backoff/circuit breaker):")
        server.reset()
        naive = await run_naive(server)
        print(f"      collected {naive.records}/{naive.target}; stranded after "
              f"{naive.http_200} served pages by the IP-reputation ban ({naive.http_403} x 403).")

        _print_table(engine, naive)

        engine_won = engine.records == engine.target
        naive_failed = naive.records < naive.target
        print()
        if engine_won and naive_failed:
            print("[demo-resilience] RESULT: PASS — resilience demonstrated "
                  "(engine collected ALL records; naive baseline did NOT).")
        else:
            print("[demo-resilience] RESULT: FAIL — "
                  f"engine_collected_all={engine_won}, naive_failed={naive_failed}")

        # Fast deterministic assertions (double as a CI smoke check).
        assert engine.records == engine.target, "engine did not collect all target records"
        assert naive.records < naive.target, "naive baseline unexpectedly succeeded"
        assert engine.rotations >= 1, "engine performed no proxy rotation"
        assert engine.bans_survived >= 1, "engine survived no IP ban"

        return 0 if (engine_won and naive_failed) else 1
    finally:
        server.stop()


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
