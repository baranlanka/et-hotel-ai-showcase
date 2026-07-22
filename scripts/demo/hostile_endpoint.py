# scripts/demo/hostile_endpoint.py
"""Deterministic, OFFLINE hostile HTTP endpoint simulating adaptive anti-bot defenses.

This is a *self-contained* local server (stdlib :mod:`http.server` + threading,
bound to ``127.0.0.1`` on an ephemeral port) used only to exercise the extracted
fetcher engine's resilience components. It talks to nothing on the network, hits
no real target, and contains NO site-specific logic — it returns generic,
GraphQL-shaped JSON over a fictional catalog.

Every decision is a *pure function of the request sequence* (seeded), so two runs
produce byte-identical behaviour regardless of wall-clock timing. That is what
makes the accompanying ``demo_resilience.py`` metrics reproducible.

Simulated defences (POST endpoint, keyed off headers that stand in for a proxy's
identity + browser fingerprint — ``X-Proxy-Id`` and ``User-Agent``):

* **Bot challenge (fingerprint):** a missing / default (non-browser) ``User-Agent``
  or a missing ``X-Proxy-Id`` is rejected ``403``. A plausible browser UA is
  required — the engine gets one from each leased proxy profile.
* **Session challenge (``202`` + token):** the first request from a fresh
  ``X-Proxy-Id`` is answered ``202`` with a one-time ``challenge_token`` that must
  be echoed (header ``X-Challenge``) on the next request from that identity.
* **Per-proxy IP-reputation ban:** after ``k_ban`` *served* pages from one
  ``X-Proxy-Id`` the server returns ``403`` for that id forever — forcing rotation
  to a fresh proxy.
* **Global rate limit (token bucket):** a logical-clock token bucket; when
  exhausted the server returns ``429`` + ``Retry-After``.
* **Transient failures:** a deterministic, low-rate ``500`` keyed off the request
  index (transient — a retry lands on a different index).
* **Success:** a non-banned identity, under rate, with a valid fingerprint and a
  cleared challenge receives a valid ``200`` page of fictional records.

Run standalone (prints its URL and serves until Ctrl-C)::

    PYTHONPATH=. .venv/bin/python scripts/demo/hostile_endpoint.py
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

# User-Agent values treated as "not a real browser" -> bot-challenge 403. The
# engine's HTTP clients default to "resilient-fetcher/1.0"; a real browser UA only
# arrives when a proxy *profile* supplies it, so this asserts the fingerprint path.
_DEFAULT_BAD_UAS = {
    "",
    "resilient-fetcher/1.0",
    "python-httpx/0.27.0",
    "python-requests/2.31.0",
    "curl/8.0.0",
}


def _det_unit(seed: int, tag: str, idx: int) -> float:
    """A deterministic pseudo-random float in ``[0, 1)`` from (seed, tag, idx).

    Uses SHA-256 rather than :func:`hash` so the value is stable across processes
    and Python's per-process hash randomisation. Same inputs -> same output, always.
    """
    digest = hashlib.sha256(f"{seed}:{tag}:{idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _challenge_token(seed: int, proxy_id: str) -> str:
    """Deterministic per-identity session-challenge token."""
    return hashlib.sha256(f"{seed}:challenge:{proxy_id}".encode("utf-8")).hexdigest()[:12]


@dataclass
class HostileConfig:
    """Tunable knobs for :class:`HostileServer` (all deterministic)."""

    total_records: int = 60          # size of the fictional catalog / fetch target
    page_size: int = 5               # records returned per successful page
    k_ban: int = 3                   # served pages per proxy id before an IP-rep ban
    seed: int = 1337                 # seeds every pseudo-random decision
    rl_capacity: float = 3.0         # token-bucket capacity (global rate limit)
    rl_refill_per_tick: float = 0.6  # tokens regained per request reaching the bucket
    retry_after: int = 2             # Retry-After value (nominal seconds) sent with 429
    transient_rate: float = 0.12     # probability a served-eligible request 500s
    enable_challenge: bool = True    # issue the one-time 202 session challenge


class _State:
    """Mutable, resettable server state (guarded by a lock)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.total_requests = 0
        self.status_counts: Dict[int, int] = {200: 0, 202: 0, 403: 0, 429: 0, 500: 0}
        self.served_by_proxy: Dict[str, int] = {}   # X-Proxy-Id -> served (200) pages
        self.banned: set = set()                    # X-Proxy-Id values now banned
        self.challenge_cleared: set = set()         # identities that echoed their token
        self.rl_tokens: float = 0.0                 # current token count (filled on start)
        self.rl_ticks = 0                           # requests that reached the bucket
        self.attempt_seq = 0                        # requests that reached the 500 gate


class HostileServer:
    """A deterministic, offline anti-bot endpoint for resilience demos."""

    def __init__(self, config: Optional[HostileConfig] = None):
        self.config = config or HostileConfig()
        self._state = _State()
        self._state.rl_tokens = self.config.rl_capacity
        self._catalog = self._build_catalog()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> str:
        """Bind an ephemeral 127.0.0.1 port, serve in a daemon thread, return URL."""
        server = self  # capture for the handler closure

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:  # silence stdlib access logs
                return

            def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except (ValueError, UnicodeDecodeError):
                    payload = {}
                variables = (payload.get("variables") or {}) if isinstance(payload, dict) else {}
                status, body, extra_headers = server.decide(
                    proxy_id=self.headers.get("X-Proxy-Id", ""),
                    user_agent=self.headers.get("User-Agent", ""),
                    challenge=self.headers.get("X-Challenge", ""),
                    offset=int(variables.get("offset", 0) or 0),
                    limit=int(variables.get("limit", server.config.page_size) or server.config.page_size),
                )
                encoded = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                for key, value in extra_headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(encoded)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/graphql"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("server not started")
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/graphql"

    def reset(self) -> None:
        """Restore initial deterministic state (used between engine / naive runs)."""
        with self._state.lock:
            self._state.reset()
            self._state.rl_tokens = self.config.rl_capacity

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the server-side counters (per-status + totals)."""
        with self._state.lock:
            return {
                "total_requests": self._state.total_requests,
                "status_counts": dict(self._state.status_counts),
                "banned_ids": len(self._state.banned),
            }

    # ---- catalog -----------------------------------------------------------
    def _build_catalog(self) -> List[Dict[str, Any]]:
        cities = ["Springfield", "Rivertown", "Sampleton", "Testville", "Placeholder"]
        countries = ["US", "GB", "DE", "CA", "FR"]
        catalog: List[Dict[str, Any]] = []
        for i in range(self.config.total_records):
            catalog.append(
                {
                    "id": str(i + 1),
                    "name": f"Demo Property {i + 1}",
                    "pageName": f"demo-property-{i + 1}",
                    "city": cities[i % len(cities)],
                    "countryCode": countries[i % len(countries)],
                    "reviewScore": round(7.5 + (i % 20) * 0.1, 1),
                    "reviewCount": 40 + (i * 7) % 400,
                    "starRating": 3.0 + (i % 5) * 0.5,
                    "photos": {"main": f"/img/{i + 1}.jpg"},
                }
            )
        return catalog

    # ---- decision logic (pure function of request sequence) ----------------
    def decide(
        self,
        proxy_id: str,
        user_agent: str,
        challenge: str,
        offset: int,
        limit: int,
    ) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        """Return ``(status_code, body, extra_headers)`` for one request.

        The order of checks is deliberate: fingerprint -> session challenge ->
        IP-reputation ban -> global rate limit -> transient failure -> success.
        A banned identity therefore always sees ``403`` (never a 429/500), which
        is what strands a non-rotating naive client.
        """
        cfg = self.config
        st = self._state
        with st.lock:
            st.total_requests += 1

            # 1) Fingerprint / bot challenge --------------------------------
            if not proxy_id or user_agent in _DEFAULT_BAD_UAS:
                st.status_counts[403] += 1
                return 403, self._error("Forbidden: automated client fingerprint rejected"), {}

            # 2) One-time session challenge (202 + token to echo) -----------
            if cfg.enable_challenge and proxy_id not in st.challenge_cleared:
                expected = _challenge_token(cfg.seed, proxy_id)
                if challenge and challenge == expected:
                    st.challenge_cleared.add(proxy_id)
                    # fall through and serve this same request normally
                else:
                    st.status_counts[202] += 1
                    body = {
                        "data": None,
                        "challenge_required": True,
                        "challenge_token": expected,
                    }
                    return 202, body, {"X-Challenge-Token": expected}

            # 3) Per-proxy IP-reputation ban --------------------------------
            if proxy_id in st.banned or st.served_by_proxy.get(proxy_id, 0) >= cfg.k_ban:
                st.banned.add(proxy_id)
                st.status_counts[403] += 1
                return 403, self._error("Forbidden: IP reputation banned (rotate identity)"), {}

            # 4) Global rate limit (logical-clock token bucket) -------------
            st.rl_ticks += 1
            st.rl_tokens = min(cfg.rl_capacity, st.rl_tokens + cfg.rl_refill_per_tick)
            if st.rl_tokens < 1.0:
                st.status_counts[429] += 1
                return (
                    429,
                    self._error("Too Many Requests: global rate limit"),
                    {"Retry-After": str(cfg.retry_after)},
                )
            st.rl_tokens -= 1.0

            # 5) Transient failure (deterministic, low rate) ----------------
            st.attempt_seq += 1
            if _det_unit(cfg.seed, "transient", st.attempt_seq) < cfg.transient_rate:
                st.status_counts[500] += 1
                return 500, self._error("Internal Server Error (transient)"), {}

            # 6) Success -> a real page of fictional records ----------------
            st.served_by_proxy[proxy_id] = st.served_by_proxy.get(proxy_id, 0) + 1
            page = self._catalog[offset : offset + max(1, limit)]
            st.status_counts[200] += 1
            body = {"data": {"search": {"results": page}}}
            return 200, body, {}

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        """GraphQL-shaped error envelope."""
        return {"data": None, "errors": [{"message": message}]}


def _main() -> None:
    server = HostileServer()
    url = server.start()
    print(f"[hostile] serving on {url}")
    print(f"[hostile] config: {server.config}")
    print("[hostile] POST a GraphQL body with variables.offset/limit; Ctrl-C to stop.")
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        print("\n[hostile] shutting down")
    finally:
        server.stop()


if __name__ == "__main__":
    _main()
