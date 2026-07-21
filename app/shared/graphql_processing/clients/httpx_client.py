# app/shared/graphql_processing/clients/httpx_client.py
"""
Curl subprocess + httpx transport layer for GraphQL and HTML fetches.

This module provides two transport strategies for making authenticated HTTP
requests to remote endpoints:

1. **httpx async client** — connection-pooled async HTTP transport for GraphQL
   queries (``HTTPXGraphQLClient``).
2. **curl subprocess transport** — kernel-enforced wall-clock timeouts via curl,
   used for both GraphQL POST requests (``ProxiedHTTPXGraphQLClient``) and raw
   HTML GET requests (``fetch_proxied_html``). curl's ``--max-time`` cannot be
   evaded by proxy TLS hangs or drip-feed responses, unlike any Python-level
   timeout.

Architecture:
    - Async context manager for automatic connection lifecycle
    - Connection pooling (20 keepalive, 100 max connections) for httpx path
    - Standard GraphQL request/response format
    - Error detection in GraphQL responses
    - Proxy + cookie support on both transport paths
    - ``fetch_proxied_html`` — module-level async function for authenticated
      GET fetches returning raw HTML; used by sitemap feeder activities

References:
    - EPIC: docs/research/EPIC_GraphQL_Multi_Backend_Extraction_System.md
    - Story: Story 1.1 from POC_GraphQL_Scraper_Implementation_Plan_V2_Functional.md
    - Pattern: Async httpx with connection pooling for concurrent requests

Examples:
    >>> async with HTTPXGraphQLClient(
    ...     endpoint="https://api.example.com/graphql",
    ...     cookies="session=abc123"
    ... ) as client:
    ...     query = "query { hotels { id name } }"
    ...     variables = {"limit": 10}
    ...     response = await client.execute(query, variables)
    ...     print(response["data"]["hotels"])

    >>> html = await fetch_proxied_html(
    ...     url="https://www.example.com/page.html",
    ...     cookies="session=abc123",
    ...     proxy_url="http://proxy.example.com:8080",
    ... )
"""
import asyncio as _asyncio
import httpx
import json as _json
import time as _time
from typing import Dict, Any, Optional, Tuple
import random

from curl_cffi import requests as _curl_cffi_requests
from curl_cffi.requests.errors import RequestsError as _CurlCffiRequestsError

from ..errors import RateLimitError, NetworkError, BackendBlockedError
from common.observability import trace
from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("graphql-processing", context="activity")
logger = _obs.logger


def _truncate_for_log(value: Any, limit: int = 200) -> str:
    """Compact one-line preview of a value for logs / error context.

    Keeps error-context payloads well under Temporal's 2 MB per-payload
    limit when they end up in workflow failure events. Collapses newlines
    so the preview is a single greppable line.
    """
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


# curl exit codes → human-readable error context (module-level for reuse)
_CURL_ERRORS: Dict[int, str] = {
    6: "dns_error",
    7: "connect_error",
    28: "curl_timeout",
    35: "ssl_error",
    52: "empty_reply",
    56: "recv_error",
}


async def fetch_proxied_html(
    url: str,
    cookies: Optional[str],
    proxy_url: Optional[str],
) -> str:
    """Fetch raw HTML from a URL via curl subprocess, optionally through a proxy.

    Why: curl's ``--max-time`` is a kernel-enforced wall-clock deadline that
    cannot be evaded by proxy TLS hangs or drip-feed responses.  Python-level
    timeout mechanisms (asyncio.wait_for, httpx timeout, threading.Timer) all
    fail for CONNECT-proxy TLS hangs.  Mirrors the battle-tested subprocess
    pattern already used by ``ProxiedHTTPXGraphQLClient._curl_post``.

    The HTTP status code is captured via ``-w "\\n%{http_code}"`` appended to
    stdout.  The last line is parsed and stripped before returning the body.
    Non-200 responses raise a descriptive exception rather than returning
    potentially incomplete HTML silently.

    Args:
        url: Fully-qualified HTTP/HTTPS URL to fetch.
        cookies: Raw ``Cookie`` header value (e.g. ``"bkng=abc123; sid=xyz"``).
            Pass ``None`` to omit the Cookie header entirely.
        proxy_url: HTTP/HTTPS proxy URL (e.g. ``"http://host:port"``).
            Pass ``None`` to fetch directly without a proxy.

    Returns:
        Raw HTML body as a string (UTF-8 decoded, trailing status line removed).

    Raises:
        NetworkError: On curl exit code != 0 (DNS failure, connect error,
            timeout, SSL error, etc.) or on non-200 HTTP status codes.

    Example:
        >>> html = await fetch_proxied_html(
        ...     url="https://www.example.com/page.html",
        ...     cookies="session=abc123",
        ...     proxy_url="http://proxy.example.com:8080",
        ... )
        >>> assert "<html" in html
    """
    max_time = 40

    cmd = ["curl", "-s", "-S", "--max-time", str(max_time)]

    if proxy_url is not None:
        cmd.extend(["--proxy", proxy_url])

    if cookies is not None:
        cmd.extend(["-H", f"Cookie: {cookies}"])

    # Append HTTP status code as the last line of stdout for reliable parsing.
    cmd.extend(["-w", "\n%{http_code}"])
    cmd.append(url)

    try:
        proc = await _asyncio.create_subprocess_exec(
            *cmd,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await _asyncio.wait_for(
            proc.communicate(),
            timeout=max_time + 10,
        )
    except _asyncio.TimeoutError:
        raise NetworkError(
            f"curl process killed after {max_time + 10}s for url={_truncate_for_log(url)}",
            context={"timeout_type": "subprocess_kill", "url": url},
        )

    exit_code = proc.returncode if proc.returncode is not None else -1

    if exit_code != 0:
        err_type = _CURL_ERRORS.get(exit_code, f"exit_{exit_code}")
        detail = stderr_bytes.decode("utf-8", errors="replace").strip()[:200] if stderr_bytes else ""
        msg = f"curl {err_type}: {detail}" if detail else f"curl {err_type}"
        raise NetworkError(
            msg,
            context={
                "timeout_type": err_type,
                "exit_code": exit_code,
                "url": _truncate_for_log(url),
            },
        )

    raw = stdout_bytes.decode("utf-8", errors="replace")

    # The last line is the HTTP status code written by -w "\n%{http_code}".
    # Split on the final newline; handle edge cases where the body itself
    # ends with a newline (rsplit limit=1 is robust here).
    if "\n" in raw:
        body, status_line = raw.rsplit("\n", 1)
    else:
        body = raw
        status_line = ""

    try:
        status_code = int(status_line.strip())
    except ValueError:
        # Unexpected format — treat as success and return whatever we got.
        logger.warning(
            "fetch_proxied_html: could not parse HTTP status from curl output "
            f"url={_truncate_for_log(url)} status_line={_truncate_for_log(status_line)}"
        )
        return body

    if status_code != 200:
        body_preview = _truncate_for_log(body, 300)
        logger.warning(
            f"fetch_proxied_html: HTTP {status_code} for url={_truncate_for_log(url)} "
            f"body={body_preview}"
        )
        raise NetworkError(
            f"HTTP {status_code} fetching url={_truncate_for_log(url, 150)}",
            context={
                "status_code": status_code,
                "url": url,
                "body_preview": body_preview,
            },
        )

    return body


class HTTPXGraphQLClient:
    """
    Async GraphQL client using httpx with connection pooling.

    Provides HTTP transport layer for GraphQL queries with async/await support.
    Implements the async context manager protocol for automatic connection cleanup.

    Architecture:
        - Connection pooling for performance (reuse TCP connections)
        - Async/await enables concurrent requests without threading
        - Standard GraphQL POST request format (query + variables + operationName)
        - GraphQL error detection in response payload

    Design Decisions:
        - Uses httpx instead of aiohttp for HTTP/2 support and better async patterns
        - Connection pooling configured for hotel scraping workload (20-100 connections)
        - Context manager ensures connections are properly closed

    References:
        - Story 1.1: GraphQL Client Infrastructure (POC_GraphQL_Scraper_Implementation_Plan_V2_Functional.md)
        - EPIC: HTTP client selection (EPIC_GraphQL_Multi_Backend_Extraction_System.md)

    Examples:
        >>> async with HTTPXGraphQLClient(
        ...     endpoint="https://api.example.com/graphql",
        ...     timeout=30,
        ...     cookies="session_id=abc123"
        ... ) as client:
        ...     query = "query Search($limit: Int!) { search(limit: $limit) { results { id name } } }"
        ...     variables = {"limit": 25}
        ...     result = await client.execute(query, variables, "Search")
        ...     records = result["data"]["search"]["results"]
    """

    def __init__(
        self,
        endpoint: str,
        timeout: int = 30,
        cookies: Optional[str] = None,
        user_agent: str = "resilient-fetcher/1.0"
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.cookies = cookies
        self.user_agent = user_agent
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100
            ),
        )
        logger.info(f"GraphQL client initialized: {self.endpoint}")
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            logger.info("GraphQL client closed")

    @trace(span_name="graphql.http_execute")
    async def execute(
        self,
        query: str,
        variables: Dict[str, Any],
        operation_name: str = "Search"
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query via HTTP POST.

        Sends a GraphQL query to the configured endpoint and returns the parsed JSON
        response. Follows the GraphQL over HTTP specification with query, variables,
        and operationName in the POST body.

        Args:
            query: GraphQL query string (generated by query builders)
            variables: GraphQL variable values (dict matching query $variables)
            operation_name: Name of the GraphQL operation (e.g., "Search")

        Returns:
            Dict[str, Any]: Parsed JSON response from GraphQL endpoint
                Format: {"data": {...}, "errors": [...]} (standard GraphQL response)

        Raises:
            httpx.HTTPError: On network errors or HTTP status errors (4xx, 5xx)
            httpx.TimeoutException: If request exceeds configured timeout

        Note:
            GraphQL errors in the response ({"errors": [...]}) are logged but not
            raised as exceptions. Caller must check response["errors"] if needed.

        Examples:
            >>> query = "query GetHotel($id: ID!) { hotel(id: $id) { name } }"
            >>> variables = {"id": "12345"}
            >>> response = await client.execute(query, variables, "GetHotel")
            >>> if "errors" in response:
            ...     print("GraphQL errors:", response["errors"])
            >>> hotel_name = response["data"]["hotel"]["name"]
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

        if self.cookies:
            headers["Cookie"] = self.cookies

        payload = {
            "operationName": operation_name,
            "query": query,
            "variables": variables,
        }

        logger.info(f"Executing GraphQL query: {operation_name}")
        logger.debug(f"Query length: {len(query)} chars")
        logger.debug(f"Variables: {variables}")

        try:
            # Enforce an absolute overarching timeout to prevent indefinite hangs
            # if httpx connection pooling or read states lose track of the timeout.
            response = await _asyncio.wait_for(
                self._client.post(self.endpoint, json=payload, headers=headers),
                timeout=self.timeout + 2.0
            )

            # Detect rate limit (429 Too Many Requests)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                try:
                    retry_after_seconds = int(retry_after)
                except (ValueError, TypeError):
                    retry_after_seconds = 60

                # Add random jitter (±20%) to prevent thundering herd
                jitter = random.uniform(0.8, 1.2)
                delay = retry_after_seconds * jitter

                logger.warning(
                    f"Rate limit exceeded (429), retry after {delay:.1f}s",
                    extra={
                        "error_type": "rate_limit",
                        "retry_after": retry_after_seconds,
                        "delay_with_jitter": delay,
                        "endpoint": self.endpoint,
                        "operation": operation_name
                    }
                )

                raise RateLimitError(
                    f"Rate limit exceeded, retry after {delay:.1f}s",
                    context={
                        "retry_after": retry_after_seconds,
                        "delay_with_jitter": delay,
                        "endpoint": self.endpoint,
                        "operation": operation_name
                    }
                )

            response.raise_for_status()

            result = response.json()
            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")

            return result

        except (_asyncio.TimeoutError, httpx.TimeoutException) as e:
            logger.error(
                f"Request timeout after {self.timeout}s",
                extra={
                    "error_type": "timeout",
                    "timeout_seconds": self.timeout,
                    "endpoint": self.endpoint,
                    "operation": operation_name
                }
            )
            raise NetworkError(
                f"Request timeout after {self.timeout}s",
                context={
                    "timeout": self.timeout,
                    "endpoint": self.endpoint,
                    "operation": operation_name
                }
            ) from e

        except httpx.HTTPError as e:
            # Re-raise RateLimitError without wrapping
            if isinstance(e, RateLimitError):
                raise

            logger.error(
                f"HTTP error: {e}",
                extra={
                    "error_type": "http_error",
                    "error_message": str(e),
                    "endpoint": self.endpoint,
                    "operation": operation_name
                }
            )
            raise NetworkError(
                f"HTTP error: {e}",
                context={
                    "error": str(e),
                    "endpoint": self.endpoint,
                    "operation": operation_name
                }
            ) from e


class ProxiedHTTPXGraphQLClient(HTTPXGraphQLClient):
    """HTTPXGraphQLClient subclass that routes requests through a rotating proxy.

    Uses ``curl_cffi`` as the transport (libcurl bindings with browser TLS
    impersonation). Each call creates a fresh ``curl_cffi.requests.Session``
    — short lifetimes keep a rotating-proxy provider's IP selection cycling.
    Split timeouts: short ``connect_timeout`` (catches broken proxy IPs fast)
    and longer ``read_timeout`` (allows slow responses). The timeout is
    enforced at libcurl level — a kernel-level wall-clock deadline that a
    Python-level timeout cannot guarantee for CONNECT-proxy TLS hangs.

    Why ``curl_cffi`` (not raw ``curl`` subprocess or native ``httpx``):
    it combines libcurl's hang-proof timeout semantics with a consistent,
    realistic browser TLS fingerprint (``impersonate="chrome120"``) that
    matches the accompanying HTTP headers — so the transport presents a
    coherent client profile rather than a mismatched one.
    """

    # Browser TLS profile used for libcurl's impersonation. Updating this
    # is fine — valid values are documented in
    # ``curl_cffi.requests.session.BrowserType`` (chrome99–chrome120,
    # safari15_3–safari17_0). Keep it pinned rather than "chrome" (alias)
    # so tests and observability can rely on a deterministic fingerprint.
    _IMPERSONATE = "chrome120"

    def __init__(
        self,
        endpoint: str,
        proxy_url: str,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        cookies: Optional[str] = None,
        user_agent: str = "resilient-fetcher/1.0",
        timeout: int = 10,  # legacy — ignored when connect/read are set
        custom_headers: Optional[Dict[str, str]] = None,
        impersonate: Optional[str] = None,
    ):
        super().__init__(endpoint, timeout, cookies, user_agent)
        self.proxy_url = proxy_url
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        # Present a coherent client profile: HTTP headers (Sec-CH-UA, User-Agent,
        # …) that AGREE with the TLS impersonate — both sourced from the leased
        # proxy profile (profile_headers / browser_fingerprint). Defaults preserve
        # prior behavior, so non-proxied callers are unaffected.
        self._custom_headers: Dict[str, str] = custom_headers or {}
        self._impersonate: str = impersonate or self._IMPERSONATE

    async def __aenter__(self):
        logger.info(f"GraphQL client initialized via proxy: {self.endpoint}")
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        logger.info("GraphQL client closed")

    def _curl_post(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Tuple[int, str, Dict[str, str]]:
        """HTTP POST via ``curl_cffi`` — libcurl timeout + browser TLS.

        Method name kept for historical continuity: the prior implementation
        shelled out to ``curl``, and the existing test suite patches this
        method by name. The transport changed; the contract did not.

        Returns ``(http_status_code, response_body_str, response_meta)``.
        ``response_meta["content_type"]`` is populated from the response
        headers; ``response_meta["remote_ip"]`` stays empty because
        ``curl_cffi`` does not surface the proxy-remote IP — downstream
        observability falls back to ``"?"`` when empty.

        Raises ``NetworkError`` for libcurl-level failures (DNS, connect,
        SSL, timeout). The inline classifier maps the error message into
        the same ``timeout_type`` buckets the subprocess version used, so
        alerting/metrics keyed on that field keep working.
        """
        try:
            with _curl_cffi_requests.Session() as session:
                response = session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    proxies={"https": self.proxy_url, "http": self.proxy_url},
                    impersonate=self._impersonate,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
        except _CurlCffiRequestsError as exc:
            # Classify into the same buckets the old curl-exit-code table
            # used. Observability signal (``timeout_type``) stays stable.
            msg = str(exc).lower()
            if "timeout" in msg or "timed out" in msg:
                err_type = "curl_timeout"
            elif "resolve" in msg or "could not resolve" in msg or "dns" in msg:
                err_type = "dns_error"
            elif "connect" in msg:
                err_type = "connect_error"
            elif "ssl" in msg or "tls" in msg or "certificate" in msg:
                err_type = "ssl_error"
            else:
                err_type = "libcurl_error"
            raise NetworkError(
                f"curl_cffi {err_type}: {exc}",
                context={"timeout_type": err_type},
            )

        meta: Dict[str, str] = {
            "remote_ip": "",  # not exposed by curl_cffi
            "content_type": response.headers.get("Content-Type", ""),
        }
        return response.status_code, response.text, meta

    async def execute(
        self,
        query: str,
        variables: Dict[str, Any],
        operation_name: str = "Search",
    ) -> Dict[str, Any]:
        """Execute GraphQL query via curl subprocess in a thread."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.cookies:
            headers["Cookie"] = self.cookies
        # Coherent client profile: fingerprint headers (Sec-CH-UA, a real Chrome
        # User-Agent, sec-fetch-*, …) override the defaults so the HTTP headers
        # agree with the TLS impersonate set in _curl_post.
        if self._custom_headers:
            headers.update(self._custom_headers)

        payload = {
            "operationName": operation_name,
            "query": query,
            "variables": variables,
        }

        start = _time.monotonic()

        try:
            status_code, body, meta = await _asyncio.to_thread(
                self._curl_post, payload, headers
            )
        except NetworkError:
            raise

        elapsed = _time.monotonic() - start
        remote_ip = meta.get("remote_ip") or "?"
        content_type = meta.get("content_type") or "?"

        # --- HTTP status handling ---
        if status_code == 429:
            logger.warning(
                f"PROXY 429 RATE LIMITED via {remote_ip} op={operation_name} "
                f"({elapsed:.1f}s) body={_truncate_for_log(body)}"
            )
            raise RateLimitError(
                "Rate limit exceeded",
                context={
                    "retry_after": 60,
                    "remote_ip": remote_ip,
                    "operation_name": operation_name,
                    "body_preview": _truncate_for_log(body, 200),
                    "content_type": content_type,
                },
            )

        if status_code == 403:
            logger.warning(
                f"PROXY 403 FORBIDDEN via {remote_ip} op={operation_name} "
                f"({elapsed:.1f}s) body={_truncate_for_log(body)} "
                f"- proxy IP likely banned"
            )

        if status_code >= 400:
            # Diagnostic context for triage without replicating locally: proxy IP,
            # the error body, the Content-Type (JSON vs an HTML error page), the op
            # name, and a variable preview (NOT the whole query).
            body_preview = _truncate_for_log(body, 400)
            var_preview = _truncate_for_log(_json.dumps(variables), 200)
            logger.warning(
                f"PROXY HTTP {status_code} via {remote_ip} "
                f"op={operation_name} ct={content_type} "
                f"({elapsed:.1f}s) "
                f"vars={var_preview} body={body_preview}"
            )
            # A 403/405 is an explicit refusal by the upstream, not a transient
            # network error. Classify it as BackendBlockedError (non-retryable) so
            # the caller requeues with a fresh proxy rather than hammering the same
            # IP with in-run retries (which only degrades that IP further).
            if status_code in (403, 405):
                raise BackendBlockedError(
                    f"Access blocked HTTP {status_code} via {remote_ip} op={operation_name}",
                    context={
                        "operation_name": operation_name,
                        "record_id": "",
                        "status_code": status_code,
                        "api_message": f"Upstream refused request (HTTP {status_code})",
                        "body_preview": body_preview,
                        "response_type": "AccessBlocked",
                    },
                )
            raise NetworkError(
                f"HTTP {status_code} via {remote_ip} op={operation_name}",
                context={
                    "status_code": status_code,
                    "elapsed": elapsed,
                    "remote_ip": remote_ip,
                    "operation_name": operation_name,
                    "content_type": content_type,
                    "body_preview": body_preview,
                    "vars_preview": var_preview,
                },
            )

        # --- Parse JSON response ---
        try:
            result = _json.loads(body)
        except (_json.JSONDecodeError, ValueError) as e:
            raise NetworkError(
                f"Invalid JSON from proxy via {remote_ip}: {e}",
                context={
                    "timeout_type": "json_error",
                    "elapsed": elapsed,
                    "remote_ip": remote_ip,
                    "content_type": content_type,
                    "body_preview": _truncate_for_log(body, 400),
                },
            ) from e

        logger.debug(f"PROXY OK via {remote_ip} op={operation_name} ({elapsed:.1f}s)")
        if "errors" in result:
            logger.error(
                f"GraphQL errors via {remote_ip} op={operation_name}: {result['errors']}"
            )
        return result
