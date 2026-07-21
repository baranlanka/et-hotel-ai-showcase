"""Proxy manager for domain-aware proxy rotation and metrics tracking.

This module provides proxy selection and metrics update functionality
for multi-domain fetching operations with per-domain rate limiting
and LRU-based proxy rotation.

In production this reads from a shared proxy store with row-locking so many
workers coordinate. For this showcase, the store is abstracted behind
:class:`ProxyRepositoryProtocol`, and a runnable in-memory fake
(:class:`InMemoryProxyRepository`) backed by ``data/synthetic/proxies.json``
lets proxy selection run standalone with no database.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from urllib.parse import urlparse
from uuid import UUID, uuid4

from app.core.observability.factory import ObservabilityFactory

_obs = ObservabilityFactory.create_unified("proxy-manager", context="activity")
logger = _obs.logger

# Domain-specific rate limits (seconds between requests). These are neutral demo
# hosts — replace with your own target domains.
DOMAIN_RATE_LIMITS = {
    'example.com': 6,          # 10 requests/minute (6-second intervals)
    'api.example.com': 12,     # 5 requests/minute (12-second intervals)
    'demo.example.org': 10,    # 6 requests/minute (10-second intervals)
}


@runtime_checkable
class ProxyRepositoryProtocol(Protocol):
    """Structural interface for the proxy store the manager coordinates against.

    Production ships a database-backed implementation (row-locked LRU selection);
    the showcase ships :class:`InMemoryProxyRepository`. Any object implementing
    these four coroutine methods is accepted.
    """

    async def get_lru_proxy(self) -> Optional[Any]:
        """Return the least-recently-used proxy row (or ``None`` if the pool is empty)."""
        ...

    async def get_domain_lru_proxy(
        self,
        domain: str,
        workflow_id: Optional[str] = None,
        rate_limit_seconds: Optional[int] = None,
    ) -> Optional[Any]:
        """Return the least-recently-used proxy for ``domain`` honouring the rate limit."""
        ...

    async def update_proxy_metrics(
        self, proxy_id: UUID, success: bool, status_code: Optional[int] = None
    ) -> None:
        """Record the outcome of a request that used the given proxy."""
        ...

    async def update_domain_status(
        self,
        proxy_id: UUID,
        domain: str,
        success: bool,
        status_code: Optional[int] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """Record a domain-scoped outcome for the given proxy."""
        ...


class _FakeProxyRow:
    """Lightweight stand-in for a proxy row returned by the store.

    Exposes the same attribute surface (``id``, ``proxy_username``,
    ``proxy_password``, ``ip_address_port``, ``proxy_type``, ``country_code``,
    ``city``, ``timezone``, ``browser_fingerprint``, ``profile_headers``) that
    :class:`ProxyManager` reads when building a :class:`ProxyProfile`.
    """

    def __init__(self, data: Dict[str, Any]):
        raw_id = data.get("proxy_id")
        try:
            self.id: UUID = UUID(str(raw_id)) if raw_id else uuid4()
        except (ValueError, TypeError):
            self.id = uuid4()
        self.proxy_username: Optional[str] = data.get("proxy_username")
        self.proxy_password: Optional[str] = data.get("proxy_password")
        self.ip_address_port: str = data.get("ip_address_port", "")
        self.proxy_type: str = data.get("proxy_type", "datacenter")
        self.country_code: Optional[str] = data.get("country_code")
        self.city: Optional[str] = data.get("city")
        self.timezone: Optional[str] = data.get("timezone")
        self.browser_fingerprint: str = data.get("browser_fingerprint", "chrome125")
        self.profile_headers: Dict[str, str] = data.get("profile_headers", {})


def _default_proxies_path() -> Path:
    """Locate ``data/synthetic/proxies.json`` at the repo root (overridable via env)."""
    override = os.environ.get("SHOWCASE_PROXIES_PATH")
    if override:
        return Path(override)
    # app/shared/scraping/proxy_manager.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "data" / "synthetic" / "proxies.json"


class InMemoryProxyRepository:
    """In-memory :class:`ProxyRepositoryProtocol` fake backed by a JSON fixture.

    Implements a simple least-recently-used rotation (global and per-domain) so the
    proxy-selection logic is fully runnable without CockroachDB. Metrics updates are
    recorded in-process only.
    """

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None):
        if rows is None:
            rows = self._load_rows(_default_proxies_path())
        self._rows: List[_FakeProxyRow] = [_FakeProxyRow(r) for r in rows]
        # LRU bookkeeping: monotonic "last used" ticks, global and per-domain.
        self._last_used: Dict[UUID, float] = {}
        self._domain_last_used: Dict[str, Dict[UUID, float]] = {}

    @staticmethod
    def _load_rows(path: Path) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data = data.get("proxies", [])
            return list(data)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "synthetic proxies fixture unavailable",
                extra={"path": str(path), "error": str(exc)},
            )
            return []

    async def get_lru_proxy(self) -> Optional[_FakeProxyRow]:
        if not self._rows:
            return None
        row = min(self._rows, key=lambda r: self._last_used.get(r.id, 0.0))
        self._last_used[row.id] = time.monotonic()
        return row

    async def get_domain_lru_proxy(
        self,
        domain: str,
        workflow_id: Optional[str] = None,
        rate_limit_seconds: Optional[int] = None,
    ) -> Optional[_FakeProxyRow]:
        if not self._rows:
            return None
        seen = self._domain_last_used.setdefault(domain, {})
        now = time.monotonic()

        # Honour the per-domain rate limit: only offer proxies whose last use for
        # this domain is older than rate_limit_seconds.
        candidates = self._rows
        if rate_limit_seconds:
            eligible = [
                r for r in self._rows
                if (now - seen.get(r.id, 0.0)) >= rate_limit_seconds
            ]
            if eligible:
                candidates = eligible

        row = min(candidates, key=lambda r: seen.get(r.id, 0.0))
        seen[row.id] = now
        return row

    async def update_proxy_metrics(
        self, proxy_id: UUID, success: bool, status_code: Optional[int] = None
    ) -> None:
        logger.debug(
            "proxy metrics updated",
            extra={"proxy_id": str(proxy_id), "success": success, "status_code": status_code},
        )

    async def update_domain_status(
        self,
        proxy_id: UUID,
        domain: str,
        success: bool,
        status_code: Optional[int] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        logger.debug(
            "proxy domain status updated",
            extra={
                "proxy_id": str(proxy_id),
                "domain": domain,
                "success": success,
                "status_code": status_code,
            },
        )


def normalize_domain(url: str) -> str:
    """Normalize URL to domain name for consistent tracking.

    Extracts domain from URL and normalizes it by:
    - Removing www. prefix
    - Removing protocol (https://, http://)
    - Removing port numbers
    - Converting to lowercase

    Args:
        url: URL or domain string

    Returns:
        Normalized domain name

    Examples:
        >>> normalize_domain('https://www.example.com/path')
        'example.com'
        >>> normalize_domain('http://api.example.com:443/v1')
        'api.example.com'
        >>> normalize_domain('EXAMPLE.COM/test')
        'example.com'
    """
    # Remove protocol if present
    if '://' in url:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
    else:
        domain = url

    # Remove port if present
    domain = domain.split(':')[0]

    # Remove www. prefix
    if domain.startswith('www.'):
        domain = domain[4:]

    # Remove path components
    domain = domain.split('/')[0]

    # Convert to lowercase
    return domain.lower()


class ProxyManager:
    """Manages proxy selection and domain-aware metrics tracking.

    Provides LRU-based proxy rotation with per-domain rate limiting,
    success/failure metrics, and plaintext credential handling.
    Proxy seeding is handled externally by a seeder step.
    """

    async def get_next_proxy(
        self, repository: ProxyRepositoryProtocol
    ) -> Optional['ProxyProfile']:
        """Get next proxy using simple LRU (least recently used).

        In production the store's row locking ensures no race conditions across
        workers. Reads plaintext credentials and builds a ProxyProfile.

        Args:
            repository: proxy store implementing :class:`ProxyRepositoryProtocol`

        Returns:
            ProxyProfile or None if no proxies available

        Example:
            >>> manager = ProxyManager()
            >>> repository = InMemoryProxyRepository()
            >>> profile = await manager.get_next_proxy(repository)
            >>> if profile:
            ...     print(profile.network.proxy_type)
        """
        # Simple LRU query - the store handles coordination
        proxy = await repository.get_lru_proxy()

        if not proxy:
            return None

        # Convert to ProxyProfile (Pydantic model)
        from app.shared.scraping.models import ProxyProfile, GeoMetadata, NetworkConfig

        # Use credentials directly (stored as plaintext)
        username = proxy.proxy_username if proxy.proxy_username else None
        password = proxy.proxy_password if proxy.proxy_password else None

        # Build proxy URL
        if username and password:
            proxy_url = f"http://{username}:{password}@{proxy.ip_address_port}"
        else:
            proxy_url = f"http://{proxy.ip_address_port}"

        profile = ProxyProfile(
            proxy_id=proxy.id,
            network=NetworkConfig(
                proxy_url=proxy_url,
                proxy_type=proxy.proxy_type
            ),
            geo_metadata=GeoMetadata(
                country_code=proxy.country_code,
                city=proxy.city,
                timezone=proxy.timezone
            ),
            browser_fingerprint=proxy.browser_fingerprint,
            profile_headers=proxy.profile_headers
        )

        return profile

    async def get_next_proxy_for_domain(
        self,
        repository: ProxyRepositoryProtocol,
        domain: str,
        workflow_id: Optional[str] = None
    ) -> Optional['ProxyProfile']:
        """Get next proxy for specific domain using domain-aware LRU selection.

        This method provides domain-specific proxy management where a proxy that is
        rate-limited or degraded for one domain can still serve other domains.
        Implements automatic rate limiting per domain based on DOMAIN_RATE_LIMITS.

        Args:
            repository: proxy store implementing :class:`ProxyRepositoryProtocol`
            domain: Domain name or URL (will be normalized)
            workflow_id: Optional correlation/workflow id for tracking

        Returns:
            ProxyProfile or None if no proxies available

        Example:
            >>> manager = ProxyManager()
            >>> repository = InMemoryProxyRepository()
            >>> profile = await manager.get_next_proxy_for_domain(
            ...     repository=repository,
            ...     domain='https://api.example.com/v1',
            ...     workflow_id='fetch-123'
            ... )
            >>> if profile:
            ...     print(profile.network.proxy_url)
        """
        # Normalize domain for consistent tracking
        normalized_domain = normalize_domain(domain)

        # Get rate limit for this domain (if configured)
        rate_limit_seconds = DOMAIN_RATE_LIMITS.get(normalized_domain)

        # Domain-aware LRU query with rate limiting
        proxy = await repository.get_domain_lru_proxy(
            domain=normalized_domain,
            workflow_id=workflow_id,
            rate_limit_seconds=rate_limit_seconds
        )

        if not proxy:
            logger.warning(
                "proxy pool empty for domain",
                extra={
                    "domain": normalized_domain,
                    "workflow_id": workflow_id,
                    "rate_limit_seconds": rate_limit_seconds,
                },
            )
            return None

        # Convert to ProxyProfile (Pydantic model)
        from app.shared.scraping.models import ProxyProfile, GeoMetadata, NetworkConfig

        # Use credentials directly (stored as plaintext)
        username = proxy.proxy_username if proxy.proxy_username else None
        password = proxy.proxy_password if proxy.proxy_password else None

        # Build proxy URL
        if username and password:
            proxy_url = f"http://{username}:{password}@{proxy.ip_address_port}"
        else:
            proxy_url = f"http://{proxy.ip_address_port}"

        profile = ProxyProfile(
            proxy_id=proxy.id,
            network=NetworkConfig(
                proxy_url=proxy_url,
                proxy_type=proxy.proxy_type
            ),
            geo_metadata=GeoMetadata(
                country_code=proxy.country_code,
                city=proxy.city,
                timezone=proxy.timezone
            ),
            browser_fingerprint=proxy.browser_fingerprint,
            profile_headers=proxy.profile_headers
        )

        # Lease observability — host:port only, credentials stripped (security).
        # Pairs with the "proxy released" log emitted by update_proxy_metrics_for_domain.
        logger.info(
            "proxy leased",
            extra={
                "proxy_id": str(proxy.id),
                "proxy_host": proxy.ip_address_port,
                "country_code": proxy.country_code,
                "domain": normalized_domain,
                "workflow_id": workflow_id,
            },
        )

        return profile

    async def update_proxy_metrics(
        self,
        proxy_id: UUID,
        success: bool,
        repository: ProxyRepositoryProtocol,
        status_code: Optional[int] = None
    ) -> None:
        """Update proxy metrics after batch completion (legacy, domain-agnostic).

        This method updates global proxy metrics. For domain-specific tracking,
        use update_proxy_metrics_for_domain() instead.

        Args:
            proxy_id: UUID of proxy to update
            success: True if batch succeeded, False if failed
            repository: proxy store implementing :class:`ProxyRepositoryProtocol`
            status_code: HTTP status code (used to detect 429 rate limiting)

        Example:
            >>> manager = ProxyManager()
            >>> await manager.update_proxy_metrics(proxy_id, True, repository)
            >>> await manager.update_proxy_metrics(proxy_id, False, repository, status_code=429)
        """
        await repository.update_proxy_metrics(proxy_id, success, status_code)

    async def update_proxy_metrics_for_domain(
        self,
        proxy_id: UUID,
        domain: str,
        success: bool,
        repository: ProxyRepositoryProtocol,
        status_code: Optional[int] = None,
        workflow_id: Optional[str] = None
    ) -> None:
        """Update domain-specific proxy metrics after request completion.

        This method implements domain-specific status tracking where a proxy
        degraded for one domain can still serve other domains.

        Args:
            proxy_id: UUID of proxy to update
            domain: Domain name or URL (will be normalized)
            success: True if request succeeded, False if failed
            repository: proxy store implementing :class:`ProxyRepositoryProtocol`
            status_code: HTTP status code for error classification
            workflow_id: Optional correlation/workflow id for tracking

        Example:
            >>> manager = ProxyManager()
            >>> await manager.update_proxy_metrics_for_domain(
            ...     proxy_id=proxy_id,
            ...     domain='https://api.example.com/v1',
            ...     success=False,
            ...     repository=repository,
            ...     status_code=403,
            ...     workflow_id='fetch-123'
            ... )
            >>> # Proxy degraded for api.example.com but still usable elsewhere
        """
        # Normalize domain for consistent tracking
        normalized_domain = normalize_domain(domain)

        # Update domain-specific status
        await repository.update_domain_status(
            proxy_id=proxy_id,
            domain=normalized_domain,
            success=success,
            status_code=status_code,
            workflow_id=workflow_id
        )

        # Release observability — pairs with the "proxy leased" log emitted by
        # get_next_proxy_for_domain. Lease duration is computed by the caller
        # when it's known; not tracked here.
        logger.info(
            "proxy released",
            extra={
                "proxy_id": str(proxy_id),
                "domain": normalized_domain,
                "success": success,
                "status_code": status_code,
                "workflow_id": workflow_id,
            },
        )


def _derive_backbone_url(proxy_url: str) -> str:
    """Strip a ``-rotate`` suffix from the username of a rotating-proxy URL.

    Why: many rotating-proxy providers append ``-rotate`` to the username to
    activate per-request IP rotation on a shared gateway. A session-pinned
    (backbone) URL — where every request exits from the same IP — is obtained by
    removing ``-rotate``, switching to backbone mode on the same account without any
    credential change.

    None / type guards are the caller's responsibility: callers should validate the
    input before invoking this helper. This helper has no logging by design.

    Args:
        proxy_url: Full proxy URL string, e.g.
            ``"http://user-rotate:pass@proxy.example.com:80"``.

    Returns:
        The same URL with ``-rotate`` removed from the username component, e.g.
        ``"http://user:pass@proxy.example.com:80"``.

    Raises:
        ValueError: If the username portion does not end with ``-rotate``.
        TypeError: If ``proxy_url`` is not a string (caller responsibility).

    Example:
        >>> _derive_backbone_url("http://alice-rotate:<REDACTED>@proxy.example.com:9999")
        'http://alice:<REDACTED>@proxy.example.com:9999'
    """
    parsed = urlparse(proxy_url)
    username: Optional[str] = parsed.username

    if not username or not username.endswith("-rotate"):
        raise ValueError(
            f"proxy_url username must end with '-rotate' to derive backbone URL; "
            f"got username={username!r}"
        )

    backbone_username = username[: -len("-rotate")]
    password: Optional[str] = parsed.password
    credentials = (
        f"{backbone_username}:{password}" if password else backbone_username
    )
    netloc = f"{credentials}@{parsed.hostname}"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return parsed._replace(netloc=netloc).geturl()


# Lazy singleton pattern
_proxy_manager_instance = None


def get_proxy_manager() -> ProxyManager:
    """Get or create singleton ProxyManager instance.

    Returns:
        ProxyManager: Singleton instance of ProxyManager
    """
    global _proxy_manager_instance
    if _proxy_manager_instance is None:
        _proxy_manager_instance = ProxyManager()
    return _proxy_manager_instance


# Lazy singleton for the in-memory repository (convenience for demos/tests).
_in_memory_proxy_repository = None


def get_in_memory_proxy_repository() -> InMemoryProxyRepository:
    """Get or create a singleton in-memory proxy repository backed by the fixture."""
    global _in_memory_proxy_repository
    if _in_memory_proxy_repository is None:
        _in_memory_proxy_repository = InMemoryProxyRepository()
    return _in_memory_proxy_repository
