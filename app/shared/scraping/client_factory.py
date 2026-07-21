"""Factory for creating curl_cffi sessions with browser fingerprinting.

This module provides a factory for creating stealth-configured curl_cffi
AsyncSession instances with browser impersonation, proxy configuration,
and custom headers for web scraping.
"""

from curl_cffi.requests import AsyncSession
from app.shared.scraping.models import ProxyProfile


class StealthClientFactory:
    """Creates curl_cffi sessions with browser fingerprinting.

    This factory creates AsyncSession instances configured with:
    - Browser impersonation (TLS fingerprinting)
    - Custom HTTP headers
    - Proxy configuration with authentication
    - Timeout and redirect settings
    """

    @staticmethod
    async def create_session(profile: ProxyProfile) -> AsyncSession:
        """Create curl_cffi AsyncSession with stealth configuration.

        Args:
            profile: ProxyProfile with all configuration (network, headers, browser)

        Returns:
            Configured AsyncSession ready for downloading

        Example:
            >>> profile = ProxyProfile(...)
            >>> session = await StealthClientFactory.create_session(profile)
            >>> response = await session.get('https://example.com')
        """
        session = AsyncSession(
            impersonate=profile.browser_fingerprint,  # chrome125, safari17_0, etc.
            headers=profile.profile_headers,          # Complete header dict from DB
            proxies={
                "http": profile.network.proxy_url,
                "https": profile.network.proxy_url
            },
            timeout=30,
            max_redirects=5
        )

        return session
