"""Pydantic models for proxy management and stealth scraping.

This module defines type-safe models for proxy profiles, network configuration,
and geographic metadata used in batch download operations.
"""

from pydantic import BaseModel
from typing import Optional, Dict
from uuid import UUID


class NetworkConfig(BaseModel):
    """Network configuration for proxy.

    Attributes:
        proxy_url: Full proxy URL (http://user:pass@ip:port or http://ip:port)
        proxy_type: Type of proxy (residential, mobile, datacenter)
    """
    proxy_url: str  # http://user:pass@ip:port or http://ip:port
    proxy_type: str  # residential, mobile, datacenter


class GeoMetadata(BaseModel):
    """Geographic metadata for proxy.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code (US, DE, CA, etc.) — optional;
            stored for reference only, not used for proxy selection.
        city: City name (optional)
        timezone: IANA timezone string (America/New_York, Europe/Berlin, optional)
    """
    country_code: Optional[str] = None  # US, DE, CA, etc.
    city: Optional[str] = None
    timezone: Optional[str] = None  # America/New_York, Europe/Berlin


class ProxyProfile(BaseModel):
    """Complete proxy profile for batch download.

    This model contains all information needed to create a configured
    curl_cffi session for stealth scraping operations.

    Attributes:
        proxy_id: UUID of the proxy in the store
        network: Network configuration with proxy URL and type
        geo_metadata: Geographic metadata (stored for reference; not used for proxy selection)
        browser_fingerprint: curl_cffi impersonate string (chrome125, safari17_0, etc.)
        profile_headers: Complete HTTP header dictionary
    """
    proxy_id: UUID
    network: NetworkConfig
    geo_metadata: GeoMetadata
    browser_fingerprint: str  # chrome125, safari17_0, etc.
    profile_headers: Dict[str, str]  # Complete header dict
    session_token: Optional[str] = None  # Ephemeral carrier; never persisted
