# app/shared/graphql_processing/contracts.py
"""
Pydantic data contracts for GraphQL scraping system.

This module defines type-safe data models for configuration and parsed hotel data,
ensuring data integrity across the GraphQL processing pipeline. All models use
Pydantic 2.x with modern ConfigDict patterns.

Architecture:
    - Type safety enforced via Pydantic validation
    - Field aliasing for API response mapping (camelCase → snake_case)
    - Quality scoring to measure data completeness
    - JSON serialization support for Temporal workflow parameters

References:
    - EPIC: docs/research/EPIC_GraphQL_Multi_Backend_Extraction_System.md
    - Story: Story 1.3 from POC_GraphQL_Scraper_Implementation_Plan_V2_Functional.md
    - Pattern: Pydantic 2.x contracts for type-safe data flow

Examples:
    >>> config = ScraperConfig(
    ...     endpoint="https://api.example.com/graphql",
    ...     cookies="session=abc123",
    ...     timeout=30
    ... )
    >>> hotel = HotelDataContract(
    ...     id="12345",
    ...     name="Grand Hotel",
    ...     page_name="grand-hotel",
    ...     location=HotelLocation(city="Springfield", country_code="US")
    ... )
    >>> hotel.calculate_quality_score()
    40.0
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ScraperConfig(BaseModel):
    """
    Configuration contract for GraphQL scraper execution.

    Defines runtime parameters for GraphQL API communication including endpoint,
    timeouts, authentication cookies, and user agent. This contract is passed to
    the orchestrator and client to configure scraping behavior.

    Architecture:
        - Pydantic validation ensures valid URLs and timeout ranges
        - Temporal-safe: serializable to JSON for workflow parameters
        - Secrets (cookies) should be loaded from environment/vault

    Attributes:
        endpoint: GraphQL API endpoint URL
        timeout: HTTP request timeout in seconds (1-300s)
        cookies: Optional authentication cookies as semicolon-separated string
        user_agent: HTTP User-Agent header for API requests

    Examples:
        >>> # Load endpoint/cookies from your own settings/vault, e.g. env vars.
        >>> import os
        >>> config = ScraperConfig(
        ...     endpoint=os.environ["GRAPHQL_ENDPOINT"],
        ...     cookies=os.environ.get("GRAPHQL_COOKIES"),
        ...     timeout=30
        ... )
    """
    endpoint: HttpUrl
    timeout: int = Field(default=30, ge=1, le=300)
    cookies: Optional[str] = None
    user_agent: str = "resilient-fetcher/1.0"
    proxy_url: Optional[str] = None

    model_config = ConfigDict(populate_by_alias=True)


class HotelLocation(BaseModel):
    """
    Hotel location data contract.

    Represents parsed geographic and address information for a hotel property.
    Supports field aliasing to map API response camelCase to Python snake_case.

    Architecture:
        - Field aliases maintain API compatibility (countryCode → country_code)
        - All fields optional to handle partial data gracefully
        - Used as nested model in HotelDataContract

    Attributes:
        address: Street address of the hotel
        city: City name where hotel is located
        country_code: ISO country code (e.g., "US", "TR")
        display_location: Human-readable location string

    Note:
        The `alias` parameter maps API response keys to Python field names.
        API returns "countryCode", Python model uses "country_code".

    Examples:
        >>> location = HotelLocation(
        ...     address="123 Main St",
        ...     city="Istanbul",
        ...     countryCode="TR"  # API key name
        ... )
        >>> location.country_code  # Python attribute name
        'TR'
    """
    address: Optional[str] = None
    city: Optional[str] = None
    country_code: Optional[str] = Field(None, alias="countryCode")
    display_location: Optional[str] = Field(None, alias="displayLocation")

    model_config = ConfigDict(populate_by_alias=True, populate_by_name=True)


class HotelReview(BaseModel):
    """
    Hotel review summary contract.

    Aggregated review data including scores and counts. Used to represent
    customer feedback metrics for a hotel property.

    Architecture:
        - Field aliasing for API compatibility
        - All fields optional for partial data scenarios
        - Nested within HotelDataContract

    Attributes:
        score: Numerical review score (e.g., 8.5 out of 10)
        review_count: Total number of reviews
        score_text: Textual score description (e.g., "Excellent")

    Examples:
        >>> reviews = HotelReview(
        ...     score=8.5,
        ...     reviewCount=1234  # API key
        ... )
        >>> reviews.review_count  # Python attribute
        1234
    """
    score: Optional[float] = None
    review_count: Optional[int] = Field(None, alias="reviewCount")
    score_text: Optional[str] = None

    model_config = ConfigDict(populate_by_alias=True, populate_by_name=True)


class HotelDataContract(BaseModel):
    """
    Complete hotel data contract with quality scoring.

    Primary data model representing a parsed hotel property from GraphQL API responses.
    Includes quality scoring to measure data completeness, supporting data quality
    monitoring and filtering in the ET Hotel AI pipeline.

    Architecture:
        - Type-safe representation of hotel data
        - Quality score (0-100) measures data completeness
        - Used as return type from backend parse_response() methods
        - Serializable to JSON for Temporal workflows and storage

    Attributes:
        id: Unique hotel identifier from source platform
        name: Hotel name
        page_name: URL-friendly slug for hotel page
        ufi: Optional numeric facility/location identifier (target-specific)
        location: Hotel location data (address, city, country)
        reviews: Review summary (score, count)
        star_rating: Star rating (e.g., 4.5)
        photos: Photo data dictionary (URLs)
        quality_score: Data completeness score (0-100)

    Quality Score Calculation:
        - 5 critical fields: name, city, review score, star rating, photos
        - Each present field = 20 points
        - Target: ≥70% for production data quality gates

    References:
        - Story 1.3: Pydantic Contracts (POC_GraphQL_Scraper_Implementation_Plan_V2_Functional.md)
        - EPIC: Quality targets >90% completeness (EPIC_GraphQL_Multi_Backend_Extraction_System.md)

    Examples:
        >>> hotel = HotelDataContract(
        ...     id="12345",
        ...     name="Grand Hotel",
        ...     page_name="grand-hotel",
        ...     location=HotelLocation(city="Istanbul", country_code="TR"),
        ...     reviews=HotelReview(score=8.5, review_count=1000),
        ...     star_rating=4.5,
        ...     photos={"high_res_url": "/img/hotel.jpg"}
        ... )
        >>> hotel.quality_score = hotel.calculate_quality_score()
        >>> hotel.quality_score
        100.0
    """
    id: str
    name: str
    page_name: str
    ufi: Optional[int] = None
    location: HotelLocation
    reviews: Optional[HotelReview] = None
    star_rating: Optional[float] = None
    photos: Optional[dict] = None
    quality_score: float = 0.0

    def calculate_quality_score(self) -> float:
        """
        Calculate data completeness score (0-100).

        Measures the presence of 5 critical hotel data fields and returns a
        percentage score. Used to filter low-quality data and monitor scraping
        effectiveness.

        Scoring Logic:
            - name present: +20 points
            - location.city present: +20 points
            - reviews.score present: +20 points
            - star_rating present: +20 points
            - photos present: +20 points
            - Total: 100 points maximum

        Returns:
            float: Quality score from 0.0 to 100.0

        Note:
            Quality gates in production workflows may reject data below 70%.
            This ensures downstream content generation has sufficient input data.

        Examples:
            >>> hotel = HotelDataContract(
            ...     id="1", name="Hotel", page_name="hotel",
            ...     location=HotelLocation(city="NYC")
            ... )
            >>> hotel.calculate_quality_score()
            40.0  # Only name and city present (2/5 fields)
        """
        fields_present = sum([
            1 if self.name else 0,
            1 if self.location.city else 0,
            1 if self.reviews and self.reviews.score else 0,
            1 if self.star_rating else 0,
            1 if self.photos else 0,
        ])
        return (fields_present / 5.0) * 100.0

    model_config = ConfigDict(populate_by_alias=True)
