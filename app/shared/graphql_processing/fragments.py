"""
GraphQL Fragment Library - Reusable fragment management system.

Provides centralized fragment library with:
- Singleton pattern for global fragment registry
- Fragment registration with validation
- Circular dependency detection
- Pre-registered common fragments (Hotel domain)
- Fragment composition and nesting support

Usage:
    from app.shared.graphql_processing.fragments import FragmentLibrary

    # Get singleton instance
    library = FragmentLibrary()

    # Register custom fragment
    library.register("CustomFragment", "CustomType", ["field1", "field2"])

    # Retrieve fragment definition
    fragment_def = library.get("HotelBasicInfo")

    # Validate fragment (checks circular dependencies)
    is_valid = library.validate("CustomFragment")

    # List all registered fragments
    all_fragments = library.list_all()

Pre-registered Fragments:
    - HotelBasicInfo: Hotel basic fields (id, name, slug)
    - HotelLocation: Hotel location data (city, country, coordinates)
    - HotelReviews: Hotel review data (score, count, distribution)
"""

from typing import List, Dict, Set, Optional
from functools import lru_cache
import re


class FragmentLibrary:
    """
    Singleton fragment library for reusable GraphQL fragments.

    Manages a global registry of GraphQL fragments with validation,
    circular dependency detection, and fragment composition support.

    Singleton Pattern:
        Only one instance exists per application lifecycle.
        All calls to FragmentLibrary() return the same instance.

    Example:
        >>> library = FragmentLibrary()
        >>> library.register("UserInfo", "User", ["id", "name", "email"])
        >>> fragment = library.get("UserInfo")
        >>> print(fragment)
        fragment UserInfo on User {
          id
          name
          email
        }
    """

    _instance: Optional["FragmentLibrary"] = None
    _fragments: Dict[str, tuple[str, List[str]]] = {}
    _protected_fragments: Set[str] = set()

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_pre_registered_fragments()
        return cls._instance

    def _initialize_pre_registered_fragments(self) -> None:
        """Initialize pre-registered common fragments."""
        # Reset fragments on initialization
        self._fragments = {}
        self._protected_fragments = set()

        # Pre-register Hotel domain fragments
        pre_registered = {
            "HotelBasicInfo": ("Hotel", ["id", "name", "slug"]),
            "HotelLocation": ("Hotel", ["city", "country", "coordinates"]),
            "HotelReviews": ("Hotel", ["score", "count", "distribution"]),
        }

        for name, (type_on, fields) in pre_registered.items():
            self._fragments[name] = (type_on, fields)
            self._protected_fragments.add(name)

    def register(self, name: str, type_on: str, fields: List[str]) -> None:
        """
        Register new fragment in library.

        Args:
            name: Fragment name (e.g., "UserBasicInfo")
            type_on: GraphQL type fragment applies to (e.g., "User")
            fields: List of field names to include in fragment

        Raises:
            ValueError: If name already registered, type_on invalid, or fields empty

        Example:
            >>> library = FragmentLibrary()
            >>> library.register("PostInfo", "Post", ["id", "title", "content"])
            >>> library.validate("PostInfo")
            True
        """
        # Validate inputs
        if name in self._fragments:
            if name in self._protected_fragments:
                raise ValueError(
                    f"Fragment '{name}' is a pre-registered protected fragment and cannot be overwritten"
                )
            raise ValueError(f"Fragment '{name}' already registered")

        if type_on is None:
            raise ValueError("type_on cannot be None")

        if not type_on or not type_on.strip():
            raise ValueError("type_on cannot be empty")

        if fields is None:
            raise ValueError("fields cannot be None")

        if not fields:
            raise ValueError("fields cannot be empty")

        # Register fragment
        self._fragments[name] = (type_on, fields)

    @lru_cache(maxsize=128)
    def get(self, name: str) -> str:
        """
        Retrieve formatted fragment definition (cached).

        Args:
            name: Fragment name to retrieve

        Returns:
            str: Formatted GraphQL fragment definition

        Raises:
            KeyError: If fragment not found in library

        Example:
            >>> library = FragmentLibrary()
            >>> fragment = library.get("HotelBasicInfo")
            >>> print(fragment)
            fragment HotelBasicInfo on Hotel {
              id
              name
              slug
            }
        """
        if name not in self._fragments:
            raise KeyError(f"Fragment '{name}' not found in library")

        type_on, fields = self._fragments[name]

        # Format fragment definition
        fields_formatted = "\n  ".join(fields)
        return f"fragment {name} on {type_on} {{\n  {fields_formatted}\n}}"

    def validate(self, name: str) -> bool:
        """
        Validate fragment exists and is well-formed.

        Checks:
        - Fragment exists in library
        - No circular dependencies
        - Valid field syntax

        Args:
            name: Fragment name to validate

        Returns:
            bool: True if fragment is valid

        Raises:
            KeyError: If fragment not found
            ValueError: If circular dependency or invalid syntax detected

        Example:
            >>> library = FragmentLibrary()
            >>> library.register("ValidFragment", "Type", ["field1", "field2"])
            >>> library.validate("ValidFragment")
            True
        """
        if name not in self._fragments:
            raise KeyError(f"Fragment '{name}' not found in library")

        # Check for circular dependencies
        self._check_circular_dependencies(name, visited=set(), path=[])

        # Check field syntax
        type_on, fields = self._fragments[name]
        for field in fields:
            if not self._is_valid_field_syntax(field):
                raise ValueError(f"Invalid field syntax: {field}")

        return True

    def list_all(self) -> List[str]:
        """
        List all registered fragment names.

        Returns:
            List[str]: Sorted list of fragment names

        Example:
            >>> library = FragmentLibrary()
            >>> library.register("CustomFragment", "Type", ["field1"])
            >>> library.list_all()
            ['CustomFragment', 'HotelBasicInfo', 'HotelLocation', 'HotelReviews']
        """
        return sorted(self._fragments.keys())

    def _check_circular_dependencies(
        self, name: str, visited: Set[str], path: List[str]
    ) -> None:
        """
        Check for circular dependencies in fragment references.

        Uses depth-first search to detect cycles in fragment dependency graph.

        Args:
            name: Fragment name to check
            visited: Set of visited fragment names
            path: Current path in dependency graph

        Raises:
            ValueError: If circular dependency detected
        """
        if name in path:
            cycle = " -> ".join(path + [name])
            raise ValueError(f"Circular dependency detected: {cycle}")

        if name not in self._fragments:
            # Reference to external fragment (not in library)
            return

        if name in visited:
            return

        visited.add(name)
        path.append(name)

        # Check all fragment references in fields
        type_on, fields = self._fragments[name]
        for field in fields:
            if field.startswith("..."):
                # Fragment spread reference
                referenced_fragment = field[3:].strip()
                self._check_circular_dependencies(referenced_fragment, visited, path)

        path.pop()

    def _is_valid_field_syntax(self, field: str) -> bool:
        """
        Validate field syntax follows GraphQL spec.

        Accepts:
        - Simple field names: "id", "name"
        - Fragment spreads: "...FragmentName"
        - Nested fields with arguments: "field(arg: value)"

        Args:
            field: Field string to validate

        Returns:
            bool: True if syntax is valid
        """
        # Check for invalid characters that would break GraphQL
        invalid_patterns = [
            r"\{\{\{",  # Triple braces
            r"\}\}\}",  # Triple braces
            r"[^\w\s\.\(\)\:\,\$\!\[\]\{\}]",  # Invalid GraphQL characters
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, field):
                return False

        return True
