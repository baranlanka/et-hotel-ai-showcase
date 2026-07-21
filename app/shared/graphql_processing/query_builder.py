"""
GraphQL Query Builder Framework.

Provides extensible, chainable query building with validation, fragment integration,
and variable management.

Key Features:
- Fluent interface (chainable methods)
- Field aliasing and arguments
- Nested field selection with dot notation
- Fragment integration (GraphQL spec compliant)
- Variable management with type validation
- Query validation on build()
- Pretty-print formatted output

Usage:
    >>> builder = QueryBuilder()
    >>> query = (builder
    ...     .add_variable("limit", "Int!", 10)
    ...     .add_field("id")
    ...     .add_field("name", alias="hotelName")
    ...     .add_fragment("...LocationData")
    ...     .add_field("reviews.score")
    ...     .build())
"""

from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
from app.shared.graphql_processing.errors import QueryBuildError


class QueryBuilder:
    """
    Extensible GraphQL query builder with validation.

    Provides chainable methods for constructing GraphQL queries with fields,
    fragments, and variables. Validation occurs in build() method.

    Attributes:
        _fields: List of field specifications (name, alias, arguments, nested)
        _fragments: List of fragment spread strings
        _variables: Dictionary of variable definitions (name -> type, value)
    """

    def __init__(self):
        """Initialize empty query builder."""
        self._fields: List[Dict[str, Any]] = []
        self._fragments: List[str] = []
        self._variables: Dict[str, Dict[str, Any]] = {}

    def add_field(
        self,
        field_name: str,
        alias: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> "QueryBuilder":
        """
        Add a field to the query (chainable).

        Supports:
        - Simple fields: add_field("id")
        - Aliased fields: add_field("name", alias="hotelName")
        - Fields with arguments: add_field("search", arguments={"limit": 10})
        - Nested fields: add_field("user.profile.avatar")

        Args:
            field_name: Field name or nested path (e.g., "user.name")
            alias: Optional field alias for response mapping
            arguments: Optional field arguments (key-value pairs)

        Returns:
            QueryBuilder: Self for method chaining

        Raises:
            QueryBuildError: If field_name is None or empty

        Example:
            >>> builder = QueryBuilder()
            >>> builder.add_field("id").add_field("name", alias="hotelName")
            >>> query = builder.build()
        """
        if field_name is None:
            raise QueryBuildError(
                "Field name cannot be None",
                context={"alias": alias, "arguments": arguments}
            )

        if not field_name or not field_name.strip():
            raise QueryBuildError(
                "Field name cannot be empty",
                context={"field_name": field_name, "alias": alias}
            )

        self._fields.append({
            "name": field_name,
            "alias": alias,
            "arguments": arguments or {}
        })
        return self

    def add_fragment(self, fragment: str) -> "QueryBuilder":
        """
        Add a fragment spread to the query (chainable).

        Fragment must follow GraphQL spec: ...FragmentName

        Args:
            fragment: Fragment spread string (e.g., "...UserFragment")

        Returns:
            QueryBuilder: Self for method chaining

        Example:
            >>> builder = QueryBuilder()
            >>> builder.add_field("id").add_fragment("...LocationData")
            >>> query = builder.build()
        """
        self._fragments.append(fragment)
        return self

    def add_variable(
        self,
        name: str,
        type_: str,
        value: Any
    ) -> "QueryBuilder":
        """
        Add a variable definition to the query (chainable).

        Variables follow GraphQL spec: $variableName: Type!

        Args:
            name: Variable name (without $ prefix)
            type_: GraphQL type (e.g., "ID!", "String", "Int")
            value: Variable value (validated on build())

        Returns:
            QueryBuilder: Self for method chaining

        Example:
            >>> builder = QueryBuilder()
            >>> builder.add_variable("userId", "ID!", "123")
            >>> builder.add_variable("limit", "Int", 10)
            >>> query = builder.build()
        """
        self._variables[name] = {
            "type": type_,
            "value": value
        }
        return self

    def build(self) -> str:
        """
        Generate final query string with validation.

        Performs validation:
        - Duplicate field detection
        - Invalid fragment syntax (must start with '...')
        - Missing required variable values (for non-nullable types)

        Returns:
            str: Formatted GraphQL query string

        Raises:
            QueryBuildError: If validation fails

        Example:
            >>> builder = QueryBuilder()
            >>> builder.add_field("id").add_field("name")
            >>> query = builder.build()
            >>> print(query)
            {
              id
              name
            }
        """
        # Validate before building
        self._validate()

        # Build query components
        query_parts = []

        # Add query operation with variables if present
        if self._variables:
            variables_str = self._build_variables_declaration()
            query_parts.append(f"query({variables_str})")

        # Build query body
        query_body = self._build_query_body()
        query_parts.append(query_body)

        return " ".join(query_parts) if self._variables else query_body

    def _validate(self) -> None:
        """
        Validate query configuration.

        Checks:
        - No duplicate fields
        - Fragment syntax is valid
        - Required variable values are provided

        Raises:
            QueryBuildError: If validation fails
        """
        # Check for duplicate fields
        field_names = [f["name"] for f in self._fields]
        seen = set()
        for name in field_names:
            if name in seen:
                raise QueryBuildError(
                    f"Duplicate field: {name}",
                    context={"fields": field_names}
                )
            seen.add(name)

        # Validate fragment syntax
        for fragment in self._fragments:
            if not fragment.startswith("..."):
                raise QueryBuildError(
                    f"Fragment must start with '...': {fragment}",
                    context={"fragment": fragment}
                )

        # Validate required variables have values
        for var_name, var_data in self._variables.items():
            type_ = var_data["type"]
            value = var_data["value"]

            # Check if variable is required (ends with !)
            if type_.endswith("!") and value is None:
                raise QueryBuildError(
                    f"Variable value not provided for required variable: ${var_name}",
                    context={"variable": var_name, "type": type_}
                )

    def _build_variables_declaration(self) -> str:
        """
        Build variables declaration string.

        Returns:
            str: Variables declaration (e.g., "$userId: ID!, $limit: Int")
        """
        var_parts = []
        for var_name, var_data in self._variables.items():
            var_parts.append(f"${var_name}: {var_data['type']}")
        return ", ".join(var_parts)

    def _build_query_body(self) -> str:
        """
        Build query body with fields and fragments.

        Handles:
        - Simple fields
        - Aliased fields
        - Fields with arguments
        - Nested fields (dot notation)
        - Fragment spreads

        Returns:
            str: Formatted query body with proper indentation
        """
        if not self._fields and not self._fragments:
            return "{\n}"

        # Build nested structure from dot notation fields
        nested_structure = self._build_nested_structure()

        # Render the structure
        lines = self._render_structure(nested_structure, indent_level=0)

        # Add fragments
        if self._fragments:
            for fragment in self._fragments:
                lines.append(f"  {fragment}")

        # Wrap in braces
        result = "{\n" + "\n".join(lines) + "\n}"
        return result

    def _build_nested_structure(self) -> Dict[str, Any]:
        """
        Build nested dictionary structure from flat field list.

        Converts dot notation (e.g., "user.profile.name") into nested structure.

        Returns:
            Dict[str, Any]: Nested structure representing query fields
        """
        structure: Dict[str, Any] = {}

        for field_spec in self._fields:
            field_name = field_spec["name"]
            alias = field_spec["alias"]
            arguments = field_spec["arguments"]

            # Handle nested fields (dot notation)
            if "." in field_name:
                parts = field_name.split(".")
                current = structure

                # Navigate/create nested structure
                for i, part in enumerate(parts[:-1]):
                    if part not in current:
                        current[part] = {"_fields": {}, "_is_object": True}
                    current = current[part]["_fields"]

                # Add final field
                final_field = parts[-1]
                current[final_field] = {
                    "_alias": alias,
                    "_arguments": arguments,
                    "_is_object": False
                }
            else:
                # Simple field
                structure[field_name] = {
                    "_alias": alias,
                    "_arguments": arguments,
                    "_is_object": False
                }

        return structure

    def _render_structure(
        self,
        structure: Dict[str, Any],
        indent_level: int
    ) -> List[str]:
        """
        Render nested structure to formatted GraphQL lines.

        Args:
            structure: Nested field structure
            indent_level: Current indentation level

        Returns:
            List[str]: List of formatted query lines
        """
        lines = []
        indent = "  " * (indent_level + 1)

        for field_name, field_data in structure.items():
            if field_data.get("_is_object"):
                # Nested object
                nested_fields = field_data.get("_fields", {})
                lines.append(f"{indent}{field_name} {{")
                lines.extend(self._render_structure(nested_fields, indent_level + 1))
                lines.append(f"{indent}}}")
            else:
                # Leaf field
                alias = field_data.get("_alias")
                arguments = field_data.get("_arguments", {})

                # Build field string
                field_str = field_name

                # Add arguments if present
                if arguments:
                    args_str = ", ".join(
                        f"{k}: {self._format_argument_value(v)}"
                        for k, v in arguments.items()
                    )
                    field_str = f"{field_name}({args_str})"

                # Add alias if present
                if alias:
                    field_str = f"{alias}: {field_str}"

                lines.append(f"{indent}{field_str}")

        return lines

    def _format_argument_value(self, value: Any) -> str:
        """
        Format argument value for GraphQL syntax.

        Args:
            value: Argument value (str, int, bool, etc.)

        Returns:
            str: Formatted value string
        """
        if isinstance(value, str):
            # Check if already quoted
            if value.startswith('"') and value.endswith('"'):
                return value
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif value is None:
            return "null"
        else:
            return str(value)
