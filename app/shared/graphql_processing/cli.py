#!/usr/bin/env python3
"""
GraphQL Processing CLI Tool.

Command-line interface for GraphQL query generation, validation, fragment management,
and query testing.

Subcommands:
    generate    - Generate GraphQL query from parameters
    validate    - Validate query syntax and structure
    fragments   - List/manage registered fragments
    test-query  - Test query against mock data

Global Options:
    --output/-o   - Output file path
    --format/-f   - Output format (json, graphql, compact)
    --verbose/-v  - Verbose output with debug info

Usage:
    # Generate query
    python cli.py generate --fields id name location

    # Validate query
    python cli.py validate --query "{ id name }"

    # List fragments
    python cli.py fragments list

    # Test query with mock data
    python cli.py test-query --query "{ id }" --mock-data '{"data": {"id": "123"}}'

Author: ET Hotel AI Team
Version: 1.0.0
"""

import click
import json
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.shared.graphql_processing.query_builder import QueryBuilder
from app.shared.graphql_processing.fragments import FragmentLibrary
from app.shared.graphql_processing.errors import QueryBuildError


@click.group()
@click.version_option(version='1.0.0', prog_name='graphql-cli')
def cli():
    """
    GraphQL Processing CLI Tool.

    Command-line interface for query generation, validation, fragment management,
    and query testing.
    """
    pass


@cli.command()
@click.argument('fields', nargs=-1, required=True)
@click.option('--fragments', '-g', multiple=True, help='Fragment spreads to include (e.g., ...FragmentName)')
@click.option('--variable', '-var', multiple=True, help='Variable definition (format: name:type:value)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'graphql', 'compact']), default='graphql', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output with debug info')
def generate(fields: tuple, fragments: tuple, variable: tuple, output: Optional[str], format: str, verbose: bool):
    """
    Generate GraphQL query from parameters.

    Examples:
        # Basic query
        graphql-cli generate --fields id name location

        # Query with fragments
        graphql-cli generate --fields id --fragments ...HotelBasicInfo

        # Query with variables
        graphql-cli generate --fields id --variable userId:ID!:123

        # Save to file
        graphql-cli generate --fields id name --output query.graphql
    """
    try:
        # Build query
        builder = QueryBuilder()

        # Add fields
        for field in fields:
            if ':' in field:
                # Aliased field: alias:fieldName
                alias, field_name = field.split(':', 1)
                builder.add_field(field_name, alias=alias)
            else:
                builder.add_field(field)

        # Add fragments
        for fragment in fragments:
            if not fragment.startswith('...'):
                fragment = f'...{fragment}'
            builder.add_fragment(fragment)

        # Add variables
        for var in variable:
            parts = var.split(':', 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid variable format: {var}. Expected name:type:value")
            name, type_, value = parts
            builder.add_variable(name, type_, value)

        # Generate query
        query = builder.build()

        # Format output
        if format == 'json':
            output_data = {
                "query": query,
                "variables": {name: builder._variables[name]['value'] for name in builder._variables},
                "metadata": {
                    "field_count": len(fields),
                    "fragment_count": len(fragments),
                    "variable_count": len(variable)
                }
            }
            output_str = json.dumps(output_data, indent=2)
        elif format == 'compact':
            output_str = query.replace('\n', ' ').replace('  ', ' ')
        else:  # graphql
            output_str = query

        # Verbose info
        if verbose:
            click.echo("=" * 50)
            click.echo("GraphQL Query Generator")
            click.echo("=" * 50)
            click.echo(f"Fields: {len(fields)}")
            click.echo(f"Fragments: {len(fragments)}")
            click.echo(f"Variables: {len(variable)}")
            click.echo(f"Query size: {len(query)} bytes")
            click.echo("=" * 50)
            click.echo()

        # Output
        if output:
            Path(output).write_text(output_str)
            click.echo(f"Query saved to: {output}")
        else:
            click.echo(output_str)

    except QueryBuildError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--query', '-q', required=True, help='GraphQL query to validate')
@click.option('--check-fragments', is_flag=True, help='Check for circular fragment dependencies')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output with detailed errors')
def validate(query: str, check_fragments: bool, verbose: bool):
    """
    Validate GraphQL query syntax and structure.

    Examples:
        # Validate query
        graphql-cli validate --query "{ id name location }"

        # Check circular fragments
        graphql-cli validate --query "..." --check-fragments

        # Verbose validation
        graphql-cli validate --query "..." --verbose
    """
    try:
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # Basic syntax validation
        if not query.strip():
            validation_results["valid"] = False
            validation_results["errors"].append("Query is empty")

        # Check for balanced braces
        open_braces = query.count('{')
        close_braces = query.count('}')
        if open_braces != close_braces:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

        # Check for variables without values (warning only)
        if '$' in query and 'query(' in query:
            validation_results["warnings"].append("Query defines variables - ensure values are provided at execution")

        # Check for circular fragments if requested
        if check_fragments and 'fragment' in query.lower():
            # Extract fragment names and check for circular references
            import re
            fragment_pattern = r'fragment\s+(\w+)\s+on\s+\w+\s*\{[^}]*\.\.\.(\w+)'
            fragments_found = re.findall(fragment_pattern, query)

            if fragments_found:
                # Build dependency graph
                deps = {}
                for frag_name, ref_name in fragments_found:
                    if frag_name not in deps:
                        deps[frag_name] = []
                    deps[frag_name].append(ref_name)

                # Simple cycle detection
                def has_cycle(node, visited, stack):
                    visited.add(node)
                    stack.add(node)

                    if node in deps:
                        for neighbor in deps[node]:
                            if neighbor not in visited:
                                if has_cycle(neighbor, visited, stack):
                                    return True
                            elif neighbor in stack:
                                return True

                    stack.remove(node)
                    return False

                visited = set()
                for frag_name in deps:
                    if frag_name not in visited:
                        if has_cycle(frag_name, visited, set()):
                            validation_results["valid"] = False
                            validation_results["errors"].append(f"Circular fragment dependency detected involving: {frag_name}")

        # Display results
        if validation_results["valid"]:
            click.echo("✓ Query is valid")
            if validation_results["warnings"]:
                click.echo("\nWarnings:")
                for warning in validation_results["warnings"]:
                    click.echo(f"  - {warning}")
        else:
            click.echo("✗ Query is invalid", err=True)
            click.echo("\nErrors:")
            for error in validation_results["errors"]:
                click.echo(f"  - {error}", err=True)

            if verbose:
                click.echo("\nQuery:")
                click.echo(query)

            sys.exit(1)

        # Verbose output
        if verbose:
            click.echo("\nValidation Details:")
            click.echo(f"Query length: {len(query)} bytes")
            click.echo(f"Open braces: {open_braces}")
            click.echo(f"Close braces: {close_braces}")
            click.echo(f"Variables: {'Yes' if '$' in query else 'No'}")
            click.echo(f"Fragments: {'Yes' if 'fragment' in query.lower() else 'No'}")

    except Exception as e:
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error during validation: {e}", err=True)
        sys.exit(1)


@cli.group()
def fragments():
    """
    Manage GraphQL fragment library.

    Examples:
        # List all fragments
        graphql-cli fragments list

        # Show specific fragment
        graphql-cli fragments show HotelBasicInfo

        # Register new fragment
        graphql-cli fragments register CustomFragment --type CustomType --fields field1 field2
    """
    pass


@fragments.command('list')
@click.option('--verbose', '-v', is_flag=True, help='Show fragment definitions')
def fragments_list(verbose: bool):
    """List all registered fragments."""
    try:
        library = FragmentLibrary()
        all_fragments = library.list_all()

        if not all_fragments:
            click.echo("No fragments registered")
            return

        click.echo(f"Registered Fragments ({len(all_fragments)}):")
        click.echo("=" * 50)

        for frag_name in sorted(all_fragments):
            click.echo(f"  - {frag_name}")
            if verbose:
                frag_def = library.get(frag_name)
                click.echo(f"    {frag_def}")
                click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fragments.command('show')
@click.argument('name')
def fragments_show(name: str):
    """Show specific fragment definition."""
    try:
        library = FragmentLibrary()
        fragment = library.get(name)

        if fragment:
            click.echo(fragment)
        else:
            click.echo(f"Error: Fragment '{name}' not found", err=True)
            sys.exit(1)

    except KeyError:
        click.echo(f"Error: Fragment '{name}' not found", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fragments.command('register')
@click.argument('name')
@click.option('--type', '-t', 'type_on', required=True, help='GraphQL type name')
@click.option('--fields', '-f', multiple=True, required=True, help='Fragment fields')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def fragments_register(name: str, type_on: str, fields: tuple, verbose: bool):
    """Register new custom fragment."""
    try:
        library = FragmentLibrary()
        library.register(name, type_on, list(fields))

        click.echo(f"✓ Fragment '{name}' registered successfully")

        if verbose:
            fragment = library.get(name)
            click.echo("\nRegistered fragment:")
            click.echo(fragment)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('test-query')
@click.option('--query', '-q', required=True, help='GraphQL query to test')
@click.option('--mock-data', '-m', required=True, help='Mock data as JSON string')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output with detailed results')
def test_query(query: str, mock_data: str, verbose: bool):
    """
    Test GraphQL query against mock data.

    Examples:
        # Test simple query
        graphql-cli test-query --query "{ id }" --mock-data '{"data": {"id": "123"}}'

        # Test with verbose output
        graphql-cli test-query --query "..." --mock-data "..." --verbose
    """
    try:
        # Parse mock data
        try:
            data = json.loads(mock_data)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON in mock data - {e}", err=True)
            sys.exit(1)

        # Basic validation
        if not query.strip():
            click.echo("Error: Query is empty", err=True)
            sys.exit(1)

        # Extract expected fields from query
        import re
        # Simple field extraction (doesn't handle nested queries perfectly)
        fields_in_query = re.findall(r'\b(\w+)\b(?!\s*:)', query)
        fields_in_query = [f for f in fields_in_query if f not in ['query', 'mutation', 'fragment', 'on']]

        # Check if mock data has expected structure
        test_results = {
            "success": True,
            "fields_requested": len(fields_in_query),
            "fields_matched": 0,
            "missing_fields": []
        }

        # Check fields in mock data
        data_fields = data.get('data', {})
        for field in fields_in_query:
            if field in data_fields:
                test_results["fields_matched"] += 1
            else:
                test_results["missing_fields"].append(field)

        # Display results
        if test_results["missing_fields"]:
            click.echo("⚠ Test completed with warnings")
            click.echo(f"\nMissing fields in mock data:")
            for field in test_results["missing_fields"]:
                click.echo(f"  - {field}")
        else:
            click.echo("✓ Test passed successfully")

        # Verbose output
        if verbose:
            click.echo("\nTest Details:")
            click.echo(f"Query length: {len(query)} bytes")
            click.echo(f"Fields requested: {test_results['fields_requested']}")
            click.echo(f"Fields matched: {test_results['fields_matched']}")
            click.echo("\nMock data structure:")
            click.echo(json.dumps(data, indent=2))

    except Exception as e:
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error during test: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
