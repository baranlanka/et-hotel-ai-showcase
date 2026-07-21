# app/shared/graphql_processing/backends/__init__.py
"""GraphQL backends (Strategy pattern).

Only the generic ABC and the neutral demo backend are exported here. Concrete
target-specific backends live outside this public showcase.
"""
from .base import GraphQLBackend
from .demo_backend import DemoBackend

__all__ = ["GraphQLBackend", "DemoBackend"]
