"""Showcase shim for ``app.core.observability.telemetry_context``.

Production ``inject_telemetry_id`` propagates a correlation id into the tracing
context. Here it is a transparent no-op decorator (bare or parametrized, sync or
async).
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

__all__ = ["inject_telemetry_id"]


def _passthrough(func: Callable) -> Callable:
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _async(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return _async

    @functools.wraps(func)
    def _sync(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _sync


def inject_telemetry_id(*dargs: Any, **dkwargs: Any) -> Any:
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _passthrough(dargs[0])

    def _decorator(func: Callable) -> Callable:
        return _passthrough(func)

    return _decorator
