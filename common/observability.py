"""Showcase shim for ``common.observability``.

In the production platform these decorators wire structured logging and
OpenTelemetry tracing around functions. For this public showcase they are
reduced to transparent no-ops so the extracted code runs standalone without the
observability stack (OTel Collector / Tempo / Loki / Prometheus / Grafana).

They accept both bare (``@trace``) and parametrized (``@trace("name")``) usage,
on sync or async callables.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

__all__ = ["trace", "log_io"]


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


def _flexible(*dargs: Any, **dkwargs: Any) -> Any:
    # Bare usage: @trace
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _passthrough(dargs[0])

    # Parametrized usage: @trace("name", key=value)
    def _decorator(func: Callable) -> Callable:
        return _passthrough(func)

    return _decorator


trace = _flexible
log_io = _flexible
