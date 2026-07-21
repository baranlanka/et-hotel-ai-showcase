"""Showcase shim for ``app.core.observability.factory``.

The production ``ObservabilityFactory`` builds a unified logging + tracing +
metrics facade backed by OpenTelemetry. For this showcase it returns a
logging-only facade so extracted modules that do
``_obs = ObservabilityFactory.create_unified(...)`` at import time keep working
offline. Unknown attributes resolve to no-ops so the exact production surface
does not need to be reproduced here.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


class _NoopSpan:
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_event(self, *args: Any, **kwargs: Any) -> None:
        return None

    def end(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoopTracer:
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()

    def start_span(self, *args: Any, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()


class UnifiedObservability:
    def __init__(self, name: str = "showcase") -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        self.tracer = _NoopTracer()
        self.meter = None

    def __getattr__(self, item: str) -> Any:
        # Permissive no-op for any other attribute the production API exposes.
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


class ObservabilityFactory:
    @staticmethod
    def create_unified(
        name: str = "showcase", context: Optional[Any] = None, **kwargs: Any
    ) -> UnifiedObservability:
        return UnifiedObservability(name)

    @classmethod
    def create(cls, name: str = "showcase", **kwargs: Any) -> UnifiedObservability:
        return cls.create_unified(name, **kwargs)
