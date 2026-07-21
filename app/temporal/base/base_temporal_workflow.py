"""Showcase shim for ``app.temporal.base.base_temporal_workflow``.

The production ``BaseTemporalWorkflow`` mixes in telemetry setup and operational
alerting (e.g. ops notifications) for every durable workflow. For the showcase
it is a logging-only no-op base, so the outreach workflow imports cleanly and
reads as an architecture artifact without pulling in the alerting stack.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("workflow")


class BaseTemporalWorkflow:
    def setup_telemetry(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_workflow_start(self, *args: Any, **kwargs: Any) -> None:
        logger.info("workflow start")

    def log_workflow_complete(self, *args: Any, **kwargs: Any) -> None:
        logger.info("workflow complete")

    def send_workflow_alert(self, *args: Any, **kwargs: Any) -> None:
        logger.warning("workflow alert: args=%s kwargs=%s", args, kwargs)

    def __getattr__(self, item: str) -> Any:
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop
