"""轻量 tracing：记录 span 耗时。后续可替换为 OpenTelemetry。"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager

from app.observability.logging import get_logger

log = get_logger("trace")


@contextmanager
def trace_span(name: str, **attrs):
    start = time.perf_counter()
    try:
        yield
    finally:
        log.debug("span", name=name, ms=round((time.perf_counter() - start) * 1000, 1), **attrs)


@asynccontextmanager
async def atrace_span(name: str, **attrs):
    start = time.perf_counter()
    try:
        yield
    finally:
        log.debug("span", name=name, ms=round((time.perf_counter() - start) * 1000, 1), **attrs)
