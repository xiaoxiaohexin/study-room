"""FastAPI Lifespan 上下文管理器：初始化容器、注册生命周期后台任务与优雅退出。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.container import AppContainer
from app.observability.logging import get_logger

log = get_logger(__name__)


async def _background_memory_maintenance(container: AppContainer, interval_sec: int) -> None:
    log.info("background_memory_maintenance_worker_started", interval_sec=interval_sec)
    try:
        while True:
            await asyncio.sleep(interval_sec)
            log.info("triggering_periodic_memory_maintenance")
            # 维护任务：定期清理工作记忆与过期情景记忆
    except asyncio.CancelledError:
        log.info("background_memory_maintenance_worker_stopped")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("fastapi_startup_begun", app_name="agent-backend")

    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    # 启动后台定期任务
    bg_task = asyncio.create_task(
        _background_memory_maintenance(container, settings.lifecycle_interval_seconds)
    )

    log.info("fastapi_startup_ready")
    try:
        yield
    finally:
        log.info("fastapi_shutdown_begun")
        bg_task.cancel()
        try:
            await bg_task
        except asyncio.CancelledError:
            pass
        await container.aclose()
        log.info("fastapi_shutdown_completed")
