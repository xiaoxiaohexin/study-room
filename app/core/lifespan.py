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
            # 真正的维护：按重要度淘汰过期情景记忆，而不是只打印日志（审计 P1-6）。
            try:
                user_ids = await container.sqlite.list_user_ids()
            except Exception as e:  # noqa: BLE001 - 维护失败不影响服务
                log.warning("memory_maintenance_list_users_failed", error=str(e))
                continue
            for user_id in user_ids:
                try:
                    removed = await container.memory_lifecycle.forget(user_id, strategy="importance")
                    removed += await container.memory_lifecycle.forget(user_id, strategy="capacity")
                    removed += await container.memory_lifecycle.forget(user_id, strategy="age")
                    if removed:
                        log.info("memory_maintenance_pruned", user_id=user_id, removed=removed)
                except Exception as e:  # noqa: BLE001
                    log.warning("memory_maintenance_failed", user_id=user_id, error=str(e))
    except asyncio.CancelledError:
        log.info("background_memory_maintenance_worker_stopped")


async def _background_staging_maintenance(container: AppContainer, interval_sec: int = 3600) -> None:
    """定期清理超过 TTL 的临时知识库文档（不阻塞启动）。"""
    log.info("staging_cleanup_worker_started", interval_sec=interval_sec)
    try:
        # 启动后稍等再首次清理，避免和容器的其它初始化竞争
        await asyncio.sleep(30)
        while True:
            try:
                await container.web_staging.cleanup_expired()
            except Exception as e:  # noqa: BLE001 - 清理失败不影响服务
                log.warning("staging_cleanup_failed", error=str(e))
            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        log.info("staging_cleanup_worker_stopped")


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
    staging_task = asyncio.create_task(_background_staging_maintenance(container))

    log.info("fastapi_startup_ready")
    try:
        yield
    finally:
        log.info("fastapi_shutdown_begun")
        for task in (bg_task, staging_task):
            task.cancel()
        for task in (bg_task, staging_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
        await container.aclose()
        log.info("fastapi_shutdown_completed")
