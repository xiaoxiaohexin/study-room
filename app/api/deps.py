"""FastAPI 依赖注入辅助函数。"""

from __future__ import annotations

from fastapi import Request

from app.core.config import get_settings
from app.core.container import AppContainer


def get_container(request: Request) -> AppContainer:
    if not hasattr(request.app.state, "container") or request.app.state.container is None:
        # 兼容无 lifespan 启动的测试环境
        container = AppContainer(get_settings())
        request.app.state.container = container
    return request.app.state.container
