"""FastAPI 依赖注入辅助函数。"""

from __future__ import annotations

from fastapi import Header, Request

from app.core.config import get_settings
from app.core.container import AppContainer

DEFAULT_USER_ID = "default_user"


def get_container(request: Request) -> AppContainer:
    if not hasattr(request.app.state, "container") or request.app.state.container is None:
        # 兼容无 lifespan 启动的测试环境
        container = AppContainer(get_settings())
        request.app.state.container = container
    return request.app.state.container


def get_user_id(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> str:
    """调用方身份标识（本地单用户默认 default_user）。

    这不是身份认证；它只用于数据层的作用域归属校验，配合可选 API_TOKEN 使用。
    """
    cleaned = (x_user_id or "").strip()
    return cleaned or DEFAULT_USER_ID

