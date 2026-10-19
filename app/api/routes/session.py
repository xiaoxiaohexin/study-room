"""会话管理路由：列出与删除用户会话。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.core.container import AppContainer
from app.observability.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.get("/{user_id}")
async def list_user_sessions(user_id: str, container: AppContainer = Depends(get_container)):
    """列出指定用户的历史会话。"""
    sessions = await container.state_manager.list_sessions(user_id=user_id)
    return {"user_id": user_id, "sessions": sessions}


@router.delete("/{user_id}/{session_id}")
async def delete_user_session(
    user_id: str, session_id: str, container: AppContainer = Depends(get_container)
):
    """删除指定的会话及关联的消息历史。"""
    await container.state_manager.delete_session(user_id=user_id, session_id=session_id)
    return {"ok": True, "user_id": user_id, "deleted_session_id": session_id}
