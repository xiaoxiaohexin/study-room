"""会话管理路由：创建、获取、更新、列出与删除用户会话。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_container
from app.core.container import AppContainer
from app.observability.logging import get_logger
from app.session.state_manager import SessionState

log = get_logger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    title: str | None = None
    course_id: str | None = None
    rag_namespace: str = "default"


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    course_id: str | None = None
    rag_namespace: str | None = None


@router.post("", response_model=SessionState)
async def create_session(
    req: CreateSessionRequest,
    container: AppContainer = Depends(get_container),
):
    """创建或获取会话，并支持绑定知识域与知识库命名空间。"""
    try:
        return await container.state_manager.get_or_create_session(
            user_id=req.user_id,
            session_id=req.session_id,
            course_id=req.course_id,
            rag_namespace=req.rag_namespace,
            title=req.title,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/{user_id}")
async def list_user_sessions(user_id: str, container: AppContainer = Depends(get_container)):
    """列出指定用户的历史会话。"""
    sessions = await container.state_manager.list_sessions(user_id=user_id)
    return {"user_id": user_id, "sessions": sessions}


@router.get("/{user_id}/{session_id}", response_model=SessionState)
async def get_user_session(
    user_id: str, session_id: str, container: AppContainer = Depends(get_container)
):
    """获取指定会话详情。"""
    session = await container.state_manager.get_session(user_id=user_id, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{user_id}/{session_id}", response_model=SessionState)
async def update_user_session(
    user_id: str,
    session_id: str,
    req: UpdateSessionRequest,
    container: AppContainer = Depends(get_container),
):
    """更新指定会话属性（标题、知识域绑定、知识库命名空间）。"""
    try:
        return await container.state_manager.update_session(
            user_id=user_id,
            session_id=session_id,
            title=req.title,
            course_id=req.course_id,
            rag_namespace=req.rag_namespace,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/{user_id}/{session_id}")
async def delete_user_session(
    user_id: str, session_id: str, container: AppContainer = Depends(get_container)
):
    """删除指定的会话及关联的消息历史。"""
    await container.state_manager.delete_session(user_id=user_id, session_id=session_id)
    return {"ok": True, "user_id": user_id, "deleted_session_id": session_id}
