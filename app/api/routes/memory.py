"""记忆系统路由：查询、新增、整合与遗忘接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_container
from app.core.container import AppContainer
from app.observability.logging import get_logger
from app.schemas.requests import ForgetRequest, MemoryAddRequest

log = get_logger(__name__)
router = APIRouter(prefix="/api/memory", tags=["Memory"])


@router.get("/{user_id}")
async def list_memories(
    user_id: str,
    type: str | None = Query(None, description="working | episodic | semantic | perceptual"),
    limit: int = Query(50, ge=1, le=200),
    container: AppContainer = Depends(get_container),
):
    """查询指定用户的长期/短期记忆列表。"""
    items = await container.memory_manager.list_items(user_id=user_id, type=type, limit=limit)
    return {"user_id": user_id, "count": len(items), "items": items}


@router.post("/{user_id}/add")
async def add_memory(
    user_id: str,
    req: MemoryAddRequest,
    container: AppContainer = Depends(get_container),
):
    """手动向用户的记忆库中添加一条记忆（支持 perceptual 图片路径）。"""
    meta = {"image_path": req.image_path} if req.image_path else {}
    mem_id = await container.memory_manager.add(
        type=req.type,
        content=req.content,
        user_id=user_id,
        session_id=req.session_id,
        importance=req.importance,
        metadata=meta,
    )
    return {"ok": True, "memory_id": mem_id, "user_id": user_id}


@router.post("/{user_id}/consolidate")
async def consolidate_memories(
    user_id: str,
    container: AppContainer = Depends(get_container),
):
    """手动触发用户的记忆强化与沉淀流转（Working -> Episodic -> Semantic）。"""
    res = await container.memory_manager.consolidate(user_id)
    return {"user_id": user_id, "result": res}


@router.post("/{user_id}/forget")
async def forget_memories(
    user_id: str,
    req: ForgetRequest,
    container: AppContainer = Depends(get_container),
):
    """根据指定的遗忘策略清理用户过期或低价值的情景记忆。"""
    removed = await container.memory_manager.forget(
        user_id=user_id,
        strategy=req.strategy,
        threshold=req.threshold,
        max_age_days=req.max_age_days,
        capacity=req.capacity,
    )
    return {"user_id": user_id, "strategy": req.strategy, "removed_count": removed}
