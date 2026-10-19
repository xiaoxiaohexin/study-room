"""系统健康监测与工具自省路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.core.container import AppContainer
from app.schemas.responses import HealthResponse

router = APIRouter(tags=["Health & Tools"])


@router.get("/health", response_model=HealthResponse)
async def health_check(container: AppContainer = Depends(get_container)):
    """健康监测：检查 SQLite / Qdrant / Neo4j / Redis / LLM 各组件的就绪与降级状态。"""
    overall = "ok"
    for v in container.status.values():
        if v.startswith("degraded") or v.startswith("error"):
            overall = "degraded"
            break
    return HealthResponse(status=overall, components=container.status)


@router.get("/api/tools")
async def list_tools(container: AppContainer = Depends(get_container)):
    """返回当前注册的所有工具列表及其参数 JSON Schema。"""
    specs = container.tool_registry.specs()
    return {"count": len(specs), "tools": [s.model_dump() for s in specs]}
