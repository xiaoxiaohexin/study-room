"""联网导入 / 临时知识库（Staging）/ 提纯 / 晋升 API。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_container, get_user_id
from app.core.container import AppContainer
from app.observability.logging import get_logger
from app.web.distill import DistillError
from app.web.fetcher import WebFetchError
from app.web.promotion import PromotionError
from app.web.search import WebSearchError
from app.web.staging import WebStagingError

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Web & Staging"])


class WebPreviewRequest(BaseModel):
    url: str = Field(..., min_length=3)


class WebStageRequest(BaseModel):
    url: str = Field(..., min_length=3)
    session_id: str = ""
    conversation_id: str = ""
    query: str = ""
    target_course_id: str = ""


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)
    stage: bool = False
    session_id: str = ""


class PromoteRequest(BaseModel):
    staging_doc_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    target_course_id: str = "default"
    target_namespace: str | None = None


class RevokeRequest(BaseModel):
    staging_doc_id: str = Field(..., min_length=1)


def _require_web(container: AppContainer) -> None:
    if not container.settings.web_enabled:
        raise HTTPException(
            status_code=403,
            detail="联网功能未开启（在 .env 中设置 WEB_ENABLED=true 后重启后端）",
        )


@router.get("/web/status")
async def web_status(container: AppContainer = Depends(get_container)):
    """联网能力状态（前端据此显示/禁用联网面板，并展示搜索源诊断信息）。"""
    settings = container.settings
    return {
        "enabled": bool(settings.web_enabled),
        "search_available": bool(container.web_staging.search_available),
        "search_providers": [
            provider.name for provider in container.web_staging.search_providers
        ],
        "search_primary": settings.web_search_provider,
        "search_fallback": settings.web_search_fallback_provider,
        "search_base_url": settings.web_search_base_url,
        "mcp_configured": bool(container.mcp_web_client.configured),
        "fallback_mode": settings.web_fallback_mode,
        "allowed_domains": settings.web_allowed_domains,
        "max_bytes": settings.web_max_bytes,
        "staging_ttl_days": settings.staging_ttl_days,
    }


@router.post("/web/preview")
async def web_preview(
    req: WebPreviewRequest,
    container: AppContainer = Depends(get_container),
):
    """抓取网页并抽取正文预览（不落库）。"""
    _require_web(container)
    try:
        data = await container.web_staging.preview_url(req.url)
    except WebFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", **data}


@router.post("/web/stage")
async def web_stage(
    req: WebStageRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """抓取网页并写入临时知识库。"""
    _require_web(container)
    try:
        doc = await container.web_staging.stage_url(
            req.url,
            session_id=req.session_id,
            user_id=user_id,
            origin="web_import",
            query=req.query,
            conversation_id=req.conversation_id,
            target_course_id=req.target_course_id,
        )
    except (WebFetchError, WebStagingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "document": doc}


@router.post("/web/search")
async def web_search(
    req: WebSearchRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """联网搜索（可选同时抓取 Top-N 暂存到临时知识库）。"""
    _require_web(container)
    try:
        results = await container.web_staging.search(req.query, top_k=req.top_k)
        payload: dict[str, Any] = {
            "status": "ok",
            "query": req.query,
            "results": [r.__dict__ for r in results],
            "staged": [],
        }
        if req.stage and results:
            staged = await container.web_staging.search_and_stage(
                req.query,
                session_id=req.session_id,
                user_id=user_id,
                top_k=min(req.top_k, 3),
                origin="web_search_import",
            )
            payload["staged"] = staged
        return payload
    except WebSearchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/staging/documents")
async def list_staging_documents(
    session_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """列出临时知识库文档（默认按会话过滤；未传 session_id 时列出当前用户全部）。"""
    docs = await container.web_staging.list_documents(
        session_id=session_id, user_id=user_id, status=status, limit=limit
    )
    return {"status": "ok", "count": len(docs), "documents": docs}


@router.delete("/staging/documents/{doc_id}")
async def delete_staging_document(
    doc_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """删除（丢弃）临时知识库中的文档及其提纯候选。"""
    row = await container.web_staging.get_staging_document(doc_id)
    if not row or (row.get("user_id") or "default_user") != user_id:
        raise HTTPException(status_code=404, detail="临时文档不存在")
    if row.get("status") == "promoted":
        raise HTTPException(status_code=409, detail="该文档已晋升，请先撤销晋升")
    deleted = await container.web_staging.delete_document(doc_id)
    return {"status": "ok", "deleted": deleted, "doc_id": doc_id}


@router.get("/staging/stats")
async def staging_stats(
    session_id: str | None = Query(None),
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return {
        "status": "ok",
        **(
            await container.web_staging.stats(session_id=session_id, user_id=user_id)
        ),
    }


@router.post("/staging/documents/{doc_id}/distill")
async def distill_document(
    doc_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """把临时文档提纯成知识节点候选（规则清洗 + LLM 结构化提炼 + 质量分）。"""
    row = await container.web_staging.get_staging_document(doc_id)
    if not row or (row.get("user_id") or "default_user") != user_id:
        raise HTTPException(status_code=404, detail="临时文档不存在")
    try:
        candidates = await container.web_distiller.distill_document(doc_id, user_id=user_id)
    except DistillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "doc_id": doc_id,
        "quality_score": (await container.web_staging.get_staging_document(doc_id) or {}).get(
            "quality_score", 0
        ),
        "candidates": candidates,
    }


@router.get("/staging/candidates")
async def list_candidates(
    session_id: str | None = Query(None),
    status: str | None = Query("pending"),
    limit: int = Query(100, ge=1, le=500),
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """列出提纯候选（待入库 inbox）。"""
    candidates = await container.web_distiller.list_candidates(
        session_id=session_id, user_id=user_id, status=status, limit=limit
    )
    return {"status": "ok", "count": len(candidates), "candidates": candidates}


@router.post("/staging/candidates/{candidate_id}/reject")
async def reject_candidate(
    candidate_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    row = await container.sqlite.get_staging_candidate(candidate_id)
    if not row or (row.get("user_id") or "default_user") != user_id:
        raise HTTPException(status_code=404, detail="候选不存在")
    rejected = await container.web_distiller.reject_candidate(candidate_id)
    return {"status": "ok", "rejected": rejected, "candidate_id": candidate_id}


@router.post("/staging/promote")
async def promote_staging(
    req: PromoteRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """把临时文档 + 提纯候选晋升到标准知识库（文档 + 知识节点卡片）。"""
    try:
        result = await container.web_promoter.promote(
            staging_doc_ids=req.staging_doc_ids,
            candidate_ids=req.candidate_ids,
            target_course_id=req.target_course_id,
            target_namespace=req.target_namespace,
            user_id=user_id,
        )
    except PromotionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.post("/staging/revoke")
async def revoke_promotion(
    req: RevokeRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """撤销某临时文档的晋升（删除标准文档与由它创建的知识节点）。"""
    row = await container.web_staging.get_staging_document(req.staging_doc_id)
    if not row or (row.get("user_id") or "default_user") != user_id:
        raise HTTPException(status_code=404, detail="临时文档不存在")
    try:
        result = await container.web_promoter.revoke(req.staging_doc_id, user_id=user_id)
    except PromotionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
