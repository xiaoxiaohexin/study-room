"""RAG 知识库路由：文件导入与检索接口。"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_container
from app.core.container import AppContainer
from app.observability.logging import get_logger
from app.schemas.requests import RagSearchRequest
from app.schemas.responses import IngestResponse

log = get_logger(__name__)
router = APIRouter(prefix="/api/rag", tags=["RAG"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    namespace: str = Form("default"),
    container: AppContainer = Depends(get_container),
):
    """上传并导入多格式文档（PDF / Markdown / 代码等）到指定 namespace。"""
    upload_dir = Path(container.settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / (file.filename or "uploaded_doc")

    log.info("uploading_document", filename=file.filename, namespace=namespace)
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_id, chunk_count = await container.ingestion_pipeline.ingest_file(target_path, namespace=namespace)
    return IngestResponse(
        namespace=namespace,
        filename=file.filename or "unknown",
        chunks=chunk_count,
        doc_id=doc_id,
    )


@router.post("/search")
async def search_rag(
    request: RagSearchRequest,
    container: AppContainer = Depends(get_container),
):
    """调试与验证知识库语义检索结果。"""
    queries = [request.query]
    if request.expand:
        exp = await container.query_expander.expand(request.query)
        queries = exp.all_queries

    docs = await container.rag_retriever.search(queries, namespace=request.namespace, top_k=request.top_k)
    return {"query": request.query, "namespace": request.namespace, "count": len(docs), "results": docs}


@router.get("/documents")
async def list_documents(
    namespace: str = "default",
    container: AppContainer = Depends(get_container),
):
    """获取指定命名空间下已导入的所有知识库文档列表与切块统计。"""
    col = container.ingestion_pipeline._collection_name(namespace)
    cursor = await container.sqlite.conn.execute(
        "SELECT id, payload_json, updated_at FROM vector_points WHERE collection = ?",
        (col,),
    )
    rows = await cursor.fetchall()
    import json

    docs_map: dict[str, dict] = {}
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except Exception:
            payload = {}
        doc_id = payload.get("doc_id") or r["id"].split("_")[0]
        filename = payload.get("filename") or "未命名文档"
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": 0,
                "created_at": r["updated_at"],
                "namespace": namespace,
            }
        docs_map[doc_id]["chunks"] += 1
        docs_map[doc_id]["created_at"] = max(docs_map[doc_id]["created_at"], r["updated_at"])

    doc_list = sorted(docs_map.values(), key=lambda d: d["created_at"], reverse=True)
    return {
        "namespace": namespace,
        "total_docs": len(doc_list),
        "total_chunks": sum(d["chunks"] for d in doc_list),
        "documents": doc_list,
    }


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    namespace: str = "default",
    container: AppContainer = Depends(get_container),
):
    """从知识库中删除指定文档及其对应的所有向量切块。"""
    col = container.ingestion_pipeline._collection_name(namespace)
    cursor = await container.sqlite.conn.execute(
        "SELECT id FROM vector_points WHERE collection = ?",
        (col,),
    )
    rows = await cursor.fetchall()
    ids_to_del = [r["id"] for r in rows if r["id"].startswith(f"{doc_id}_") or r["id"] == doc_id]
    if ids_to_del:
        await container.vector_store.delete(col, ids_to_del)
    log.info("document_deleted_from_rag", doc_id=doc_id, chunks_removed=len(ids_to_del), namespace=namespace)
    return {
        "ok": True,
        "doc_id": doc_id,
        "deleted_chunks": len(ids_to_del),
        "namespace": namespace,
    }


@router.get("/stats")
async def get_rag_stats(
    namespace: str = "default",
    container: AppContainer = Depends(get_container),
):
    """获取当前知识库的组件状态、命名空间与向量数。"""
    col = container.ingestion_pipeline._collection_name(namespace)
    cursor = await container.sqlite.conn.execute(
        "SELECT count(*) as cnt FROM vector_points WHERE collection = ?",
        (col,),
    )
    row = await cursor.fetchone()
    total_chunks = row["cnt"] if row else 0
    return {
        "namespace": namespace,
        "collection": col,
        "total_chunks": total_chunks,
        "embedder": container.embedder.tag,
        "embedder_dim": container.embedder.dim,
        "vector_store_status": container.status.get("vector_store", "unknown"),
    }

