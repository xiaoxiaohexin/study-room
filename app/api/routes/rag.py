"""RAG 知识库路由：文件导入、检索归因、文档注册表与物理页原文预览接口。"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_container
from app.core.container import AppContainer
from app.observability.logging import get_logger
from app.schemas.requests import RagAttributeRequest, RagSearchRequest
from app.schemas.responses import IngestResponse

log = get_logger(__name__)
router = APIRouter(tags=["RAG"])

_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv; charset=utf-8",
}

_NORMALIZE_RE = re.compile(
    r"[\s\u3000，。！？、；：,.!?;:·\"'“”‘’()（）\[\]【】<>《》\-—_/\\|]+"
)


def _safe_upload_path(upload_dir: Path, filename: str | None) -> tuple[Path, str]:
    """Return (deduplicated target path, original basename) without path traversal."""
    original = Path(filename or "uploaded_doc").name.strip() or "uploaded_doc"
    target = upload_dir / f"{uuid.uuid4().hex[:8]}_{original}"
    return target, original


def _within(base: Path, candidate: Path) -> bool:
    try:
        return candidate.resolve().is_relative_to(base.resolve())
    except (OSError, ValueError):
        return False


async def _get_document_or_404(container: AppContainer, doc_id: str, namespace: str) -> dict:
    doc = await container.sqlite.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    if namespace and doc.get("namespace") and doc["namespace"] != namespace:
        # Do not leak the existence of a document from another knowledge base.
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc


def _document_file_path(container: AppContainer, doc: dict) -> Path:
    raw = doc.get("file_path") or ""
    data_dir = Path(container.settings.data_dir)
    if not raw:
        raise HTTPException(status_code=404, detail="Original file is not available for this document")
    candidate = Path(raw)
    if not _within(data_dir, candidate) or not candidate.exists():
        raise HTTPException(status_code=404, detail="Original file is not available for this document")
    return candidate


def _trigrams(text: str) -> set[str]:
    normalized = _NORMALIZE_RE.sub("", text or "").lower()
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)}


def _quote_containment(quote: str, content: str) -> float:
    q_grams = _trigrams(quote)
    if not q_grams:
        return 0.0
    c_grams = _trigrams(content)
    return len(q_grams & c_grams) / len(q_grams)


@router.post("/api/rag/ingest", response_model=IngestResponse)
@router.post("/api/v1/rag/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    namespace: str = Form("default"),
    container: AppContainer = Depends(get_container),
):
    """上传并导入多格式文档（PDF / Markdown / 代码等）到指定 namespace。"""
    upload_dir = Path(container.settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path, original_name = _safe_upload_path(upload_dir, file.filename)

    log.info("uploading_document", filename=original_name, namespace=namespace)
    with target_path.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    doc_id, chunk_count = await container.ingestion_pipeline.ingest_file(
        target_path, namespace=namespace, original_filename=original_name
    )
    return IngestResponse(
        namespace=namespace,
        filename=original_name,
        chunks=chunk_count,
        doc_id=doc_id,
    )


@router.post("/api/rag/search")
@router.post("/api/v1/rag/search")
async def search_rag(
    request: RagSearchRequest,
    container: AppContainer = Depends(get_container),
):
    """知识库语义检索接口，支持单查询/多查询融合检索与真实物理页码返回。"""
    query_text = request.query.strip()
    if not query_text:
        return {"query": request.query, "namespace": request.namespace, "count": 0, "results": []}

    queries = [query_text]
    if request.expand:
        exp = await container.query_expander.expand(query_text)
        queries = exp.all_queries

    docs = await container.rag_retriever.search(queries, namespace=request.namespace, top_k=request.top_k)
    return {"query": request.query, "namespace": request.namespace, "count": len(docs), "results": docs}


@router.post("/api/rag/attribute")
@router.post("/api/v1/rag/attribute")
async def attribute_quote(
    request: RagAttributeRequest,
    container: AppContainer = Depends(get_container),
):
    """将一段划词/摘录归因到知识库中的原文切片与物理页码。

    只有字符三元组包含度达到阈值时才返回 ``matched=True``，避免伪造出处。
    """
    quote = request.quote.strip()
    if len(quote) < 2:
        return {"matched": False, "overlap": 0.0, "reason": "quote_too_short"}

    docs = await container.rag_retriever.search(
        [quote], namespace=request.namespace, top_k=request.top_k
    )
    threshold = (
        request.min_overlap
        if request.min_overlap is not None
        else container.settings.rag_attribute_min_overlap
    )

    best_doc = None
    best_overlap = 0.0
    for doc in docs:
        overlap = _quote_containment(quote, doc.content)
        if overlap > best_overlap:
            best_doc, best_overlap = doc, overlap

    if best_doc is None or best_overlap < threshold:
        return {
            "matched": False,
            "overlap": round(best_overlap, 4),
            "threshold": threshold,
            "reason": "no_confident_match",
            "candidates_count": len(docs),
        }

    return {
        "matched": True,
        "overlap": round(best_overlap, 4),
        "threshold": threshold,
        "chunk": {
            "doc_id": best_doc.doc_id,
            "chunk_id": best_doc.id,
            "chunk_index": best_doc.chunk_index,
            "filename": best_doc.filename,
            "page_start": best_doc.page_start,
            "page_end": best_doc.page_end,
            "page_label": best_doc.page_label,
            "headers": best_doc.headers,
            "content": best_doc.content[:600],
        },
    }


@router.get("/api/rag/documents")
@router.get("/api/v1/rag/documents")
async def list_documents(
    namespace: str = "default",
    container: AppContainer = Depends(get_container),
):
    """获取指定命名空间下已导入的文档列表（优先读注册表，兼容旧数据扫描）。"""
    registered = await container.sqlite.list_documents(namespace)
    if registered:
        documents = [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "chunks": int(d.get("chunk_count") or 0),
                "page_count": int(d.get("page_count") or 1),
                "file_available": bool(d.get("file_path")) and Path(d["file_path"]).exists(),
                "created_at": d["created_at"],
                "namespace": d["namespace"],
            }
            for d in registered
        ]
        return {
            "namespace": namespace,
            "total_docs": len(documents),
            "total_chunks": sum(d["chunks"] for d in documents),
            "documents": documents,
        }

    # Legacy fallback: scan persisted vector payloads directly.
    col = container.ingestion_pipeline._collection_name(namespace)
    cursor = await container.sqlite.conn.execute(
        "SELECT id, payload_json, updated_at FROM vector_points WHERE collection = ?",
        (col,),
    )
    rows = await cursor.fetchall()
    docs_map: dict[str, dict] = {}
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        doc_id = payload.get("doc_id") or r["id"].split("_")[0]
        filename = payload.get("filename") or "未命名文档"
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": 0,
                "page_count": 1,
                "file_available": False,
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


@router.get("/api/rag/documents/{doc_id}/file")
@router.get("/api/v1/rag/documents/{doc_id}/file")
async def get_document_file(
    doc_id: str,
    namespace: str = "default",
    container: AppContainer = Depends(get_container),
):
    """返回已导入文档的原始文件字节（供前端 PDF 页渲染）。"""
    doc = await _get_document_or_404(container, doc_id, namespace)
    path = _document_file_path(container, doc)
    media_type = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=doc.get("filename") or path.name,
    )


@router.get("/api/rag/documents/{doc_id}/page")
@router.get("/api/v1/rag/documents/{doc_id}/page")
async def get_document_page(
    doc_id: str,
    page: int = 1,
    namespace: str = "default",
    container: AppContainer = Depends(get_container),
):
    """返回文档指定物理页的纯文本（非 PDF 文档返回分页文本）。"""
    doc = await _get_document_or_404(container, doc_id, namespace)
    path = _document_file_path(container, doc)
    try:
        text, total_pages = await asyncio.to_thread(
            container.doc_loader.read_page_text_sync, path, page
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "doc_id": doc_id,
        "filename": doc.get("filename"),
        "page": page,
        "total_pages": total_pages,
        "text": text,
    }


@router.delete("/api/rag/documents/{doc_id}")
@router.delete("/api/v1/rag/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    namespace: str = "default",
    container: AppContainer = Depends(get_container),
):
    """从知识库中删除指定文档及其向量切块与注册表记录。"""
    doc = await container.sqlite.get_document(doc_id)
    chunk_count = int(doc.get("chunk_count") or 0) if doc else 0
    if chunk_count > 0:
        # Chunk ids are deterministic (``{doc_id}_{index}``), so the registry is enough
        # to delete vectors even when the store is not SQLite-backed.
        ids_to_del = [f"{doc_id}_{i}" for i in range(chunk_count)]
    else:
        col = container.ingestion_pipeline._collection_name(namespace)
        cursor = await container.sqlite.conn.execute(
            "SELECT id FROM vector_points WHERE collection = ?",
            (col,),
        )
        rows = await cursor.fetchall()
        ids_to_del = [r["id"] for r in rows if r["id"].startswith(f"{doc_id}_") or r["id"] == doc_id]
    if ids_to_del:
        col = container.ingestion_pipeline._collection_name(namespace)
        await container.vector_store.delete(col, ids_to_del)
    await container.sqlite.delete_document(doc_id, namespace)
    log.info("document_deleted_from_rag", doc_id=doc_id, chunks_removed=len(ids_to_del), namespace=namespace)
    return {
        "ok": True,
        "doc_id": doc_id,
        "deleted_chunks": len(ids_to_del),
        "namespace": namespace,
    }


@router.get("/api/rag/stats")
@router.get("/api/v1/rag/stats")
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
    registered = await container.sqlite.list_documents(namespace)
    return {
        "namespace": namespace,
        "collection": col,
        "total_chunks": total_chunks,
        "total_documents": len(registered),
        "embedder": container.embedder.tag,
        "embedder_dim": container.embedder.dim,
        "vector_store_status": container.status.get("vector_store", "unknown"),
    }
