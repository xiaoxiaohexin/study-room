"""离线文档处理管道（Ingestion Pipeline）：加载 -> 结构切块 -> 统一嵌入 -> 向量入库。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.llm.embedding import Embedder
from app.observability.logging import get_logger
from app.rag.chunker import MarkdownChunker
from app.rag.loader import DocumentLoader
from app.storage.sqlite import SQLiteStore
from app.storage.vector import VectorPoint, VectorStore

log = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: MarkdownChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        sqlite: SQLiteStore | None = None,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.sqlite = sqlite

    def _collection_name(self, namespace: str, tier: str = "standard") -> str:
        if tier == "staging":
            return f"staging_{self.embedder.tag}"
        return f"rag_{namespace}_{self.embedder.tag}"

    async def ingest_file(
        self,
        file_path: Path,
        namespace: str = "default",
        original_filename: str | None = None,
        *,
        tier: str = "standard",
        metadata: dict | None = None,
    ) -> tuple[str, int]:
        """将文件导入到指定的 namespace/tier，返回 (doc_id, chunk_count)。

        ``tier='staging'`` 写入独立的临时知识库 collection，不参与标准 RAG 检索。
        """
        doc_id = uuid.uuid4().hex[:12]
        filename = original_filename or file_path.name
        meta = dict(metadata or {})
        log.info(
            "ingest_file_started",
            doc_id=doc_id,
            filename=filename,
            namespace=namespace,
            tier=tier,
        )

        # 1. 转换为 Markdown
        md_text = await self.loader.load_to_markdown(file_path)

        # 2. 结构感知切块
        chunks = self.chunker.split_text(md_text)
        if not chunks:
            log.warning("no_chunks_extracted", doc_id=doc_id, filename=filename)
            await self._register_document(
                doc_id, namespace, filename, file_path, 1, 0, tier=tier, metadata=meta
            )
            return doc_id, 0

        # 3. 统一向量化
        texts = [c.content for c in chunks]
        vecs = await self.embedder.embed(texts)

        # 4. 存入向量库
        col = self._collection_name(namespace, tier)
        await self.vector_store.ensure_collection(col, self.embedder.dim)

        points = []
        max_page = 1
        for idx, c in enumerate(chunks):
            # 兼容带有 page_start/page_end 的 Chunk 模型及历史无页码 Chunk
            p_start = int(getattr(c, "page_start", 1) or 1)
            p_end = int(getattr(c, "page_end", p_start) or p_start)
            max_page = max(max_page, p_end)
            c_index = int(getattr(c, "index", idx))

            points.append(
                VectorPoint(
                    id=f"{doc_id}_{c_index}",
                    vector=vecs[idx],
                    payload={
                        "doc_id": doc_id,
                        "filename": filename,
                        "namespace": namespace,
                        "tier": tier,
                        "chunk_index": c_index,
                        "content": c.content,
                        "headers": getattr(c, "headers", []),
                        "tokens": getattr(c, "tokens", 0),
                        "page_start": p_start,
                        "page_end": p_end,
                        "chapter": getattr(c, "chapter", None) or "",
                        **meta,
                    },
                )
            )
        await self.vector_store.upsert(col, points)
        await self._register_document(
            doc_id, namespace, filename, file_path, max_page, len(points), tier=tier, metadata=meta
        )
        log.info(
            "ingest_file_completed",
            doc_id=doc_id,
            filename=filename,
            chunks_saved=len(points),
            tier=tier,
        )
        return doc_id, len(points)

    async def _register_document(
        self,
        doc_id: str,
        namespace: str,
        filename: str,
        file_path: Path,
        page_count: int,
        chunk_count: int,
        *,
        tier: str = "standard",
        metadata: dict | None = None,
    ) -> None:
        if self.sqlite is None:
            return
        meta = dict(metadata or {})
        try:
            await self.sqlite.register_document(
                doc_id=doc_id,
                namespace=namespace,
                filename=filename,
                file_path=str(file_path.resolve()),
                page_count=page_count,
                chunk_count=chunk_count,
                tier=tier,
                source_type=meta.get("source_type", "local"),
                url=meta.get("url", ""),
                fetched_at=meta.get("fetched_at", 0),
                content_hash=meta.get("content_hash", ""),
                origin=meta.get("origin", ""),
                conversation_id=meta.get("conversation_id", ""),
                session_id=meta.get("session_id", ""),
                user_id=meta.get("user_id", "default_user"),
                status="staged" if tier == "staging" else "standard",
                target_namespace=meta.get("target_namespace", ""),
                metadata_json=meta.get("metadata_json", "{}"),
            )
        except Exception as e:  # noqa: BLE001 - registry must never break ingestion
            log.warning("document_registry_write_failed", doc_id=doc_id, error=str(e))

    @classmethod
    async def backfill_document_registry(
        cls,
        sqlite: SQLiteStore,
        data_dir: Path,
        embedder_tag: str,
    ) -> int:
        """Register documents ingested before the registry existed.

        Scans ``vector_points`` payloads (doc_id/filename/page_end), maps collection
        names back to namespaces and only registers entries whose original upload
        file still exists. Idempotent and safe to run on every startup.
        """
        uploads = (data_dir / "uploads").resolve()
        cursor = await sqlite.conn.execute(
            "SELECT collection, payload_json FROM vector_points WHERE collection LIKE 'rag_%'"
        )
        rows = await cursor.fetchall()
        suffix = f"_{embedder_tag}"
        grouped: dict[tuple[str, str], dict] = {}
        for row in rows:
            collection = row["collection"]
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:  # noqa: BLE001
                continue
            doc_id = payload.get("doc_id")
            filename = payload.get("filename")
            if not doc_id or not filename:
                continue
            if collection.startswith("rag_") and suffix and collection.endswith(suffix):
                namespace = collection[len("rag_") : -len(suffix)]
            else:
                namespace = "default"
            namespace = namespace or "default"
            key = (namespace, doc_id)
            entry = grouped.setdefault(
                key,
                {
                    "filename": filename,
                    "page_count": 1,
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
            entry["page_count"] = max(
                entry["page_count"], int(payload.get("page_end", 1) or 1)
            )

        registered = 0
        for (namespace, doc_id), entry in grouped.items():
            candidate = uploads / Path(entry["filename"]).name
            if not candidate.exists():
                continue
            existing = await sqlite.get_document(doc_id)
            if existing and existing.get("file_path"):
                continue
            await sqlite.register_document(
                doc_id=doc_id,
                namespace=namespace,
                filename=entry["filename"],
                file_path=str(candidate),
                page_count=entry["page_count"],
                chunk_count=entry["chunk_count"],
            )
            registered += 1
        if registered:
            log.info("document_registry_backfilled", count=registered)
        return registered
