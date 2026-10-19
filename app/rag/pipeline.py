"""离线文档处理管道（Ingestion Pipeline）：加载 -> 结构切块 -> 统一嵌入 -> 向量入库。"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.llm.embedding import Embedder
from app.observability.logging import get_logger
from app.rag.chunker import MarkdownChunker
from app.rag.loader import DocumentLoader
from app.storage.vector import VectorPoint, VectorStore

log = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: MarkdownChunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def _collection_name(self, namespace: str) -> str:
        return f"rag_{namespace}_{self.embedder.tag}"

    async def ingest_file(self, file_path: Path, namespace: str = "default") -> tuple[str, int]:
        """将文件导入到指定的 namespace，返回 (doc_id, chunk_count)。"""
        doc_id = uuid.uuid4().hex[:12]
        filename = file_path.name
        log.info("ingest_file_started", doc_id=doc_id, filename=filename, namespace=namespace)

        # 1. 转换为 Markdown
        md_text = await self.loader.load_to_markdown(file_path)

        # 2. 结构感知切块
        chunks = self.chunker.split_text(md_text)
        if not chunks:
            log.warning("no_chunks_extracted", doc_id=doc_id, filename=filename)
            return doc_id, 0

        # 3. 统一向量化
        texts = [c.content for c in chunks]
        vecs = await self.embedder.embed(texts)

        # 4. 存入向量库
        col = self._collection_name(namespace)
        await self.vector_store.ensure_collection(col, self.embedder.dim)

        points = [
            VectorPoint(
                id=f"{doc_id}_{c.index}",
                vector=vecs[idx],
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": c.index,
                    "content": c.content,
                    "headers": c.headers,
                    "tokens": c.tokens,
                },
            )
            for idx, c in enumerate(chunks)
        ]
        await self.vector_store.upsert(col, points)
        log.info("ingest_file_completed", doc_id=doc_id, filename=filename, chunks_saved=len(points))
        return doc_id, len(points)
