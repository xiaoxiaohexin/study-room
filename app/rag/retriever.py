"""RAG 检索器：支持多查询（MQE + HyDE）融合检索与 RRF 分值重排。"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from app.llm.embedding import Embedder
from app.observability.logging import get_logger
from app.storage.vector import VectorStore

log = get_logger(__name__)


class DocChunk(BaseModel):
    id: str
    content: str
    score: float
    doc_id: str
    filename: str
    headers: list[str] = Field(default_factory=list)
    tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def _collection_name(self, namespace: str) -> str:
        return f"rag_{namespace}_{self.embedder.tag}"

    async def search(
        self,
        queries: list[str],
        namespace: str = "default",
        top_k: int = 5,
    ) -> list[DocChunk]:
        """对传入的一个或多个查询词并行检索，并应用 RRF 算法进行分值融合。"""
        if not queries:
            return []

        col = self._collection_name(namespace)
        log.info("rag_search_started", namespace=namespace, query_count=len(queries), top_k=top_k)

        # 批量向量化
        vecs = await self.embedder.embed(queries)

        tasks = [self.vector_store.search(col, vector=v, top_k=top_k * 2) for v in vecs]
        results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

        # Reciprocal Rank Fusion (RRF: k=60)
        rrf_scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        rrf_k = 60.0

        for res in results_per_query:
            if isinstance(res, Exception):
                log.warning("rag_search_single_query_failed", error=str(res))
                continue
            for rank, pt in enumerate(res):
                rrf_scores[pt.id] = rrf_scores.get(pt.id, 0.0) + (1.0 / (rrf_k + rank + 1))
                payloads[pt.id] = pt.payload

        if not rrf_scores:
            log.info("rag_search_empty_result", namespace=namespace)
            return []

        # 归一化排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        max_rrf = max(rrf_scores.values()) or 1.0

        doc_chunks = []
        for cid in sorted_ids:
            p = payloads[cid]
            doc_chunks.append(
                DocChunk(
                    id=cid,
                    content=p.get("content", ""),
                    score=float(rrf_scores[cid] / max_rrf),
                    doc_id=p.get("doc_id", ""),
                    filename=p.get("filename", ""),
                    headers=p.get("headers", []),
                    tokens=int(p.get("tokens", 0)),
                    metadata=p,
                )
            )

        log.info("rag_search_completed", namespace=namespace, returned_count=len(doc_chunks))
        return doc_chunks
