"""RAG 检索器：支持多查询（MQE + HyDE）融合检索与 RRF 分值重排。"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from pydantic import BaseModel, Field

from app.llm.embedding import Embedder
from app.observability.logging import get_logger
from app.storage.vector import VectorStore

log = get_logger(__name__)


def _extract_lexical_tokens(queries: list[str]) -> list[str]:
    tokens = set()
    for q in queries:
        for w in re.findall(r"[\u4e00-\u9fa5]{2,4}|[a-zA-Z0-9_]{2,}", q.lower()):
            tokens.add(w)
    return list(tokens)


def _calibrate_chunk_score(
    chunk_text: str,
    filename: str,
    query_tokens: list[str],
    raw_vector_score: float,
    rrf_relative_rank: float,
) -> float:
    vec_sim = max(0.0, raw_vector_score)
    if not query_tokens:
        if vec_sim > 0.0:
            return float(min(1.0, max(0.0, vec_sim * 0.7 + 0.3 * rrf_relative_rank)))
        return float(min(1.0, max(0.0, rrf_relative_rank)))

    search_target = f"{filename} {chunk_text}".lower()
    matched_count = sum(1 for tok in query_tokens if tok in search_target)
    lexical_ratio = matched_count / max(1, len(query_tokens))

    if matched_count == 0:
        # 零词法重合：若向量相似度也较低，代表仅因 top_k 强制召回的无关噪点，打压虚高分
        if vec_sim < 0.65:
            return float(min(0.35, vec_sim * 0.5 if vec_sim > 0.0 else 0.20))
        return float(min(0.55, vec_sim * 0.65))

    # 词法重合较弱（如多词仅命中 1 个通用词）且向量相似度极低 (< 0.45)
    if lexical_ratio < 0.3 and vec_sim < 0.45:
        return float(min(0.38, vec_sim * 0.7 + 0.1 * rrf_relative_rank))

    # 具备实质匹配：综合向量相似度与 RRF 相对分，并赋予重合度奖励
    if vec_sim > 0.0:
        base = vec_sim * 0.70 + 0.30 * rrf_relative_rank
    else:
        base = rrf_relative_rank
    boost = 1.0 + 0.20 * lexical_ratio
    return float(min(1.0, max(0.0, base * boost)))


class DocChunk(BaseModel):
    id: str
    content: str
    score: float
    doc_id: str
    filename: str
    headers: list[str] = Field(default_factory=list)
    tokens: int = 0
    chunk_index: int = 0
    page_start: int = 1
    page_end: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def page_label(self) -> str:
        """返回规范化页码字符串，例如 'P.42' 或 'P.42-43'。"""
        if self.page_start == self.page_end:
            return f"P.{self.page_start}"
        return f"P.{self.page_start}-{self.page_end}"

    def format_citation(self, include_score: bool = False) -> str:
        """返回用于系统提示词的引用标题行。"""
        hdr = f" ({' > '.join(self.headers)})" if self.headers else ""
        if include_score:
            return f"[DOCUMENT: {self.filename} {self.page_label}{hdr} | score={self.score:.2f}]"
        return f"[DOCUMENT: {self.filename} {self.page_label}{hdr}]"


class RagRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def _collection_name(self, namespace: str) -> str:
        return f"rag_{namespace}_{self.embedder.tag}"

    async def confidence(
        self,
        queries: list[str],
        namespace: str = "default",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """本地知识库检索置信度（供联网补充门控使用）。

        与 ``search`` 的 RRF 归一化分不同，这里返回**绝对向量相似度**：
        - ``max_similarity``：所有命中切片中的最大余弦相似度；
        - ``hits``：命中切片数量；
        - ``avg_top_similarity``：前 top_k 平均相似度。
        """
        if not queries:
            return {"max_similarity": 0.0, "avg_top_similarity": 0.0, "hits": 0}
        col = self._collection_name(namespace)
        try:
            vecs = await self.embedder.embed(queries[:2])
            scores: list[float] = []
            for vec in vecs:
                points = await self.vector_store.search(col, vector=vec, top_k=max(1, top_k))
                scores.extend(float(p.score) for p in points)
        except Exception as exc:  # noqa: BLE001 - 置信度失败时按“不足”处理由调用方决定
            log.warning("rag_confidence_failed", namespace=namespace, error=str(exc))
            return {"max_similarity": 0.0, "avg_top_similarity": 0.0, "hits": 0}
        if not scores:
            return {"max_similarity": 0.0, "avg_top_similarity": 0.0, "hits": 0}
        hits = len(scores)
        top = sorted(scores, reverse=True)[: max(1, top_k)]
        return {
            "max_similarity": max(scores),
            "avg_top_similarity": sum(top) / len(top),
            "hits": hits,
        }

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

        # Reciprocal Rank Fusion (RRF: k=60) + 真实向量相似度记录
        rrf_scores: dict[str, float] = {}
        max_vector_scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        rrf_k = 60.0

        for res in results_per_query:
            if isinstance(res, Exception):
                log.warning("rag_search_single_query_failed", error=str(res))
                continue
            for rank, pt in enumerate(res):
                rrf_scores[pt.id] = rrf_scores.get(pt.id, 0.0) + (1.0 / (rrf_k + rank + 1))
                payloads[pt.id] = pt.payload
                sim = float(getattr(pt, "score", 0.0) or 0.0)
                if pt.id not in max_vector_scores or sim > max_vector_scores[pt.id]:
                    max_vector_scores[pt.id] = sim

        if not rrf_scores:
            log.info("rag_search_empty_result", namespace=namespace)
            return []

        # 归一化排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        max_rrf = max(rrf_scores.values()) or 1.0
        query_tokens = _extract_lexical_tokens(queries)

        doc_chunks = []
        for cid in sorted_ids:
            p = payloads[cid]
            p_start = int(p.get("page_start", 1) or 1)
            p_end = int(p.get("page_end", p_start) or p_start)
            content = p.get("content", "")
            filename = p.get("filename", "")
            rel_rrf = rrf_scores[cid] / max_rrf
            sim = max_vector_scores.get(cid, 0.0)
            calibrated_score = _calibrate_chunk_score(content, filename, query_tokens, sim, rel_rrf)

            doc_chunks.append(
                DocChunk(
                    id=cid,
                    content=content,
                    score=calibrated_score,
                    doc_id=p.get("doc_id", ""),
                    filename=filename,
                    headers=p.get("headers", []),
                    tokens=int(p.get("tokens", 0)),
                    chunk_index=int(p.get("chunk_index", 0)),
                    page_start=p_start,
                    page_end=p_end,
                    metadata=p,
                )
            )

        log.info("rag_search_completed", namespace=namespace, returned_count=len(doc_chunks))
        return doc_chunks

    async def search_staging(
        self,
        queries: list[str],
        session_id: str | None = None,
        top_k: int = 3,
    ) -> list[DocChunk]:
        """检索临时知识库（staging_{embedder_tag}）中的已抓取/暂存网页切片。
        若提供 session_id，则优先过滤该会话的暂存内容，无命中时回退到全局暂存。
        """
        if not queries:
            return []

        col = f"staging_{self.embedder.tag}"
        log.info("staging_search_started", query_count=len(queries), session_id=session_id, top_k=top_k)
        vecs = await self.embedder.embed(queries)
        filters = {"session_id": session_id} if session_id else None

        tasks = [
            self.vector_store.search(col, vector=v, top_k=top_k * 2, filters=filters)
            for v in vecs
        ]
        results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

        total_pts = sum(len(r) for r in results_per_query if isinstance(r, list))
        if total_pts == 0 and filters is not None:
            tasks = [
                self.vector_store.search(col, vector=v, top_k=top_k * 2)
                for v in vecs
            ]
            results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

        rrf_scores: dict[str, float] = {}
        max_vector_scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        rrf_k = 60.0

        for res in results_per_query:
            if isinstance(res, Exception):
                log.warning("staging_search_single_query_failed", error=str(res))
                continue
            for rank, pt in enumerate(res):
                rrf_scores[pt.id] = rrf_scores.get(pt.id, 0.0) + (1.0 / (rrf_k + rank + 1))
                payloads[pt.id] = pt.payload
                sim = float(getattr(pt, "score", 0.0) or 0.0)
                if pt.id not in max_vector_scores or sim > max_vector_scores[pt.id]:
                    max_vector_scores[pt.id] = sim

        if not rrf_scores:
            return []

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        max_rrf = max(rrf_scores.values()) or 1.0
        query_tokens = _extract_lexical_tokens(queries)

        doc_chunks = []
        for cid in sorted_ids:
            p = payloads[cid]
            p_start = int(p.get("page_start", 1) or 1)
            p_end = int(p.get("page_end", p_start) or p_start)
            content = p.get("content", "")
            filename = p.get("filename", "")
            rel_rrf = rrf_scores[cid] / max_rrf
            sim = max_vector_scores.get(cid, 0.0)
            calibrated_score = _calibrate_chunk_score(content, filename, query_tokens, sim, rel_rrf)

            doc_chunks.append(
                DocChunk(
                    id=cid,
                    content=content,
                    score=calibrated_score,
                    doc_id=p.get("doc_id", ""),
                    filename=filename,
                    headers=p.get("headers", []),
                    tokens=int(p.get("tokens", 0)),
                    chunk_index=int(p.get("chunk_index", 0)),
                    page_start=p_start,
                    page_end=p_end,
                    metadata=p,
                )
            )

        log.info("staging_search_completed", returned_count=len(doc_chunks))
        return doc_chunks

        log.info("rag_search_completed", namespace=namespace, returned_count=len(doc_chunks))
        return doc_chunks

    @staticmethod
    def format_citation(chunk: DocChunk, include_score: bool = False) -> str:
        """格式化单个文档切片的引用标头。"""
        return chunk.format_citation(include_score=include_score)

    @staticmethod
    def format_prompt_citations(chunks: list[DocChunk]) -> str:
        """将文档切片列表格式化为注入系统提示词的标准 Markdown 文本块。"""
        blocks = []
        for c in chunks:
            blocks.append(f"{c.format_citation()}\n{c.content}")
        return "\n\n".join(blocks)

