"""语义记忆（Semantic Memory）：Qdrant 概念向量库 + Neo4j 拓扑图谱（图谱可降级）。

打分公式：(内容相似度 * 0.7 + 图拓扑密度/关联 * 0.3) * (0.8 + importance * 0.4)
"""

from __future__ import annotations

import time

from app.llm.embedding import Embedder
from app.memory.base import BaseMemory, MemoryItem, ScoredMemory
from app.observability.logging import get_logger
from app.storage.graph import GraphStore
from app.storage.vector import VectorPoint, VectorStore

log = get_logger(__name__)


class SemanticMemory(BaseMemory):
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, embedder: Embedder):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.collection_name = f"semantic_{self.embedder.tag}"
        self._initialized = False
        # 本地内存字典辅助（用于存储 payload 文本）
        self._items: dict[str, MemoryItem] = {}

    async def _ensure_store(self) -> None:
        if not self._initialized:
            await self.vector_store.ensure_collection(self.collection_name, self.embedder.dim)
            self._initialized = True

    async def add(self, item: MemoryItem) -> str:
        await self._ensure_store()
        vecs = await self.embedder.embed([item.content])
        pt = VectorPoint(
            id=item.id,
            vector=vecs[0],
            payload={
                "user_id": item.user_id,
                "importance": item.importance,
                "content": item.content,
                "created_at": item.created_at,
                "metadata": item.metadata,
            },
        )
        await self.vector_store.upsert(self.collection_name, [pt])
        self._items[item.id] = item

        # 尝试将实体写入图谱（如果有实体标注）
        entities = item.metadata.get("entities", [])
        for ent in entities:
            await self.graph_store.upsert_entity(item.user_id, name=str(ent), label="Concept")

        log.info("semantic_memory_added", user_id=item.user_id, memory_id=item.id, entities=len(entities))
        return item.id

    async def retrieve(self, query: str, user_id: str, top_k: int = 5) -> list[ScoredMemory]:
        await self._ensure_store()
        vecs = await self.embedder.embed([query])
        if not vecs:
            return []

        scored_pts = await self.vector_store.search(
            self.collection_name,
            vector=vecs[0],
            top_k=top_k,
            filters={"user_id": user_id},
        )
        scored: list[ScoredMemory] = []
        now = time.time()

        for pt in scored_pts:
            payload = pt.payload
            importance = float(payload.get("importance", 0.5))
            content = str(payload.get("content", ""))
            sim_score = (
                max(0.0, min(1.0, (pt.score + 1.0) / 2.0)) if -1.0 <= pt.score <= 1.0 else max(0.0, pt.score)
            )

            # 查询图谱拓扑关联系数（如有）
            graph_score = 0.0
            neighbors = await self.graph_store.neighbors(user_id, name=query[:20], depth=1, limit=5)
            if neighbors:
                graph_score = min(1.0, len(neighbors) * 0.2)

            final_score = (sim_score * 0.7 + graph_score * 0.3) * (0.8 + importance * 0.4)

            item = MemoryItem(
                id=pt.id,
                user_id=user_id,
                type="semantic",
                content=content,
                importance=importance,
                created_at=float(payload.get("created_at", now)),
                last_accessed=now,
                metadata=payload.get("metadata", {}),
            )
            scored.append(
                ScoredMemory(
                    item=item,
                    score=float(final_score),
                    breakdown={
                        "content_sim": sim_score,
                        "graph_density": graph_score,
                        "importance_factor": (0.8 + importance * 0.4),
                    },
                )
            )

        scored.sort(key=lambda s: s.score, reverse=True)
        res = scored[:top_k]
        log.debug("semantic_memory_retrieved", user_id=user_id, count=len(res))
        return res

    async def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._ensure_store()
        await self.vector_store.delete(self.collection_name, ids)
        for i in ids:
            self._items.pop(i, None)
        log.info("semantic_memories_removed", count=len(ids))

    async def list_items(self, user_id: str, limit: int = 50) -> list[MemoryItem]:
        items = [it for it in self._items.values() if it.user_id == user_id]
        return sorted(items, key=lambda x: x.created_at, reverse=True)[:limit]
