"""感知记忆（Perceptual Memory）：复用主多模态模型与向量库进行跨模态记忆存储与检索。"""

from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.embedding import Embedder
from app.memory.base import BaseMemory, MemoryItem, ScoredMemory
from app.observability.logging import get_logger
from app.schemas.messages import Message
from app.storage.vector import VectorPoint, VectorStore

log = get_logger(__name__)


class PerceptualMemory(BaseMemory):
    """多模态感知记忆：直接通过主多模态 LLM 与 Embedder + VectorStore 进行跨模态向量检索。"""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self._items: dict[str, list[MemoryItem]] = {}

    def _collection(self) -> str:
        tag = self.embedder.tag if self.embedder else "raw"
        return f"mem_perceptual_{tag}"

    async def add(self, item: MemoryItem) -> str:
        # 如果手动调用 add_memory 仅传了 image_path 而未经过对话理解，复用主多模态 LLM 提取视觉语义
        img_source = item.metadata.get("image_path") or item.metadata.get("image_url")
        if img_source and self.llm:
            try:
                resp = await self.llm.chat(
                    [
                        Message(
                            role="user",
                            content=f"请结合备注（{item.content or '无'}）简要描述这张图片的核心内容。",
                            images=[str(img_source)],
                        )
                    ]
                )
                desc = (resp.content or "").strip()
                if desc:
                    item.metadata["visual_description"] = desc
                    item.content = f"[图像感知记忆] {item.content}\n视觉内容：{desc}".strip()
            except Exception as e:  # noqa: BLE001
                log.warning("perceptual_llm_describe_failed", error=str(e))

        self._items.setdefault(item.user_id, []).append(item)

        if self.vector_store and self.embedder and item.content.strip():
            col = self._collection()
            await self.vector_store.ensure_collection(col, self.embedder.dim)
            vecs = await self.embedder.embed([item.content])
            await self.vector_store.upsert(
                col,
                [
                    VectorPoint(
                        id=item.id,
                        vector=vecs[0],
                        payload={
                            "user_id": item.user_id,
                            "session_id": item.session_id or "",
                            "content": item.content,
                            "importance": item.importance,
                            "created_at": item.created_at,
                            "metadata": item.metadata,
                        },
                    )
                ],
            )

        log.info("perceptual_memory_added", user_id=item.user_id, memory_id=item.id)
        return item.id

    async def retrieve(
        self, query: str, user_id: str, top_k: int = 5, session_id: str | None = None
    ) -> list[ScoredMemory]:
        if self.vector_store and self.embedder and query.strip():
            col = self._collection()
            await self.vector_store.ensure_collection(col, self.embedder.dim)
            q_vec = (await self.embedder.embed([query]))[0]
            hits = await self.vector_store.search(
                col, vector=q_vec, top_k=top_k, filters={"user_id": user_id}
            )
            if hits:
                scored_hits: list[ScoredMemory] = []
                for h in hits:
                    p = h.payload
                    it = MemoryItem(
                        id=h.id,
                        type="perceptual",
                        user_id=p.get("user_id", user_id),
                        session_id=p.get("session_id") or None,
                        content=p.get("content", ""),
                        importance=float(p.get("importance", 0.5)),
                        created_at=float(p.get("created_at", 0.0)),
                        metadata=p.get("metadata") or {},
                    )
                    sim = max(0.0, float(h.score))
                    final_score = sim * (0.8 + it.importance * 0.4)
                    scored_hits.append(
                        ScoredMemory(
                            item=it,
                            score=final_score,
                            breakdown={"vector_sim": round(sim, 4), "importance": it.importance},
                        )
                    )
                return scored_hits

        items = self._items.get(user_id, [])
        scored: list[ScoredMemory] = []
        for it in items:
            scored.append(
                ScoredMemory(
                    item=it,
                    score=0.5 * (0.8 + it.importance * 0.4),
                    breakdown={"perceptual_match": 0.5},
                )
            )
        return scored[:top_k]

    async def remove(
        self,
        ids: list[str],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        id_set = set(ids)
        for uid in list(self._items.keys()):
            if user_id is not None and uid != user_id:
                continue
            items = self._items[uid]
            kept = [
                it
                for it in items
                if not (it.id in id_set and (session_id is None or it.session_id == session_id))
            ]
            self._items[uid] = kept
        if self.vector_store and self.embedder and ids:
            await self.vector_store.delete(self._collection(), ids)

    async def list_items(
        self, user_id: str, limit: int = 50, session_id: str | None = None
    ) -> list[MemoryItem]:
        return list(reversed(self._items.get(user_id, [])))[:limit]
