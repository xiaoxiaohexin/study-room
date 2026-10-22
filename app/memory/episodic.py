"""情景记忆（Episodic Memory）：SQLite 关系库 + VectorStore 向量库双引擎存储。

打分公式：(向量相似度 * 0.8 + 时间近因 * 0.2) * (0.8 + importance * 0.4)
"""

from __future__ import annotations

import json
import math
import re
import time

from app.llm.embedding import Embedder
from app.memory.base import BaseMemory, MemoryItem, ScoredMemory
from app.observability.logging import get_logger
from app.storage.sqlite import SQLiteStore
from app.storage.vector import VectorPoint, VectorStore

log = get_logger(__name__)


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return {w for w in words if len(w) > 1}


class EpisodicMemory(BaseMemory):
    def __init__(self, sqlite: SQLiteStore, vector_store: VectorStore, embedder: Embedder):
        self.sqlite = sqlite
        self.vector_store = vector_store
        self.embedder = embedder
        self.collection_name = f"episodic_{self.embedder.tag}"
        self._initialized = False

    async def _ensure_store(self) -> None:
        if not self._initialized:
            await self.vector_store.ensure_collection(self.collection_name, self.embedder.dim)
            self._initialized = True

    async def add(self, item: MemoryItem) -> str:
        await self._ensure_store()
        meta = dict(item.metadata or {})
        if item.session_id and "session_id" not in meta:
            meta["session_id"] = item.session_id

        # 1. 向量化并写入向量库
        vecs = await self.embedder.embed([item.content])
        pt = VectorPoint(
            id=item.id,
            vector=vecs[0],
            payload={
                "user_id": item.user_id,
                "session_id": item.session_id,
                "importance": item.importance,
                "created_at": item.created_at,
                "metadata": meta,
            },
        )
        await self.vector_store.upsert(self.collection_name, [pt])

        # 2. 写入 SQLite
        query = """
        INSERT OR REPLACE INTO episodic_memories
        (id, user_id, session_id, content, importance, created_at, last_accessed, access_count, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.sqlite.conn.execute(
            query,
            (
                item.id,
                item.user_id,
                item.session_id,
                item.content,
                item.importance,
                item.created_at,
                item.last_accessed,
                item.access_count,
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        await self.sqlite.conn.commit()
        log.info("episodic_memory_added", user_id=item.user_id, memory_id=item.id, importance=item.importance)
        return item.id

    async def retrieve(
        self, query: str, user_id: str, top_k: int = 5, session_id: str | None = None
    ) -> list[ScoredMemory]:
        await self._ensure_store()
        vecs = await self.embedder.embed([query])
        if not vecs:
            return []

        # 向量检索，按 user_id 过滤（扩大候选窗口以便重排与会话过滤）
        candidate_k = max(top_k * 4, 20)
        scored_pts = await self.vector_store.search(
            self.collection_name,
            vector=vecs[0],
            top_k=candidate_k,
            filters={"user_id": user_id},
        )
        if not scored_pts:
            return []

        ids = [p.id for p in scored_pts]
        placeholders = ",".join("?" for _ in ids)
        sql = f"SELECT * FROM episodic_memories WHERE id IN ({placeholders}) AND user_id = ?"
        cursor = await self.sqlite.conn.execute(sql, (*ids, user_id))
        rows = await cursor.fetchall()
        rows_by_id = {r["id"]: r for r in rows}

        query_kw = _extract_keywords(query)
        now = time.time()
        scored: list[ScoredMemory] = []
        updated_ids: list[str] = []

        for pt in scored_pts:
            row = rows_by_id.get(pt.id)
            if not row:
                continue

            importance = float(row["importance"])
            created_at = float(row["created_at"])
            age_days = max(0.0, (now - created_at) / 86400.0)
            recency = math.exp(-0.1 * age_days)
            content_kw = _extract_keywords(row["content"])
            # 对字符 N-gram 散列兜底方案 (HashingEmbedder)，在词面完全无交集时抑制桶饱和碰撞噪声
            if query_kw and not (query_kw & content_kw) and "hash" in getattr(self.embedder, "tag", ""):
                sim_score = 0.0
            else:
                sim_score = (
                    max(0.0, min(1.0, (pt.score + 1.0) / 2.0)) if -1.0 <= pt.score <= 1.0 else max(0.0, pt.score)
                )

            # 打分公式：(向量 * 0.8 + 时间近因 * 0.2) * (0.8 + importance * 0.4)
            base_score = (sim_score * 0.8 + recency * 0.2) * (0.8 + importance * 0.4)

            # 会话优先加权与跨会话过滤
            item_sess = row["session_id"]
            if session_id is not None and session_id != "":
                if item_sess == session_id:
                    session_factor = 1.35
                    final_score = base_score * session_factor
                else:
                    session_factor = 0.65
                    final_score = base_score * session_factor
                    # 跨会话且最终分值低于 0.50 强制舍弃，消除跨学科污染
                    if final_score < 0.50:
                        continue
            else:
                session_factor = 1.0
                final_score = base_score

            meta = json.loads(row["metadata_json"] or "{}")
            sess_id = row["session_id"] or meta.get("session_id")
            item = MemoryItem(
                id=row["id"],
                user_id=row["user_id"],
                session_id=sess_id,
                type="episodic",
                content=row["content"],
                importance=importance,
                created_at=created_at,
                last_accessed=now,
                access_count=int(row["access_count"]) + 1,
                metadata=meta,
            )
            scored.append(
                ScoredMemory(
                    item=item,
                    score=float(final_score),
                    breakdown={
                        "vector_sim": sim_score,
                        "recency": recency,
                        "importance_factor": (0.8 + importance * 0.4),
                        "session_affinity": session_factor,
                    },
                )
            )
            updated_ids.append(row["id"])

        # 异步更新访问记录
        if updated_ids:
            up_ph = ",".join("?" for _ in updated_ids)
            await self.sqlite.conn.execute(
                f"UPDATE episodic_memories SET last_accessed = ?, access_count = access_count + 1 WHERE id IN ({up_ph})",
                (now, *updated_ids),
            )
            await self.sqlite.conn.commit()

        scored.sort(key=lambda s: s.score, reverse=True)
        res = scored[:top_k]
        log.debug("episodic_memory_retrieved", user_id=user_id, session_id=session_id, count=len(res))
        return res

    async def remove(
        self,
        ids: list[str],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if not ids:
            return
        await self._ensure_store()
        ph = ",".join("?" for _ in ids)
        where = [f"id IN ({ph})"]
        params: list[object] = list(ids)
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        cursor = await self.sqlite.conn.execute(
            f"SELECT id FROM episodic_memories WHERE {' AND '.join(where)}", tuple(params)
        )
        owned_ids = [r["id"] for r in await cursor.fetchall()]
        if not owned_ids:
            return
        await self.vector_store.delete(self.collection_name, owned_ids)
        owned_ph = ",".join("?" for _ in owned_ids)
        await self.sqlite.conn.execute(
            f"DELETE FROM episodic_memories WHERE id IN ({owned_ph})", owned_ids
        )
        await self.sqlite.conn.commit()
        log.info("episodic_memories_removed", count=len(owned_ids), user_id=user_id)

    async def list_items(
        self, user_id: str, limit: int = 50, session_id: str | None = None
    ) -> list[MemoryItem]:
        if session_id is not None and session_id != "":
            cursor = await self.sqlite.conn.execute(
                "SELECT * FROM episodic_memories WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, session_id, limit),
            )
        else:
            cursor = await self.sqlite.conn.execute(
                "SELECT * FROM episodic_memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
        rows = await cursor.fetchall()
        return [
            MemoryItem(
                id=r["id"],
                user_id=r["user_id"],
                session_id=r["session_id"],
                type="episodic",
                content=r["content"],
                importance=float(r["importance"]),
                created_at=float(r["created_at"]),
                last_accessed=float(r["last_accessed"]),
                access_count=int(r["access_count"]),
                metadata=json.loads(r["metadata_json"] or "{}"),
            )
            for r in rows
        ]
