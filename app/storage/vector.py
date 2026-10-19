"""向量存储抽象：Qdrant 主实现 + numpy 内存降级实现。"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from app.observability.logging import get_logger

log = get_logger(__name__)


class VectorPoint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class ScoredPoint(BaseModel):
    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    async def ensure_collection(self, collection: str, dim: int) -> None: ...
    async def upsert(self, collection: str, points: list[VectorPoint]) -> None: ...
    async def search(
        self, collection: str, vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[ScoredPoint]: ...
    async def delete(self, collection: str, ids: list[str]) -> None: ...
    async def aclose(self) -> None: ...


def _match(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(payload.get(k) == v for k, v in filters.items())


class InMemoryVectorStore:
    """降级实现：进程内存 + 余弦相似度。重启后数据丢失。"""

    def __init__(self):
        self._data: dict[str, dict[str, VectorPoint]] = {}
        self._lock = asyncio.Lock()

    async def ensure_collection(self, collection: str, dim: int) -> None:
        self._data.setdefault(collection, {})

    async def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        async with self._lock:
            col = self._data.setdefault(collection, {})
            for p in points:
                col[p.id] = p

    async def search(self, collection, vector, top_k, filters=None) -> list[ScoredPoint]:
        col = self._data.get(collection, {})
        cands = [p for p in col.values() if _match(p.payload, filters)]
        if not cands:
            return []
        q = np.asarray(vector, dtype=np.float32)
        m = np.asarray([p.vector for p in cands], dtype=np.float32)
        denom = (np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)) + 1e-9
        scores = (m @ q) / denom
        idx = np.argsort(-scores)[:top_k]
        return [ScoredPoint(id=cands[i].id, score=float(scores[i]), payload=cands[i].payload) for i in idx]

    async def delete(self, collection, ids) -> None:
        col = self._data.get(collection, {})
        for i in ids:
            col.pop(i, None)

    async def aclose(self) -> None:
        return None


class SQLiteVectorStore:
    """基于 SQLite + NumPy 的本地持久化向量库：零外部服务依赖，重启后数据不丢失。"""

    def __init__(self, sqlite: Any):
        self.sqlite = sqlite
        self._cache: dict[str, dict[str, VectorPoint]] = {}
        self._loaded_collections: set[str] = set()
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self, collection: str) -> dict[str, VectorPoint]:
        if collection not in self._loaded_collections:
            import json

            col_map: dict[str, VectorPoint] = {}
            cursor = await self.sqlite.conn.execute(
                "SELECT id, vector_json, payload_json FROM vector_points WHERE collection = ?",
                (collection,),
            )
            rows = await cursor.fetchall()
            for r in rows:
                col_map[r["id"]] = VectorPoint(
                    id=r["id"],
                    vector=json.loads(r["vector_json"]),
                    payload=json.loads(r["payload_json"] or "{}"),
                )
            self._cache[collection] = col_map
            self._loaded_collections.add(collection)
        return self._cache.setdefault(collection, {})

    async def ensure_collection(self, collection: str, dim: int) -> None:
        async with self._lock:
            await self._ensure_loaded(collection)

    async def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        import json
        import time

        now = time.time()
        async with self._lock:
            col = await self._ensure_loaded(collection)
            rows_to_insert = []
            for p in points:
                col[p.id] = p
                rows_to_insert.append(
                    (
                        collection,
                        p.id,
                        len(p.vector),
                        json.dumps(p.vector),
                        json.dumps(p.payload, ensure_ascii=False),
                        now,
                    )
                )
            if rows_to_insert:
                await self.sqlite.conn.executemany(
                    "INSERT OR REPLACE INTO vector_points (collection, id, dim, vector_json, payload_json, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    rows_to_insert,
                )
                await self.sqlite.conn.commit()

    async def search(
        self, collection: str, vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[ScoredPoint]:
        async with self._lock:
            col = await self._ensure_loaded(collection)
            cands = [p for p in col.values() if _match(p.payload, filters)]
        if not cands:
            return []
        q = np.asarray(vector, dtype=np.float32)
        m = np.asarray([p.vector for p in cands], dtype=np.float32)
        denom = (np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)) + 1e-9
        scores = (m @ q) / denom
        idx = np.argsort(-scores)[:top_k]
        return [ScoredPoint(id=cands[i].id, score=float(scores[i]), payload=cands[i].payload) for i in idx]

    async def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        async with self._lock:
            col = await self._ensure_loaded(collection)
            for i in ids:
                col.pop(i, None)
            placeholders = ",".join("?" for _ in ids)
            await self.sqlite.conn.execute(
                f"DELETE FROM vector_points WHERE collection = ? AND id IN ({placeholders})",
                [collection, *ids],
            )
            await self.sqlite.conn.commit()

    async def aclose(self) -> None:
        return None


class QdrantVectorStore:
    def __init__(self, url: str, api_key: str | None = None):
        from qdrant_client import AsyncQdrantClient

        self.client = AsyncQdrantClient(url=url, api_key=api_key, timeout=10)
        self._known: set[str] = set()

    async def ping(self) -> None:
        await self.client.get_collections()

    async def ensure_collection(self, collection: str, dim: int) -> None:
        if collection in self._known:
            return
        from qdrant_client import models

        if not await self.client.collection_exists(collection):
            await self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
        self._known.add(collection)

    async def upsert(self, collection, points) -> None:
        from qdrant_client import models

        await self.client.upsert(
            collection,
            points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
        )

    async def search(self, collection, vector, top_k, filters=None) -> list[ScoredPoint]:
        from qdrant_client import models

        if collection not in self._known and not await self.client.collection_exists(collection):
            return []
        qf = None
        if filters:
            qf = models.Filter(
                must=[
                    models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in filters.items()
                ]
            )
        res = await self.client.query_points(
            collection, query=vector, limit=top_k, query_filter=qf, with_payload=True
        )
        return [ScoredPoint(id=str(p.id), score=p.score, payload=p.payload or {}) for p in res.points]

    async def delete(self, collection, ids) -> None:
        from qdrant_client import models

        await self.client.delete(collection, points_selector=models.PointIdsList(points=ids))

    async def aclose(self) -> None:
        await self.client.close()
