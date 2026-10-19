"""图存储抽象（语义记忆的实体关系）。MVP：Neo4j 实现为基础版，降级为 NullGraphStore。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStore(Protocol):
    async def upsert_entity(self, user_id: str, name: str, label: str = "Entity", **props) -> None: ...
    async def upsert_relation(self, user_id: str, src: str, rel: str, dst: str, **props) -> None: ...
    async def neighbors(
        self, user_id: str, name: str, depth: int = 1, limit: int = 20
    ) -> list[dict[str, Any]]: ...
    async def aclose(self) -> None: ...


class NullGraphStore:
    async def upsert_entity(self, user_id, name, label="Entity", **props) -> None:
        return None

    async def upsert_relation(self, user_id, src, rel, dst, **props) -> None:
        return None

    async def neighbors(self, user_id, name, depth=1, limit=20) -> list[dict[str, Any]]:
        return []

    async def aclose(self) -> None:
        return None


class Neo4jGraphStore:
    """TODO: 实体抽取（LLM/NER）→ 写入；按实体扩展检索。目前只提供基础读写。"""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import AsyncGraphDatabase  # 可选依赖

        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def ping(self) -> None:
        await self.driver.verify_connectivity()

    async def upsert_entity(self, user_id, name, label="Entity", **props) -> None:
        async with self.driver.session() as s:
            await s.run(
                "MERGE (e:Entity {user_id:$uid, name:$name}) SET e.label=$label, e += $props",
                uid=user_id,
                name=name,
                label=label,
                props=props,
            )

    async def upsert_relation(self, user_id, src, rel, dst, **props) -> None:
        async with self.driver.session() as s:
            await s.run(
                "MERGE (a:Entity {user_id:$uid, name:$src}) MERGE (b:Entity {user_id:$uid, name:$dst}) "
                "MERGE (a)-[r:REL {type:$rel}]->(b) SET r += $props",
                uid=user_id,
                src=src,
                dst=dst,
                rel=rel,
                props=props,
            )

    async def neighbors(self, user_id, name, depth=1, limit=20) -> list[dict[str, Any]]:
        depth = max(1, min(depth, 3))
        async with self.driver.session() as s:
            res = await s.run(
                f"MATCH (a:Entity {{user_id:$uid, name:$name}})-[r*1..{depth}]-(b) "
                "RETURN b.name AS name, [x IN r | x.type] AS path LIMIT $limit",
                uid=user_id,
                name=name,
                limit=limit,
            )
            return [dict(rec) async for rec in res]

    async def aclose(self) -> None:
        await self.driver.close()
