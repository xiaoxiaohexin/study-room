"""多租户会话隔离与状态管理（StateManager）：用户空间与会话空间隔离、持久化与并发控制。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.observability.logging import get_logger
from app.schemas.messages import Message, ToolCall
from app.storage.sqlite import SQLiteStore

log = get_logger(__name__)


class SessionState(BaseModel):
    user_id: str
    session_id: str
    title: str = ""
    course_id: str | None = None
    rag_namespace: str = "default"
    history: list[Message] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class AsyncRLock:
    """可重入异步锁：支持同一 asyncio.Task 内重入，防止与外部会话锁产生死锁。"""

    def __init__(self) -> None:
        self._block = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._count: int = 0

    async def acquire(self) -> bool:
        task = asyncio.current_task()
        if self._owner is not None and self._owner == task:
            self._count += 1
            return True
        await self._block.acquire()
        self._owner = task
        self._count = 1
        return True

    def release(self) -> None:
        task = asyncio.current_task()
        if self._owner != task:
            raise RuntimeError(f"Cannot release un-acquired lock: owner={self._owner}, caller={task}")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._block.release()

    async def __aenter__(self) -> AsyncRLock:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def locked(self) -> bool:
        return self._count > 0


class SessionStateManager:
    def __init__(self, sqlite: SQLiteStore, history_window: int = 10):
        self.sqlite = sqlite
        self.history_window = history_window
        # 内存锁防止单会话并发争抢 (session_id -> AsyncRLock)
        self._locks: dict[str, AsyncRLock] = {}
        # 内存活跃会话缓存
        self._cache: dict[str, SessionState] = {}

    def get_lock(self, session_id: str) -> AsyncRLock:
        if session_id not in self._locks:
            self._locks[session_id] = AsyncRLock()
        return self._locks[session_id]

    async def get_or_create_session(
        self,
        user_id: str,
        session_id: str | None = None,
        course_id: str | None = None,
        rag_namespace: str | None = None,
        title: str | None = None,
    ) -> SessionState:
        now = time.time()
        sid = session_id or uuid.uuid4().hex[:16]
        effective_title = title if title is not None else "新会话"
        effective_rag_ns = rag_namespace or "default"

        async with self.get_lock(sid):
            # 1. 检查内存缓存中是否已被其他租户占用
            if sid in self._cache and self._cache[sid].user_id != user_id:
                raise ValueError(f"Session '{sid}' already belongs to another user")

            # 2. 内存命中且归属当前租户
            if sid in self._cache and self._cache[sid].user_id == user_id:
                state = self._cache[sid]
                new_title = title if title is not None else state.title
                new_course_id = course_id if course_id is not None else state.course_id
                new_rag_namespace = (rag_namespace or "default") if rag_namespace is not None else state.rag_namespace

                needs_update = (
                    new_title != state.title
                    or new_course_id != state.course_id
                    or new_rag_namespace != state.rag_namespace
                )

                if needs_update:
                    await self.sqlite.conn.execute(
                        "UPDATE sessions SET title = ?, course_id = ?, rag_namespace = ?, updated_at = ? WHERE session_id = ? AND user_id = ?",
                        (new_title, new_course_id, new_rag_namespace, now, sid, user_id),
                    )
                    await self.sqlite.conn.commit()
                    # 仅在 DB commit 成功后更新缓存对象，防止 Cache Poisoning
                    state.title = new_title
                    state.course_id = new_course_id
                    state.rag_namespace = new_rag_namespace
                    state.updated_at = now
                return state

            log.info("loading_session", user_id=user_id, session_id=sid)
            # 3. 查询数据库中是否存在该 session_id（全局租户冲突校验）
            cursor = await self.sqlite.conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (sid,),
            )
            row = await cursor.fetchone()

            if row is not None and row["user_id"] != user_id:
                raise ValueError(f"Session '{sid}' already belongs to another user")

            if row is None:
                # 创建新会话
                try:
                    await self.sqlite.conn.execute(
                        "INSERT INTO sessions (session_id, user_id, title, course_id, rag_namespace, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sid, user_id, effective_title, course_id, effective_rag_ns, now, now),
                    )
                    await self.sqlite.conn.commit()
                except Exception as e:
                    if "UNIQUE constraint failed" in str(e):
                        raise ValueError(f"Session '{sid}' already belongs to another user") from e
                    raise

                state = SessionState(
                    user_id=user_id,
                    session_id=sid,
                    title=effective_title,
                    course_id=course_id,
                    rag_namespace=effective_rag_ns,
                    created_at=now,
                    updated_at=now,
                )
                self._cache[sid] = state
                log.info(
                    "session_created",
                    user_id=user_id,
                    session_id=sid,
                    course_id=course_id,
                    rag_namespace=effective_rag_ns,
                )
                return state

            # 4. 会话属于当前用户，读取消息历史（严格限定 session_id AND user_id）
            msg_cursor = await self.sqlite.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
                (sid, user_id),
            )
            msg_rows = await msg_cursor.fetchall()
            history: list[Message] = []
            for mr in msg_rows:
                tc = None
                if mr["tool_calls_json"]:
                    raw_tc = json.loads(mr["tool_calls_json"])
                    tc = [ToolCall(**c) for c in raw_tc]
                history.append(
                    Message(
                        role=mr["role"],
                        content=mr["content"],
                        name=mr["name"],
                        tool_call_id=mr["tool_call_id"],
                        tool_calls=tc,
                    )
                )

            current_course = row["course_id"] if "course_id" in row.keys() else None
            current_rag_ns = (
                row["rag_namespace"] if "rag_namespace" in row.keys() and row["rag_namespace"] else None
            ) or "default"
            current_title = row["title"] if "title" in row.keys() else ""

            needs_update = False
            new_title = current_title
            new_course = current_course
            new_rag_ns = current_rag_ns

            if title is not None and title != current_title:
                new_title = title
                needs_update = True
            if course_id is not None and course_id != current_course:
                new_course = course_id
                needs_update = True
            if rag_namespace is not None:
                norm_ns = rag_namespace or "default"
                if norm_ns != current_rag_ns:
                    new_rag_ns = norm_ns
                    needs_update = True

            updated_at = float(row["updated_at"])
            if needs_update:
                updated_at = now
                await self.sqlite.conn.execute(
                    "UPDATE sessions SET title = ?, course_id = ?, rag_namespace = ?, updated_at = ? WHERE session_id = ? AND user_id = ?",
                    (new_title, new_course, new_rag_ns, updated_at, sid, user_id),
                )
                await self.sqlite.conn.commit()

            state = SessionState(
                user_id=user_id,
                session_id=sid,
                title=new_title,
                course_id=new_course,
                rag_namespace=new_rag_ns,
                history=history[-self.history_window * 2 :],
                created_at=float(row["created_at"]),
                updated_at=updated_at,
            )
            self._cache[sid] = state
            log.info("session_loaded", user_id=user_id, session_id=sid, history_count=len(state.history))
            return state

    async def get_session(self, user_id: str, session_id: str) -> SessionState | None:
        async with self.get_lock(session_id):
            return await self._get_session_unlocked(user_id=user_id, session_id=session_id)

    async def _get_session_unlocked(self, user_id: str, session_id: str) -> SessionState | None:
        if session_id in self._cache:
            if self._cache[session_id].user_id == user_id:
                return self._cache[session_id]
            return None

        cursor = await self.sqlite.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        msg_cursor = await self.sqlite.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
            (session_id, user_id),
        )
        msg_rows = await msg_cursor.fetchall()
        history: list[Message] = []
        for mr in msg_rows:
            tc = None
            if mr["tool_calls_json"]:
                raw_tc = json.loads(mr["tool_calls_json"])
                tc = [ToolCall(**c) for c in raw_tc]
            history.append(
                Message(
                    role=mr["role"],
                    content=mr["content"],
                    name=mr["name"],
                    tool_call_id=mr["tool_call_id"],
                    tool_calls=tc,
                )
            )

        course_id = row["course_id"] if "course_id" in row.keys() else None
        rag_ns = (
            row["rag_namespace"] if "rag_namespace" in row.keys() and row["rag_namespace"] else None
        ) or "default"
        state = SessionState(
            user_id=user_id,
            session_id=session_id,
            title=row["title"] if "title" in row.keys() else "",
            course_id=course_id,
            rag_namespace=rag_ns,
            history=history[-self.history_window * 2 :],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
        self._cache[session_id] = state
        return state

    async def update_session(
        self,
        user_id: str,
        session_id: str,
        title: str | None = None,
        course_id: str | None = None,
        rag_namespace: str | None = None,
    ) -> SessionState:
        async with self.get_lock(session_id):
            state = await self._get_session_unlocked(user_id=user_id, session_id=session_id)
            if state is None:
                raise ValueError(f"Session '{session_id}' not found for user '{user_id}'")

            now = time.time()
            new_title = title if title is not None else state.title
            new_course_id = course_id if course_id is not None else state.course_id
            new_rag_namespace = (rag_namespace or "default") if rag_namespace is not None else state.rag_namespace

            await self.sqlite.conn.execute(
                "UPDATE sessions SET title = ?, course_id = ?, rag_namespace = ?, updated_at = ? WHERE session_id = ? AND user_id = ?",
                (new_title, new_course_id, new_rag_namespace, now, session_id, user_id),
            )
            await self.sqlite.conn.commit()

            state.title = new_title
            state.course_id = new_course_id
            state.rag_namespace = new_rag_namespace
            state.updated_at = now
            self._cache[session_id] = state
            log.info(
                "session_updated",
                user_id=user_id,
                session_id=session_id,
                title=new_title,
                course_id=new_course_id,
                rag_namespace=new_rag_namespace,
            )
            return state

    async def append_message(self, user_id: str, session_id: str, message: Message) -> None:
        async with self.get_lock(session_id):
            # 校验会话存在性与租户归属，防止孤儿记录写入
            cursor = await self.sqlite.conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            if await cursor.fetchone() is None:
                log.warning("append_message_session_not_found", user_id=user_id, session_id=session_id)
                return

            now = time.time()
            msg_id = uuid.uuid4().hex[:16]
            tc_json = None
            if message.tool_calls:
                tc_json = json.dumps([tc.model_dump() for tc in message.tool_calls], ensure_ascii=False)

            await self.sqlite.conn.execute(
                "INSERT INTO messages (id, session_id, user_id, role, content, name, tool_call_id, tool_calls_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    msg_id,
                    session_id,
                    user_id,
                    message.role,
                    message.content,
                    message.name,
                    message.tool_call_id,
                    tc_json,
                    now,
                ),
            )
            await self.sqlite.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ? AND user_id = ?",
                (now, session_id, user_id),
            )
            await self.sqlite.conn.commit()

            if session_id in self._cache and self._cache[session_id].user_id == user_id:
                self._cache[session_id].history.append(message)
                if len(self._cache[session_id].history) > self.history_window * 2:
                    self._cache[session_id].history = self._cache[session_id].history[-self.history_window * 2 :]
                self._cache[session_id].updated_at = now

            log.debug("message_appended", session_id=session_id, role=message.role)

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = await self.sqlite.conn.execute(
            "SELECT session_id, user_id, title, course_id, rag_namespace, created_at, updated_at "
            "FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if not d.get("rag_namespace"):
                d["rag_namespace"] = "default"
            result.append(d)
        return result

    async def delete_session(self, user_id: str, session_id: str) -> None:
        async with self.get_lock(session_id):
            cursor = await self.sqlite.conn.execute(
                "SELECT user_id FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None or row["user_id"] != user_id:
                log.info("session_delete_skipped", user_id=user_id, session_id=session_id)
                return

            await self.sqlite.conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            await self.sqlite.conn.execute(
                "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            await self.sqlite.conn.commit()
            if session_id in self._cache and self._cache[session_id].user_id == user_id:
                self._cache.pop(session_id, None)
            log.info("session_deleted", user_id=user_id, session_id=session_id)
