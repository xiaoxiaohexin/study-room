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
    history: list[Message] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class SessionStateManager:
    def __init__(self, sqlite: SQLiteStore, history_window: int = 10):
        self.sqlite = sqlite
        self.history_window = history_window
        # 内存锁防止单会话并发争抢 (session_id -> asyncio.Lock)
        self._locks: dict[str, asyncio.Lock] = {}
        # 内存活跃会话缓存
        self._cache: dict[str, SessionState] = {}

    def get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def get_or_create_session(self, user_id: str, session_id: str | None = None) -> SessionState:
        now = time.time()
        sid = session_id or uuid.uuid4().hex[:16]

        if sid in self._cache and self._cache[sid].user_id == user_id:
            return self._cache[sid]

        log.info("loading_session", user_id=user_id, session_id=sid)
        # 查询数据库
        cursor = await self.sqlite.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
            (sid, user_id),
        )
        row = await cursor.fetchone()

        if row is None:
            # 创建新会话
            await self.sqlite.conn.execute(
                "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (sid, user_id, "新会话", now, now),
            )
            await self.sqlite.conn.commit()
            state = SessionState(
                user_id=user_id, session_id=sid, title="新会话", created_at=now, updated_at=now
            )
            self._cache[sid] = state
            log.info("session_created", user_id=user_id, session_id=sid)
            return state

        # 读取消息历史
        msg_cursor = await self.sqlite.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (sid,),
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

        state = SessionState(
            user_id=user_id,
            session_id=sid,
            title=row["title"],
            history=history[-self.history_window * 2 :],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
        self._cache[sid] = state
        log.info("session_loaded", user_id=user_id, session_id=sid, history_count=len(state.history))
        return state

    async def append_message(self, user_id: str, session_id: str, message: Message) -> None:
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
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        await self.sqlite.conn.commit()

        if session_id in self._cache:
            self._cache[session_id].history.append(message)
            if len(self._cache[session_id].history) > self.history_window * 2:
                self._cache[session_id].history = self._cache[session_id].history[-self.history_window * 2 :]
            self._cache[session_id].updated_at = now

        log.debug("message_appended", session_id=session_id, role=message.role)

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = await self.sqlite.conn.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete_session(self, user_id: str, session_id: str) -> None:
        await self.sqlite.conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self.sqlite.conn.execute(
            "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self.sqlite.conn.commit()
        self._cache.pop(session_id, None)
        log.info("session_deleted", user_id=user_id, session_id=session_id)
