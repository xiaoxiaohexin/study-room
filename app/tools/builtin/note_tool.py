"""NoteTool（结构化持久化笔记）：YAML 元数据 + Markdown 正文，持久化存储于本地文件与 SQLite 索引。"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from app.observability.logging import get_logger
from app.storage.sqlite import SQLiteStore
from app.tools.base import Tool, ToolContext, ToolResult

log = get_logger(__name__)

# user_id / note_id 只允许安全字符，杜绝 "../../outside/secret" 之类的路径穿越。
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class NoteTool(Tool):
    name = "note_tool"
    description = (
        "管理用户的永久记忆笔记。支持操作：create (创建笔记), read (读取指定笔记), "
        "update (更新笔记), search (关键词搜索笔记), list (列出笔记)。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "update", "search", "list"],
                "description": "要执行的操作",
            },
            "title": {"type": "string", "description": "笔记标题（创建/更新/搜索时使用）"},
            "content": {"type": "string", "description": "Markdown 正文（创建/更新时使用）"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"},
            "note_id": {"type": "string", "description": "笔记唯一ID（读取/更新时使用）"},
            "query": {"type": "string", "description": "搜索关键词（搜索时使用）"},
        },
        "required": ["action"],
    }

    def __init__(self, notes_root: Path, sqlite: SQLiteStore):
        self.notes_root = Path(notes_root).resolve()
        self.sqlite = sqlite
        self.notes_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value: str, field: str) -> str:
        value = (value or "").strip()
        if not _SAFE_ID_RE.match(value):
            raise ValueError(f"{field} 仅允许字母、数字、下划线与短横线（1~64 位）。")
        return value

    def _user_dir(self, user_id: str) -> Path:
        safe_user = self._safe_id(user_id, "user_id")
        p = self.notes_root / safe_user
        # 纵深防御：即使字符校验被绕过，也必须落在 notes_root 内。
        if not p.resolve().is_relative_to(self.notes_root):
            raise ValueError("user_id 解析后越出笔记根目录。")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _note_path(self, user_id: str, note_id: str) -> Path:
        safe_note = self._safe_id(note_id, "note_id")
        target = (self._user_dir(user_id) / f"{safe_note}.md").resolve()
        if not target.is_relative_to(self.notes_root):
            raise ValueError("note_id 解析后越出笔记根目录。")
        return target

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action")
        user_id = context.user_id
        log.info("note_tool_action", action=action, user_id=user_id)

        try:
            if action == "create":
                title = arguments.get("title", "未命名笔记")
                content = arguments.get("content", "")
                tags = arguments.get("tags", [])
                note_id = uuid.uuid4().hex[:12]
                now = time.time()

                file_path = self._note_path(user_id, note_id)
                meta = {"id": note_id, "title": title, "tags": tags, "created_at": now, "updated_at": now}
                doc_text = f"---\n{yaml.safe_dump(meta, allow_unicode=True)}---\n\n{content}"
                file_path.write_text(doc_text, encoding="utf-8")

                # 写入 SQLite 索引
                await self.sqlite.conn.execute(
                    "INSERT INTO notes_index (id, user_id, title, path, tags_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (note_id, user_id, title, str(file_path), json.dumps(tags, ensure_ascii=False), now, now),
                )
                await self.sqlite.conn.commit()
                return ToolResult(ok=True, output=f"笔记创建成功！ID: {note_id}, 标题: {title}")

            elif action == "read":
                note_id = arguments.get("note_id")
                if not note_id:
                    return ToolResult(ok=False, error="读取笔记必须提供 note_id。")
                file_path = self._note_path(user_id, note_id)
                if not file_path.exists():
                    return ToolResult(ok=False, error=f"笔记 {note_id} 不存在。")
                return ToolResult(ok=True, output=file_path.read_text(encoding="utf-8"))

            elif action == "update":
                note_id = arguments.get("note_id")
                if not note_id:
                    return ToolResult(ok=False, error="更新笔记必须提供 note_id。")
                file_path = self._note_path(user_id, note_id)
                if not file_path.exists():
                    return ToolResult(ok=False, error=f"笔记 {note_id} 不存在。")

                title = arguments.get("title")
                content = arguments.get("content")
                tags = arguments.get("tags")
                now = time.time()

                # 读取原文件元数据
                raw = file_path.read_text(encoding="utf-8")
                parts = raw.split("---", 2)
                meta = yaml.safe_load(parts[1]) if len(parts) >= 3 else {}
                old_body = parts[2].strip() if len(parts) >= 3 else raw

                new_title = title if title is not None else meta.get("title", "未命名")
                new_content = content if content is not None else old_body
                new_tags = tags if tags is not None else meta.get("tags", [])
                meta.update({"title": new_title, "tags": new_tags, "updated_at": now})

                new_doc = f"---\n{yaml.safe_dump(meta, allow_unicode=True)}---\n\n{new_content}"
                file_path.write_text(new_doc, encoding="utf-8")

                await self.sqlite.conn.execute(
                    "UPDATE notes_index SET title = ?, tags_json = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                    (new_title, json.dumps(new_tags, ensure_ascii=False), now, note_id, user_id),
                )
                await self.sqlite.conn.commit()
                return ToolResult(ok=True, output=f"笔记 {note_id} 更新成功。")

            elif action == "list":
                cursor = await self.sqlite.conn.execute(
                    "SELECT id, title, tags_json, updated_at FROM notes_index WHERE user_id = ? ORDER BY updated_at DESC LIMIT 20",
                    (user_id,),
                )
                rows = await cursor.fetchall()
                if not rows:
                    return ToolResult(ok=True, output="暂无笔记。")
                lines = [f"- [{r['id']}] {r['title']} (标签: {r['tags_json']})" for r in rows]
                return ToolResult(ok=True, output="\n".join(lines))

            elif action == "search":
                q = arguments.get("query", "")
                cursor = await self.sqlite.conn.execute(
                    "SELECT id, title, tags_json FROM notes_index WHERE user_id = ? AND (title LIKE ? OR tags_json LIKE ?) LIMIT 10",
                    (user_id, f"%{q}%", f"%{q}%"),
                )
                rows = await cursor.fetchall()
                if not rows:
                    return ToolResult(ok=True, output=f"未找到与 '{q}' 相关的笔记。")
                lines = [f"- [{r['id']}] {r['title']}" for r in rows]
                return ToolResult(ok=True, output="\n".join(lines))

            return ToolResult(ok=False, error=f"未知操作: {action}")
        except Exception as e:  # noqa: BLE001
            log.error("note_tool_failed", error=str(e))
            return ToolResult(ok=False, error=str(e))
