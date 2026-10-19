"""对话消息与工具调用的通用数据模型（与 OpenAI Chat Completions 格式对齐）。"""

from __future__ import annotations

import base64
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def _encode_image_url(image_input: str | Path) -> str:
    s = str(image_input).strip()
    if s.startswith(("data:image/", "http://", "https://")):
        return s
    p = Path(s)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: {s}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    images: list[str] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    def to_openai(self) -> dict[str, Any]:
        if self.images and self.role == "user":
            parts: list[dict[str, Any]] = [{"type": "text", "text": self.content or ""}]
            for img in self.images:
                parts.append({"type": "image_url", "image_url": {"url": _encode_image_url(img)}})
            d: dict[str, Any] = {"role": self.role, "content": parts}
        else:
            d = {"role": self.role, "content": self.content or ""}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return d


class ToolSpec(BaseModel):
    """工具描述，可转换为 OpenAI function calling schema。"""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }
