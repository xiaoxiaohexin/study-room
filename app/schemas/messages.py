"""对话消息与工具调用的通用数据模型（与 OpenAI Chat Completions 格式对齐）。"""

from __future__ import annotations

import base64
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# 客户端只允许引用受控目录内的本地图片；空列表表示使用默认的 <data_dir>/uploads。
_ALLOWED_IMAGE_ROOTS: list[Path] = []
_MAX_LOCAL_IMAGE_BYTES = 10 * 1024 * 1024


def set_allowed_image_roots(roots: list[Path]) -> None:
    """Configure which local directories may be read as image inputs."""
    _ALLOWED_IMAGE_ROOTS[:] = [Path(r) for r in roots]


def _allowed_image_roots() -> list[Path]:
    if _ALLOWED_IMAGE_ROOTS:
        return list(_ALLOWED_IMAGE_ROOTS)
    from app.core.config import get_settings

    return [Path(get_settings().data_dir) / "uploads"]


def _resolve_local_image(path: Path) -> Path:
    """Resolve a local image path and reject anything outside the allowed roots."""
    try:
        resolved = path.resolve()
    except (OSError, ValueError) as e:
        raise FileNotFoundError(f"Invalid image path: {path}") from e
    for root in _allowed_image_roots():
        try:
            if resolved.is_relative_to(root.resolve()):
                return resolved
        except (OSError, ValueError):
            continue
    raise PermissionError(
        f"Local image path is outside the allowed upload directory: {path}"
    )


def _check_local_image(raw: str) -> Path:
    """Resolve and validate a local image file (boundary + existence + size)."""
    p = _resolve_local_image(Path(raw))
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: {raw}")
    if p.stat().st_size > _MAX_LOCAL_IMAGE_BYTES:
        raise ValueError(f"Image file exceeds the {_MAX_LOCAL_IMAGE_BYTES} byte limit: {raw}")
    return p


def validate_image_input(image_input: str) -> None:
    """Raise when a client-supplied image reference would read outside the upload dir."""
    s = (image_input or "").strip()
    if not s or s.startswith(("data:image/", "http://", "https://")):
        return
    _check_local_image(s)


def _encode_image_url(image_input: str | Path) -> str:
    s = str(image_input).strip()
    if s.startswith(("data:image/", "http://", "https://")):
        return s
    p = _check_local_image(s)
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
