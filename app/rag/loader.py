"""多格式文档统一加载转换引擎：优先 MarkItDown，失败或纯文本时自适应解码。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.llm.base import LLMClient
from app.observability.logging import get_logger
from app.schemas.messages import Message

log = get_logger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class DocumentLoader:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm
        self._has_markitdown = False
        try:
            import markitdown  # noqa: F401

            self._has_markitdown = True
            log.info("markitdown_available")
        except ImportError:
            log.info("markitdown_not_available_using_builtins")

    async def load_to_markdown(self, file_path: Path) -> str:
        """将文档（PDF/Word/Excel/图片/TXT/MD/代码）转换为 Markdown 字符串。"""
        log.info("loading_document", path=str(file_path))
        suffix = file_path.suffix.lower()

        # 图片文件直接通过主多模态 LLM（如本地 Gemma-4）提取视觉内容与 OCR 文字
        if suffix in _IMAGE_SUFFIXES and self.llm:
            try:
                resp = await self.llm.chat(
                    [
                        Message(
                            role="user",
                            content="请将这张图片的内容完整转换为结构化 Markdown，包含所有可见文字（OCR）、图表结构与视觉细节描述。",
                            images=[str(file_path)],
                        )
                    ]
                )
                desc = (resp.content or "").strip()
                return f"# 图片文档：{file_path.name}\n\n{desc}\n"
            except Exception as e:  # noqa: BLE001
                log.warning("llm_image_convert_failed", path=str(file_path), error=str(e))

        # 纯文本 / 代码类文件直接文本读取
        if suffix in {".txt", ".md", ".json", ".csv", ".py", ".sh", ".yaml", ".yml", ".sql"}:
            return await asyncio.to_thread(self._read_text_fallback, file_path)

        # 二进制文档尝试通过 markitdown 解析
        if self._has_markitdown:
            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                res = await asyncio.to_thread(md.convert, str(file_path))
                text = res.text_content or ""
                log.info("markitdown_converted_successfully", path=str(file_path), char_count=len(text))
                return text
            except Exception as e:  # noqa: BLE001
                log.warning("markitdown_convert_failed", path=str(file_path), error=str(e))

        # 兜底纯文本读取
        return await asyncio.to_thread(self._read_text_fallback, file_path)

    def _read_text_fallback(self, file_path: Path) -> str:
        for enc in ("utf-8", "gb18030", "gbk", "latin-1"):
            try:
                with open(file_path, encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(file_path, errors="ignore") as f:
            return f.read()
