"""Markdown 结构感知切块：保留标题层级语义 + Token 重叠窗口切分。"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.core.tokens import count_tokens
from app.observability.logging import get_logger

log = get_logger(__name__)


class Chunk(BaseModel):
    index: int
    content: str
    tokens: int
    headers: list[str] = Field(default_factory=list)


class MarkdownChunker:
    def __init__(self, chunk_size_tokens: int = 500, chunk_overlap_tokens: int = 80):
        self.chunk_size = chunk_size_tokens
        self.overlap = chunk_overlap_tokens
        self._header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def split_text(self, markdown_text: str) -> list[Chunk]:
        """按标题段落分割，并进行 token 重叠滑动切块。"""
        log.debug("chunking_text_started", total_length=len(markdown_text))
        lines = markdown_text.splitlines(keepends=True)
        sections: list[tuple[list[str], str]] = []

        current_headers: list[str] = []
        current_lines: list[str] = []

        for line in lines:
            m = self._header_re.match(line)
            if m:
                if current_lines:
                    sections.append((list(current_headers), "".join(current_lines)))
                    current_lines = []
                level = len(m.group(1))
                title = m.group(2).strip()
                # 调整当前标题栈
                current_headers = current_headers[: level - 1]
                current_headers.append(title)
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((list(current_headers), "".join(current_lines)))

        chunks: list[Chunk] = []
        chunk_idx = 0

        for headers, sec_text in sections:
            header_prefix = (" > ".join(headers) + "\n\n") if headers else ""
            prefix_tokens = count_tokens(header_prefix)
            available_tokens = max(50, self.chunk_size - prefix_tokens)

            # 按段落拆解
            paragraphs = sec_text.split("\n\n")
            curr_chunk_paras: list[str] = []
            curr_chunk_tokens = 0

            for p in paragraphs:
                p_text = p.strip()
                if not p_text:
                    continue
                p_tokens = count_tokens(p_text)

                if curr_chunk_tokens + p_tokens > available_tokens and curr_chunk_paras:
                    # 固化当前 chunk
                    chunk_body = "\n\n".join(curr_chunk_paras)
                    full_content = header_prefix + chunk_body
                    chunks.append(
                        Chunk(
                            index=chunk_idx,
                            content=full_content,
                            tokens=count_tokens(full_content),
                            headers=headers,
                        )
                    )
                    chunk_idx += 1

                    # 重叠滑动：保留末尾一部分段落
                    kept_paras: list[str] = []
                    kept_tokens = 0
                    for prev_p in reversed(curr_chunk_paras):
                        prev_tok = count_tokens(prev_p)
                        if kept_tokens + prev_tok <= self.overlap:
                            kept_paras.insert(0, prev_p)
                            kept_tokens += prev_tok
                        else:
                            break
                    curr_chunk_paras = kept_paras
                    curr_chunk_tokens = kept_tokens

                curr_chunk_paras.append(p_text)
                curr_chunk_tokens += p_tokens

            if curr_chunk_paras:
                chunk_body = "\n\n".join(curr_chunk_paras)
                full_content = header_prefix + chunk_body
                chunks.append(
                    Chunk(
                        index=chunk_idx,
                        content=full_content,
                        tokens=count_tokens(full_content),
                        headers=headers,
                    )
                )
                chunk_idx += 1

        log.info("chunking_text_completed", total_chunks=len(chunks))
        return chunks
