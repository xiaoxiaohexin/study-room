"""Markdown 结构感知切块：保留标题层级语义 + 物理页码跟踪 + Token 重叠窗口切分。"""

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
    page_start: int = 1
    page_end: int = 1
    chapter: str | None = None


class MarkdownChunker:
    def __init__(self, chunk_size_tokens: int = 500, chunk_overlap_tokens: int = 80):
        self.chunk_size = chunk_size_tokens
        self.overlap = chunk_overlap_tokens
        self._header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        self._page_re = re.compile(
            r"(?:<!--\s*PAGE:?\s*(\d+)\s*-->|[\x0c\f])", re.IGNORECASE
        )

    def _split_long_paragraph(self, text: str, max_tokens: int) -> list[str]:
        """将超长段落按句子边界切分为较小片段。"""
        if count_tokens(text) <= max_tokens:
            return [text]
        parts = re.split(r"([。！？!?\n]|\.\s+)", text)
        sentences: list[str] = []
        for i in range(0, len(parts) - 1, 2):
            s = (parts[i] + parts[i + 1]).strip()
            if s:
                sentences.append(s)
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        if len(sentences) <= 1:
            from app.core.tokens import truncate_to_tokens

            parts_res = []
            rem = text
            while rem:
                sub = truncate_to_tokens(rem, max_tokens)
                if not sub or sub == rem:
                    parts_res.append(rem)
                    break
                parts_res.append(sub)
                rem = rem[len(sub) :].strip()
            return parts_res

        sub_paras: list[str] = []
        curr_s: list[str] = []
        curr_tok = 0
        for s in sentences:
            s_tok = count_tokens(s)
            if curr_tok + s_tok > max_tokens and curr_s:
                sub_paras.append(" ".join(curr_s))
                curr_s = []
                curr_tok = 0
            curr_s.append(s)
            curr_tok += s_tok
        if curr_s:
            sub_paras.append(" ".join(curr_s))
        return sub_paras

    def _parse_paged_paragraphs(
        self, sec_text: str, current_page: int
    ) -> tuple[list[tuple[str, int]], int]:
        """解析段落并跟踪物理页码，自动过滤掉页面标记注释，并将超长段落拆解。"""
        paragraphs: list[tuple[str, int]] = []
        last_end = 0
        page = current_page

        def _add_p(text_seg: str, p_num: int):
            for p in text_seg.split("\n\n"):
                p_clean = p.strip()
                if not p_clean:
                    continue
                if count_tokens(p_clean) > self.chunk_size:
                    for sub in self._split_long_paragraph(p_clean, self.chunk_size):
                        paragraphs.append((sub, p_num))
                else:
                    paragraphs.append((p_clean, p_num))

        for m in self._page_re.finditer(sec_text):
            seg = sec_text[last_end : m.start()]
            _add_p(seg, page)

            page_val = m.group(1)
            if page_val is not None:
                try:
                    page = int(page_val)
                except ValueError:
                    page += 1
            else:
                page += 1
            last_end = m.end()

        trailing = sec_text[last_end:]
        _add_p(trailing, page)

        return paragraphs, page

    def split_text(self, markdown_text: str) -> list[Chunk]:
        """按标题段落分割并跟踪物理页码，进行 token 重叠滑动切块。"""
        log.debug("chunking_text_started", total_length=len(markdown_text))
        lines = markdown_text.splitlines(keepends=True)
        sections: list[tuple[list[str], str]] = []

        current_headers: list[str] = []
        current_lines: list[str] = []

        for line in lines:
            m = self._header_re.match(line)
            if m:
                if current_lines:
                    sec_content = "".join(current_lines)
                    if sec_content.strip():
                        sections.append((list(current_headers), sec_content))
                    current_lines = []
                level = len(m.group(1))
                title = m.group(2).strip()
                # 调整当前标题栈
                current_headers = current_headers[: level - 1]
                current_headers.append(title)
            else:
                current_lines.append(line)

        if current_lines:
            sec_content = "".join(current_lines)
            if sec_content.strip():
                sections.append((list(current_headers), sec_content))

        if not sections and current_headers:
            sections.append((list(current_headers), current_headers[-1]))

        chunks: list[Chunk] = []
        chunk_idx = 0
        current_page = 1

        for headers, sec_text in sections:
            header_prefix = (" > ".join(headers) + "\n\n") if headers else ""
            prefix_tokens = count_tokens(header_prefix)
            available_tokens = max(50, self.chunk_size - prefix_tokens)

            # 按页码及段落拆解
            paragraphs, current_page = self._parse_paged_paragraphs(sec_text, current_page)
            curr_chunk_paras: list[tuple[str, int]] = []
            curr_chunk_tokens = 0

            for p_text, p_page in paragraphs:
                p_tokens = count_tokens(p_text)

                if curr_chunk_tokens + p_tokens > available_tokens and curr_chunk_paras:
                    # 固化当前 chunk
                    chunk_body = "\n\n".join(p[0] for p in curr_chunk_paras)
                    full_content = header_prefix + chunk_body
                    p_start = min(p[1] for p in curr_chunk_paras)
                    p_end = max(p[1] for p in curr_chunk_paras)
                    chunks.append(
                        Chunk(
                            index=chunk_idx,
                            content=full_content,
                            tokens=count_tokens(full_content),
                            headers=headers,
                            page_start=p_start,
                            page_end=p_end,
                            chapter=" > ".join(headers) if headers else None,
                        )
                    )
                    chunk_idx += 1

                    # 重叠滑动：保留末尾一部分段落及其所在页码
                    kept_paras: list[tuple[str, int]] = []
                    kept_tokens = 0
                    for prev_p, prev_p_page in reversed(curr_chunk_paras):
                        prev_tok = count_tokens(prev_p)
                        if kept_tokens + prev_tok <= self.overlap:
                            kept_paras.insert(0, (prev_p, prev_p_page))
                            kept_tokens += prev_tok
                        else:
                            break
                    curr_chunk_paras = kept_paras
                    curr_chunk_tokens = kept_tokens

                curr_chunk_paras.append((p_text, p_page))
                curr_chunk_tokens += p_tokens

            if curr_chunk_paras:
                chunk_body = "\n\n".join(p[0] for p in curr_chunk_paras)
                full_content = header_prefix + chunk_body
                p_start = min(p[1] for p in curr_chunk_paras)
                p_end = max(p[1] for p in curr_chunk_paras)
                chunks.append(
                    Chunk(
                        index=chunk_idx,
                        content=full_content,
                        tokens=count_tokens(full_content),
                        headers=headers,
                        page_start=p_start,
                        page_end=p_end,
                        chapter=" > ".join(headers) if headers else None,
                    )
                )
                chunk_idx += 1

        log.info("chunking_text_completed", total_chunks=len(chunks))
        return chunks

