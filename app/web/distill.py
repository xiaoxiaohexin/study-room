"""临时知识库「提纯」服务：质量打分 + LLM 结构化提炼为知识节点候选。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.structured import chat_json
from app.observability.logging import get_logger
from app.rag.chunker import MarkdownChunker
from app.schemas.messages import Message
from app.storage.sqlite import SQLiteStore

log = get_logger(__name__)

_TRUSTED_SUFFIXES = (".edu", ".edu.cn", ".gov", ".gov.cn", ".ac.cn", ".org")
_TRUSTED_HOSTS = ("wikipedia.org", "arxiv.org", "zhihu.com", "csdn.net", "github.com")

_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "content": {"type": "string"},
                    "raw_quote": {"type": "string"},
                    "difficulty": {"type": "number"},
                    "suggested_course_id": {"type": "string"},
                },
                "required": ["title", "summary", "content"],
            },
        }
    },
    "required": ["candidates"],
}


class DistillError(Exception):
    """提纯失败（文档不存在/状态不允许/内容为空）。"""


def quality_score(text: str, url: str = "") -> float:
    """启发式质量分（0~1）：正文长度 + 文本密度 + 域名可信度 + 结构化程度。"""
    if not text:
        return 0.0
    length_score = min(1.0, len(text) / 6000.0)
    paragraphs = [p for p in text.splitlines() if len(p.strip()) > 30]
    density_score = min(1.0, len(paragraphs) / 20.0)
    headings = sum(1 for line in text.splitlines() if line.strip().startswith("#"))
    structure_score = min(1.0, headings / 6.0)
    domain_score = 0.5
    host = (url.split("//")[-1].split("/")[0] if url else "").lower()
    if host:
        if any(host == h or host.endswith("." + h) for h in _TRUSTED_HOSTS):
            domain_score = 0.85
        elif host.endswith(_TRUSTED_SUFFIXES):
            domain_score = 0.9
        elif host.endswith((".cn", ".com", ".net")):
            domain_score = 0.55
        else:
            domain_score = 0.65
    score = (
        length_score * 0.3
        + density_score * 0.25
        + structure_score * 0.15
        + domain_score * 0.3
    )
    return round(min(1.0, max(0.0, score)), 4)


class DistillationService:
    def __init__(
        self,
        settings: Settings,
        sqlite: SQLiteStore,
        llm: LLMClient,
        chunker: MarkdownChunker,
    ):
        self.settings = settings
        self.sqlite = sqlite
        self.llm = llm
        self.chunker = chunker

    async def _read_snapshot(self, row: dict) -> str:
        path = Path(row.get("file_path") or "")
        if not path.exists():
            raise DistillError(f"临时快照不存在: {row.get('doc_id')}")
        return path.read_text(encoding="utf-8", errors="ignore")

    def _excerpts(self, text: str) -> list[str]:
        try:
            chunks = self.chunker.split_text(text)
            return [c.content for c in chunks if c.content.strip()][:8]
        except Exception:  # noqa: BLE001 - chunker failures fall back to paragraph windows
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            return paragraphs[:8]

    async def _llm_candidates(
        self, title: str, url: str, excerpts: list[str]
    ) -> list[dict[str, Any]]:
        material = "\n\n".join(excerpts)[:7000]
        prompt = (
            "你是一个学习资料整理助手。请把下面网页内容提炼成 1~3 个可直接用于复习的"
            "知识节点候选。要求：\n"
            "1. title 是概念名（不超过 20 字）；summary 是一句话解释；\n"
            "2. content 是结构化要点（Markdown，保留公式/关键步骤，不要照抄全文）；\n"
            "3. raw_quote 必须截取网页中一段原文（不超过 160 字）作为出处证据；\n"
            "4. difficulty 为 0.1~1.0；suggested_course_id 留空。\n"
            f"网页标题：{title}\n网页地址：{url}\n\n内容：\n{material}"
        )
        data = await chat_json(
            self.llm,
            [Message(role="user", content=prompt)],
            _CANDIDATE_SCHEMA,
            name="staging_candidates",
            temperature=0.2,
            max_tokens=1200,
        )
        raw_candidates = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(raw_candidates, list):
            raise DistillError("LLM 未返回候选列表")
        candidates: list[dict[str, Any]] = []
        for item in raw_candidates[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not title or not summary:
                continue
            difficulty = float(item.get("difficulty") or 0.5)
            candidates.append(
                {
                    "title": title[:80],
                    "summary": summary[:400],
                    "content": str(item.get("content") or summary)[:4000],
                    "raw_quote": str(item.get("raw_quote") or "")[:500],
                    "difficulty": min(1.0, max(0.1, difficulty)),
                    "suggested_course_id": str(item.get("suggested_course_id") or "").strip()[:80],
                }
            )
        return candidates

    def _fallback_candidates(
        self, row: dict, text: str, excerpts: list[str]
    ) -> list[dict[str, Any]]:
        title = (row.get("filename") or "网页资料").replace(".md", "")[:80]
        first = excerpts[0] if excerpts else text
        summary = re_first_sentence(first) or title
        return [
            {
                "title": title,
                "summary": summary[:400],
                "content": "\n\n".join(excerpts[:2])[:4000] or first[:4000],
                "raw_quote": first[:300],
                "difficulty": 0.5,
                "suggested_course_id": "",
            }
        ]

    async def distill_document(self, doc_id: str, *, user_id: str = "default_user") -> list[dict]:
        row = await self.sqlite.get_document(doc_id)
        if not row or row.get("tier") != "staging":
            raise DistillError(f"临时文档不存在: {doc_id}")
        if row.get("status") == "promoted":
            raise DistillError("该文档已晋升，无需再次提纯")

        text = await self._read_snapshot(row)
        excerpts = self._excerpts(text)
        if not excerpts:
            raise DistillError("临时文档内容为空，无法提纯")
        score = quality_score(text, row.get("url") or "")
        url = row.get("url") or ""
        title = (row.get("filename") or "网页资料").replace(".md", "")

        candidates: list[dict[str, Any]] = []
        try:
            candidates = await self._llm_candidates(title, url, excerpts)
        except Exception as exc:  # noqa: BLE001 - LLM 不可用时用规则兜底
            log.warning("staging_distill_llm_failed", doc_id=doc_id, error=str(exc))
        if not candidates:
            candidates = self._fallback_candidates(row, text, excerpts)

        await self.sqlite.delete_staging_candidates_for_doc(doc_id)
        created: list[dict[str, Any]] = []
        now = time.time()
        for item in candidates:
            candidate = {
                "id": f"cand_{uuid.uuid4().hex[:10]}",
                "doc_id": doc_id,
                "session_id": row.get("session_id") or "",
                "user_id": user_id or row.get("user_id") or "default_user",
                "title": item["title"],
                "summary": item["summary"],
                "content": item["content"],
                "raw_quote": item["raw_quote"],
                "difficulty": item["difficulty"],
                "suggested_course_id": item["suggested_course_id"],
                "quality_score": score,
                "status": "pending",
                "provenance_json": json.dumps(
                    {
                        "doc_id": doc_id,
                        "url": row.get("url") or "",
                        "fetched_at": row.get("fetched_at") or 0,
                        "origin": row.get("origin") or "",
                    },
                    ensure_ascii=False,
                ),
                "created_at": now,
                "updated_at": now,
            }
            await self.sqlite.add_staging_candidate(candidate)
            created.append(candidate)

        await self.sqlite.update_document(doc_id, status="distilled", quality_score=score)
        log.info(
            "staging_distilled",
            doc_id=doc_id,
            candidates=len(created),
            quality_score=score,
        )
        return created

    async def list_candidates(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        status: str | None = "pending",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = await self.sqlite.list_staging_candidates(
            status=status,
            session_id=session_id if session_id is not None else None,
            user_id=user_id,
            limit=limit,
        )
        enriched = []
        for row in rows:
            doc = await self.sqlite.get_document(row.get("doc_id") or "")
            enriched.append(
                {
                    "id": row["id"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "content": row["content"],
                    "raw_quote": row["raw_quote"],
                    "difficulty": float(row["difficulty"] or 0.5),
                    "quality_score": float(row["quality_score"] or 0),
                    "status": row["status"],
                    "suggested_course_id": row.get("suggested_course_id") or "",
                    "promoted_kp_id": row.get("promoted_kp_id") or "",
                    "created_at": row.get("created_at") or 0,
                    "source_title": ((doc or {}).get("filename") or "").replace(".md", ""),
                    "source_url": (doc or {}).get("url") or "",
                }
            )
        return enriched

    async def reject_candidate(self, candidate_id: str) -> bool:
        row = await self.sqlite.get_staging_candidate(candidate_id)
        if not row:
            return False
        await self.sqlite.update_staging_candidate(
            candidate_id, status="rejected", updated_at=time.time()
        )
        return True


def re_first_sentence(text: str) -> str:
    import re

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    match = re.match(r"(.{10,120}?[。！？.!?])", cleaned)
    return match.group(1) if match else cleaned[:120]
