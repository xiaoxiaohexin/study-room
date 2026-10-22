"""B 通道：本地 RAG 不足时的联网补充（MCP/搜索 Provider + 临时知识库暂存）。"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import Settings
from app.observability.logging import get_logger
from app.query.policy import is_web_query_eligible
from app.web.search import sanitize_search_query
from app.web.staging import WebStagingService

log = get_logger(__name__)


class WebSupplementService:
    def __init__(self, settings: Settings, staging: WebStagingService):
        self.settings = settings
        self.staging = staging

    @property
    def enabled(self) -> bool:
        return (
            self.settings.web_fallback_mode == "auto"
            and self.settings.web_enabled
            and self.staging.search_available
        )

    def insufficient(
        self,
        confidence: dict[str, Any],
        local_docs_count: int,
        *,
        query: str | None = None,
    ) -> bool:
        if self.settings.web_fallback_mode != "auto":
            return False
        if query is not None and not is_web_query_eligible(query):
            return False
        if local_docs_count < max(1, self.settings.web_fallback_min_hits):
            return True
        return float(confidence.get("max_similarity") or 0.0) < self.settings.web_fallback_min_similarity

    async def supplement(
        self,
        query: str,
        *,
        session_id: str,
        user_id: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """联网搜索 + 抓取 + 暂存，返回可注入 Prompt 的 [WEB] 区块与来源列表。"""
        if not self.enabled:
            return {"block": "", "sources": [], "staged": []}

        limit = max(1, min(3, top_k or self.settings.web_fallback_max_sources))
        search_q = sanitize_search_query(query) or query
        try:
            staged = await self.staging.search_and_stage(
                search_q,
                session_id=session_id,
                user_id=user_id,
                top_k=limit,
                origin="web_fallback",
            )
        except Exception as exc:  # noqa: BLE001 - 联网失败必须降级为纯本地
            log.warning("web_supplement_failed", query=query, error=str(exc))
            return {"block": "", "sources": [], "staged": []}

        if not staged:
            return {"block": "", "sources": [], "staged": []}

        lines: list[str] = []
        sources: list[dict[str, Any]] = []
        for doc in staged:
            excerpt = (doc.get("excerpt") or "").strip()[:1200]
            if not excerpt:
                continue
            fetched = float(doc.get("fetched_at") or time.time())
            when = time.strftime("%Y-%m-%d", time.localtime(fetched))
            lines.append(
                f"[WEB: {(doc.get('title') or doc.get('filename') or '网页资料')[:80]} | "
                f"{doc.get('url') or ''} | 抓取 {when}]\n{excerpt}"
            )
            sources.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "title": doc.get("title") or doc.get("filename"),
                    "url": doc.get("url"),
                    "fetched_at": fetched,
                    "origin": doc.get("origin"),
                    "reused": doc.get("reused", False),
                }
            )

        if not lines:
            return {"block": "", "sources": [], "staged": staged}

        block = (
            "### [Web Supplements - 联网补充资料（非权威，引用时必须标注 URL 与抓取时间）]\n\n"
            + "\n\n".join(lines)
        )
        log.info("web_supplement_injected", query=query, sources=len(sources))
        return {"block": block, "sources": sources, "staged": staged}
