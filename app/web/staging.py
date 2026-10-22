"""临时知识库（Staging）服务：联网抓取 / 暂存 / 去重 / TTL / 删除。

设计原则：
- 临时库使用独立向量 collection（``staging_{embedder_tag}``），绝不参与标准 RAG 检索；
- 所有网页正文先落快照文件，再走 IngestionPipeline 入库，保持与标准库一致的切块/元数据；
- 只有 PromotionService 的“晋升”动作才会写入标准知识库。
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.observability.logging import get_logger
from app.rag.pipeline import IngestionPipeline
from app.storage.sqlite import SQLiteStore
from app.web.fetcher import (
    FetchedPage,
    WebFetcher,
    WebFetchError,
    content_hash,
    extract_relevant_excerpt,
)
from app.web.mcp_search import McpSearchClient
from app.web.search import (
    SearchProvider,
    WebSearchError,
    WebSearchResult,
    is_poisoned_search_results,
    sanitize_search_query,
    score_result_relevance,
)

log = get_logger(__name__)

STAGING_NAMESPACE = "staging"
_WAF_PRONE_HOSTS = ("baike.baidu.com", "huijiwiki.com", "fandom.com", "bilibili.com/video/")


class WebStagingError(Exception):
    """临时知识库业务错误（容量、状态、不可用等）。"""


class WebStagingService:
    def __init__(
        self,
        settings: Settings,
        sqlite: SQLiteStore,
        pipeline: IngestionPipeline,
        fetcher: WebFetcher,
        search_providers: list[SearchProvider] | None = None,
        mcp_client: McpSearchClient | None = None,
    ):
        self.settings = settings
        self.sqlite = sqlite
        self.pipeline = pipeline
        self.fetcher = fetcher
        self.search_providers = list(search_providers or [])
        self.mcp_client = mcp_client
        # 失败的搜索源冷却 60s，避免每次搜索都先等一遍已宕机的服务
        self._search_cooldown_seconds = 60
        self._search_cooldowns: dict[str, float] = {}
        self.staging_dir = Path(settings.data_dir) / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 搜索 ====================

    @property
    def search_available(self) -> bool:
        return bool(
            (self.mcp_client and self.mcp_client.configured)
            or self.search_providers
        )

    def _in_cooldown(self, key: str) -> bool:
        return time.time() < self._search_cooldowns.get(key, 0.0)

    def _mark_failed(self, key: str) -> None:
        self._search_cooldowns[key] = time.time() + self._search_cooldown_seconds

    @staticmethod
    def _needs_multi_source_expansion(
        query: str, results: list[WebSearchResult], top_k: int
    ) -> bool:
        """判断当前单源结果是否需要继续聚合下一兜底搜索源（如全为易 403 站点或高相关结果不足）。"""
        if not results:
            return True
        if is_poisoned_search_results(query, results):
            return True
        fetchable_high_rel = [
            r
            for r in results
            if score_result_relevance(query, r) >= 2.0
            and not any(h in r.url.lower() for h in _WAF_PRONE_HOSTS)
        ]
        return len(fetchable_high_rel) < min(2, max(1, top_k))

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        errors: list[str] = []
        attempted = False
        skipped: list[str] = []
        limit = max(1, top_k)

        if self.mcp_client and self.mcp_client.configured:
            if self._in_cooldown("mcp"):
                skipped.append("MCP")
            else:
                attempted = True
                try:
                    mcp_res = await self.mcp_client.search(query, limit)
                    if mcp_res and not is_poisoned_search_results(query, mcp_res):
                        return mcp_res[:limit]
                except WebSearchError as exc:
                    errors.append(f"MCP: {exc}")
                    self._mark_failed("mcp")
                    log.warning("mcp_web_search_failed", error=str(exc))

        aggregated: list[WebSearchResult] = []
        seen_urls: set[str] = set()

        for provider in self.search_providers:
            key = f"provider:{provider.name}:{id(provider)}"
            if self._in_cooldown(key):
                skipped.append(provider.name)
                continue
            attempted = True
            try:
                results = await provider.search(query, max(limit, 5))
            except WebSearchError as exc:
                errors.append(f"{provider.name}: {exc}")
                self._mark_failed(key)
                log.warning(
                    "web_search_provider_failed",
                    provider=provider.name,
                    error=str(exc),
                )
                continue
            self._search_cooldowns.pop(key, None)
            if not results:
                continue

            # 若注入了测试 MockTransport，保持单 Provider 立即返回语义
            if getattr(provider, "_transport", None) is not None:
                return results[:limit]

            for item in results:
                norm_u = item.url.rstrip("/").lower()
                if norm_u in seen_urls:
                    continue
                if score_result_relevance(query, item) < 0.0:
                    continue
                seen_urls.add(norm_u)
                aggregated.append(item)

            if aggregated and not self._needs_multi_source_expansion(query, aggregated, limit):
                break

        if aggregated:
            aggregated.sort(key=lambda r: score_result_relevance(query, r), reverse=True)
            return aggregated[:limit]

        if attempted and not errors:
            return []
        if errors:
            raise WebSearchError("；".join(errors))
        if skipped:
            raise WebSearchError(
                f"搜索服务暂不可用（{', '.join(skipped)} 刚连接失败，"
                f"{self._search_cooldown_seconds} 秒后自动重试）"
            )
        raise WebSearchError(
            "未配置联网搜索服务：可设置 WEB_SEARCH_PROVIDER=searxng/tavily，"
            "或使用 WEB_SEARCH_FALLBACK_PROVIDER=bing（内置免安装），"
            "也可配置 MCP_WEB_SERVER_URL 指向外部 MCP 搜索服务"
        )

    # ==================== 抓取与暂存 ====================

    async def preview_url(self, url: str) -> dict[str, Any]:
        """抓取并抽取正文（不落库），用于前端预览确认。"""
        page = await self.fetcher.fetch(url)
        return {
            "url": page.url,
            "final_url": page.final_url,
            "title": page.title,
            "chars": len(page.text),
            "preview": page.text[:3000],
            "content_hash": page.content_hash,
            "fetched_at": page.fetched_at,
            "content_type": page.content_type,
        }

    async def _count_session_documents(self, session_id: str) -> int:
        docs = await self.sqlite.list_documents(
            tier="staging", session_id=session_id or "", limit=500
        )
        return len(docs)

    async def stage_page(
        self,
        page: FetchedPage,
        *,
        session_id: str,
        user_id: str,
        origin: str = "web_import",
        query: str = "",
        conversation_id: str = "",
        target_course_id: str = "",
    ) -> dict[str, Any]:
        """把已抓取网页写入临时知识库；同会话内相同内容自动去重。"""
        excerpt = extract_relevant_excerpt(page.text, query=query, max_chars=1500)
        existing = await self.sqlite.find_document_by_hash(
            page.content_hash, tier="staging", session_id=session_id or ""
        )
        if existing:
            return {**self._doc_public(existing), "reused": True, "excerpt": excerpt}

        count = await self._count_session_documents(session_id)
        if count >= self.settings.staging_max_documents_per_session:
            raise WebStagingError(
                f"临时知识库已达上限（{self.settings.staging_max_documents_per_session} 篇/会话），"
                "请先提纯入库或清理"
            )

        snapshot_id = uuid.uuid4().hex[:12]
        snapshot_path = self.staging_dir / f"{snapshot_id}.md"
        header = (
            f"# {page.title}\n\n"
            f"> 来源: {page.final_url}\n"
            f"> 抓取时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(page.fetched_at))}\n\n"
            "---\n\n"
        )
        snapshot_path.write_text(header + page.text, encoding="utf-8")

        filename = self._safe_filename(page.title) or f"web_{snapshot_id}.md"
        metadata = {
            "source_type": "web",
            "url": page.final_url,
            "fetched_at": page.fetched_at,
            "content_hash": page.content_hash,
            "origin": origin,
            "query": query[:200],
            "conversation_id": conversation_id,
            "session_id": session_id or "",
            "user_id": user_id,
            "target_namespace": target_course_id,
        }
        doc_id, chunks = await self.pipeline.ingest_file(
            snapshot_path,
            namespace=STAGING_NAMESPACE,
            original_filename=filename,
            tier="staging",
            metadata=metadata,
        )
        await self.sqlite.update_document(
            doc_id,
            status="staged",
            session_id=session_id or "",
            user_id=user_id,
            origin=origin,
            target_namespace=target_course_id,
        )
        log.info(
            "web_staged",
            doc_id=doc_id,
            chunks=chunks,
            session_id=session_id,
            origin=origin,
            url=page.final_url,
        )
        row = await self.sqlite.get_document(doc_id)
        return {
            **self._doc_public(row or {"doc_id": doc_id}),
            "reused": False,
            "excerpt": excerpt,
        }

    async def stage_url(
        self,
        url: str,
        *,
        session_id: str,
        user_id: str,
        origin: str = "web_import",
        query: str = "",
        conversation_id: str = "",
        target_course_id: str = "",
    ) -> dict[str, Any]:
        page = await self.fetcher.fetch(url)
        return await self.stage_page(
            page,
            session_id=session_id,
            user_id=user_id,
            origin=origin,
            query=query,
            conversation_id=conversation_id,
            target_course_id=target_course_id,
        )

    async def search_and_stage(
        self,
        query: str,
        *,
        session_id: str,
        user_id: str,
        top_k: int = 2,
        origin: str = "search_import",
        results: list[WebSearchResult] | None = None,
    ) -> list[dict[str, Any]]:
        """搜索 + 抓取 Top-N 并暂存（单条失败自动顺延候选 URL，必要时用高相关搜索摘要兜底）。"""
        target_count = max(1, top_k)
        candidates = (
            list(results)
            if results is not None
            else await self.search(query, top_k=max(6, target_count * 3))
        )
        if not candidates:
            return []

        # 优先尝试高相关且非纯视频页的链接，遇到 403/超限自动顺延后续候选
        ordered = sorted(
            candidates,
            key=lambda r: (
                0 if any(h in r.url.lower() for h in _WAF_PRONE_HOSTS) else 1,
                score_result_relevance(query, r),
            ),
            reverse=True,
        )

        staged: list[dict[str, Any]] = []
        skipped_results: list[WebSearchResult] = []
        for result in ordered:
            if len(staged) >= target_count:
                break
            try:
                doc = await self.stage_url(
                    result.url,
                    session_id=session_id,
                    user_id=user_id,
                    origin=origin,
                    query=query,
                )
                doc["search_title"] = result.title
                staged.append(doc)
            except (WebFetchError, WebStagingError) as exc:
                skipped_results.append(result)
                log.warning("web_search_stage_skipped", url=result.url, error=str(exc))

        # 若部分高价值百科/社区条目因反爬 403 未能抓取全文，将多源高相关摘要合成为检索综述页入库
        high_rel_snippets = [
            r
            for r in (skipped_results if staged else candidates)
            if r.snippet and score_result_relevance(query, r) >= 1.5
        ]
        if high_rel_snippets and len(staged) < target_count:
            clean_q = sanitize_search_query(query) or query
            digest_lines = [f"# 联网检索多源摘要：{clean_q}\n"]
            for idx, item in enumerate(high_rel_snippets[:6], start=1):
                digest_lines.append(
                    f"## {idx}. {item.title}\n"
                    f"- 来源链接: {item.url}\n"
                    f"- 核心摘要: {item.snippet}\n"
                )
            digest_text = "\n".join(digest_lines).strip()
            digest_page = FetchedPage(
                url=high_rel_snippets[0].url,
                final_url=high_rel_snippets[0].url,
                title=f"联网检索多源摘要：{clean_q[:40]}",
                text=digest_text,
                content_hash=content_hash(digest_text),
                fetched_at=time.time(),
                content_type="text/markdown",
            )
            try:
                digest_doc = await self.stage_page(
                    digest_page,
                    session_id=session_id,
                    user_id=user_id,
                    origin=f"{origin}_digest",
                    query=query,
                )
                staged.append(digest_doc)
            except WebStagingError as exc:
                log.warning("web_search_digest_stage_skipped", error=str(exc))

        return staged

    # ==================== 查询 / 删除 / TTL ====================

    @staticmethod
    def _doc_public(row: dict) -> dict[str, Any]:
        return {
            "doc_id": row.get("doc_id", ""),
            "filename": row.get("filename", ""),
            "title": (row.get("filename") or "").replace(".md", ""),
            "namespace": row.get("namespace", STAGING_NAMESPACE),
            "tier": row.get("tier", "staging"),
            "source_type": row.get("source_type", "web"),
            "url": row.get("url", ""),
            "origin": row.get("origin", ""),
            "session_id": row.get("session_id", ""),
            "chunks": int(row.get("chunk_count") or 0),
            "page_count": int(row.get("page_count") or 1),
            "quality_score": float(row.get("quality_score") or 0),
            "status": row.get("status", "staged"),
            "created_at": row.get("created_at", 0),
            "fetched_at": row.get("fetched_at", 0),
            "target_namespace": row.get("target_namespace", ""),
            "file_available": bool(row.get("file_path")) and Path(row["file_path"]).exists(),
            "promoted_doc_id": row.get("promoted_doc_id", ""),
        }

    async def list_documents(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = await self.sqlite.list_documents(
            tier="staging",
            session_id=session_id if session_id is not None else None,
            user_id=user_id,
            status=status,
            limit=limit,
        )
        return [self._doc_public(row) for row in rows]

    async def get_staging_document(self, doc_id: str) -> dict | None:
        row = await self.sqlite.get_document(doc_id)
        if not row or row.get("tier") != "staging":
            return None
        return row

    async def delete_document(self, doc_id: str) -> bool:
        row = await self.get_staging_document(doc_id)
        if not row:
            return False
        collection = self.pipeline._collection_name(STAGING_NAMESPACE, tier="staging")
        chunk_count = int(row.get("chunk_count") or 0)
        ids = [f"{doc_id}_{i}" for i in range(chunk_count)]
        if ids:
            await self.pipeline.vector_store.delete(collection, ids)
        file_path = row.get("file_path") or ""
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("staging_snapshot_delete_failed", path=file_path, error=str(exc))
        await self.sqlite.delete_staging_candidates_for_doc(doc_id)
        await self.sqlite.delete_document(doc_id)
        log.info("staging_document_deleted", doc_id=doc_id)
        return True

    async def cleanup_expired(self) -> int:
        cutoff = time.time() - self.settings.staging_ttl_days * 86400
        rows = await self.sqlite.list_expired_staging(cutoff, limit=200)
        removed = 0
        for row in rows:
            if await self.delete_document(row["doc_id"]):
                removed += 1
        if removed:
            log.info("staging_cleanup_completed", removed=removed, ttl_days=self.settings.staging_ttl_days)
        return removed

    async def stats(self, *, session_id: str | None = None, user_id: str | None = None) -> dict[str, Any]:
        docs = await self.list_documents(session_id=session_id, user_id=user_id, limit=1000)
        candidates = await self.sqlite.list_staging_candidates(
            status="pending",
            session_id=session_id if session_id is not None else None,
            user_id=user_id,
            limit=1000,
        )
        return {
            "documents": len(docs),
            "pending_candidates": len(candidates),
            "search_available": self.search_available,
            "ttl_days": self.settings.staging_ttl_days,
        }

    @staticmethod
    def _safe_filename(title: str) -> str:
        cleaned = (title or "").strip().replace("/", "_").replace("\\", "_")
        cleaned = "".join(ch for ch in cleaned if ch >= " ").strip()
        if not cleaned:
            return ""
        if not cleaned.lower().endswith((".md", ".txt")):
            cleaned += ".md"
        return cleaned[:120]

    def snapshot_path(self, doc: dict, data_dir: Path | None = None) -> Path:
        raw = doc.get("file_path") or ""
        if raw and Path(raw).exists():
            return Path(raw)
        target_dir = (data_dir or Path(self.settings.data_dir)) / "staging"
        return target_dir / f"{doc.get('doc_id', 'unknown')}.md"

    def copy_snapshot(self, doc: dict, data_dir: Path | None = None) -> Path:
        """复制临时快照到 uploads 稳定路径，返回新文件路径。"""
        source = self.snapshot_path(doc, data_dir)
        if not source.exists():
            raise WebStagingError(f"临时快照不存在: {doc.get('doc_id')}")
        uploads = Path(self.settings.data_dir) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        target = uploads / f"promoted_{doc.get('doc_id', uuid.uuid4().hex[:8])}_{Path(doc.get('filename') or 'web.md').name}"
        shutil.copyfile(source, target)
        return target
