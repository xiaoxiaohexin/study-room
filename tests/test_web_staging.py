"""临时知识库（Staging）与联网补充（A/B 通道）测试。"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes.llama_compat import _build_rag_and_memory_block
from app.core.config import Settings
from app.core.container import AppContainer
from app.llm.fake import FakeLLM
from app.main import create_app
from app.tools.base import ToolContext
from app.web.distill import quality_score
from app.web.fetcher import (
    FetchedPage,
    WebFetcher,
    WebFetchError,
    content_hash,
    validate_public_url,
)
from app.web.mcp_search import _coerce_search_results, _parse_jsonrpc_payload
from app.web.search import (
    BingProvider,
    SearxngProvider,
    WebSearchError,
    WebSearchResult,
)
from app.web.staging import WebStagingError
from app.web.supplement import WebSupplementService

PAGE_TEXT = (
    "# 光合作用详解\n\n"
    "光合作用是绿色植物利用光能，把二氧化碳和水转化为有机物并释放氧气的过程。\n"
    "它分为光反应与暗反应两个阶段：光反应在类囊体膜上进行，产生 ATP 与 NADPH；"
    "暗反应在叶绿体基质中进行，通过卡尔文循环固定二氧化碳。\n"
    "影响光合作用速率的环境因素包括光照强度、二氧化碳浓度和温度等。\n"
)


class FakeFetcher:
    def __init__(self, text: str = PAGE_TEXT, title: str = "光合作用详解"):
        self.text = text
        self.title = title
        self.calls: list[str] = []

    async def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        if "fail" in url:
            raise WebFetchError("mock fetch failure")
        return FetchedPage(
            url=url,
            final_url=url,
            title=self.title,
            text=self.text,
            content_hash=content_hash(self.text),
            fetched_at=time.time(),
            content_type="text/html",
        )


@pytest.fixture
async def staging_app(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "staging.db",
        data_dir=tmp_path / "data",
        web_enabled=True,
        web_allow_loopback=True,
        web_fallback_mode="auto",
        web_fallback_max_sources=2,
        staging_max_documents_per_session=5,
        staging_ttl_days=7,
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    container.web_staging.fetcher = FakeFetcher()  # type: ignore[assignment]
    yield app, container, tmp_path
    await container.aclose()


@pytest.fixture
async def staging_client(staging_app):
    app, _, _ = staging_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestFetcherSafety:
    @pytest.mark.asyncio
    async def test_blocks_private_and_metadata_addresses(self):
        settings = Settings(offline_mode=True, web_enabled=True)
        with pytest.raises(WebFetchError):
            await validate_public_url("http://127.0.0.1/admin", settings)
        with pytest.raises(WebFetchError):
            await validate_public_url("http://169.254.169.254/latest/meta-data", settings)
        with pytest.raises(WebFetchError):
            await validate_public_url("http://192.168.1.10/", settings)

    @pytest.mark.asyncio
    async def test_loopback_requires_explicit_opt_in(self):
        settings = Settings(offline_mode=True, web_enabled=True, web_allow_loopback=True)
        url = await validate_public_url("http://127.0.0.1:8011/page.html", settings)
        assert url.startswith("http://127.0.0.1:8011")

    @pytest.mark.asyncio
    async def test_fetch_extracts_html_via_http_transport(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            html_body = (
                "<html><head><title>示例标题</title></head><body>"
                "<nav>导航应忽略</nav><h1>光合作用</h1>"
                "<p>光合作用把光能转化为化学能。</p></body></html>"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=html_body.encode("utf-8"),
            )

        settings = Settings(offline_mode=True, web_enabled=True, web_allow_loopback=True)
        fetcher = WebFetcher(settings, transport=httpx.MockTransport(handler))
        page = await fetcher.fetch("http://127.0.0.1:9/page.html")
        assert page.title == "示例标题"
        assert "光合作用把光能转化为化学能" in page.text
        assert page.content_hash

    @pytest.mark.asyncio
    async def test_redirect_to_private_address_is_blocked_before_following(self):
        import httpx

        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            if len(requests) == 1:
                return httpx.Response(
                    302,
                    headers={"location": "http://169.254.169.254/latest/meta-data"},
                )
            return httpx.Response(
                200, headers={"content-type": "text/html"}, content=b"<p>should not reach</p>"
            )

        settings = Settings(offline_mode=True, web_enabled=True, web_allow_loopback=True)
        fetcher = WebFetcher(settings, transport=httpx.MockTransport(handler))
        with pytest.raises(WebFetchError):
            await fetcher.fetch("http://127.0.0.1:9/redirect")
        assert len(requests) == 1, "redirect target must be validated before any follow-up request"

    def test_quality_score_rewards_long_structured_text(self):
        short = quality_score("短文本", "https://example.com")
        long_text = quality_score(PAGE_TEXT * 30, "https://edu.example.edu")
        assert long_text > short
        assert 0.0 <= short <= 1.0 and 0.0 <= long_text <= 1.0


class TestMcpParsingHelpers:
    def test_parse_sse_and_json_payloads(self):
        sse = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
        messages = _parse_jsonrpc_payload(sse)
        assert messages and messages[0]["result"]["ok"] is True
        assert _parse_jsonrpc_payload('{"jsonrpc":"2.0","id":2,"result":{}}')[0]["id"] == 2
        assert _parse_jsonrpc_payload("not json") == []

    def test_coerce_common_search_shapes(self):
        results = _coerce_search_results(
            '{"results":[{"title":"A","url":"https://a.example","content":"sa"}]}'
        )
        assert results and results[0].url == "https://a.example"
        lines = _coerce_search_results(
            "标题一 https://a.example 摘要\n标题二 https://b.example"
        )
        assert {r.url for r in lines} == {"https://a.example", "https://b.example"}


class TestStagingService:
    @pytest.mark.asyncio
    async def test_stage_url_writes_staging_tier_only(self, staging_app):
        _, container, _ = staging_app
        doc = await container.web_staging.stage_url(
            "https://example.com/page", session_id="s1", user_id="default_user"
        )
        assert doc["doc_id"]
        assert doc["tier"] == "staging"
        assert doc["source_type"] == "web"

        # staging 文档不出现在标准知识库列表
        standard = await container.sqlite.list_documents(tier="standard")
        assert all(d["doc_id"] != doc["doc_id"] for d in standard)

        staging = await container.web_staging.list_documents(session_id="s1")
        assert len(staging) == 1
        # staging 向量不参与标准 RAG 检索
        hits = await container.rag_retriever.search(["光合作用"], namespace="default")
        assert hits == []

    @pytest.mark.asyncio
    async def test_same_content_is_deduplicated_per_session(self, staging_app):
        _, container, _ = staging_app
        first = await container.web_staging.stage_url(
            "https://example.com/a", session_id="s1", user_id="default_user"
        )
        second = await container.web_staging.stage_url(
            "https://example.com/b", session_id="s1", user_id="default_user"
        )
        assert first["doc_id"] == second["doc_id"]
        assert second["reused"] is True

    @pytest.mark.asyncio
    async def test_session_capacity_limit(self, staging_app):
        _, container, _ = staging_app
        fetcher = container.web_staging.fetcher
        for idx in range(5):
            fetcher.text = PAGE_TEXT + f"\n第 {idx} 篇不同内容段落。"  # type: ignore[attr-defined]
            await container.web_staging.stage_url(
                f"https://example.com/{idx}", session_id="s_cap", user_id="default_user"
            )
        fetcher.text = PAGE_TEXT + "\n第 6 篇不同内容段落。"  # type: ignore[attr-defined]
        with pytest.raises(WebStagingError):
            await container.web_staging.stage_url(
                "https://example.com/5", session_id="s_cap", user_id="default_user"
            )

    @pytest.mark.asyncio
    async def test_ttl_cleanup_removes_expired_staging_documents(self, staging_app):
        _, container, _ = staging_app
        doc = await container.web_staging.stage_url(
            "https://example.com/ttl", session_id="s_ttl", user_id="default_user"
        )
        await container.sqlite.conn.execute(
            "UPDATE documents SET created_at = ? WHERE doc_id = ?",
            (time.time() - 30 * 86400, doc["doc_id"]),
        )
        await container.sqlite.conn.commit()
        removed = await container.web_staging.cleanup_expired()
        assert removed == 1
        assert await container.sqlite.get_document(doc["doc_id"]) is None


class TestDistillAndPromote:
    @pytest.mark.asyncio
    async def test_distill_creates_candidates_and_promote_into_standard_kb(self, staging_app):
        _, container, _ = staging_app
        doc = await container.web_staging.stage_url(
            "https://example.com/photosynthesis",
            session_id="s_learn",
            user_id="default_user",
        )
        container.web_distiller.llm = FakeLLM(
            [
                '{"candidates": [{"title": "光合作用", "summary": "植物把光能转化为化学能", '
                '"content": "光反应 + 暗反应（卡尔文循环）", "raw_quote": "光合作用是绿色植物利用光能", '
                '"difficulty": 0.4}]}'
            ]
        )
        candidates = await container.web_distiller.distill_document(
            doc["doc_id"], user_id="default_user"
        )
        assert len(candidates) == 1
        assert candidates[0]["title"] == "光合作用"
        staged_rows = await container.sqlite.list_documents(
            tier="staging", session_id="s_learn"
        )
        assert staged_rows[0]["status"] == "distilled"
        assert float(staged_rows[0]["quality_score"]) > 0

        result = await container.web_promoter.promote(
            staging_doc_ids=[doc["doc_id"]],
            candidate_ids=[candidates[0]["id"]],
            target_course_id="course_bio",
            user_id="default_user",
        )
        assert result["ok"] is True
        assert len(result["promoted_documents"]) == 1
        assert len(result["promoted_knowledge_points"]) == 1

        # 标准库里出现该文档与知识节点
        standard_docs = await container.sqlite.list_documents(
            namespace="course_bio", tier="standard"
        )
        assert any(d["source_type"] == "web" and d["url"] for d in standard_docs)

        kps = await container.learning_store.list_knowledge_points(
            course_id="course_bio", user_id="default_user"
        )
        assert [kp["name"] for kp in kps] == ["光合作用"]
        assert kps[0]["provenance"]["url"].startswith("https://example.com")
        assert kps[0]["provenance"]["doc_id"]

        # 晋升后标准 RAG 可以检索到（词面/哈希向量都能命中）
        hits = await container.rag_retriever.search(["光合作用"], namespace="course_bio")
        assert hits, "promoted web document should be retrievable from the standard KB"

        # 撤销晋升：标准文档与知识节点同时删除
        revoke = await container.web_promoter.revoke(doc["doc_id"], user_id="default_user")
        assert revoke["ok"] is True
        remaining = await container.learning_store.list_knowledge_points(
            course_id="course_bio", user_id="default_user"
        )
        assert remaining == []
        standard_after = await container.sqlite.list_documents(
            namespace="course_bio", tier="standard"
        )
        assert standard_after == []
        staged_after = await container.web_staging.get_staging_document(doc["doc_id"])
        assert staged_after is not None and staged_after["status"] == "distilled"

    @pytest.mark.asyncio
    async def test_candidate_can_be_rejected(self, staging_app):
        _, container, _ = staging_app
        doc = await container.web_staging.stage_url(
            "https://example.com/reject", session_id="s_rej", user_id="default_user"
        )
        container.web_distiller.llm = FakeLLM(
            ['{"candidates": [{"title": "候选", "summary": "摘要", "content": "内容"}]}']
        )
        candidates = await container.web_distiller.distill_document(
            doc["doc_id"], user_id="default_user"
        )
        assert await container.web_distiller.reject_candidate(candidates[0]["id"]) is True
        pending = await container.web_distiller.list_candidates(session_id="s_rej")
        assert pending == []


class TestWebApi:
    @pytest.mark.asyncio
    async def test_status_and_stage_and_list_and_delete(self, staging_client, staging_app):
        _, container, _ = staging_app
        status = await staging_client.get("/api/web/status")
        assert status.status_code == 200
        assert status.json()["enabled"] is True

        preview = await staging_client.post(
            "/api/web/preview", json={"url": "https://example.com/preview"}
        )
        assert preview.status_code == 200
        assert "光合作用" in preview.json()["preview"]

        staged = await staging_client.post(
            "/api/web/stage",
            json={"url": "https://example.com/page", "session_id": "api_s1"},
        )
        assert staged.status_code == 200
        doc = staged.json()["document"]
        assert doc["doc_id"]

        listed = await staging_client.get(
            "/api/staging/documents", params={"session_id": "api_s1"}
        )
        assert listed.json()["count"] == 1

        deleted = await staging_client.delete(f"/api/staging/documents/{doc['doc_id']}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_distill_promote_revoke_via_api(self, staging_client, staging_app):
        _, container, _ = staging_app
        container.web_distiller.llm = FakeLLM(
            ['{"candidates": [{"title": "API 候选", "summary": "摘要", "content": "内容"}]}']
        )
        staged = await staging_client.post(
            "/api/web/stage",
            json={"url": "https://example.com/api-promote", "session_id": "api_s2"},
        )
        doc_id = staged.json()["document"]["doc_id"]

        distilled = await staging_client.post(f"/api/staging/documents/{doc_id}/distill")
        assert distilled.status_code == 200
        candidate_id = distilled.json()["candidates"][0]["id"]

        promoted = await staging_client.post(
            "/api/staging/promote",
            json={
                "staging_doc_ids": [doc_id],
                "candidate_ids": [candidate_id],
                "target_course_id": "course_api_web",
            },
        )
        assert promoted.status_code == 200
        payload = promoted.json()
        assert payload["ok"] is True
        assert payload["promoted_knowledge_points"]

        revoked = await staging_client.post(
            "/api/staging/revoke", json={"staging_doc_id": doc_id}
        )
        assert revoked.status_code == 200
        assert revoked.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_web_disabled_returns_403(self, tmp_path):
        settings = Settings(
            offline_mode=True,
            sqlite_path=tmp_path / "disabled.db",
            data_dir=tmp_path / "data",
            web_enabled=False,
        )
        app = create_app(settings)
        container = AppContainer(settings)
        await container.initialize()
        app.state.container = container
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/web/stage", json={"url": "https://example.com/x"}
            )
            assert resp.status_code == 403
            status = (await client.get("/api/web/status")).json()
            assert status["enabled"] is False
        await container.aclose()


class TestSupplementService:
    def test_insufficient_gate(self, tmp_path):
        services = _supplement_fixture(tmp_path)
        supplement, staging = services
        assert supplement.insufficient({"max_similarity": 0.2, "hits": 1}, 1) is True
        assert supplement.insufficient({"max_similarity": 0.9, "hits": 5}, 5) is False
        assert (
            supplement.insufficient(
                {"max_similarity": 0.1, "hits": 0},
                0,
                query="你好",
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_supplement_formats_web_block_and_stages(self, tmp_path):
        supplement, staging = _supplement_fixture(tmp_path)
        staged_doc = {
            "doc_id": "web123",
            "title": "光合作用详解",
            "filename": "光合作用详解.md",
            "url": "https://example.com/bio",
            "fetched_at": time.time(),
            "origin": "web_fallback",
            "excerpt": PAGE_TEXT[:300],
            "reused": False,
        }

        async def fake_search_and_stage(query, **kwargs):
            return [staged_doc]

        staging.search_and_stage = fake_search_and_stage  # type: ignore[assignment]
        result = await supplement.supplement(
            "光合作用", session_id="s1", user_id="default_user"
        )
        assert "[WEB:" in result["block"]
        assert "https://example.com/bio" in result["block"]
        assert result["sources"][0]["doc_id"] == "web123"


class _StubStaging:
    def __init__(self):
        self.search_available = True
        self.search_and_stage = None  # replaced per test

    async def search_and_stage(self, query, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError


class TestSearchProviders:
    @pytest.mark.asyncio
    async def test_bing_provider_parses_rss(self):
        import httpx

        rss = """<?xml version="1.0" encoding="utf-8" ?>
<rss version="2.0"><channel>
  <item>
    <title>光合作用 - 百度百科</title>
    <link>https://baike.baidu.com/item/光合作用</link>
    <description>&lt;b&gt;光合作用&lt;/b&gt;是绿色植物利用光能的过程。</description>
  </item>
  <item>
    <title>光合作用全过程 - 知乎</title>
    <link>https://www.zhihu.com/question/1</link>
    <description>光反应与暗反应。</description>
  </item>
</channel></rss>"""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["format"] == "rss"
            return httpx.Response(
                200, headers={"content-type": "application/rss+xml"}, text=rss
            )

        provider = BingProvider(transport=httpx.MockTransport(handler))
        results = await provider.search("光合作用", 2)
        assert len(results) == 2
        assert results[0].provider == "bing"
        assert results[0].url.startswith("https://baike.baidu.com/")
        assert "<b>" not in results[0].snippet
        assert "绿色植物" in results[0].snippet

    @pytest.mark.asyncio
    async def test_bing_connection_error_is_actionable(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        provider = BingProvider(transport=httpx.MockTransport(handler))
        with pytest.raises(WebSearchError) as exc_info:
            await provider.search("x")
        assert "无法连接 Bing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_searxng_connection_error_mentions_fallback(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        provider = SearxngProvider("http://127.0.0.1:8888", transport=httpx.MockTransport(handler))
        with pytest.raises(WebSearchError) as exc_info:
            await provider.search("x")
        message = str(exc_info.value)
        assert "无法连接 SearXNG" in message
        assert "WEB_SEARCH_FALLBACK_PROVIDER" in message

    @pytest.mark.asyncio
    async def test_search_chain_falls_back_to_bing(self, staging_app):
        import httpx

        _, container, _ = staging_app

        dead_calls = 0

        def dead(request: httpx.Request) -> httpx.Response:
            nonlocal dead_calls
            dead_calls += 1
            raise httpx.ConnectError("connection refused", request=request)

        rss = (
            '<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"><channel>'
            "<item><title>光合作用</title><link>https://example.com/bio</link>"
            "<description>光反应与暗反应</description></item></channel></rss>"
        )

        def bing(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "application/rss+xml"}, text=rss
            )

        container.web_staging.search_providers = [
            SearxngProvider("http://127.0.0.1:8888", transport=httpx.MockTransport(dead)),
            BingProvider(transport=httpx.MockTransport(bing)),
        ]
        results = await container.web_staging.search("光合作用", 3)
        assert results and results[0].provider == "bing"
        # 冷却期内第二次搜索不再重试宕机的主服务
        second = await container.web_staging.search("光合作用", 3)
        assert second and second[0].provider == "bing"
        assert dead_calls == 1

    @pytest.mark.asyncio
    async def test_search_cooldown_reports_temporary_unavailable(self, staging_app):
        import httpx

        _, container, _ = staging_app

        def dead(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        container.web_staging.search_providers = [
            SearxngProvider("http://127.0.0.1:8888", transport=httpx.MockTransport(dead))
        ]
        with pytest.raises(WebSearchError):
            await container.web_staging.search("x", 2)
        with pytest.raises(WebSearchError) as exc_info:
            await container.web_staging.search("x", 2)
        assert "暂不可用" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_status_reports_search_diagnostics(self, staging_client):
        status = await staging_client.get("/api/web/status")
        payload = status.json()
        assert payload["enabled"] is True
        assert payload["search_primary"] in {"none", "searxng", "tavily"}
        assert payload["search_fallback"] in {"none", "bing"}
        assert isinstance(payload["search_providers"], list)
        assert isinstance(payload["mcp_configured"], bool)


class TestBChannelAndWebTool:
    @pytest.mark.asyncio
    async def test_block_skips_all_retrieval_when_policy_is_none(self, staging_app):
        _, container, _ = staging_app
        container.rag_retriever.search = AsyncMock(return_value=[])  # type: ignore[method-assign]
        container.memory_manager.retrieve_all = AsyncMock(return_value=[])  # type: ignore[method-assign]

        block, rag_sources, web_sources = await _build_rag_and_memory_block(
            container=container,
            user_query="你好",
            user_id="default_user",
            namespace="default",
            session_id="s1",
            retrieval_mode="none",
            allow_web_fallback=False,
        )

        assert block == ""
        assert rag_sources == []
        assert web_sources == []
        container.rag_retriever.search.assert_not_awaited()
        container.memory_manager.retrieve_all.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_block_injects_web_supplement_when_local_rag_insufficient(self, staging_app):
        _, container, _ = staging_app
        container.web_staging.search_providers = [object()]  # type: ignore[list-item]
        container.rag_retriever.search = AsyncMock(return_value=[])  # type: ignore[method-assign]
        container.rag_retriever.confidence = AsyncMock(  # type: ignore[method-assign]
            return_value={"max_similarity": 0.1, "avg_top_similarity": 0.05, "hits": 0}
        )
        container.memory_manager.retrieve_all = AsyncMock(return_value=[])  # type: ignore[method-assign]
        container.learning_store.build_learner_profile_text = AsyncMock(  # type: ignore[method-assign]
            return_value=""
        )

        async def fake_supplement(query, **kwargs):
            return {
                "block": (
                    "### [Web Supplements - 联网补充资料]\n\n"
                    "[WEB: 光合作用 | https://example.com/bio | 抓取 2026-10-21]\n"
                    "光合作用分为光反应与暗反应。"
                ),
                "sources": [
                    {"doc_id": "web1", "title": "光合作用", "url": "https://example.com/bio"}
                ],
                "staged": [],
            }

        container.web_supplement.supplement = fake_supplement  # type: ignore[method-assign]
        block, rag_sources, web_sources = await _build_rag_and_memory_block(
            container=container,
            user_query="光合作用",
            user_id="default_user",
            namespace="default",
            session_id="s1",
        )
        assert "[WEB:" in block
        assert "https://example.com/bio" in block
        assert rag_sources == []
        assert web_sources and web_sources[0]["doc_id"] == "web1"

    @pytest.mark.asyncio
    async def test_web_block_absent_when_local_rag_is_confident(self, staging_app):
        _, container, _ = staging_app
        container.web_staging.search_providers = [object()]  # type: ignore[list-item]
        container.rag_retriever.search = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _doc(f"高置信本地教材切片 {idx}", doc_id=f"doc_{idx}") for idx in range(3)
            ]
        )
        container.rag_retriever.confidence = AsyncMock(  # type: ignore[method-assign]
            return_value={"max_similarity": 0.92, "avg_top_similarity": 0.9, "hits": 6}
        )
        container.memory_manager.retrieve_all = AsyncMock(return_value=[])  # type: ignore[method-assign]
        container.learning_store.build_learner_profile_text = AsyncMock(  # type: ignore[method-assign]
            return_value=""
        )
        called = False

        async def fake_supplement(query, **kwargs):
            nonlocal called
            called = True
            return {"block": "", "sources": [], "staged": []}

        container.web_supplement.supplement = fake_supplement  # type: ignore[method-assign]
        block, _sources, web_sources = await _build_rag_and_memory_block(
            container=container,
            user_query="光合作用",
            user_id="default_user",
            namespace="default",
        )
        assert "高置信本地教材切片" in block
        assert web_sources == []
        assert called is False

    @pytest.mark.asyncio
    async def test_web_tool_registered_and_returns_search_results(self, staging_app):
        _, container, _ = staging_app
        assert "web_tool" in container.tool_registry._tools

        async def fake_search(query, top_k=3):
            return [
                WebSearchResult(
                    title="光合作用讲义",
                    url="https://example.com/bio",
                    snippet="光反应与暗反应",
                    provider="mock",
                )
            ]

        container.web_staging.search = fake_search  # type: ignore[method-assign]
        tool = container.tool_registry._tools["web_tool"]
        context = ToolContext(user_id="default_user", session_id="s1")
        result = await tool.run({"action": "search", "query": "光合作用", "stage": False}, context)
        assert result.ok is True
        assert "https://example.com/bio" in result.output

    @pytest.mark.asyncio
    async def test_web_tool_reports_disabled(self, tmp_path):
        settings = Settings(
            offline_mode=True,
            sqlite_path=tmp_path / "disabled2.db",
            data_dir=tmp_path / "data",
            web_enabled=False,
        )
        container = AppContainer(settings)
        await container.initialize()
        tool = container.tool_registry._tools["web_tool"]
        result = await tool.run(
            {"action": "search", "query": "x"}, ToolContext(user_id="default_user")
        )
        assert result.ok is False
        assert "未开启" in (result.error or "")
        await container.aclose()


def _doc(content: str, doc_id: str = "doc_1"):
    from app.rag.retriever import DocChunk

    return DocChunk(
        id=f"{doc_id}_0",
        doc_id=doc_id,
        chunk_index=0,
        content=content,
        score=0.92,
        filename="textbook.pdf",
        headers=["第一章"],
        page_start=1,
        page_end=1,
    )


def _supplement_fixture(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "supp.db",
        data_dir=tmp_path / "data",
        web_enabled=True,
        web_fallback_mode="auto",
    )
    staging = _StubStaging()
    supplement = WebSupplementService(settings=settings, staging=staging)  # type: ignore[arg-type]
    return supplement, staging
