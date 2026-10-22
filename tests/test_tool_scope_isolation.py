"""Phase P0-B tests: tool-layer session scope isolation (rag namespace / course)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.llm.fake import FakeLLM
from app.main import create_app
from app.schemas.requests import ChatRequest
from app.tools.base import ToolContext, ToolResult


@pytest.fixture
async def scope_app(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "scope.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    # Two namespaced documents with distinctive filenames.
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    apple = docs_dir / "apple_notes.txt"
    apple.write_text("苹果是红色水果，富含维生素。", encoding="utf-8")
    banana = docs_dir / "banana_notes.txt"
    banana.write_text("香蕉是黄色水果，富含钾元素。", encoding="utf-8")
    await container.ingestion_pipeline.ingest_file(apple, namespace="ns_a")
    await container.ingestion_pipeline.ingest_file(banana, namespace="ns_b")

    yield app, container

    await container.aclose()


@pytest.fixture
async def scope_client(scope_app):
    app, _ = scope_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestRagToolContextOnly:
    @pytest.mark.asyncio
    async def test_rag_tool_uses_context_namespace_and_ignores_argument(self, scope_app):
        _, container = scope_app
        rag_tool = container.tool_registry.get("rag_search_tool")
        assert rag_tool is not None

        ctx_a = ToolContext(user_id="u1", session_id="s1", rag_namespace="ns_a")
        res_a = await rag_tool.run(
            {"query": "水果", "namespace": "ns_b"},  # the argument must be ignored
            ctx_a,
        )
        assert res_a.ok
        assert "apple_notes.txt" in res_a.output
        assert "banana_notes.txt" not in res_a.output

        ctx_b = ToolContext(user_id="u1", session_id="s1", rag_namespace="ns_b")
        res_b = await rag_tool.run({"query": "水果"}, ctx_b)
        assert res_b.ok
        assert "banana_notes.txt" in res_b.output
        assert "apple_notes.txt" not in res_b.output

    def test_tool_context_backwards_compatible_defaults(self):
        ctx = ToolContext(user_id="u1")
        assert ctx.rag_namespace == "default"
        assert ctx.course_id is None


class TestToolEndpointsUseSessionScope:
    @pytest.mark.asyncio
    async def test_webui_tools_endpoint_resolves_session_namespace(self, scope_client):
        await scope_client.post(
            "/api/sessions",
            json={"user_id": "default_user", "session_id": "sess_scope", "rag_namespace": "ns_a"},
        )

        resp = await scope_client.post(
            "/tools",
            json={
                "tool": "rag_search_tool",
                "params": {"query": "水果", "session_id": "sess_scope"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        text = body.get("plain_text") or ""
        assert "apple_notes.txt" in text
        assert "banana_notes.txt" not in text

        # Explicit header override wins over the persisted session binding.
        resp_override = await scope_client.post(
            "/tools",
            headers={"X-RAG-Namespace": "ns_b"},
            json={
                "tool": "rag_search_tool",
                "params": {"query": "水果", "session_id": "sess_scope"},
            },
        )
        text_override = resp_override.json().get("plain_text") or ""
        assert "banana_notes.txt" in text_override
        assert "apple_notes.txt" not in text_override

    @pytest.mark.asyncio
    async def test_mcp_tools_call_resolves_session_namespace(self, scope_client):
        await scope_client.post(
            "/api/sessions",
            json={"user_id": "default_user", "session_id": "sess_mcp", "rag_namespace": "ns_b"},
        )
        resp = await scope_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "rag_search_tool",
                    "arguments": {"query": "水果", "session_id": "sess_mcp"},
                },
            },
        )
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        assert "banana_notes.txt" in text


class TestAgentLoopToolContext:
    @pytest.mark.asyncio
    async def test_agent_loop_propagates_session_scope_to_tools(self, scope_app):
        _, container = scope_app
        session = await container.state_manager.get_or_create_session(
            user_id="u1",
            session_id="sess_loop",
            rag_namespace="ns_b",
            course_id="math_calc",
        )

        captured: list[ToolContext] = []
        original_execute = container.tool_registry.execute

        async def capture_execute(name, arguments, ctx):
            captured.append(ctx)
            return ToolResult(ok=True, output="工具结果占位")

        container.tool_registry.execute = capture_execute  # type: ignore[assignment]
        fake_llm = FakeLLM(
            responses=[
                '[TOOL_CALL: rag_search_tool: {"query": "水果"}]',
                "Finish[已整理完成]",
            ]
        )
        container.agent_loop.llm = fake_llm
        try:
            req = ChatRequest(
                user_id="u1",
                session_id=session.session_id,
                message="查一下知识库里的水果资料",
                enable_query_expansion=False,
                enable_reflection=False,
            )
            events = [ev async for ev in container.agent_loop.run(req, session)]
            assert any(ev.type.value == "final" for ev in events)
        finally:
            container.tool_registry.execute = original_execute  # type: ignore[assignment]

        assert captured, "agent loop should have executed a tool"
        assert captured[0].rag_namespace == "ns_b"
        assert captured[0].course_id == "math_calc"
        assert captured[0].user_id == "u1"
        assert captured[0].session_id == "sess_loop"
