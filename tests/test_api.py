import pytest
from httpx import ASGITransport, AsyncClient
from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app


@pytest.fixture
async def app_instance(tmp_path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "test.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app
    await container.aclose()


@pytest.mark.asyncio
async def test_health_and_tools_endpoints(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "components" in data

        res_tools = await ac.get("/api/tools")
        assert res_tools.status_code == 200
        tools_data = res_tools.json()
        assert tools_data["count"] >= 3


@pytest.mark.asyncio
async def test_chat_sync_endpoint(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req = {
            "user_id": "test_user",
            "message": "你好",
            "enable_query_expansion": False,
        }
        res = await ac.post("/api/chat", json=req)
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert "answer" in data


@pytest.mark.asyncio
async def test_llama_compat_and_mcp_endpoints(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. /props (WebUI ServerLoadingSplash 探活必需)
        r_props = await ac.get("/props")
        assert r_props.status_code == 200
        props = r_props.json()
        assert "default_generation_settings" in props
        assert "chat_template_caps" in props

        # 2. /models & /v1/models & /slots
        r_models = await ac.get("/models")
        assert r_models.status_code == 200
        assert "data" in r_models.json()
        r_slots = await ac.get("/slots")
        assert r_slots.status_code == 200
        assert isinstance(r_slots.json(), list)

        # 3. /v1/chat/completions (非流式 + 自定义 RAG 自动检索注入)
        chat_payload = {
            "messages": [{"role": "user", "content": "请解释一下动态规划"}],
            "stream": False,
            "cache_prompt": True,
            "min_p": 0.05,
            "top_k": 40,
        }
        r_chat = await ac.post("/v1/chat/completions", json=chat_payload)
        assert r_chat.status_code == 200
        c_data = r_chat.json()
        assert c_data["object"] == "chat.completion"
        assert len(c_data["choices"]) == 1

        # 4. /v1/chat/completions (SSE 流式)
        chat_stream_payload = {
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        }
        r_stream = await ac.post("/v1/chat/completions", json=chat_stream_payload)
        assert r_stream.status_code == 200
        assert "data: [DONE]" in r_stream.text

        # 5. /mcp (MCP JSON-RPC 2.0 服务端点)
        r_mcp_list = await ac.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert r_mcp_list.status_code == 200
        mcp_tools = r_mcp_list.json()["result"]["tools"]
        tool_names = {t["name"] for t in mcp_tools}
        assert "rag_search_tool" in tool_names
        assert "quiz_tool" in tool_names

