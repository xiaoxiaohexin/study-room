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


@pytest.mark.asyncio
async def test_rag_search_endpoints_and_page_preservation(app_instance, tmp_path):
    """验证 RAG 检索接口返回物理页码，且 /api/rag/search 与 /api/v1/rag/search 路由别名均正常工作。"""
    from httpx import ASGITransport, AsyncClient
    from app.storage.vector import VectorPoint

    container = app_instance.state.container
    col = container.ingestion_pipeline._collection_name("default")
    await container.vector_store.ensure_collection(col, container.embedder.dim)

    # 1. 模拟插入含有 page_start 与 page_end 标头的向量切片
    vecs = await container.embedder.embed(["以太网交换机工作在数据链路层"])
    test_point = VectorPoint(
        id="doc_mock_1",
        vector=vecs[0],
        payload={
            "doc_id": "doc_mock",
            "filename": "computer_networks.pdf",
            "chunk_index": 3,
            "content": "以太网交换机工作在数据链路层，通过 MAC 地址表进行帧转发。",
            "headers": ["第3章 数据链路层", "3.4 以太网技术"],
            "tokens": 45,
            "page_start": 42,
            "page_end": 43,
        },
    )
    await container.vector_store.upsert(col, [test_point])

    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 2. 测试 POST /api/rag/search (基础路由)
        resp_rag = await ac.post("/api/rag/search", json={"query": "以太网交换机", "top_k": 3})
        assert resp_rag.status_code == 200
        data_rag = resp_rag.json()
        assert data_rag["count"] >= 1
        item_rag = data_rag["results"][0]
        assert item_rag["doc_id"] == "doc_mock"
        assert item_rag["filename"] == "computer_networks.pdf"
        assert item_rag["page_start"] == 42
        assert item_rag["page_end"] == 43
        assert item_rag["chunk_index"] == 3

        # 3. 测试 POST /api/v1/rag/search (前端画布路由别名)
        resp_v1 = await ac.post("/api/v1/rag/search", json={"query": "以太网交换机", "top_k": 3})
        assert resp_v1.status_code == 200
        data_v1 = resp_v1.json()
        assert data_v1["count"] >= 1
        item_v1 = data_v1["results"][0]
        assert item_v1["page_start"] == 42
        assert item_v1["page_end"] == 43

        # 4. 验证空查询保护
        resp_empty = await ac.post("/api/v1/rag/search", json={"query": "   ", "top_k": 3})
        assert resp_empty.status_code == 200
        assert resp_empty.json()["count"] == 0

        # 5. 验证 DocChunk 引用文本格式化方法
        retrieved_docs = await container.rag_retriever.search(["以太网交换机"], top_k=1)
        assert len(retrieved_docs) >= 1
        doc = retrieved_docs[0]
        assert doc.page_label == "P.42-43"
        citation = doc.format_citation()
        assert "[DOCUMENT: computer_networks.pdf P.42-43 (第3章 数据链路层 > 3.4 以太网技术)]" == citation


