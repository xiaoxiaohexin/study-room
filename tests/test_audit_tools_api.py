"""审计 P0 回归：/tools 未知工具 404、执行失败 400，不能以 HTTP 200 伪装成功。"""

from __future__ import annotations

from pathlib import Path

from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app


async def _make_client(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "api.db",
        data_dir=tmp_path / "data",
        cors_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), container


async def test_tools_endpoint_unknown_tool_returns_404(tmp_path: Path):
    client, container = await _make_client(tmp_path)
    try:
        resp = await client.post("/tools", json={"tool": "nope", "params": {}})
        assert resp.status_code == 404
    finally:
        await client.aclose()
        await container.aclose()


async def test_tools_endpoint_failure_returns_400(tmp_path: Path):
    client, container = await _make_client(tmp_path)
    try:
        resp = await client.post(
            "/tools", json={"tool": "note_tool", "params": {"action": "read"}}
        )
        assert resp.status_code == 400
        assert "error" in resp.json()
    finally:
        await client.aclose()
        await container.aclose()


async def test_mcp_unknown_tool_returns_jsonrpc_error(tmp_path: Path):
    client, container = await _make_client(tmp_path)
    try:
        resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "nope", "arguments": {}},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["error"]["code"] == -32602
    finally:
        await client.aclose()
        await container.aclose()