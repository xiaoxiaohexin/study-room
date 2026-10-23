"""审计 P0 回归：CORS 默认值收紧 + 跨站写请求拦截（CSRF）。"""

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
        # 显式指定，避免本机 .env / 环境变量影响断言
        cors_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), container


def test_default_cors_is_not_wildcard():
    default = Settings.model_fields["cors_origins"].default
    assert "*" not in default
    assert any("127.0.0.1" in origin for origin in default)


async def test_cross_origin_post_is_rejected(tmp_path: Path):
    client, container = await _make_client(tmp_path)
    try:
        blocked = await client.post(
            "/tools",
            json={"tool": "note_tool", "params": {"action": "list"}},
            headers={"Origin": "http://evil.example"},
        )
        assert blocked.status_code == 403

        allowed = await client.post(
            "/tools",
            json={"tool": "note_tool", "params": {"action": "list"}},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert allowed.status_code == 200
    finally:
        await client.aclose()
        await container.aclose()