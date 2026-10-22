"""Phase P2 tests: optional API token authentication boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app


async def _make_client(tmp_path: Path, api_token: str | None):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "auth.db",
        data_dir=tmp_path / "data",
        api_token=SecretStr(api_token) if api_token else None,
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, container


class TestApiToken:
    @pytest.mark.asyncio
    async def test_disabled_by_default_allows_all_requests(self, tmp_path):
        client, container = await _make_client(tmp_path, None)
        try:
            resp = await client.get("/api/rag/stats")
            assert resp.status_code == 200
        finally:
            await client.aclose()
            await container.aclose()

    @pytest.mark.asyncio
    async def test_enabled_requires_bearer_token(self, tmp_path):
        client, container = await _make_client(tmp_path, "s3cret")
        try:
            assert (await client.get("/health")).status_code == 200

            no_token = await client.get("/api/rag/stats")
            assert no_token.status_code == 401

            wrong = await client.get(
                "/api/rag/stats", headers={"Authorization": "Bearer wrong"}
            )
            assert wrong.status_code == 401

            ok = await client.get(
                "/api/rag/stats", headers={"Authorization": "Bearer s3cret"}
            )
            assert ok.status_code == 200

            via_header = await client.get(
                "/api/rag/stats", headers={"X-API-Token": "s3cret"}
            )
            assert via_header.status_code == 200
        finally:
            await client.aclose()
            await container.aclose()

    @pytest.mark.asyncio
    async def test_write_endpoints_are_protected_too(self, tmp_path):
        client, container = await _make_client(tmp_path, "s3cret")
        try:
            resp = await client.post(
                "/api/courses/default/knowledge",
                json={"name": "未授权写入"},
            )
            assert resp.status_code == 401

            authed = await client.post(
                "/api/courses/default/knowledge",
                headers={"Authorization": "Bearer s3cret"},
                json={"name": "授权写入"},
            )
            assert authed.status_code == 200
        finally:
            await client.aclose()
            await container.aclose()
