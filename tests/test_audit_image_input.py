"""审计 P1-4 回归：本地图片只允许从受控上传目录读取。"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app
from app.schemas.messages import _encode_image_url, set_allowed_image_roots


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


def test_local_image_path_outside_upload_dir_is_rejected(tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"\x89PNG\r\n")
    set_allowed_image_roots([uploads])
    try:
        with pytest.raises(PermissionError):
            _encode_image_url(secret)

        inside = uploads / "ok.png"
        inside.write_bytes(b"\x89PNG\r\n")
        assert _encode_image_url(inside).startswith("data:image/png;base64,")
    finally:
        set_allowed_image_roots([])


async def test_chat_api_rejects_local_image_outside_upload_dir(tmp_path: Path):
    client, container = await _make_client(tmp_path)
    try:
        secret = tmp_path / "secret.png"
        secret.write_bytes(b"\x89PNG\r\n")
        resp = await client.post(
            "/api/chat",
            json={"user_id": "u1", "message": "看看这张图", "images": [str(secret)]},
        )
        assert resp.status_code == 400
    finally:
        await client.aclose()
        await container.aclose()