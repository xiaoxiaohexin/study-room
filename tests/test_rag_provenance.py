"""Phase P0-C tests: RAG document registry, real page preview and quote attribution."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app
from app.rag.pipeline import IngestionPipeline
from app.storage.vector import VectorPoint

FIXTURE_PDF = (
    Path(__file__).resolve().parents[1]
    / "llama-webui"
    / "tools"
    / "ui"
    / "tests"
    / "stories"
    / "fixtures"
    / "assets"
    / "example.pdf"
)


@pytest.fixture
async def rag_app(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "rag_prov.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app, container, tmp_path
    await container.aclose()


@pytest.fixture
async def rag_client(rag_app):
    app, _, _ = rag_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _ingest_pdf(client: AsyncClient, namespace: str) -> dict:
    content = FIXTURE_PDF.read_bytes()
    resp = await client.post(
        "/api/rag/ingest",
        data={"namespace": namespace},
        files={"file": ("example.pdf", content, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _ingest_markdown(client: AsyncClient, namespace: str, content: str | None = None) -> dict:
    md = content or (
        "第一页：苹果是红色水果。\n\n<!-- PAGE: 2 -->\n第二页：香蕉是黄色水果。"
    )
    resp = await client.post(
        "/api/rag/ingest",
        data={"namespace": namespace},
        files={"file": ("notes.md", md.encode("utf-8"), "text/markdown")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestDocumentRegistryAndPreview:
    @pytest.mark.asyncio
    async def test_ingest_registers_document_with_page_count_and_file(self, rag_client):
        result = await _ingest_pdf(rag_client, "ns_pdf")
        assert result["filename"] == "example.pdf"
        assert result["chunks"] > 0

        docs = (await rag_client.get("/api/rag/documents", params={"namespace": "ns_pdf"})).json()
        assert docs["total_docs"] == 1
        doc = docs["documents"][0]
        assert doc["doc_id"] == result["doc_id"]
        assert doc["page_count"] >= 3
        assert doc["file_available"] is True

    @pytest.mark.asyncio
    async def test_document_file_endpoint_serves_original_bytes(self, rag_client):
        result = await _ingest_pdf(rag_client, "ns_pdf")
        resp = await rag_client.get(
            f"/api/rag/documents/{result['doc_id']}/file", params={"namespace": "ns_pdf"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/pdf")
        assert resp.content[:4] == b"%PDF"

        missing_ns = await rag_client.get(
            f"/api/rag/documents/{result['doc_id']}/file", params={"namespace": "other"}
        )
        assert missing_ns.status_code == 404

    @pytest.mark.asyncio
    async def test_document_page_endpoint_returns_real_page_text(self, rag_client):
        result = await _ingest_pdf(rag_client, "ns_pdf")
        resp = await rag_client.get(
            f"/api/rag/documents/{result['doc_id']}/page",
            params={"namespace": "ns_pdf", "page": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_pages"] >= 3
        assert "USENIX" in body["text"]

        bad_page = await rag_client.get(
            f"/api/rag/documents/{result['doc_id']}/page",
            params={"namespace": "ns_pdf", "page": 999},
        )
        assert bad_page.status_code == 400

    @pytest.mark.asyncio
    async def test_markdown_page_markers_are_addressable(self, rag_client):
        result = await _ingest_markdown(rag_client, "ns_md")
        resp = await rag_client.get(
            f"/api/rag/documents/{result['doc_id']}/page",
            params={"namespace": "ns_md", "page": 2},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_pages"] == 2
        assert "香蕉" in body["text"]

    @pytest.mark.asyncio
    async def test_delete_removes_registry_and_file_access(self, rag_client):
        result = await _ingest_markdown(rag_client, "ns_del")
        delete_resp = await rag_client.delete(
            f"/api/rag/documents/{result['doc_id']}", params={"namespace": "ns_del"}
        )
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted_chunks"] >= 1

        file_resp = await rag_client.get(
            f"/api/rag/documents/{result['doc_id']}/file", params={"namespace": "ns_del"}
        )
        assert file_resp.status_code == 404
        docs = (await rag_client.get("/api/rag/documents", params={"namespace": "ns_del"})).json()
        assert docs["total_docs"] == 0


class TestQuoteAttribution:
    @pytest.mark.asyncio
    async def test_matching_quote_returns_real_chunk_and_page(self, rag_client):
        await _ingest_pdf(rag_client, "ns_pdf")
        resp = await rag_client.post(
            "/api/rag/attribute",
            json={
                "quote": "USENIX Example Paper Pekka Nikander Aalto University",
                "namespace": "ns_pdf",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matched"] is True
        chunk = body["chunk"]
        assert chunk["filename"] == "example.pdf"
        assert chunk["page_start"] == 1
        assert chunk["doc_id"]

    @pytest.mark.asyncio
    async def test_unrelated_quote_is_not_fabricated(self, rag_client):
        await _ingest_pdf(rag_client, "ns_pdf")
        resp = await rag_client.post(
            "/api/rag/attribute",
            json={"quote": "量子纠缠与超导材料的能隙关系", "namespace": "ns_pdf"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matched"] is False
        assert "chunk" not in body

    @pytest.mark.asyncio
    async def test_attribution_respects_namespace_isolation(self, rag_client):
        await _ingest_markdown(rag_client, "ns_a", "第一页：苹果是红色水果，富含维生素。")
        await _ingest_markdown(rag_client, "ns_b", "第一页：香蕉是黄色水果，富含钾元素。")
        resp_a = await rag_client.post(
            "/api/rag/attribute",
            json={"quote": "香蕉是黄色水果", "namespace": "ns_a"},
        )
        body_a = resp_a.json()
        # ns_a only has the apple document, so the banana quote must not be attributed there.
        assert body_a["matched"] is False

        resp_b = await rag_client.post(
            "/api/rag/attribute",
            json={"quote": "香蕉是黄色水果", "namespace": "ns_b"},
        )
        assert resp_b.json()["matched"] is True


class TestRegistryBackfill:
    @pytest.mark.asyncio
    async def test_backfill_registers_legacy_vector_documents(self, rag_app):
        _, container, tmp_path = rag_app
        uploads = tmp_path / "data" / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        legacy_file = uploads / "legacy.pdf"
        legacy_file.write_bytes(FIXTURE_PDF.read_bytes())

        collection = f"rag_default_{container.embedder.tag}"
        await container.vector_store.ensure_collection(collection, container.embedder.dim)
        await container.vector_store.upsert(
            collection,
            [
                VectorPoint(
                    id="legacydoc_0",
                    vector=[0.0] * container.embedder.dim,
                    payload={
                        "doc_id": "legacydoc",
                        "filename": "legacy.pdf",
                        "page_start": 1,
                        "page_end": 3,
                        "content": "legacy content",
                    },
                )
            ],
        )
        # Backfill reads the persisted SQLite vector table (legacy stores may be SQLite-backed).
        await container.sqlite.conn.execute(
            "INSERT OR REPLACE INTO vector_points (collection, id, dim, vector_json, payload_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                collection,
                "legacydoc_0",
                container.embedder.dim,
                "[]",
                '{"doc_id": "legacydoc", "filename": "legacy.pdf", "page_start": 1, "page_end": 3}',
                0.0,
            ),
        )
        await container.sqlite.conn.commit()

        count = await IngestionPipeline.backfill_document_registry(
            container.sqlite, tmp_path / "data", container.embedder.tag
        )
        assert count == 1
        doc = await container.sqlite.get_document("legacydoc")
        assert doc is not None
        assert doc["page_count"] == 3
        assert doc["file_path"].endswith("legacy.pdf")

        # Idempotent: a second run must not duplicate.
        count_again = await IngestionPipeline.backfill_document_registry(
            container.sqlite, tmp_path / "data", container.embedder.tag
        )
        assert count_again == 0

    @pytest.mark.asyncio
    async def test_file_endpoint_rejects_paths_outside_data_dir(self, rag_app):
        _, container, tmp_path = rag_app
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        await container.sqlite.register_document(
            doc_id="evil",
            namespace="default",
            filename="evil.txt",
            file_path=str(outside),
            page_count=1,
            chunk_count=1,
        )
        app, _, _ = rag_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/rag/documents/evil/file", params={"namespace": "default"}
            )
        assert resp.status_code == 404
