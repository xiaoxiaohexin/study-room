"""Adversarial Verification & Stress Test Suite for Milestone 1.

Authored by Empirical Challenger M1-2.
Tests RAG Physical Page Preservation, Citation Formatting, and Route Aliases (/api/rag vs /api/v1/rag).
"""

from __future__ import annotations

from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app
from app.rag.retriever import DocChunk, RagRetriever
from app.context.builder import ContextBuilder
from app.storage.vector import SQLiteVectorStore


@pytest.fixture
async def app_instance(tmp_path: Path):
    settings = Settings(
        offline_mode=False,
        qdrant_url="http://localhost:63333",  # deliberately unreachable port to trigger SQLiteVectorStore fallback
        sqlite_path=tmp_path / "test.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    # Explicitly ensure SQLiteVectorStore is wired to pipeline and retriever
    if not isinstance(container.vector_store, SQLiteVectorStore):
        container.vector_store = SQLiteVectorStore(container.sqlite)
        container.ingestion_pipeline.vector_store = container.vector_store
        container.rag_retriever.vector_store = container.vector_store
    app.state.container = container
    yield app
    await container.aclose()


class TestDocChunkCitationPermutations:
    """Verifies DocChunk citation formatting across all single-page, multi-page, and fallback permutations."""

    def test_single_page_citation(self):
        chunk = DocChunk(
            id="c1",
            content="Hello world",
            score=0.95,
            doc_id="d1",
            filename="Calculus.pdf",
            page_start=42,
            page_end=42,
        )
        assert chunk.page_label == "P.42"
        assert chunk.format_citation(include_score=False) == "[DOCUMENT: Calculus.pdf P.42]"
        assert chunk.format_citation(include_score=True) == "[DOCUMENT: Calculus.pdf P.42 | score=0.95]"

    def test_multi_page_citation(self):
        chunk = DocChunk(
            id="c2",
            content="Cross page content",
            score=0.88,
            doc_id="d1",
            filename="Calculus.pdf",
            page_start=42,
            page_end=45,
            headers=["Chapter 3", "Derivatives"],
        )
        assert chunk.page_label == "P.42-45"
        assert chunk.format_citation(include_score=False) == "[DOCUMENT: Calculus.pdf P.42-45 (Chapter 3 > Derivatives)]"
        assert chunk.format_citation(include_score=True) == "[DOCUMENT: Calculus.pdf P.42-45 (Chapter 3 > Derivatives) | score=0.88]"

    def test_default_fallback_page(self):
        chunk = DocChunk(
            id="c3",
            content="Unpaged legacy doc",
            score=0.70,
            doc_id="d2",
            filename="notes.txt",
        )
        assert chunk.page_start == 1
        assert chunk.page_end == 1
        assert chunk.page_label == "P.1"
        assert chunk.format_citation() == "[DOCUMENT: notes.txt P.1]"

    def test_rag_retriever_format_prompt_citations(self):
        c1 = DocChunk(
            id="c1",
            content="Page 10 content",
            score=0.9,
            doc_id="d1",
            filename="book.pdf",
            page_start=10,
            page_end=10,
        )
        c2 = DocChunk(
            id="c2",
            content="Page 11-12 content",
            score=0.85,
            doc_id="d1",
            filename="book.pdf",
            headers=["Part 1"],
            page_start=11,
            page_end=12,
        )
        formatted = RagRetriever.format_prompt_citations([c1, c2])
        assert "[DOCUMENT: book.pdf P.10]\nPage 10 content" in formatted
        assert "[DOCUMENT: book.pdf P.11-12 (Part 1)]\nPage 11-12 content" in formatted

    def test_context_builder_doc_citation_formatting(self):
        builder = ContextBuilder(max_tokens=4000)
        c1 = DocChunk(
            id="c1",
            content="Some textbook text",
            score=0.92,
            doc_id="d1",
            filename="physics.pdf",
            page_start=5,
            page_end=7,
            headers=["Mechanics", "Kinematics"],
        )
        packets = builder.gather(
            system_prompt="You are a tutor",
            user_query="Explain acceleration",
            history=[],
            docs=[c1],
        )
        doc_packets = [p for p in packets if p.source == "rag"]
        assert len(doc_packets) == 1
        assert "[DOCUMENT: physics.pdf P.5-7 (Mechanics > Kinematics)]\nSome textbook text" in doc_packets[0].content


class TestRouteAliasesAndEmpiricalSearch:
    """Empirical queries against /api/rag/search and /api/v1/rag/search with edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query_handling_parity(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for route in ("/api/rag/search", "/api/v1/rag/search"):
                # Empty string
                res1 = await client.post(route, json={"query": ""})
                assert res1.status_code == 200
                data1 = res1.json()
                assert data1["count"] == 0
                assert data1["results"] == []

                # Whitespace string
                res2 = await client.post(route, json={"query": "    \t\n   "})
                assert res2.status_code == 200
                data2 = res2.json()
                assert data2["count"] == 0
                assert data2["results"] == []

    @pytest.mark.asyncio
    async def test_special_characters_queries(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            special_inputs = [
                "'; DROP TABLE vector_points; --",
                "<script>alert('xss')</script>",
                "<!-- PAGE: 999 -->",
                "🔍 ⚡ 📖 🚀 \U0001F600",
                "English, 中文, 日本語, Русский, العربية",
                "A" * 5000,  # Long query
            ]
            for inp in special_inputs:
                for route in ("/api/rag/search", "/api/v1/rag/search"):
                    res = await client.post(route, json={"query": inp, "top_k": 3})
                    assert res.status_code == 200
                    data = res.json()
                    assert "results" in data
                    assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_varying_top_k_values_valid_and_boundaries(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Valid top_k ranges: 1 <= top_k <= 50
            for k in (1, 5, 20, 50):
                for route in ("/api/rag/search", "/api/v1/rag/search"):
                    res = await client.post(route, json={"query": "test query", "top_k": k})
                    assert res.status_code == 200
                    data = res.json()
                    assert len(data["results"]) <= k

            # Invalid top_k boundaries: < 1 or > 50
            for invalid_k in (-10, 0, 51, 100):
                for route in ("/api/rag/search", "/api/v1/rag/search"):
                    res = await client.post(route, json={"query": "test query", "top_k": invalid_k})
                    assert res.status_code in {400, 422}

    @pytest.mark.asyncio
    async def test_full_ingestion_and_retrieval_page_structure(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            markdown_content = (
                "# Introduction to Agents\n\n"
                "<!-- PAGE: 10 -->\n"
                "Autonomous agents perceive environments and take actions.\n\n"
                "<!-- PAGE: 11 -->\n"
                "Memory architectures allow agents to retain past experiences.\n\n"
                "<!-- PAGE: 12 -->\n"
                "Reinforcement learning optimizes policy decisions over time.\n"
            )

            # Ingest file via /api/v1/rag/ingest
            files = {
                "file": ("agents_textbook.md", markdown_content.encode("utf-8"), "text/markdown"),
            }
            ingest_res = await client.post("/api/v1/rag/ingest", files=files, data={"namespace": "agents_ns"})
            assert ingest_res.status_code == 200
            ingest_data = ingest_res.json()
            assert ingest_data["chunks"] >= 1
            doc_id = ingest_data["doc_id"]

            # Search via /api/v1/rag/search
            docs_res = await client.post("/api/v1/rag/search", json={
                "query": "Autonomous agents perceive environments",
                "namespace": "agents_ns",
                "top_k": 5
            })
            assert docs_res.status_code == 200
            search_data = docs_res.json()
            assert search_data["count"] > 0
            first_chunk = search_data["results"][0]

            # Verify JSON structure contains page_start, page_end, chunk_index
            assert "page_start" in first_chunk
            assert "page_end" in first_chunk
            assert "chunk_index" in first_chunk
            assert isinstance(first_chunk["page_start"], int)
            assert isinstance(first_chunk["page_end"], int)
            assert isinstance(first_chunk["chunk_index"], int)
            assert first_chunk["page_start"] == 10
            assert first_chunk["filename"] == "agents_textbook.md"

            # Verify parity with legacy /api/rag/search
            legacy_res = await client.post("/api/rag/search", json={
                "query": "Autonomous agents perceive environments",
                "namespace": "agents_ns",
                "top_k": 5
            })
            assert legacy_res.status_code == 200
            legacy_data = legacy_res.json()
            assert legacy_data["count"] == search_data["count"]
            assert legacy_data["results"][0]["page_start"] == first_chunk["page_start"]
            assert legacy_data["results"][0]["page_end"] == first_chunk["page_end"]
            assert legacy_data["results"][0]["chunk_index"] == first_chunk["chunk_index"]

            # Verify documents endpoint and stats endpoint on both /api/v1 and /api
            for prefix in ("/api", "/api/v1"):
                doc_list_res = await client.get(f"{prefix}/rag/documents?namespace=agents_ns")
                assert doc_list_res.status_code == 200
                doc_list = doc_list_res.json()
                assert doc_list["total_docs"] >= 1
                stats_res = await client.get(f"{prefix}/rag/stats?namespace=agents_ns")
                assert stats_res.status_code == 200
                stats = stats_res.json()
                assert stats["total_chunks"] >= 1

            # Cleanup via delete endpoint
            del_res = await client.delete(f"/api/v1/rag/documents/{doc_id}?namespace=agents_ns")
            assert del_res.status_code == 200
            assert del_res.json()["ok"] is True
