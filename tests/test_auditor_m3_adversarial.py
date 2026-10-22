"""Adversarial stress-testing suite for Milestone M3: Backend API Routes & Context Retrieval Isolation.
Independently authored by Forensic Auditor.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.llm.embedding import HashingEmbedder
from app.main import create_app
from app.rag.retriever import DocChunk, RagRetriever
from app.storage.sqlite import SQLiteStore
from app.storage.vector import SQLiteVectorStore, VectorPoint


@pytest.fixture
async def auditor_app(tmp_path: Path):
    db_path = tmp_path / "auditor_test.db"
    settings = Settings(
        offline_mode=True,
        sqlite_path=db_path,
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app
    await container.aclose()


@pytest.fixture
async def auditor_client(auditor_app):
    transport = ASGITransport(app=auditor_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestAuditorM3PrecedenceHierarchy:
    """Stress tests the 4-tier precedence hierarchy across all chat endpoints."""

    @pytest.mark.asyncio
    async def test_api_chat_precedence_header_over_body_and_session(self, auditor_client, auditor_app):
        """Header X-RAG-Namespace MUST override body rag_namespace and session rag_namespace."""
        container: AppContainer = auditor_app.state.container

        # 1. Create a session bound to session_ns
        sess = await container.state_manager.get_or_create_session(
            user_id="u_audit_1",
            session_id="s_audit_1",
            course_id="course_session",
            rag_namespace="session_ns",
        )
        assert sess.rag_namespace == "session_ns"

        # Intercept rag_retriever.search to inspect the actual namespace passed
        searched_namespaces = []
        original_search = container.rag_retriever.search

        async def mock_search(queries, namespace="default", top_k=5):
            searched_namespaces.append(namespace)
            return []

        with patch.object(container.rag_retriever, "search", side_effect=mock_search):
            # Send with both Header and Body
            resp = await auditor_client.post(
                "/api/chat",
                headers={
                    "X-RAG-Namespace": "header_override_ns",
                    "X-Course-Id": "header_course",
                },
                json={
                    "user_id": "u_audit_1",
                    "session_id": "s_audit_1",
                    "message": "Test question",
                    "rag_namespace": "body_ns",
                    "course_id": "body_course",
                    "enable_query_expansion": False,
                },
            )
            assert resp.status_code == 200
            assert "header_override_ns" in searched_namespaces

    @pytest.mark.asyncio
    async def test_api_chat_precedence_body_over_session_when_no_header(self, auditor_client, auditor_app):
        """Body rag_namespace MUST override session rag_namespace when Header is absent."""
        container: AppContainer = auditor_app.state.container

        await container.state_manager.get_or_create_session(
            user_id="u_audit_2",
            session_id="s_audit_2",
            rag_namespace="session_ns_2",
        )

        searched_namespaces = []

        async def mock_search(queries, namespace="default", top_k=5):
            searched_namespaces.append(namespace)
            return []

        with patch.object(container.rag_retriever, "search", side_effect=mock_search):
            resp = await auditor_client.post(
                "/api/chat",
                json={
                    "user_id": "u_audit_2",
                    "session_id": "s_audit_2",
                    "message": "Test question",
                    "rag_namespace": "body_ns_2",
                    "enable_query_expansion": False,
                },
            )
            assert resp.status_code == 200
            assert "body_ns_2" in searched_namespaces

    @pytest.mark.asyncio
    async def test_api_chat_precedence_session_fallback_when_omitted(self, auditor_client, auditor_app):
        """Session rag_namespace MUST be used when Header and Body are omitted."""
        container: AppContainer = auditor_app.state.container

        await container.state_manager.get_or_create_session(
            user_id="u_audit_3",
            session_id="s_audit_3",
            rag_namespace="bound_course_ns",
        )

        searched_namespaces = []

        async def mock_search(queries, namespace="default", top_k=5):
            searched_namespaces.append(namespace)
            return []

        with patch.object(container.rag_retriever, "search", side_effect=mock_search):
            resp = await auditor_client.post(
                "/api/chat",
                json={
                    "user_id": "u_audit_3",
                    "session_id": "s_audit_3",
                    "message": "Test question",
                    "enable_query_expansion": False,
                },
            )
            assert resp.status_code == 200
            assert "bound_course_ns" in searched_namespaces

    @pytest.mark.asyncio
    async def test_api_chat_whitespace_header_falls_back(self, auditor_client, auditor_app):
        """Whitespace header '   ' MUST be sanitized and fall back to body or session."""
        container: AppContainer = auditor_app.state.container

        await container.state_manager.get_or_create_session(
            user_id="u_audit_4",
            session_id="s_audit_4",
            rag_namespace="session_valid_ns",
        )

        searched_namespaces = []

        async def mock_search(queries, namespace="default", top_k=5):
            searched_namespaces.append(namespace)
            return []

        with patch.object(container.rag_retriever, "search", side_effect=mock_search):
            resp = await auditor_client.post(
                "/api/chat",
                headers={"X-RAG-Namespace": "   "},
                json={
                    "user_id": "u_audit_4",
                    "session_id": "s_audit_4",
                    "message": "Test question",
                    "enable_query_expansion": False,
                },
            )
            assert resp.status_code == 200
            assert "session_valid_ns" in searched_namespaces


class TestAuditorM3LlamaCompatIsolation:
    """Stress tests session pre-fetch and isolation in /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_llama_compat_session_prefetch_and_isolation(self, auditor_client, auditor_app):
        """Pre-created session in SQLite must be loaded before RAG retrieval, applying its namespace."""
        container: AppContainer = auditor_app.state.container

        # Pre-seed session in database
        await container.state_manager.get_or_create_session(
            user_id="u_llama_pre",
            session_id="sess_llama_pre",
            course_id="advanced_calc",
            rag_namespace="calc_advanced_ns",
        )

        searched_namespaces = []

        async def mock_search(queries, namespace="default", top_k=5):
            searched_namespaces.append(namespace)
            return []

        with patch.object(container.rag_retriever, "search", side_effect=mock_search):
            resp = await auditor_client.post(
                "/v1/chat/completions",
                headers={"X-User-Id": "u_llama_pre", "X-Conversation-Id": "sess_llama_pre"},
                json={
                    "messages": [{"role": "user", "content": "How to compute Green's theorem?"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            assert "calc_advanced_ns" in searched_namespaces

    @pytest.mark.asyncio
    async def test_llama_compat_header_overrides_session(self, auditor_client, auditor_app):
        """X-RAG-Namespace header on /v1/chat/completions overrides session-bound namespace."""
        container: AppContainer = auditor_app.state.container

        await container.state_manager.get_or_create_session(
            user_id="u_llama_override",
            session_id="sess_llama_override",
            rag_namespace="biology_ns",
        )

        searched_namespaces = []

        async def mock_search(queries, namespace="default", top_k=5):
            searched_namespaces.append(namespace)
            return []

        with patch.object(container.rag_retriever, "search", side_effect=mock_search):
            resp = await auditor_client.post(
                "/v1/chat/completions",
                headers={
                    "X-User-Id": "u_llama_override",
                    "X-Conversation-Id": "sess_llama_override",
                    "X-RAG-Namespace": "chemistry_ns",
                },
                json={
                    "messages": [{"role": "user", "content": "Organic synthesis reaction"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            assert "chemistry_ns" in searched_namespaces

    @pytest.mark.asyncio
    async def test_llama_compat_dialogue_archival_preserves_bindings(self, auditor_client, auditor_app):
        """New session created via /v1/chat/completions preserves course_id & rag_namespace in SQLite."""
        container: AppContainer = auditor_app.state.container

        resp = await auditor_client.post(
            "/v1/chat/completions",
            headers={
                "X-User-Id": "student_new_sess",
                "X-Conversation-Id": "sess_autocreated_1",
                "X-RAG-Namespace": "discrete_math",
                "X-Course-Id": "cs_discrete",
            },
            json={
                "messages": [{"role": "user", "content": "Explain graph isomorphism"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200

        # Verify session is persisted with Discrete Math bindings in SQLite
        sess = await container.state_manager.get_session("student_new_sess", "sess_autocreated_1")
        assert sess is not None
        assert sess.rag_namespace == "discrete_math"
        assert sess.course_id == "cs_discrete"


class TestAuditorM3ConcurrencyAndRobustness:
    """Stress tests concurrency locks and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_request_conflict_returns_409(self, auditor_client, auditor_app):
        """When an operation holds the session lock, /api/chat must reject concurrent call with 409."""
        container: AppContainer = auditor_app.state.container
        session_id = "sess_conflict_409"

        # Explicitly acquire the session lock
        lock = container.state_manager.get_lock(session_id)
        await lock.acquire()

        try:
            resp = await auditor_client.post(
                "/api/chat",
                json={
                    "user_id": "u_lock",
                    "session_id": session_id,
                    "message": "Should fail with 409",
                },
            )
            assert resp.status_code == 409
            assert "already running" in resp.json()["detail"]

            # Stream route must also return 409
            resp_stream = await auditor_client.post(
                "/api/chat/stream",
                json={
                    "user_id": "u_lock",
                    "session_id": session_id,
                    "message": "Stream should also fail with 409",
                },
            )
            assert resp_stream.status_code == 409
        finally:
            lock.release()

    @pytest.mark.asyncio
    async def test_special_characters_and_cjk_namespace_handling(self, auditor_client, auditor_app):
        """CJK and URL-encoded-like namespaces must be handled safely without error."""
        from urllib.parse import quote

        container: AppContainer = auditor_app.state.container
        cjk_ns = "离散数学_图论#2@2026"

        searched_namespaces = []

        async def mock_search(queries, namespace="default", top_k=5):
            searched_namespaces.append(namespace)
            return []

        with patch.object(container.rag_retriever, "search", side_effect=mock_search):
            resp = await auditor_client.post(
                "/api/chat",
                headers={"X-RAG-Namespace": quote(cjk_ns)},
                json={
                    "user_id": "u_cjk",
                    "message": "What is Euler path?",
                    "enable_query_expansion": False,
                },
            )
            assert resp.status_code == 200
            assert cjk_ns in searched_namespaces
