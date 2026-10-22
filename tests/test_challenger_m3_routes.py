"""Adversarial Stress Test Suite for Milestone M3: llama_compat.py & Context Retrieval Isolation.

Executed by Challenger 2 to empirically verify:
1. Session pre-fetch before RAG retrieval (inherits bound rag_namespace without explicit header)
2. Header override precedence (Header > Body > Session > "default")
3. Streaming vs non-streaming equivalence in retrieval and context injection
4. Turn archiving: persistence of course_id and rag_namespace in SQLite for brand-new and existing sessions
5. Adversarial edge cases: whitespace headers, multi-tenant isolation, tool callback rounds, enable_rag=False
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app
from app.storage.vector import VectorPoint


@pytest.fixture
async def app_fixture(tmp_path: Path):
    """Creates an isolated app with fresh SQLite DB and in-memory stores."""
    db_file = tmp_path / "adversarial_m3.db"
    settings = Settings(
        offline_mode=True,
        sqlite_path=db_file,
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app, container
    await container.aclose()


@pytest.fixture
async def client_fixture(app_fixture):
    app, container = app_fixture
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, container


async def index_test_doc(
    container: AppContainer,
    namespace: str,
    doc_id: str,
    content: str,
    filename: str = "doc.pdf",
    page_start: int = 1,
    page_end: int = 1,
) -> None:
    """Helper to cleanly index documents using the container's active embedder."""
    col = container.rag_retriever._collection_name(namespace)
    dim = container.embedder.dim
    await container.vector_store.ensure_collection(col, dim)
    vecs = await container.embedder.embed([content])
    await container.vector_store.upsert(
        col,
        [
            VectorPoint(
                id=doc_id,
                vector=vecs[0],
                payload={
                    "content": content,
                    "filename": filename,
                    "page_start": page_start,
                    "page_end": page_end,
                },
            )
        ],
    )


class TestSessionPreFetchInheritance:
    """Verify that a session bound to a custom rag_namespace inherits that namespace during /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_session_prefetch_inherits_rag_namespace_non_streaming(self, client_fixture):
        """A conversation previously created with a custom rag_namespace inherits that namespace without explicit header."""
        client, container = client_fixture

        user_id = "student_alpha"
        session_id = "sess_bio_101"
        await container.state_manager.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            course_id="course_bio_intro",
            rag_namespace="biology_101",
        )

        bio_doc_text = "Mitochondria generates ATP via oxidative phosphorylation."
        default_doc_text = "General encyclopedic knowledge of no specific subject."

        await index_test_doc(
            container,
            namespace="biology_101",
            doc_id="bio_1",
            content=bio_doc_text,
            filename="cell_biology.pdf",
            page_start=42,
            page_end=43,
        )
        await index_test_doc(
            container,
            namespace="default",
            doc_id="def_1",
            content=default_doc_text,
            filename="encyclopedia.pdf",
            page_start=1,
            page_end=1,
        )

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Explain ATP production"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "1"

        last_call_messages = container.llm.calls[-1]
        user_msg = next(m for m in last_call_messages if m.role == "user")
        assert bio_doc_text in user_msg.content
        assert default_doc_text not in user_msg.content
        assert "P.42-43" in user_msg.content

    @pytest.mark.asyncio
    async def test_session_prefetch_inherits_rag_namespace_streaming(self, client_fixture):
        """Streaming completions also pre-fetch session and respect namespace boundaries."""
        client, container = client_fixture

        user_id = "student_beta"
        session_id = "sess_chem_301"
        await container.state_manager.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            course_id="course_chem_advanced",
            rag_namespace="chemistry_301",
        )

        chem_doc_text = "Le Chatelier's principle predicts shift in dynamic equilibrium."
        default_doc_text = "General encyclopedia volume 5."

        await index_test_doc(
            container,
            namespace="chemistry_301",
            doc_id="chem_1",
            content=chem_doc_text,
            filename="physical_chem.pdf",
            page_start=88,
            page_end=88,
        )
        await index_test_doc(
            container,
            namespace="default",
            doc_id="def_chem_1",
            content=default_doc_text,
            filename="encyclopedia.pdf",
            page_start=1,
            page_end=1,
        )

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "messages": [{"role": "user", "content": "What happens if temperature increases?"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "1"

        body_text = resp.text
        assert "data: " in body_text
        assert "data: [DONE]" in body_text

        last_call_messages = container.llm.calls[-1]
        user_msg = next(m for m in last_call_messages if m.role == "user")
        assert chem_doc_text in user_msg.content
        assert default_doc_text not in user_msg.content


class TestHeaderOverridePrecedence:
    """Verify Header > Body > Session > default precedence hierarchy."""

    @pytest.mark.asyncio
    async def test_header_overrides_session_bound_namespace(self, client_fixture):
        """Header X-RAG-Namespace overrides session's bound namespace."""
        client, container = client_fixture

        user_id = "student_gamma"
        session_id = "sess_override_test"
        await container.state_manager.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            course_id="course_history",
            rag_namespace="history_101",
        )

        await index_test_doc(
            container,
            namespace="history_101",
            doc_id="h1",
            content="The Magna Carta was signed in 1215.",
            filename="history.pdf",
            page_start=10,
        )
        await index_test_doc(
            container,
            namespace="physics_101",
            doc_id="p1",
            content="Newtonian universal gravitation F=G(m1m2)/r^2.",
            filename="physics.pdf",
            page_start=50,
        )

        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-RAG-Namespace": "physics_101"},
            json={
                "user_id": user_id,
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Explain gravitational force"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "1"

        user_msg = next(m for m in container.llm.calls[-1] if m.role == "user")
        assert "Newtonian universal gravitation" in user_msg.content
        assert "Magna Carta" not in user_msg.content

    @pytest.mark.asyncio
    async def test_header_overrides_body_rag_namespace(self, client_fixture):
        """Header X-RAG-Namespace overrides body parameter rag_namespace (Header > Body)."""
        client, container = client_fixture

        user_id = "student_delta"
        session_id = "sess_header_vs_body"

        await index_test_doc(
            container,
            namespace="header_ns",
            doc_id="hdr1",
            content="TARGET_FROM_HEADER_DOC",
            filename="h.pdf",
            page_start=1,
        )
        await index_test_doc(
            container,
            namespace="body_ns",
            doc_id="bdy1",
            content="WRONG_FROM_BODY_DOC",
            filename="b.pdf",
            page_start=1,
        )

        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-RAG-Namespace": "header_ns"},
            json={
                "user_id": user_id,
                "session_id": session_id,
                "rag_namespace": "body_ns",
                "messages": [{"role": "user", "content": "Test precedence"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "1"

        user_msg = next(m for m in container.llm.calls[-1] if m.role == "user")
        assert "TARGET_FROM_HEADER_DOC" in user_msg.content
        assert "WRONG_FROM_BODY_DOC" not in user_msg.content

    @pytest.mark.asyncio
    async def test_body_overrides_session_namespace_when_header_absent(self, client_fixture):
        """Body parameter rag_namespace overrides session's bound namespace when header is absent (Body > Session)."""
        client, container = client_fixture

        user_id = "student_epsilon"
        session_id = "sess_body_vs_session"
        await container.state_manager.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            course_id="c_sess",
            rag_namespace="sess_ns",
        )

        await index_test_doc(
            container,
            namespace="sess_ns",
            doc_id="s1",
            content="FROM_SESSION_NS_DOC",
            filename="s.pdf",
            page_start=1,
        )
        await index_test_doc(
            container,
            namespace="body_only_ns",
            doc_id="b1",
            content="FROM_BODY_NS_DOC",
            filename="b.pdf",
            page_start=1,
        )

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "rag_namespace": "body_only_ns",
                "messages": [{"role": "user", "content": "Test precedence"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "1"

        user_msg = next(m for m in container.llm.calls[-1] if m.role == "user")
        assert "FROM_BODY_NS_DOC" in user_msg.content
        assert "FROM_SESSION_NS_DOC" not in user_msg.content

    @pytest.mark.asyncio
    async def test_whitespace_header_falls_back_to_session_namespace(self, client_fixture):
        """A whitespace-only header X-RAG-Namespace: '   ' must cleanly fallback to session namespace."""
        client, container = client_fixture

        user_id = "student_zeta"
        session_id = "sess_whitespace_hdr"
        await container.state_manager.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            rag_namespace="actual_ns",
        )

        await index_test_doc(
            container,
            namespace="actual_ns",
            doc_id="act1",
            content="CLEAN_FALLBACK_CONTENT",
            filename="a.pdf",
            page_start=1,
        )

        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-RAG-Namespace": "   "},
            json={
                "user_id": user_id,
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Test whitespace"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "1"

        user_msg = next(m for m in container.llm.calls[-1] if m.role == "user")
        assert "CLEAN_FALLBACK_CONTENT" in user_msg.content


class TestTurnArchivingAndPersistence:
    """Verify that sessions created or interacted with via /v1/chat/completions persist metadata to SQLite."""

    @pytest.mark.asyncio
    async def test_new_session_persists_course_and_rag_namespace_non_streaming(self, client_fixture):
        """Calling /v1/chat/completions with new session_id persists course_id and rag_namespace in SQLite."""
        client, container = client_fixture

        user_id = "student_arch_1"
        session_id = "sess_auto_create_nonstream"

        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "X-RAG-Namespace": "robotics_401",
                "X-Course-Id": "course_robotics",
            },
            json={
                "user_id": user_id,
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Inverse kinematics"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200

        cursor = await container.sqlite.conn.execute(
            "SELECT session_id, user_id, course_id, rag_namespace FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        assert row is not None, "Session record must exist in SQLite sessions table"
        assert row["user_id"] == user_id
        assert row["course_id"] == "course_robotics"
        assert row["rag_namespace"] == "robotics_401"

        msg_cursor = await container.sqlite.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        msg_rows = await msg_cursor.fetchall()
        assert len(msg_rows) == 2, "Expected 1 user message and 1 assistant message"
        assert msg_rows[0]["role"] == "user"
        assert msg_rows[0]["content"] == "Inverse kinematics"
        assert msg_rows[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_new_session_persists_course_and_rag_namespace_streaming(self, client_fixture):
        """Calling /v1/chat/completions with stream=True persists course_id and rag_namespace upon completion."""
        client, container = client_fixture

        user_id = "student_arch_2"
        session_id = "sess_auto_create_stream"

        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "X-RAG-Namespace": "astronomy_201",
                "X-Course-Id": "course_astronomy",
            },
            json={
                "user_id": user_id,
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Hubble constant estimation"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        stream_text = resp.text
        assert "data: [DONE]" in stream_text

        cursor = await container.sqlite.conn.execute(
            "SELECT session_id, user_id, course_id, rag_namespace FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        assert row is not None, "Streaming completion must archive session to SQLite"
        assert row["user_id"] == user_id
        assert row["course_id"] == "course_astronomy"
        assert row["rag_namespace"] == "astronomy_201"

        msg_cursor = await container.sqlite.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        msg_rows = await msg_cursor.fetchall()
        assert len(msg_rows) == 2
        assert msg_rows[0]["role"] == "user"
        assert msg_rows[0]["content"] == "Hubble constant estimation"
        assert msg_rows[1]["role"] == "assistant"


class TestAdversarialAndEdgeCases:
    """Stress-test edge cases, malformed payloads, and multi-tenant security."""

    @pytest.mark.asyncio
    async def test_enable_rag_false_skips_vector_retrieval(self, client_fixture):
        """When enable_rag=False, vector retrieval must be bypassed completely."""
        client, container = client_fixture

        user_id = "student_no_rag"
        session_id = "sess_no_rag"

        await index_test_doc(
            container,
            namespace="default",
            doc_id="d1",
            content="SOME_CONTENT",
            filename="f.pdf",
            page_start=1,
        )

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "enable_rag": False,
                "messages": [{"role": "user", "content": "Direct conversation without RAG"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "0"

        user_msg = next(m for m in container.llm.calls[-1] if m.role == "user")
        assert "SOME_CONTENT" not in user_msg.content
        assert "Retrieved Course Materials" not in user_msg.content

    @pytest.mark.asyncio
    async def test_tool_followup_turn_bypasses_rag_retrieval(self, client_fixture):
        """Tool callback round (last message role == 'tool') must not trigger RAG retrieval."""
        client, container = client_fixture

        user_id = "student_tool"
        session_id = "sess_tool_followup"

        await index_test_doc(
            container,
            namespace="default",
            doc_id="d1",
            content="RETRIEVED_TEXT",
            filename="f.pdf",
            page_start=1,
        )

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "calc", "arguments": '{"x": 2}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
        ]

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "messages": messages,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RAG-Hits") == "0"

    @pytest.mark.asyncio
    async def test_cross_tenant_session_spoofing_prevented(self, client_fixture):
        """User B must not be able to interact with or overwrite User A's session."""
        client, container = client_fixture

        await container.state_manager.get_or_create_session(
            user_id="user_alice",
            session_id="alice_private_session",
            course_id="c_alice",
            rag_namespace="alice_ns",
        )

        with pytest.raises(ValueError, match="already belongs to another user"):
            await client.post(
                "/v1/chat/completions",
                headers={"X-User-Id": "user_bob"},
                json={
                    "session_id": "alice_private_session",
                    "messages": [{"role": "user", "content": "Malicious intrusion"}],
                    "stream": False,
                },
            )
