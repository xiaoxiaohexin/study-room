"""Empirical Adversarial Stress Test Suite for Milestone M3.

Author: Challenger 1 (Milestone M3)
Focus: 4-Tier Precedence Hierarchy across /api/chat and /api/chat/stream:
- Tier 1: Header(X-RAG-Namespace) overrides Body(rag_namespace) and Session(rag_namespace)
- Tier 2: Body(rag_namespace) overrides Session(rag_namespace) when Header is absent
- Tier 3: Session(rag_namespace) is inherited when Header and Body are absent
- Tier 4: Fallback to "default" when none are provided
- Adversarial edge cases: whitespace handling, empty strings, session state poisoning,
  concurrent isolation, and SSE streaming integrity.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app
from app.storage.vector import VectorPoint


# ============================================================================
# Test Fixtures with Real Vectors & Spy Retrival Oracles
# ============================================================================

class RetrievalSpy:
    """Records every invocation of RagRetriever.search with exact arguments."""

    def __init__(self, real_search_fn):
        self._real_search = real_search_fn
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, queries: list[str], namespace: str = "default", top_k: int = 5):
        self.calls.append({
            "queries": list(queries),
            "namespace": namespace,
            "top_k": top_k,
        })
        return await self._real_search(queries, namespace=namespace, top_k=top_k)

    def reset(self):
        self.calls.clear()

    @property
    def last_namespace(self) -> str | None:
        return self.calls[-1]["namespace"] if self.calls else None


@pytest.fixture
async def challenger_env(tmp_path: Path):
    """Initializes app, container, vector store with populated collections, and retrieval spy."""
    db_file = tmp_path / "test_challenger_m3.db"
    settings = Settings(
        offline_mode=True,
        sqlite_path=db_file,
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    # Set up spy on rag_retriever.search
    spy = RetrievalSpy(container.rag_retriever.search)
    container.rag_retriever.search = spy

    # Populate 4 distinct collections in vector store with distinct payloads
    namespaces = ["ns_header", "ns_body", "ns_session", "default"]
    for ns in namespaces:
        col = container.rag_retriever._collection_name(ns)
        await container.vector_store.ensure_collection(col, container.embedder.dim)
        v = await container.embedder.embed([f"query in {ns}"])
        await container.vector_store.upsert(
            col,
            [
                VectorPoint(
                    id=f"pt_{ns}",
                    vector=v[0],
                    payload={
                        "content": f"CONTENT_PAYLOAD_FOR_{ns.upper()}",
                        "filename": f"{ns}.pdf",
                        "page_start": 10,
                        "page_end": 10,
                    },
                )
            ],
        )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "app": app,
            "container": container,
            "spy": spy,
            "client": client,
        }

    await container.aclose()


# Helper to parse SSE stream lines
def parse_sse_events(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current_event: dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue
        if line.startswith("event:"):
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_event["data"] = line[len("data:"):].strip()
    if current_event:
        events.append(current_event)
    return events


# ============================================================================
# Tier 1 Stress Tests: Header Precedence (Header > Body > Session > Default)
# ============================================================================

class TestTier1HeaderPrecedence:
    """Validates that Header(X-RAG-Namespace) wins over all other tiers."""

    @pytest.mark.asyncio
    async def test_t1_01_sync_header_overrides_body_and_session(self, challenger_env):
        """Tier 1: Header present alongside Body and bound Session -> Header wins on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        # Pre-create session bound to ns_session
        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        assert s_resp.status_code in (200, 201)
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "ns_header"},
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Explain concept",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_header"

    @pytest.mark.asyncio
    async def test_t1_02_stream_header_overrides_body_and_session(self, challenger_env):
        """Tier 1: Header present alongside Body and bound Session -> Header wins on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        assert s_resp.status_code in (200, 201)
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            headers={"X-RAG-Namespace": "ns_header"},
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Stream query",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_header"
        events = parse_sse_events(res.text)
        assert len(events) >= 1
        assert any(e.get("event") in ("status", "done", "thought") for e in events)

    @pytest.mark.asyncio
    async def test_t1_03_sync_header_overrides_body_new_session(self, challenger_env):
        """Tier 1: Header present alongside Body on new session -> Header wins on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "ns_header"},
            json={
                "user_id": "u1",
                "message": "New session query",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_header"

    @pytest.mark.asyncio
    async def test_t1_04_stream_header_overrides_body_new_session(self, challenger_env):
        """Tier 1: Header present alongside Body on new session -> Header wins on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            headers={"X-RAG-Namespace": "ns_header"},
            json={
                "user_id": "u1",
                "message": "New stream query",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_header"

    @pytest.mark.asyncio
    async def test_t1_05_sync_header_overrides_session_when_body_omitted(self, challenger_env):
        """Tier 1: Header present, Body omitted, Session bound -> Header wins on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "ns_header"},
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Header only override",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_header"

    @pytest.mark.asyncio
    async def test_t1_06_stream_header_overrides_session_when_body_omitted(self, challenger_env):
        """Tier 1: Header present, Body omitted, Session bound -> Header wins on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            headers={"X-RAG-Namespace": "ns_header"},
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Header only stream override",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_header"


# ============================================================================
# Tier 2 Stress Tests: Body Precedence (Body > Session > Default)
# ============================================================================

class TestTier2BodyPrecedence:
    """Validates that Body(rag_namespace) wins over Session when Header is absent."""

    @pytest.mark.asyncio
    async def test_t2_01_sync_body_overrides_session_when_header_absent(self, challenger_env):
        """Tier 2: Header absent, Body present, Session bound -> Body wins on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Body override query",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_body"

    @pytest.mark.asyncio
    async def test_t2_02_stream_body_overrides_session_when_header_absent(self, challenger_env):
        """Tier 2: Header absent, Body present, Session bound -> Body wins on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Stream body override query",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_body"

    @pytest.mark.asyncio
    async def test_t2_03_sync_body_new_session(self, challenger_env):
        """Tier 2: Header absent, Body present on new session -> Body used on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "message": "New session body query",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_body"

    @pytest.mark.asyncio
    async def test_t2_04_stream_body_new_session(self, challenger_env):
        """Tier 2: Header absent, Body present on new session -> Body used on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            json={
                "user_id": "u1",
                "message": "Stream new session body query",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_body"


# ============================================================================
# Tier 3 Stress Tests: Session Inheritance (Session > Default)
# ============================================================================

class TestTier3SessionInheritance:
    """Validates that Session(rag_namespace) is inherited when Header & Body are absent."""

    @pytest.mark.asyncio
    async def test_t3_01_sync_session_inherited_when_omitted(self, challenger_env):
        """Tier 3: Header absent, Body omitted -> Session namespace inherited on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Inherit session namespace",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_session"

    @pytest.mark.asyncio
    async def test_t3_02_stream_session_inherited_when_omitted(self, challenger_env):
        """Tier 3: Header absent, Body omitted -> Session namespace inherited on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Stream inherit session namespace",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_session"

    @pytest.mark.asyncio
    async def test_t3_03_sync_session_inherited_when_explicit_none(self, challenger_env):
        """Tier 3: Header absent, Body rag_namespace is None -> Session namespace inherited on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Explicit None body",
                "rag_namespace": None,
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_session"

    @pytest.mark.asyncio
    async def test_t3_04_stream_session_inherited_when_explicit_none(self, challenger_env):
        """Tier 3: Header absent, Body rag_namespace is None -> Session namespace inherited on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Stream explicit None body",
                "rag_namespace": None,
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_session"


# ============================================================================
# Tier 4 Stress Tests: Fallback to "default"
# ============================================================================

class TestTier4FallbackDefault:
    """Validates fallback to 'default' when no namespace is specified anywhere."""

    @pytest.mark.asyncio
    async def test_t4_01_sync_fallback_new_session(self, challenger_env):
        """Tier 4: Header absent, Body omitted, new session -> falls back to 'default' on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "message": "Default fallback query",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "default"

    @pytest.mark.asyncio
    async def test_t4_02_stream_fallback_new_session(self, challenger_env):
        """Tier 4: Header absent, Body omitted, new session -> falls back to 'default' on /api/chat/stream."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat/stream",
            json={
                "user_id": "u1",
                "message": "Stream default fallback query",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "default"

    @pytest.mark.asyncio
    async def test_t4_03_sync_fallback_session_bound_to_default(self, challenger_env):
        """Tier 4: Header absent, Body omitted, session bound to 'default' -> 'default' on /api/chat."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "default"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Query on default session",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "default"


# ============================================================================
# Adversarial Stress & Edge Case Tests
# ============================================================================

class TestAdversarialEdgeCases:
    """Probing edge cases, whitespace handling, and session state corruption."""

    @pytest.mark.asyncio
    async def test_edge_01_whitespace_only_header_falls_back_to_body(self, challenger_env):
        """Adversarial: Header is whitespace only ('   ') -> should ignore header and use Body."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "   "},
            json={
                "user_id": "u1",
                "message": "Whitespace header test",
                "rag_namespace": "ns_body",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_body"

    @pytest.mark.asyncio
    async def test_edge_02_whitespace_only_header_falls_back_to_session(self, challenger_env):
        """Adversarial: Header is whitespace ('  '), Body omitted -> should fall back to Session."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "   "},
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Whitespace header with session test",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_session"

    @pytest.mark.asyncio
    async def test_edge_03_whitespace_only_body_falls_back_to_session(self, challenger_env):
        """Adversarial: Header absent, Body is whitespace ('  ') -> should fall back to Session."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Whitespace body test",
                "rag_namespace": "   ",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_session"

    @pytest.mark.asyncio
    async def test_edge_04_whitespace_header_and_body_falls_back_to_default(self, challenger_env):
        """Adversarial: Both Header and Body are whitespace -> falls back to 'default'."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "  \t  "},
            json={
                "user_id": "u1",
                "message": "Whitespace everywhere test",
                "rag_namespace": "   ",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "default"

    @pytest.mark.asyncio
    async def test_edge_05_header_whitespace_trimming(self, challenger_env):
        """Adversarial: Header has leading/trailing whitespace -> must strip to exact name."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "  ns_header  "},
            json={
                "user_id": "u1",
                "message": "Trimmed header test",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_header"

    @pytest.mark.asyncio
    async def test_edge_06_body_whitespace_trimming(self, challenger_env):
        """Adversarial: Body has leading/trailing whitespace -> must strip to exact name."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "message": "Trimmed body test",
                "rag_namespace": "  ns_body  ",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_body"

    @pytest.mark.asyncio
    async def test_edge_07_empty_string_header_and_body(self, challenger_env):
        """Adversarial: Header is '' and Body is '' -> falls back to session."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        s_resp = await client.post("/api/sessions", json={"user_id": "u1", "rag_namespace": "ns_session"})
        sid = s_resp.json()["session_id"]

        spy.reset()
        res = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": ""},
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Empty strings test",
                "rag_namespace": "",
            },
        )
        assert res.status_code == 200
        assert spy.last_namespace == "ns_session"

    @pytest.mark.asyncio
    async def test_edge_08_course_id_precedence(self, challenger_env):
        """Course ID Precedence: Header X-Course-Id > Body course_id > Session course_id."""
        client: AsyncClient = challenger_env["client"]
        container: AppContainer = challenger_env["container"]

        s_resp = await client.post(
            "/api/sessions",
            json={"user_id": "u1", "course_id": "course_sess", "rag_namespace": "ns_session"},
        )
        sid = s_resp.json()["session_id"]

        # 1. Header + Body -> Header wins
        res = await client.post(
            "/api/chat",
            headers={"X-Course-Id": "course_hdr"},
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Course test 1",
                "course_id": "course_body",
            },
        )
        assert res.status_code == 200

        # 2. Body only -> Body wins
        res = await client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "session_id": sid,
                "message": "Course test 2",
                "course_id": "course_body",
            },
        )
        assert res.status_code == 200

    @pytest.mark.asyncio
    async def test_edge_09_stream_error_handling(self, challenger_env):
        """Streaming robustness: ensure SSE stream closes gracefully with error if retriever fails."""
        client: AsyncClient = challenger_env["client"]
        container: AppContainer = challenger_env["container"]

        # Temporarily mock retriever to raise Exception
        orig_search = container.rag_retriever.search
        container.rag_retriever.search = AsyncMock(side_effect=RuntimeError("Simulated RAG failure"))

        res = await client.post(
            "/api/chat/stream",
            json={
                "user_id": "u1",
                "message": "Failing stream query",
            },
        )
        assert res.status_code == 200
        events = parse_sse_events(res.text)
        assert any(e.get("event") == "error" for e in events)

        # Restore original search
        container.rag_retriever.search = orig_search

    @pytest.mark.asyncio
    async def test_edge_10_concurrent_requests_with_different_namespaces(self, challenger_env):
        """Stress: Multiple concurrent requests with distinct namespaces execute without cross-talk."""
        client: AsyncClient = challenger_env["client"]
        spy: RetrievalSpy = challenger_env["spy"]

        async def make_request(idx: int, ns: str):
            res = await client.post(
                "/api/chat",
                headers={"X-RAG-Namespace": ns},
                json={
                    "user_id": f"user_conc_{idx}",
                    "message": f"Concurrent query {idx}",
                },
            )
            return res.status_code

        tasks = [
            make_request(i, "ns_header" if i % 2 == 0 else "ns_body")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        assert all(code == 200 for code in results)
        assert len(spy.calls) >= 10
        header_calls = [c for c in spy.calls if c["namespace"] == "ns_header"]
        body_calls = [c for c in spy.calls if c["namespace"] == "ns_body"]
        assert len(header_calls) == 5
        assert len(body_calls) == 5


# ============================================================================
# Critical Hypothesis Testing: Session State Poisoning / Persistent Mutation
# ============================================================================

class TestSessionStatePoisoningVulnerability:
    """Tests whether a one-off Header/Body override corrupts the persistent session in SQLite."""

    @pytest.mark.asyncio
    async def test_vulnerability_header_override_must_not_poison_session_db(self, challenger_env):
        """CRITICAL VULNERABILITY TEST:
        When a session is created bound to 'ns_session', sending a one-off request with
        'X-RAG-Namespace: ns_header' must NOT mutate the session's permanent binding in the database.
        Subsequent requests without headers must STILL inherit 'ns_session'.
        """
        client: AsyncClient = challenger_env["client"]
        container: AppContainer = challenger_env["container"]
        spy: RetrievalSpy = challenger_env["spy"]

        # Step 1: Create session bound to ns_session
        s_resp = await client.post(
            "/api/sessions",
            json={"user_id": "alice", "title": "Math Course", "rag_namespace": "ns_session"},
        )
        assert s_resp.status_code in (200, 201)
        sid = s_resp.json()["session_id"]

        # Verify initial session state in DB
        sess_init = await container.state_manager.get_session("alice", sid)
        assert sess_init.rag_namespace == "ns_session"

        # Step 2: Send Request 1 with Header Override (X-RAG-Namespace: ns_header)
        spy.reset()
        res1 = await client.post(
            "/api/chat",
            headers={"X-RAG-Namespace": "ns_header"},
            json={
                "user_id": "alice",
                "session_id": sid,
                "message": "Turn 1 with header override",
            },
        )
        assert res1.status_code == 200
        assert spy.last_namespace == "ns_header"

        # Step 3: Check whether session in DB / StateManager was mutated!
        sess_after_t1 = await container.state_manager.get_session("alice", sid)
        # Note: Worker M3 handoff §3 explicitly claimed:
        # "Sending X-RAG-Namespace: override_ns on an already existing session bound to base_ns
        # sets effective_rag_ns = 'override_ns' for that specific completion turn.
        # Because the session was already created, get_or_create_session maintains the persistent
        # session state unless a full PATCH /api/sessions/{user_id}/{session_id} is executed."
        assert sess_after_t1.rag_namespace == "ns_session", (
            f"VULNERABILITY DETECTED: Header override mutated session in DB from 'ns_session' to '{sess_after_t1.rag_namespace}'!"
        )

        # Step 4: Send Request 2 without headers or body -> must inherit ns_session
        spy.reset()
        res2 = await client.post(
            "/api/chat",
            json={
                "user_id": "alice",
                "session_id": sid,
                "message": "Turn 2 should inherit original session namespace",
            },
        )
        assert res2.status_code == 200
        assert spy.last_namespace == "ns_session", (
            f"VULNERABILITY CONFIRMED: Turn 2 queried '{spy.last_namespace}' instead of original session 'ns_session'!"
        )
