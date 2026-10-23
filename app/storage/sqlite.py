"""SQLite 关系型存储管理器：管理 sessions、messages、episodic_memories 与 notes 元数据。"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite

from app.observability.logging import get_logger

log = get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    course_id TEXT DEFAULT NULL,
    rag_namespace TEXT DEFAULT 'default',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_course ON sessions(course_id);
CREATE INDEX IF NOT EXISTS idx_sessions_rag_ns ON sessions(rag_namespace);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    name TEXT,
    tool_call_id TEXT,
    tool_calls_json TEXT,
    parent_id TEXT DEFAULT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at ASC);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON episodic_memories(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON episodic_memories(user_id, importance DESC);

CREATE TABLE IF NOT EXISTS notes_index (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    tags_json TEXT DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_user ON notes_index(user_id, updated_at DESC);

-- 本地持久化向量表（当未启动 Qdrant 时自动持久化向量）
CREATE TABLE IF NOT EXISTS vector_points (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    payload_json TEXT DEFAULT '{}',
    updated_at REAL NOT NULL,
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS idx_vector_collection ON vector_points(collection);

-- ==================== 学习领域模型表 ====================
CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    namespace TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_courses_user ON courses(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_points (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    user_id TEXT DEFAULT 'default_user',
    name TEXT NOT NULL,
    summary TEXT DEFAULT '',
    difficulty REAL DEFAULT 0.5,
    raw_quote TEXT DEFAULT '',
    content TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    provenance_json TEXT DEFAULT '{}',
    history_json TEXT DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_kp_course ON knowledge_points(course_id);
CREATE INDEX IF NOT EXISTS idx_kp_user_course ON knowledge_points(user_id, course_id);

CREATE TABLE IF NOT EXISTS kp_relations (
    id TEXT,
    src_id TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    user_id TEXT DEFAULT 'default_user',
    metadata_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (src_id, rel_type, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_kp_rel_id ON kp_relations(id);
CREATE INDEX IF NOT EXISTS idx_kp_rel_dst ON kp_relations(dst_id, rel_type);
CREATE INDEX IF NOT EXISTS idx_kp_rel_user ON kp_relations(user_id);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    filename TEXT NOT NULL,
    file_path TEXT DEFAULT '',
    page_count INTEGER DEFAULT 1,
    chunk_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    -- 来源分层：tier=standard 为正式知识库；tier=staging 为临时知识库（待提纯）
    tier TEXT DEFAULT 'standard',
    source_type TEXT DEFAULT 'local',
    url TEXT DEFAULT '',
    fetched_at REAL DEFAULT 0,
    content_hash TEXT DEFAULT '',
    origin TEXT DEFAULT '',
    conversation_id TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    user_id TEXT DEFAULT 'default_user',
    quality_score REAL DEFAULT 0,
    status TEXT DEFAULT 'standard',
    target_namespace TEXT DEFAULT '',
    promoted_doc_id TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_tier_session ON documents(tier, session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(tier, status);

CREATE TABLE IF NOT EXISTS staging_candidates (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    session_id TEXT DEFAULT '',
    user_id TEXT DEFAULT 'default_user',
    title TEXT NOT NULL,
    summary TEXT DEFAULT '',
    content TEXT DEFAULT '',
    raw_quote TEXT DEFAULT '',
    difficulty REAL DEFAULT 0.5,
    suggested_course_id TEXT DEFAULT '',
    quality_score REAL DEFAULT 0,
    status TEXT DEFAULT 'pending',
    provenance_json TEXT DEFAULT '{}',
    promoted_kp_id TEXT DEFAULT '',
    promoted_doc_id TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_staging_candidates_status
    ON staging_candidates(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_staging_candidates_doc ON staging_candidates(doc_id);

CREATE TABLE IF NOT EXISTS kp_prereq (
    kp_id TEXT NOT NULL,
    prereq_id TEXT NOT NULL,
    PRIMARY KEY (kp_id, prereq_id)
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default_user',
    kp_id TEXT NOT NULL,
    course_id TEXT DEFAULT 'default',
    type TEXT NOT NULL,
    stem TEXT NOT NULL,
    options_json TEXT DEFAULT '[]',
    answer TEXT NOT NULL,
    explanation TEXT DEFAULT '',
    source_chunk_id TEXT DEFAULT '',
    difficulty REAL DEFAULT 0.5,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_kp ON questions(kp_id);
CREATE INDEX IF NOT EXISTS idx_questions_user ON questions(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    kp_id TEXT NOT NULL,
    answer TEXT NOT NULL,
    is_correct INTEGER NOT NULL,
    score REAL NOT NULL,
    error_type TEXT DEFAULT '',
    feedback TEXT DEFAULT '',
    hints_used INTEGER DEFAULT 0,
    elapsed_sec REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS mastery (
    user_id TEXT NOT NULL,
    kp_id TEXT NOT NULL,
    mastery REAL DEFAULT 0.0,
    ef REAL DEFAULT 2.5,
    interval_days REAL DEFAULT 0.0,
    repetitions INTEGER DEFAULT 0,
    next_review_at REAL DEFAULT 0.0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, kp_id)
);
CREATE INDEX IF NOT EXISTS idx_mastery_review ON mastery(user_id, next_review_at ASC);

CREATE TABLE IF NOT EXISTS mistakes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    kp_id TEXT NOT NULL,
    error_type TEXT DEFAULT 'concept',
    note TEXT DEFAULT '',
    resolved INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mistakes_user ON mistakes(user_id, resolved, updated_at DESC);

CREATE TABLE IF NOT EXISTS study_plans (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    course_id TEXT DEFAULT 'default',
    goal TEXT NOT NULL,
    deadline TEXT DEFAULT '',
    items_json TEXT DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_user ON study_plans(user_id, updated_at DESC);
"""


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        log.info("initializing_sqlite", db_path=str(self.db_path))
        async with self._lock:
            if self._db is None:
                self._db = await aiosqlite.connect(self.db_path)
                self._db.row_factory = aiosqlite.Row
            await self._migrate_schema_unlocked()
            await self._db.executescript(SCHEMA_SQL)
            await self._migrate_schema_unlocked()
            await self._db.commit()
        log.info("sqlite_initialized_successfully", db_path=str(self.db_path))

    async def _safe_add_column(self, table: str, col_name: str, col_type: str) -> None:
        """Safely execute ALTER TABLE ADD COLUMN, ignoring duplicate column errors during concurrent races."""
        if self._db is None:
            return
        try:
            await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
            log.info(f"migrated_{table}_column", column=col_name)
        except (aiosqlite.OperationalError, sqlite3.OperationalError) as exc:
            if "duplicate column name" in str(exc).lower():
                log.debug("column_already_exists_ignored", table=table, column=col_name, error=str(exc))
            else:
                raise

    async def _migrate_schema_unlocked(self) -> None:
        """Idempotently migrate pre-existing tables to the latest schema without acquiring lock."""
        if self._db is None:
            return

        # 1. Migrate knowledge_points columns
        cur = await self._db.execute("PRAGMA table_info(knowledge_points)")
        rows = await cur.fetchall()
        kp_cols = {row["name"] if hasattr(row, "__getitem__") and "name" in row.keys() else row[1] for row in rows}

        if kp_cols:
            kp_additions = [
                ("raw_quote", "TEXT DEFAULT ''"),
                ("content", "TEXT DEFAULT ''"),
                ("version", "INTEGER DEFAULT 1"),
                ("provenance_json", "TEXT DEFAULT '{}'"),
                ("history_json", "TEXT DEFAULT '[]'"),
                ("updated_at", "REAL DEFAULT 0.0"),
                ("user_id", "TEXT DEFAULT 'default_user'"),
            ]
            for col_name, col_type in kp_additions:
                if col_name not in kp_cols:
                    await self._safe_add_column("knowledge_points", col_name, col_type)
            # Backfill legacy knowledge points from their owning course when possible.
            await self._db.execute(
                "UPDATE knowledge_points SET user_id = 'default_user' "
                "WHERE user_id IS NULL OR user_id = ''"
            )
            await self._db.execute(
                "UPDATE knowledge_points SET user_id = ("
                "  SELECT user_id FROM courses WHERE courses.id = knowledge_points.course_id"
                ") WHERE EXISTS ("
                "  SELECT 1 FROM courses WHERE courses.id = knowledge_points.course_id"
                ") AND (user_id IS NULL OR user_id = '' OR user_id = 'default_user')"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kp_user_course ON knowledge_points(user_id, course_id)"
            )

        # 2. Migrate messages columns
        cur = await self._db.execute("PRAGMA table_info(messages)")
        rows = await cur.fetchall()
        msg_cols = {row["name"] if hasattr(row, "__getitem__") and "name" in row.keys() else row[1] for row in rows}
        if msg_cols:
            if "parent_id" not in msg_cols:
                await self._safe_add_column("messages", "parent_id", "TEXT DEFAULT NULL")
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id)")

        # 3. Migrate kp_relations columns and indexes (in case pre-created without id)
        cur = await self._db.execute("PRAGMA table_info(kp_relations)")
        rows = await cur.fetchall()
        rel_cols = {row["name"] if hasattr(row, "__getitem__") and "name" in row.keys() else row[1] for row in rows}
        if rel_cols:
            if "id" not in rel_cols:
                await self._safe_add_column("kp_relations", "id", "TEXT")
            if "user_id" not in rel_cols:
                await self._safe_add_column("kp_relations", "user_id", "TEXT DEFAULT 'default_user'")
            await self._db.execute(
                "UPDATE kp_relations SET user_id = 'default_user' "
                "WHERE user_id IS NULL OR user_id = ''"
            )
            await self._db.execute(
                "UPDATE kp_relations SET user_id = ("
                "  SELECT kp.user_id FROM knowledge_points kp WHERE kp.id = kp_relations.src_id"
                ") WHERE EXISTS ("
                "  SELECT 1 FROM knowledge_points kp WHERE kp.id = kp_relations.src_id"
                ") AND (user_id IS NULL OR user_id = '' OR user_id = 'default_user')"
            )
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_kp_rel_id ON kp_relations(id)")
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_kp_rel_dst ON kp_relations(dst_id, rel_type)")
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_kp_rel_user ON kp_relations(user_id)")

        # 4. Migrate sessions columns and indexes
        cur = await self._db.execute("PRAGMA table_info(sessions)")
        rows = await cur.fetchall()
        sess_cols = {row["name"] if hasattr(row, "__getitem__") and "name" in row.keys() else row[1] for row in rows}
        if sess_cols:
            if "course_id" not in sess_cols:
                await self._safe_add_column("sessions", "course_id", "TEXT DEFAULT NULL")
            if "rag_namespace" not in sess_cols:
                await self._safe_add_column("sessions", "rag_namespace", "TEXT DEFAULT 'default'")
            await self._db.execute(
                "UPDATE sessions SET rag_namespace = 'default' WHERE rag_namespace IS NULL OR rag_namespace = ''"
            )
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_course ON sessions(course_id)")
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_rag_ns ON sessions(rag_namespace)")

        # 5. Migrate documents columns for the staging/tier model
        cur = await self._db.execute("PRAGMA table_info(documents)")
        rows = await cur.fetchall()
        doc_cols = {row["name"] if hasattr(row, "__getitem__") and "name" in row.keys() else row[1] for row in rows}
        if doc_cols:
            doc_additions = [
                ("tier", "TEXT DEFAULT 'standard'"),
                ("source_type", "TEXT DEFAULT 'local'"),
                ("url", "TEXT DEFAULT ''"),
                ("fetched_at", "REAL DEFAULT 0"),
                ("content_hash", "TEXT DEFAULT ''"),
                ("origin", "TEXT DEFAULT ''"),
                ("conversation_id", "TEXT DEFAULT ''"),
                ("session_id", "TEXT DEFAULT ''"),
                ("user_id", "TEXT DEFAULT 'default_user'"),
                ("quality_score", "REAL DEFAULT 0"),
                ("status", "TEXT DEFAULT 'standard'"),
                ("target_namespace", "TEXT DEFAULT ''"),
                ("promoted_doc_id", "TEXT DEFAULT ''"),
                ("metadata_json", "TEXT DEFAULT '{}'"),
            ]
            for col_name, col_type in doc_additions:
                if col_name not in doc_cols:
                    await self._safe_add_column("documents", col_name, col_type)
            await self._db.execute(
                "UPDATE documents SET tier = 'standard' WHERE tier IS NULL OR tier = ''"
            )
            await self._db.execute(
                "UPDATE documents SET status = 'standard' WHERE status IS NULL OR status = ''"
            )
            await self._db.execute(
                "UPDATE documents SET source_type = 'local' WHERE source_type IS NULL OR source_type = ''"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tier_session "
                "ON documents(tier, session_id, created_at DESC)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(tier, status)"
            )

        # 6. Migrate questions: per-user ownership so a generated question can never
        #    be answered/graded by another user (audit P1-5).
        cur = await self._db.execute("PRAGMA table_info(questions)")
        rows = await cur.fetchall()
        q_cols = {row["name"] if hasattr(row, "__getitem__") and "name" in row.keys() else row[1] for row in rows}
        if q_cols:
            if "user_id" not in q_cols:
                await self._safe_add_column("questions", "user_id", "TEXT DEFAULT 'default_user'")
            await self._db.execute(
                "UPDATE questions SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = ''"
            )
            # Backfill from the owning knowledge point when it is unambiguous.
            await self._db.execute(
                "UPDATE questions SET user_id = ("
                "  SELECT kp.user_id FROM knowledge_points kp WHERE kp.id = questions.kp_id"
                ") WHERE EXISTS ("
                "  SELECT 1 FROM knowledge_points kp WHERE kp.id = questions.kp_id"
                ") AND user_id = 'default_user'"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_questions_user ON questions(user_id, created_at DESC)"
            )
        await self._db.commit()

    async def _migrate_schema(self) -> None:
        """Idempotently migrate pre-existing tables to the latest schema (thread/task safe)."""
        async with self._lock:
            await self._migrate_schema_unlocked()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteStore is not initialized. Call initialize() first.")
        return self._db

    # ==================== RAG 文档注册表 / 临时知识库 ====================

    _DOCUMENT_FIELDS = {
        "namespace",
        "filename",
        "file_path",
        "page_count",
        "chunk_count",
        "tier",
        "source_type",
        "url",
        "fetched_at",
        "content_hash",
        "origin",
        "conversation_id",
        "session_id",
        "user_id",
        "quality_score",
        "status",
        "target_namespace",
        "promoted_doc_id",
        "metadata_json",
    }

    async def register_document(
        self,
        doc_id: str,
        namespace: str,
        filename: str,
        file_path: str,
        page_count: int,
        chunk_count: int,
        created_at: float | None = None,
        **fields: object,
    ) -> None:
        """登记文档（标准库或临时库）的原始文件、来源与分层元数据。"""
        import time as _time

        values: dict[str, object] = {
            "doc_id": doc_id,
            "namespace": namespace or "default",
            "filename": filename,
            "file_path": file_path,
            "page_count": max(1, int(page_count or 1)),
            "chunk_count": int(chunk_count or 0),
            "created_at": float(created_at if created_at is not None else _time.time()),
            "tier": "standard",
            "source_type": "local",
            "url": "",
            "fetched_at": 0.0,
            "content_hash": "",
            "origin": "",
            "conversation_id": "",
            "session_id": "",
            "user_id": "default_user",
            "quality_score": 0.0,
            "status": "standard",
            "target_namespace": "",
            "promoted_doc_id": "",
            "metadata_json": "{}",
        }
        for key, value in fields.items():
            if key in self._DOCUMENT_FIELDS and value is not None:
                values[key] = value

        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        await self.conn.execute(
            f"INSERT OR REPLACE INTO documents ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        await self.conn.commit()

    async def update_document(self, doc_id: str, **fields: object) -> bool:
        update = {k: v for k, v in fields.items() if k in self._DOCUMENT_FIELDS}
        if not update:
            return False
        assignments = ", ".join(f"{key} = ?" for key in update)
        cursor = await self.conn.execute(
            f"UPDATE documents SET {assignments} WHERE doc_id = ?",
            (*update.values(), doc_id),
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    async def get_document(self, doc_id: str) -> dict | None:
        cursor = await self.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_documents(
        self,
        namespace: str | None = None,
        *,
        tier: str | None = "standard",
        session_id: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await self.conn.execute(
            f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT ?",
            (*params, max(1, int(limit))),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def find_document_by_hash(
        self,
        content_hash: str,
        *,
        tier: str | None = "staging",
        session_id: str | None = None,
    ) -> dict | None:
        if not content_hash:
            return None
        clauses = ["content_hash = ?"]
        params: list[object] = [content_hash]
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        cursor = await self.conn.execute(
            f"SELECT * FROM documents WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at DESC LIMIT 1",
            tuple(params),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_expired_staging(self, cutoff_ts: float, limit: int = 200) -> list[dict]:
        cursor = await self.conn.execute(
            "SELECT * FROM documents WHERE tier = 'staging' AND created_at < ? "
            "AND status != 'promoted' ORDER BY created_at ASC LIMIT ?",
            (cutoff_ts, max(1, int(limit))),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def delete_document(self, doc_id: str, namespace: str | None = None) -> None:
        if namespace is None:
            await self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        else:
            await self.conn.execute(
                "DELETE FROM documents WHERE doc_id = ? AND namespace = ?", (doc_id, namespace)
            )
        await self.conn.commit()

    # ==================== 临时知识库提纯候选 ====================

    _CANDIDATE_FIELDS = {
        "doc_id",
        "session_id",
        "user_id",
        "title",
        "summary",
        "content",
        "raw_quote",
        "difficulty",
        "suggested_course_id",
        "quality_score",
        "status",
        "provenance_json",
        "promoted_kp_id",
        "promoted_doc_id",
        "updated_at",
    }

    async def add_staging_candidate(self, candidate: dict) -> None:
        import time as _time

        values = {
            "id": candidate["id"],
            "doc_id": candidate.get("doc_id", ""),
            "session_id": candidate.get("session_id", ""),
            "user_id": candidate.get("user_id", "default_user"),
            "title": candidate.get("title", "未命名候选"),
            "summary": candidate.get("summary", ""),
            "content": candidate.get("content", ""),
            "raw_quote": candidate.get("raw_quote", ""),
            "difficulty": float(candidate.get("difficulty", 0.5) or 0.5),
            "suggested_course_id": candidate.get("suggested_course_id", ""),
            "quality_score": float(candidate.get("quality_score", 0) or 0),
            "status": candidate.get("status", "pending"),
            "provenance_json": candidate.get("provenance_json", "{}"),
            "promoted_kp_id": "",
            "promoted_doc_id": "",
            "created_at": float(candidate.get("created_at") or _time.time()),
            "updated_at": float(candidate.get("updated_at") or _time.time()),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        await self.conn.execute(
            f"INSERT OR REPLACE INTO staging_candidates ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        await self.conn.commit()

    async def get_staging_candidate(self, candidate_id: str) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM staging_candidates WHERE id = ?", (candidate_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_staging_candidates(
        self,
        *,
        status: str | None = "pending",
        session_id: str | None = None,
        user_id: str | None = None,
        doc_id: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await self.conn.execute(
            f"SELECT * FROM staging_candidates {where} ORDER BY created_at DESC LIMIT ?",
            (*params, max(1, int(limit))),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def update_staging_candidate(self, candidate_id: str, **fields: object) -> bool:
        update = {k: v for k, v in fields.items() if k in self._CANDIDATE_FIELDS}
        if not update:
            return False
        assignments = ", ".join(f"{key} = ?" for key in update)
        cursor = await self.conn.execute(
            f"UPDATE staging_candidates SET {assignments} WHERE id = ?",
            (*update.values(), candidate_id),
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    async def delete_staging_candidate(self, candidate_id: str) -> None:
        await self.conn.execute("DELETE FROM staging_candidates WHERE id = ?", (candidate_id,))
        await self.conn.commit()

    async def delete_staging_candidates_for_doc(self, doc_id: str) -> None:
        await self.conn.execute("DELETE FROM staging_candidates WHERE doc_id = ?", (doc_id,))
        await self.conn.commit()

    async def list_user_ids(self, limit: int = 500) -> list[str]:
        """Distinct users that own any state (used by the background memory maintenance)."""
        cur = await self.conn.execute(
            """
            SELECT user_id FROM (
                SELECT user_id FROM sessions
                UNION
                SELECT user_id FROM mastery
                UNION
                SELECT user_id FROM episodic_memories
            )
            WHERE user_id IS NOT NULL AND user_id <> ''
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        return [r["user_id"] for r in rows]

    async def aclose(self) -> None:
        async with self._lock:
            if self._db:
                await self._db.close()
                self._db = None
                log.info("sqlite_closed")
