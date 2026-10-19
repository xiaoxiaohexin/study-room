"""SQLite 关系型存储管理器：管理 sessions、messages、episodic_memories 与 notes 元数据。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

from app.observability.logging import get_logger

log = get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    name TEXT,
    tool_call_id TEXT,
    tool_calls_json TEXT,
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
    name TEXT NOT NULL,
    summary TEXT DEFAULT '',
    difficulty REAL DEFAULT 0.5,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kp_course ON knowledge_points(course_id);

CREATE TABLE IF NOT EXISTS kp_prereq (
    kp_id TEXT NOT NULL,
    prereq_id TEXT NOT NULL,
    PRIMARY KEY (kp_id, prereq_id)
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
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
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        log.info("sqlite_initialized_successfully", db_path=str(self.db_path))

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteStore is not initialized. Call initialize() first.")
        return self._db

    async def aclose(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            log.info("sqlite_closed")
