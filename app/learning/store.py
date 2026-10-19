"""学习领域数据存储（LearningStore）：管理课程、知识点、题目、作答、掌握度（SM-2）、错题本与学习计划。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.learning.algorithms import compute_sm2, score_to_quality, update_mastery_ema
from app.observability.logging import get_logger
from app.storage.sqlite import SQLiteStore

log = get_logger(__name__)


class LearningStore:
    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    # ==================== 课程管理 ====================
    async def create_course(
        self,
        user_id: str,
        name: str,
        namespace: str | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        cid = f"course_{uuid.uuid4().hex[:10]}"
        ns = namespace or cid
        now = time.time()
        await self.sqlite.conn.execute(
            "INSERT INTO courses (id, user_id, name, namespace, description, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, user_id, name, ns, description, now),
        )
        await self.sqlite.conn.commit()
        return {
            "id": cid,
            "user_id": user_id,
            "name": name,
            "namespace": ns,
            "description": description,
            "created_at": now,
        }

    async def list_courses(self, user_id: str) -> list[dict[str, Any]]:
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM courses WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ==================== 知识点管理 ====================
    async def upsert_knowledge_point(
        self,
        course_id: str,
        name: str,
        summary: str = "",
        difficulty: float = 0.5,
        prereq_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM knowledge_points WHERE course_id = ? AND name = ?",
            (course_id, name),
        )
        existing = await cur.fetchone()
        now = time.time()
        if existing:
            kp_id = existing["id"]
            await self.sqlite.conn.execute(
                "UPDATE knowledge_points SET summary = ?, difficulty = ? WHERE id = ?",
                (summary or existing["summary"], difficulty, kp_id),
            )
        else:
            kp_id = f"kp_{uuid.uuid4().hex[:10]}"
            await self.sqlite.conn.execute(
                "INSERT INTO knowledge_points (id, course_id, name, summary, difficulty, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (kp_id, course_id, name, summary, difficulty, now),
            )
        if prereq_ids:
            for pid in prereq_ids:
                await self.sqlite.conn.execute(
                    "INSERT OR IGNORE INTO kp_prereq (kp_id, prereq_id) VALUES (?, ?)",
                    (kp_id, pid),
                )
        await self.sqlite.conn.commit()
        return {
            "id": kp_id,
            "course_id": course_id,
            "name": name,
            "summary": summary,
            "difficulty": difficulty,
        }

    async def list_knowledge_points(self, course_id: str | None = None) -> list[dict[str, Any]]:
        if course_id:
            cur = await self.sqlite.conn.execute(
                "SELECT * FROM knowledge_points WHERE course_id = ? ORDER BY created_at ASC",
                (course_id,),
            )
        else:
            cur = await self.sqlite.conn.execute("SELECT * FROM knowledge_points ORDER BY created_at ASC")
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_knowledge_point(self, kp_id_or_name: str, course_id: str = "default") -> dict[str, Any] | None:
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM knowledge_points WHERE id = ? OR (name = ? AND course_id = ?) OR name = ?",
            (kp_id_or_name, kp_id_or_name, course_id, kp_id_or_name),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    # ==================== 题库与出题 ====================
    async def add_question(
        self,
        kp_id: str,
        q_type: str,
        stem: str,
        answer: str,
        explanation: str = "",
        options: list[str] | None = None,
        source_chunk_id: str = "",
        difficulty: float = 0.5,
        course_id: str = "default",
    ) -> dict[str, Any]:
        qid = f"q_{uuid.uuid4().hex[:10]}"
        now = time.time()
        opts_json = json.dumps(options or [], ensure_ascii=False)
        await self.sqlite.conn.execute(
            "INSERT INTO questions (id, kp_id, course_id, type, stem, options_json, answer, explanation, source_chunk_id, difficulty, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (qid, kp_id, course_id, q_type, stem, opts_json, answer, explanation, source_chunk_id, difficulty, now),
        )
        await self.sqlite.conn.commit()
        return {
            "id": qid,
            "kp_id": kp_id,
            "course_id": course_id,
            "type": q_type,
            "stem": stem,
            "options": options or [],
            "answer": answer,
            "explanation": explanation,
            "source_chunk_id": source_chunk_id,
            "difficulty": difficulty,
        }

    async def get_question(self, question_id: str) -> dict[str, Any] | None:
        cur = await self.sqlite.conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,))
        row = await cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["options"] = json.loads(d.pop("options_json", "[]") or "[]")
        return d

    async def get_latest_unanswered_question(self, user_id: str) -> dict[str, Any] | None:
        """获取最近生成但该用户尚未作答的一道题（用于会话内直接作答自动关联）。"""
        cur = await self.sqlite.conn.execute(
            """
            SELECT q.* FROM questions q
            WHERE NOT EXISTS (
                SELECT 1 FROM attempts a WHERE a.question_id = q.id AND a.user_id = ?
            )
            ORDER BY q.created_at DESC LIMIT 1
            """,
            (user_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["options"] = json.loads(d.pop("options_json", "[]") or "[]")
        return d

    # ==================== 作答判分、掌握度与 SM-2 更新 ====================
    async def record_attempt_and_update_mastery(
        self,
        user_id: str,
        question_id: str,
        user_answer: str,
        is_correct: bool,
        score: float,
        error_type: str = "",
        feedback: str = "",
        hints_used: int = 0,
        elapsed_sec: float = 0.0,
    ) -> dict[str, Any]:
        q = await self.get_question(question_id)
        kp_id = q["kp_id"] if q else "kp_general"
        difficulty = float(q["difficulty"]) if q else 0.5
        now = time.time()

        attempt_id = f"att_{uuid.uuid4().hex[:10]}"
        await self.sqlite.conn.execute(
            "INSERT INTO attempts (id, user_id, question_id, kp_id, answer, is_correct, score, error_type, feedback, hints_used, elapsed_sec, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                user_id,
                question_id,
                kp_id,
                user_answer,
                1 if is_correct else 0,
                score,
                error_type,
                feedback,
                hints_used,
                elapsed_sec,
                now,
            ),
        )

        # 读取现有掌握度与 SM-2 状态
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?",
            (user_id, kp_id),
        )
        m_row = await cur.fetchone()
        prev_mastery = float(m_row["mastery"]) if m_row else 0.3
        prev_ef = float(m_row["ef"]) if m_row else 2.5
        prev_interval = float(m_row["interval_days"]) if m_row else 0.0
        prev_reps = int(m_row["repetitions"]) if m_row else 0

        quality = score_to_quality(is_correct, score, hints_used=hints_used, elapsed_sec=elapsed_sec)
        sm2 = compute_sm2(prev_ef, prev_interval, prev_reps, quality, now=now)
        new_mastery = update_mastery_ema(prev_mastery, score, difficulty=difficulty)

        await self.sqlite.conn.execute(
            """
            INSERT INTO mastery (user_id, kp_id, mastery, ef, interval_days, repetitions, next_review_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, kp_id) DO UPDATE SET
                mastery = excluded.mastery,
                ef = excluded.ef,
                interval_days = excluded.interval_days,
                repetitions = excluded.repetitions,
                next_review_at = excluded.next_review_at,
                updated_at = excluded.updated_at
            """,
            (user_id, kp_id, new_mastery, sm2.ef, sm2.interval_days, sm2.repetitions, sm2.next_review_at, now),
        )

        # 答错自动记入错题本
        mistake_id = None
        if not is_correct or score < 0.6:
            mistake_id = f"mis_{uuid.uuid4().hex[:10]}"
            await self.sqlite.conn.execute(
                "INSERT INTO mistakes (id, user_id, question_id, kp_id, error_type, note, resolved, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (mistake_id, user_id, question_id, kp_id, error_type or "concept", feedback[:300], now, now),
            )

        await self.sqlite.conn.commit()
        return {
            "attempt_id": attempt_id,
            "question_id": question_id,
            "kp_id": kp_id,
            "is_correct": is_correct,
            "score": score,
            "quality": quality,
            "prev_mastery": prev_mastery,
            "new_mastery": new_mastery,
            "interval_days": sm2.interval_days,
            "repetitions": sm2.repetitions,
            "next_review_at": sm2.next_review_at,
            "mistake_id": mistake_id,
        }

    # ==================== 复习与错题本 ====================
    async def get_due_reviews(self, user_id: str, now: float | None = None, limit: int = 10) -> list[dict[str, Any]]:
        ts = now if now is not None else time.time()
        cur = await self.sqlite.conn.execute(
            """
            SELECT m.*, COALESCE(kp.name, m.kp_id) AS kp_name, COALESCE(kp.summary, '') AS kp_summary
            FROM mastery m
            LEFT JOIN knowledge_points kp ON kp.id = m.kp_id
            WHERE m.user_id = ? AND m.next_review_at <= ?
            ORDER BY m.mastery ASC, m.next_review_at ASC
            LIMIT ?
            """,
            (user_id, ts, limit),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def list_mistakes(self, user_id: str, resolved: bool = False, limit: int = 20) -> list[dict[str, Any]]:
        cur = await self.sqlite.conn.execute(
            """
            SELECT m.*, q.stem, q.answer AS expected_answer, COALESCE(kp.name, m.kp_id) AS kp_name
            FROM mistakes m
            LEFT JOIN questions q ON q.id = m.question_id
            LEFT JOIN knowledge_points kp ON kp.id = m.kp_id
            WHERE m.user_id = ? AND m.resolved = ?
            ORDER BY m.updated_at DESC
            LIMIT ?
            """,
            (user_id, 1 if resolved else 0, limit),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def resolve_mistake(self, user_id: str, mistake_id: str) -> bool:
        now = time.time()
        cur = await self.sqlite.conn.execute(
            "UPDATE mistakes SET resolved = 1, updated_at = ? WHERE id = ? AND user_id = ?",
            (now, mistake_id, user_id),
        )
        await self.sqlite.conn.commit()
        return cur.rowcount > 0

    # ==================== 学习进度与画像 ====================
    async def get_progress_report(self, user_id: str) -> dict[str, Any]:
        cur_m = await self.sqlite.conn.execute(
            """
            SELECT m.*, COALESCE(kp.name, m.kp_id) AS kp_name
            FROM mastery m
            LEFT JOIN knowledge_points kp ON kp.id = m.kp_id
            WHERE m.user_id = ?
            ORDER BY m.mastery ASC
            """,
            (user_id,),
        )
        mastery_rows = [dict(r) for r in await cur_m.fetchall()]

        cur_a = await self.sqlite.conn.execute(
            "SELECT COUNT(*) AS total, SUM(is_correct) AS correct, AVG(score) AS avg_score FROM attempts WHERE user_id = ?",
            (user_id,),
        )
        stat_row = await cur_a.fetchone()
        total_attempts = int(stat_row["total"] or 0) if stat_row else 0
        correct_attempts = int(stat_row["correct"] or 0) if stat_row else 0
        avg_score = round(float(stat_row["avg_score"] or 0.0), 3) if stat_row else 0.0

        due = await self.get_due_reviews(user_id, limit=20)
        mistakes = await self.list_mistakes(user_id, resolved=False, limit=10)

        return {
            "user_id": user_id,
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "accuracy": round(correct_attempts / total_attempts, 3) if total_attempts else 0.0,
            "avg_score": avg_score,
            "knowledge_mastery": mastery_rows,
            "weakest_kps": mastery_rows[:5],
            "due_review_count": len(due),
            "due_reviews": due[:5],
            "unresolved_mistakes_count": len(mistakes),
            "recent_mistakes": mistakes[:5],
        }

    async def build_learner_profile_text(self, user_id: str) -> str:
        """为 GSSC ContextBuilder 构建简洁的学习者画像文本。"""
        report = await self.get_progress_report(user_id)
        pending_q = await self.get_latest_unanswered_question(user_id)

        lines: list[str] = []
        if report["total_attempts"] > 0:
            lines.append(
                f"- 累计作答: {report['total_attempts']} 题 | 正确率: {report['accuracy'] * 100:.1f}% | 待复习知识点: {report['due_review_count']} 个 | 未消灭错题: {report['unresolved_mistakes_count']} 道"
            )
        if report["weakest_kps"]:
            weak_str = "、".join(f"{w['kp_name']}(掌握度:{w['mastery']:.2f})" for w in report["weakest_kps"][:3])
            lines.append(f"- 当前薄弱知识点: {weak_str}")
        if report["recent_mistakes"]:
            mis_str = "；".join(
                f"[{m['error_type']}] {m.get('kp_name', '')}: {(m.get('stem') or '')[:40]}"
                for m in report["recent_mistakes"][:2]
            )
            lines.append(f"- 近期典型错题: {mis_str}")
        if pending_q:
            opts = f" 选项: {pending_q['options']}" if pending_q.get("options") else ""
            lines.append(
                f"- 当前待作答题目 (question_id={pending_q['id']}, 类型={pending_q['type']}): {pending_q['stem']}{opts}"
            )
        return "\n".join(lines)

    # ==================== 学习计划 ====================
    async def save_study_plan(
        self,
        user_id: str,
        goal: str,
        items: list[dict[str, Any]],
        course_id: str = "default",
        deadline: str = "",
    ) -> dict[str, Any]:
        pid = f"plan_{uuid.uuid4().hex[:10]}"
        now = time.time()
        await self.sqlite.conn.execute(
            "INSERT INTO study_plans (id, user_id, course_id, goal, deadline, items_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, user_id, course_id, goal, deadline, json.dumps(items, ensure_ascii=False), now, now),
        )
        await self.sqlite.conn.commit()
        return {
            "id": pid,
            "user_id": user_id,
            "course_id": course_id,
            "goal": goal,
            "deadline": deadline,
            "items": items,
            "created_at": now,
        }

    async def list_study_plans(self, user_id: str) -> list[dict[str, Any]]:
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM study_plans WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["items"] = json.loads(d.pop("items_json", "[]") or "[]")
            out.append(d)
        return out
