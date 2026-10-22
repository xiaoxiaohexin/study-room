"""学习领域数据存储（LearningStore）：管理课程、知识点、题目、作答、掌握度（SM-2）、错题本与学习计划。"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

from app.learning.algorithms import compute_sm2, score_to_quality, update_mastery_ema
from app.observability.logging import get_logger
from app.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from app.storage.graph import GraphStore

log = get_logger(__name__)


class LearningStore:
    def __init__(self, sqlite: SQLiteStore, graph_store: GraphStore | None = None):
        self.sqlite = sqlite
        self.graph_store = graph_store

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

    async def ensure_course(
        self,
        user_id: str,
        course_id: str | None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Ensure a course row exists for the given (user, course_id).

        The frontend may bind preset course ids (e.g. ``math_calc``) that were never
        explicitly created. ``default`` is a virtual course and never gets a row, so
        knowledge points under it stay scoped by ``user_id`` only.
        """
        cid = (course_id or "default").strip() or "default"
        if cid == "default":
            return {
                "id": "default",
                "user_id": user_id,
                "name": "默认课程",
                "namespace": "default",
                "description": "",
                "virtual": True,
            }

        cur = await self.sqlite.conn.execute("SELECT * FROM courses WHERE id = ?", (cid,))
        row = await cur.fetchone()
        if row is not None:
            if row["user_id"] != user_id:
                raise PermissionError(f"Course '{cid}' belongs to another user")
            return dict(row)

        now = time.time()
        try:
            await self.sqlite.conn.execute(
                "INSERT INTO courses (id, user_id, name, namespace, description, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cid, user_id, name or cid, cid, "", now),
            )
            await self.sqlite.conn.commit()
        except Exception as exc:  # noqa: BLE001 - concurrent creation race
            if "UNIQUE" not in str(exc).upper() and "PRIMARY" not in str(exc).upper():
                raise
            cur = await self.sqlite.conn.execute("SELECT * FROM courses WHERE id = ?", (cid,))
            row = await cur.fetchone()
            if row is None:
                raise
            if row["user_id"] != user_id:
                raise PermissionError(f"Course '{cid}' belongs to another user") from exc
            return dict(row)
        return {
            "id": cid,
            "user_id": user_id,
            "name": name or cid,
            "namespace": cid,
            "description": "",
            "created_at": now,
        }

    async def _get_owned_kp(
        self, kp_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Fetch a knowledge point only when it belongs to ``user_id``."""
        cur = await self.sqlite.conn.execute(
            "SELECT id, user_id, course_id FROM knowledge_points WHERE id = ?",
            (kp_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        if (row["user_id"] or "default_user") != user_id:
            raise PermissionError(f"Knowledge point '{kp_id}' belongs to another user")
        return dict(row)

    # ==================== 知识点管理 ====================
    async def upsert_knowledge_point(
        self,
        course_id: str,
        name: str,
        summary: str = "",
        difficulty: float = 0.5,
        prereq_ids: list[str] | None = None,
        *,
        kp_id: str | None = None,
        raw_quote: str = "",
        content: str = "",
        provenance_dict: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        user_id: str = "default_user",
        relations: list[dict[str, Any]] | None = None,
        change_type: str | None = None,
    ) -> dict[str, Any]:
        prov = provenance_dict if provenance_dict is not None else (provenance or {})
        prov_json = json.dumps(prov, ensure_ascii=False)
        now = time.time()
        diff = min(1.0, max(0.1, float(difficulty)))

        # Auto-register explicit (non-default) course ids and verify ownership.
        await self.ensure_course(user_id, course_id)

        if kp_id:
            cur = await self.sqlite.conn.execute(
                "SELECT * FROM knowledge_points WHERE id = ?", (kp_id,)
            )
        else:
            cur = await self.sqlite.conn.execute(
                "SELECT * FROM knowledge_points WHERE course_id = ? AND name = ? AND user_id = ?",
                (course_id, name, user_id),
            )
        existing = await cur.fetchone()

        if existing is not None:
            existing_user = existing["user_id"] or "default_user"
            if existing_user != user_id:
                raise PermissionError(f"Knowledge point '{existing['id']}' belongs to another user")
            existing_course = existing["course_id"] or "default"
            if existing_course != (course_id or "default"):
                raise ValueError(
                    f"Knowledge point '{existing['id']}' belongs to course '{existing_course}', "
                    "refusing cross-course update"
                )

        if existing is None:
            target_id = kp_id or f"kp_{uuid.uuid4().hex[:10]}"
            init_version = 1
            init_content = content or summary or ""
            init_raw_quote = raw_quote or ""
            init_summary = summary or (init_content[:200] if init_content else "")
            initial_snapshot = {
                "version": 1,
                "timestamp": now,
                "content": init_content,
                "raw_quote": init_raw_quote,
                "summary": init_summary,
                "change_type": change_type or "create",
                "changeType": change_type or "create",
            }
            history = [initial_snapshot]
            init_history_json = json.dumps(history, ensure_ascii=False)

            await self.sqlite.conn.execute(
                """
                INSERT INTO knowledge_points (
                    id, course_id, user_id, name, summary, difficulty,
                    raw_quote, content, version, provenance_json, history_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    course_id,
                    user_id,
                    name,
                    init_summary,
                    diff,
                    init_raw_quote,
                    init_content,
                    init_version,
                    prov_json,
                    init_history_json,
                    now,
                    now,
                ),
            )
            await self.sqlite.conn.execute(
                """
                INSERT INTO mastery (
                    user_id, kp_id, mastery, ef, interval_days, repetitions, next_review_at, updated_at
                ) VALUES (?, ?, 0.3, 2.5, 1.0, 0, ?, ?)
                ON CONFLICT(user_id, kp_id) DO NOTHING
                """,
                (user_id, target_id, now, now),
            )
            final_summary = init_summary
            final_raw_quote = init_raw_quote
            final_content = init_content
            final_version = init_version
            final_prov = prov
            final_created_at = now
        else:
            target_id = existing["id"]
            existing_raw = existing["raw_quote"] or ""
            final_raw_quote = existing_raw if existing_raw else (raw_quote or "")

            if content:
                final_content = content
            elif summary and not (existing["content"] or ""):
                final_content = summary
            else:
                final_content = existing["content"] or existing["summary"] or ""

            final_summary = summary or existing["summary"] or ""

            if prov:
                final_prov = prov
                final_prov_json = prov_json
            else:
                try:
                    final_prov = json.loads(existing["provenance_json"] or "{}")
                except Exception:
                    final_prov = {}
                final_prov_json = existing["provenance_json"] or "{}"

            curr_version = int(existing["version"] or 1)
            final_version = curr_version + 1

            try:
                history = json.loads(existing["history_json"] or "[]")
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

            if not history:
                history.append({
                    "version": curr_version,
                    "timestamp": existing["created_at"] or now,
                    "content": existing["content"] or existing["summary"] or "",
                    "raw_quote": existing["raw_quote"] or "",
                    "summary": existing["summary"] or "",
                    "change_type": "create",
                    "changeType": "create",
                })

            new_snapshot = {
                "version": final_version,
                "timestamp": now,
                "content": final_content,
                "raw_quote": final_raw_quote,
                "summary": final_summary,
                "change_type": change_type or "edit",
                "changeType": change_type or "edit",
            }
            history.append(new_snapshot)
            final_history_json = json.dumps(history, ensure_ascii=False)

            await self.sqlite.conn.execute(
                """
                UPDATE knowledge_points SET
                    name = ?,
                    summary = ?,
                    difficulty = ?,
                    raw_quote = ?,
                    content = ?,
                    version = ?,
                    provenance_json = ?,
                    history_json = ?,
                    updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    name,
                    final_summary,
                    diff,
                    final_raw_quote,
                    final_content,
                    final_version,
                    final_prov_json,
                    final_history_json,
                    now,
                    target_id,
                    user_id,
                ),
            )
            await self.sqlite.conn.execute(
                """
                INSERT INTO mastery (
                    user_id, kp_id, mastery, ef, interval_days, repetitions, next_review_at, updated_at
                ) VALUES (?, ?, 0.3, 2.5, 1.0, 0, ?, ?)
                ON CONFLICT(user_id, kp_id) DO NOTHING
                """,
                (user_id, target_id, now, now),
            )
            final_created_at = existing["created_at"] or now

        # Prerequisite relations dual-write
        if prereq_ids:
            for pid in prereq_ids:
                if not pid or pid == target_id:
                    continue
                # Cross-user prerequisite links are never allowed (dangling ids stay allowed
                # so offline canvas edges can be reconciled later).
                try:
                    await self._get_owned_kp(pid, user_id)
                except PermissionError:
                    log.warning(
                        "cross_user_prereq_rejected",
                        kp_id=target_id,
                        prereq_id=pid,
                        user_id=user_id,
                    )
                    continue
                await self.sqlite.conn.execute(
                    "INSERT OR IGNORE INTO kp_prereq (kp_id, prereq_id) VALUES (?, ?)",
                    (target_id, pid),
                )
                rel_id = f"rel_prereq_{pid}_{target_id}"
                await self.sqlite.conn.execute(
                    """
                    INSERT INTO kp_relations (id, src_id, rel_type, dst_id, user_id, metadata_json, created_at)
                    VALUES (?, ?, 'prerequisite', ?, ?, '{}', ?)
                    ON CONFLICT(src_id, rel_type, dst_id) DO NOTHING
                    """,
                    (rel_id, pid, target_id, user_id, now),
                )
                if self.graph_store:
                    try:
                        await self.graph_store.upsert_relation(
                            user_id=user_id,
                            src=pid,
                            rel="prerequisite",
                            dst=target_id,
                        )
                    except Exception as e:
                        log.warning("graph_store_prereq_sync_failed", error=str(e))

        # Batch relations sync
        if relations:
            for rel in relations:
                s_id = rel.get("src_id") or target_id
                d_id = rel.get("dst_id")
                r_type = rel.get("rel_type") or "prerequisite"
                r_meta = rel.get("metadata") or {}
                if s_id and d_id and s_id != d_id:
                    await self.upsert_relation(
                        src_id=s_id,
                        rel_type=r_type,
                        dst_id=d_id,
                        metadata=r_meta,
                        user_id=user_id,
                        course_id=course_id,
                    )

        await self.sqlite.conn.commit()
        return {
            "id": target_id,
            "course_id": course_id,
            "user_id": user_id,
            "name": name,
            "summary": final_summary,
            "difficulty": diff,
            "raw_quote": final_raw_quote,
            "content": final_content,
            "version": final_version,
            "provenance": final_prov,
            "history": history,
            "prereq_ids": prereq_ids or [],
            "created_at": final_created_at,
            "updated_at": now,
        }

    async def rollback_knowledge_point(
        self, kp_id: str, target_version: int, *, user_id: str = "default_user"
    ) -> dict[str, Any]:
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM knowledge_points WHERE id = ? AND user_id = ?", (kp_id, user_id)
        )
        row = await cur.fetchone()
        if not row:
            raise KeyError(f"Knowledge point '{kp_id}' not found")

        try:
            history = json.loads(row["history_json"] or "[]")
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []

        target_snapshot = next((s for s in history if int(s.get("version", -1)) == int(target_version)), None)
        if not target_snapshot:
            raise ValueError(f"Target version {target_version} does not exist in history of '{kp_id}'")

        curr_version = int(row["version"] or 1)
        next_version = curr_version + 1
        now = time.time()

        target_content = target_snapshot.get("content", "")
        target_raw_quote = target_snapshot.get("raw_quote", row["raw_quote"] or "")
        target_summary = target_snapshot.get("summary", row["summary"] or "")

        rollback_snapshot = {
            "version": next_version,
            "timestamp": now,
            "content": target_content,
            "raw_quote": target_raw_quote,
            "summary": target_summary,
            "change_type": f"rollback_to_v{target_version}",
            "changeType": "rollback",
            "target_version": target_version,
        }
        history.append(rollback_snapshot)

        await self.sqlite.conn.execute(
            """
            UPDATE knowledge_points SET
                content = ?,
                raw_quote = ?,
                summary = ?,
                version = ?,
                history_json = ?,
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                target_content,
                target_raw_quote,
                target_summary,
                next_version,
                json.dumps(history, ensure_ascii=False),
                now,
                kp_id,
                user_id,
            ),
        )
        await self.sqlite.conn.commit()

        prov = {}
        try:
            prov = json.loads(row["provenance_json"] or "{}")
        except Exception:
            pass

        return {
            "id": kp_id,
            "course_id": row["course_id"],
            "name": row["name"],
            "summary": target_summary,
            "difficulty": float(row["difficulty"]),
            "raw_quote": target_raw_quote,
            "content": target_content,
            "version": next_version,
            "history": history,
            "provenance": prov,
            "updated_at": now,
        }

    async def upsert_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        rel_id: str | None = None,
        user_id: str = "default_user",
        course_id: str | None = None,
    ) -> dict[str, Any]:
        valid_types = {"prerequisite", "causal", "similar", "branch_followup"}
        if rel_type not in valid_types:
            raise ValueError(f"Invalid rel_type '{rel_type}'. Must be one of {valid_types}")

        # Ownership / course consistency: existing endpoints must belong to the same
        # user and the same course. Dangling endpoints stay allowed so offline canvas
        # edges can be reconciled once the knowledge points land.
        owned_courses: list[str] = []
        for endpoint in (src_id, dst_id):
            owned = await self._get_owned_kp(endpoint, user_id)
            if owned is not None:
                owned_courses.append(owned["course_id"] or "default")
        if len(set(owned_courses)) > 1:
            raise ValueError(
                f"Cross-course relation rejected: {src_id} and {dst_id} belong to "
                f"different courses {sorted(set(owned_courses))}"
            )
        expected_course = course_id.strip() if course_id and course_id.strip() else None
        if (
            expected_course
            and expected_course not in {"default", "all"}
            and owned_courses
            and any(c != expected_course for c in owned_courses)
        ):
            raise ValueError(
                f"Knowledge points do not belong to course '{expected_course}'"
            )

        rid = rel_id or f"rel_{uuid.uuid4().hex[:10]}"
        now = time.time()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        await self.sqlite.conn.execute(
            """
            INSERT INTO kp_relations (id, src_id, rel_type, dst_id, user_id, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(src_id, rel_type, dst_id) DO UPDATE SET
                metadata_json = excluded.metadata_json
            """,
            (rid, src_id, rel_type, dst_id, user_id, meta_json, now),
        )

        if rel_type == "prerequisite":
            await self.sqlite.conn.execute(
                "INSERT OR IGNORE INTO kp_prereq (kp_id, prereq_id) VALUES (?, ?)",
                (dst_id, src_id),
            )

        if self.graph_store:
            try:
                await self.graph_store.upsert_relation(
                    user_id=user_id,
                    src=src_id,
                    rel=rel_type,
                    dst=dst_id,
                    **(metadata or {}),
                )
            except Exception as e:
                log.warning("graph_store_relation_sync_failed", error=str(e))

        await self.sqlite.conn.commit()
        return {
            "id": rid,
            "src_id": src_id,
            "rel_type": rel_type,
            "dst_id": dst_id,
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": now,
        }

    async def get_relations(
        self,
        kp_id: str,
        rel_type: str | None = None,
        direction: Literal["out", "in", "all"] = "all",
        *,
        user_id: str = "default_user",
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["user_id = ?"]
        params: list[Any] = [user_id]

        if direction == "out":
            clauses.append("src_id = ?")
            params.append(kp_id)
        elif direction == "in":
            clauses.append("dst_id = ?")
            params.append(kp_id)
        else:
            clauses.append("(src_id = ? OR dst_id = ?)")
            params.extend([kp_id, kp_id])

        if rel_type:
            clauses.append("rel_type = ?")
            params.append(rel_type)

        where_sql = " AND ".join(clauses)
        cur = await self.sqlite.conn.execute(
            f"SELECT * FROM kp_relations WHERE {where_sql} ORDER BY created_at DESC",
            tuple(params),
        )
        rows = await cur.fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
            except Exception:
                d["metadata"] = {}
            results.append(d)
        return results

    async def list_course_relations(
        self,
        course_id: str | None = None,
        rel_type: str | None = None,
        *,
        user_id: str = "default_user",
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["user_id = ?"]
        params: list[Any] = [user_id]

        if course_id:
            clauses.append(
                "(src_id IN (SELECT id FROM knowledge_points WHERE course_id = ? AND user_id = ?) "
                "OR dst_id IN (SELECT id FROM knowledge_points WHERE course_id = ? AND user_id = ?))"
            )
            params.extend([course_id, user_id, course_id, user_id])

        if rel_type:
            clauses.append("rel_type = ?")
            params.append(rel_type)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = await self.sqlite.conn.execute(
            f"SELECT * FROM kp_relations {where_sql} ORDER BY created_at DESC",
            tuple(params),
        )
        rows = await cur.fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
            except Exception:
                d["metadata"] = {}
            results.append(d)
        return results

    async def delete_relation(
        self,
        src_id_or_id: str,
        rel_type: str | None = None,
        dst_id: str | None = None,
        *,
        user_id: str = "default_user",
    ) -> bool:
        if rel_type is None and dst_id is None:
            cur_find = await self.sqlite.conn.execute(
                "SELECT src_id, rel_type, dst_id FROM kp_relations WHERE id = ? AND user_id = ?",
                (src_id_or_id, user_id),
            )
            row = await cur_find.fetchone()
            if not row:
                return False
            src_id, rel_type, dst_id = row["src_id"], row["rel_type"], row["dst_id"]
            cur = await self.sqlite.conn.execute(
                "DELETE FROM kp_relations WHERE id = ? AND user_id = ?",
                (src_id_or_id, user_id),
            )
        else:
            src_id = src_id_or_id
            cur = await self.sqlite.conn.execute(
                "DELETE FROM kp_relations WHERE src_id = ? AND rel_type = ? AND dst_id = ? AND user_id = ?",
                (src_id, rel_type, dst_id, user_id),
            )

        if rel_type == "prerequisite" and dst_id:
            await self.sqlite.conn.execute(
                "DELETE FROM kp_prereq WHERE kp_id = ? AND prereq_id = ?",
                (dst_id, src_id),
            )

        await self.sqlite.conn.commit()
        return cur.rowcount > 0

    async def get_revisions(
        self, kp_id: str, *, user_id: str = "default_user"
    ) -> list[dict[str, Any]]:
        cur = await self.sqlite.conn.execute(
            "SELECT history_json, version, content, raw_quote, summary, created_at, updated_at "
            "FROM knowledge_points WHERE id = ? AND user_id = ?",
            (kp_id, user_id),
        )
        row = await cur.fetchone()
        if not row:
            return []
        try:
            history = json.loads(row["history_json"] or "[]")
            if isinstance(history, list) and history:
                return history
        except Exception:
            pass
        return [{
            "version": int(row["version"] or 1),
            "timestamp": row["updated_at"] or row["created_at"] or time.time(),
            "content": row["content"] or "",
            "raw_quote": row["raw_quote"] or "",
            "summary": row["summary"] or "",
            "change_type": "create",
            "changeType": "create",
        }]

    async def list_knowledge_points(
        self, course_id: str | None = None, *, user_id: str = "default_user"
    ) -> list[dict[str, Any]]:
        if course_id:
            cur = await self.sqlite.conn.execute(
                "SELECT * FROM knowledge_points WHERE course_id = ? AND user_id = ? "
                "ORDER BY created_at ASC",
                (course_id, user_id),
            )
        else:
            cur = await self.sqlite.conn.execute(
                "SELECT * FROM knowledge_points WHERE user_id = ? ORDER BY created_at ASC",
                (user_id,),
            )
        rows = await cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["provenance"] = json.loads(d.pop("provenance_json", "{}") or "{}")
            except Exception:
                d["provenance"] = {}
            try:
                d["history"] = json.loads(d.pop("history_json", "[]") or "[]")
            except Exception:
                d["history"] = []
            out.append(d)
        return out

    async def get_knowledge_point(
        self,
        kp_id_or_name: str,
        course_id: str = "default",
        *,
        user_id: str = "default_user",
    ) -> dict[str, Any] | None:
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM knowledge_points WHERE user_id = ? "
            "AND (id = ? OR (name = ? AND course_id = ?))",
            (user_id, kp_id_or_name, kp_id_or_name, course_id),
        )
        row = await cur.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["provenance"] = json.loads(d.pop("provenance_json", "{}") or "{}")
        except Exception:
            d["provenance"] = {}
        try:
            d["history"] = json.loads(d.pop("history_json", "[]") or "[]")
        except Exception:
            d["history"] = []

        cur_p = await self.sqlite.conn.execute(
            "SELECT prereq_id FROM kp_prereq WHERE kp_id = ?", (d["id"],)
        )
        p_rows = await cur_p.fetchall()
        d["prereq_ids"] = [r["prereq_id"] for r in p_rows]
        return d

    async def delete_knowledge_point(
        self, kp_id: str, *, user_id: str = "default_user"
    ) -> bool:
        """Delete a knowledge point plus its relations, prerequisite links and mastery rows."""
        cur = await self.sqlite.conn.execute(
            "SELECT id FROM knowledge_points WHERE id = ? AND user_id = ?", (kp_id, user_id)
        )
        row = await cur.fetchone()
        if row is None:
            return False

        await self.sqlite.conn.execute(
            "DELETE FROM kp_prereq WHERE kp_id = ? OR prereq_id = ?", (kp_id, kp_id)
        )
        await self.sqlite.conn.execute(
            "DELETE FROM kp_relations WHERE src_id = ? OR dst_id = ?", (kp_id, kp_id)
        )
        await self.sqlite.conn.execute("DELETE FROM mastery WHERE kp_id = ?", (kp_id,))
        await self.sqlite.conn.execute(
            "DELETE FROM knowledge_points WHERE id = ? AND user_id = ?", (kp_id, user_id)
        )
        await self.sqlite.conn.commit()
        log.info("knowledge_point_deleted", kp_id=kp_id, user_id=user_id)
        return True

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

    async def review_knowledge_point(
        self, kp_id: str, quality: int, *, user_id: str = "default_user"
    ) -> dict[str, Any]:
        """按 SM-2 质量分推进某个知识点的复习排期（闪卡自评，不产生作答/错题记录）。"""
        kp = await self.get_knowledge_point(kp_id, user_id=user_id)
        if not kp:
            raise KeyError(f"Knowledge point '{kp_id}' not found")

        now = time.time()
        q = max(0, min(5, int(quality)))
        cur = await self.sqlite.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?",
            (user_id, kp["id"]),
        )
        row = await cur.fetchone()
        prev_mastery = float(row["mastery"]) if row else 0.3
        prev_ef = float(row["ef"]) if row else 2.5
        prev_interval = float(row["interval_days"]) if row else 0.0
        prev_reps = int(row["repetitions"]) if row else 0

        sm2 = compute_sm2(prev_ef, prev_interval, prev_reps, q, now=now)
        new_mastery = update_mastery_ema(
            prev_mastery, q / 5.0, difficulty=float(kp.get("difficulty") or 0.5)
        )

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
            (
                user_id,
                kp["id"],
                new_mastery,
                sm2.ef,
                sm2.interval_days,
                sm2.repetitions,
                sm2.next_review_at,
                now,
            ),
        )
        await self.sqlite.conn.commit()
        return {
            "kp_id": kp["id"],
            "name": kp.get("name") or kp["id"],
            "quality": q,
            "prev_mastery": prev_mastery,
            "new_mastery": new_mastery,
            "ef": sm2.ef,
            "interval_days": sm2.interval_days,
            "repetitions": sm2.repetitions,
            "next_review_at": sm2.next_review_at,
        }

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
