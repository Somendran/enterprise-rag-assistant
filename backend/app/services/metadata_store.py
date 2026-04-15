"""SQLite-backed metadata store for documents, feedback, and admin summaries."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.config import settings


def _db_path() -> Path:
    path = Path(settings.metadata_db_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / "backend" / path
    return path.resolve()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            file_hash TEXT PRIMARY KEY,
            document_id TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            embedding_model TEXT NOT NULL DEFAULT '',
            indexed_at INTEGER NOT NULL DEFAULT 0,
            parsing_method TEXT NOT NULL DEFAULT 'unknown',
            upload_path TEXT NOT NULL DEFAULT '',
            upload_status TEXT NOT NULL DEFAULT 'indexed',
            vision_calls_used INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            comment TEXT NOT NULL DEFAULT '',
            confidence_score REAL,
            sources_json TEXT NOT NULL DEFAULT '[]',
            diagnostics_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            status TEXT NOT NULL,
            stage TEXT NOT NULL DEFAULT '',
            message TEXT NOT NULL DEFAULT '',
            total_files INTEGER NOT NULL DEFAULT 0,
            processed_files INTEGER NOT NULL DEFAULT 0,
            total_chunks_indexed INTEGER NOT NULL DEFAULT 0,
            results_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '[]',
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            confidence_score REAL,
            confidence_level TEXT,
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            status TEXT NOT NULL,
            total INTEGER NOT NULL DEFAULT 0,
            passed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            results_json TEXT NOT NULL DEFAULT '[]',
            message TEXT NOT NULL DEFAULT ''
        )
        """
    )


def upsert_document(
    *,
    file_hash: str,
    filename: str,
    chunk_count: int,
    document_id: str,
    embedding_model: str,
    indexed_at: int,
    parsing_method: str,
    upload_path: str,
    upload_status: str,
    vision_calls_used: int,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                file_hash, filename, chunk_count, document_id, embedding_model,
                indexed_at, parsing_method, upload_path, upload_status, vision_calls_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                filename = excluded.filename,
                chunk_count = excluded.chunk_count,
                document_id = excluded.document_id,
                embedding_model = excluded.embedding_model,
                indexed_at = excluded.indexed_at,
                parsing_method = excluded.parsing_method,
                upload_path = excluded.upload_path,
                upload_status = excluded.upload_status,
                vision_calls_used = excluded.vision_calls_used
            """,
            (
                file_hash,
                filename,
                int(chunk_count),
                document_id,
                embedding_model,
                int(indexed_at),
                parsing_method,
                upload_path,
                upload_status,
                int(vision_calls_used or 0),
            ),
        )


def get_document(file_hash: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
    return dict(row) if row else None


def document_exists(file_hash: str) -> bool:
    return get_document(file_hash) is not None


def list_documents() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY lower(filename), indexed_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(file_hash: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM documents WHERE file_hash = ?", (file_hash,))
    return dict(row)


def clear_documents() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents")


def record_feedback(
    *,
    question: str,
    answer: str,
    rating: str,
    reason: str = "",
    comment: str = "",
    confidence_score: float | None = None,
    sources: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = int(time.time())
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback (
                created_at, question, answer, rating, reason, comment,
                confidence_score, sources_json, diagnostics_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                question,
                answer,
                rating,
                reason,
                comment,
                confidence_score,
                json.dumps(sources or []),
                json.dumps(diagnostics or {}),
            ),
        )
        feedback_id = int(cursor.lastrowid)
    return {
        "id": feedback_id,
        "created_at": created_at,
        "rating": rating,
        "reason": reason,
        "comment": comment,
    }


def list_feedback(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, question, rating, reason, comment, confidence_score
            FROM feedback
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 100)),),
        ).fetchall()
    return [dict(row) for row in rows]


def admin_summary() -> dict[str, Any]:
    with _connect() as conn:
        doc_row = conn.execute(
            "SELECT COUNT(*) AS count, COALESCE(SUM(chunk_count), 0) AS chunks FROM documents"
        ).fetchone()
        feedback_row = conn.execute(
            "SELECT COUNT(*) AS count FROM feedback"
        ).fetchone()
        recent_feedback = conn.execute(
            """
            SELECT id, created_at, question, rating, reason, confidence_score
            FROM feedback
            ORDER BY created_at DESC
            LIMIT 5
            """
        ).fetchall()
        sessions_row = conn.execute("SELECT COUNT(*) AS count FROM chat_sessions").fetchone()
        eval_row = conn.execute("SELECT COUNT(*) AS count FROM eval_runs").fetchone()

    return {
        "document_count": int(doc_row["count"] if doc_row else 0),
        "chunk_count": int(doc_row["chunks"] if doc_row else 0),
        "feedback_count": int(feedback_row["count"] if feedback_row else 0),
        "chat_session_count": int(sessions_row["count"] if sessions_row else 0),
        "eval_run_count": int(eval_row["count"] if eval_row else 0),
        "recent_feedback": [dict(row) for row in recent_feedback],
        "metadata_db_path": str(_db_path()),
    }


def create_ingestion_job(job_id: str, total_files: int) -> dict[str, Any]:
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_jobs (
                id, created_at, updated_at, status, stage, message, total_files
            )
            VALUES (?, ?, ?, 'queued', 'queued', 'Queued for ingestion.', ?)
            """,
            (job_id, now, now, int(total_files)),
        )
    return get_ingestion_job(job_id) or {}


def update_ingestion_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    message: str | None = None,
    processed_files: int | None = None,
    total_chunks_indexed: int | None = None,
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    current = get_ingestion_job(job_id) or {}
    payload = {
        "status": status if status is not None else current.get("status", "queued"),
        "stage": stage if stage is not None else current.get("stage", ""),
        "message": message if message is not None else current.get("message", ""),
        "processed_files": processed_files if processed_files is not None else current.get("processed_files", 0),
        "total_chunks_indexed": (
            total_chunks_indexed
            if total_chunks_indexed is not None
            else current.get("total_chunks_indexed", 0)
        ),
        "results_json": json.dumps(results if results is not None else current.get("results", [])),
    }
    with _connect() as conn:
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET updated_at = ?, status = ?, stage = ?, message = ?,
                processed_files = ?, total_chunks_indexed = ?, results_json = ?
            WHERE id = ?
            """,
            (
                int(time.time()),
                payload["status"],
                payload["stage"],
                payload["message"],
                int(payload["processed_files"] or 0),
                int(payload["total_chunks_indexed"] or 0),
                payload["results_json"],
                job_id,
            ),
        )
    return get_ingestion_job(job_id) or {}


def get_ingestion_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["results"] = json.loads(item.pop("results_json") or "[]")
    return item


def create_chat_session(session_id: str, title: str) -> dict[str, Any]:
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
    return get_chat_session(session_id) or {}


def get_chat_session(session_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_chat_sessions(limit: int = 30) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(int(limit), 100)),),
        ).fetchall()
    return [dict(row) for row in rows]


def add_chat_message(
    *,
    message_id: str,
    session_id: str,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    confidence_score: float | None = None,
    confidence_level: str | None = None,
) -> dict[str, Any]:
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (
                id, session_id, role, content, created_at, sources_json,
                diagnostics_json, confidence_score, confidence_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                role,
                content,
                now,
                json.dumps(sources or []),
                json.dumps(diagnostics or {}),
                confidence_score,
                confidence_level,
            ),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ?, title = CASE WHEN title = 'New chat' THEN ? ELSE title END WHERE id = ?",
            (now, content[:80] if role == "user" else "New chat", session_id),
        )
    return {"id": message_id, "session_id": session_id, "role": role, "content": content, "created_at": now}


def list_chat_messages(session_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        ).fetchall()
    messages: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json") or "[]")
        item["diagnostics"] = json.loads(item.pop("diagnostics_json") or "{}")
        messages.append(item)
    return messages


def create_eval_run(run_id: str, total: int) -> dict[str, Any]:
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO eval_runs (id, created_at, status, total, passed, failed, results_json, message)
            VALUES (?, ?, 'running', ?, 0, 0, '[]', 'Eval run started.')
            """,
            (run_id, now, int(total)),
        )
    return get_eval_run(run_id) or {}


def update_eval_run(
    run_id: str,
    *,
    status: str,
    results: list[dict[str, Any]],
    message: str = "",
) -> dict[str, Any]:
    passed = sum(1 for item in results if item.get("passed"))
    failed = len(results) - passed
    with _connect() as conn:
        conn.execute(
            """
            UPDATE eval_runs
            SET status = ?, total = ?, passed = ?, failed = ?, results_json = ?, message = ?
            WHERE id = ?
            """,
            (status, len(results), passed, failed, json.dumps(results), message, run_id),
        )
    return get_eval_run(run_id) or {}


def get_eval_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["results"] = json.loads(item.pop("results_json") or "[]")
    return item


def list_eval_runs(limit: int = 10) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?",
            (max(1, min(int(limit), 50)),),
        ).fetchall()
    runs: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["results"] = json.loads(item.pop("results_json") or "[]")
        runs.append(item)
    return runs
