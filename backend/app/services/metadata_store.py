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

    return {
        "document_count": int(doc_row["count"] if doc_row else 0),
        "chunk_count": int(doc_row["chunks"] if doc_row else 0),
        "feedback_count": int(feedback_row["count"] if feedback_row else 0),
        "recent_feedback": [dict(row) for row in recent_feedback],
        "metadata_db_path": str(_db_path()),
    }
