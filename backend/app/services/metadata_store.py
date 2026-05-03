"""SQLite-backed metadata store for documents, feedback, and admin summaries."""

from __future__ import annotations

import json
import hashlib
import hmac
import secrets
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
            vision_calls_used INTEGER NOT NULL DEFAULT 0,
            owner_user_id TEXT NOT NULL DEFAULT '',
            visibility TEXT NOT NULL DEFAULT 'shared',
            allowed_roles_json TEXT NOT NULL DEFAULT '[]',
            ocr_applied INTEGER NOT NULL DEFAULT 0,
            text_coverage_ratio REAL NOT NULL DEFAULT 0.0,
            low_text_pages INTEGER NOT NULL DEFAULT 0,
            ingestion_warnings_json TEXT NOT NULL DEFAULT '[]',
            is_demo INTEGER NOT NULL DEFAULT 0,
            demo_session_id TEXT NOT NULL DEFAULT '',
            expires_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    _ensure_column(conn, "documents", "owner_user_id", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "documents", "visibility", "TEXT NOT NULL DEFAULT 'shared'")
    _ensure_column(conn, "documents", "allowed_roles_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "documents", "ocr_applied", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "documents", "text_coverage_ratio", "REAL NOT NULL DEFAULT 0.0")
    _ensure_column(conn, "documents", "low_text_pages", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "documents", "ingestion_warnings_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "documents", "is_demo", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "documents", "demo_session_id", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "documents", "expires_at", "INTEGER NOT NULL DEFAULT 0")
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
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            user_id TEXT NOT NULL DEFAULT ''
        )
        """
    )
    _ensure_column(conn, "feedback", "user_id", "TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_diagnostics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            user_id TEXT NOT NULL DEFAULT '',
            question TEXT NOT NULL,
            answer_preview TEXT NOT NULL DEFAULT '',
            confidence_score REAL,
            confidence_level TEXT NOT NULL DEFAULT '',
            query_type TEXT NOT NULL DEFAULT '',
            sources_json TEXT NOT NULL DEFAULT '[]',
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            latency_ms REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_query_diagnostics_created_at ON query_diagnostics(created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_query_diagnostics_query_type ON query_diagnostics(query_type)")
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
            results_json TEXT NOT NULL DEFAULT '[]',
            created_by_user_id TEXT NOT NULL DEFAULT ''
        )
        """
    )
    _ensure_column(conn, "ingestion_jobs", "created_by_user_id", "TEXT NOT NULL DEFAULT ''")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            user_id TEXT NOT NULL DEFAULT ''
        )
        """
    )
    _ensure_column(conn, "chat_sessions", "user_id", "TEXT NOT NULL DEFAULT ''")
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'user',
            password_hash TEXT NOT NULL,
            disabled INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(lower(email))")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            revoked_at INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            actor_user_id TEXT NOT NULL DEFAULT '',
            actor_email TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL DEFAULT '',
            resource_id TEXT NOT NULL DEFAULT '',
            detail_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events(created_at DESC)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rate_limits (
            key TEXT NOT NULL,
            action TEXT NOT NULL,
            window_start INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (key, action)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_start)")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _json_list(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _decode_document_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["allowed_roles"] = _json_list(str(item.pop("allowed_roles_json", "[]")))
    item["ingestion_warnings"] = _json_list(str(item.pop("ingestion_warnings_json", "[]")))
    item["ocr_applied"] = bool(item.get("ocr_applied", 0))
    item["text_coverage_ratio"] = float(item.get("text_coverage_ratio", 0.0) or 0.0)
    item["low_text_pages"] = int(item.get("low_text_pages", 0) or 0)
    return item


def can_user_access_document(user: dict[str, Any], document: dict[str, Any], *, write: bool = False) -> bool:
    if not user:
        return False
    if user.get("role") == "admin" or user.get("is_system_admin"):
        return True
    if write:
        return str(document.get("owner_user_id", "")) == str(user.get("id", ""))
    visibility = str(document.get("visibility") or "shared")
    if visibility == "shared":
        return True
    if visibility == "private":
        return str(document.get("owner_user_id", "")) == str(user.get("id", ""))
    if visibility == "role":
        allowed_roles = document.get("allowed_roles") or _json_list(str(document.get("allowed_roles_json", "[]")))
        return str(user.get("role", "")).lower() in {str(role).lower() for role in allowed_roles}
    return False


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
    owner_user_id: str = "",
    visibility: str = "shared",
    allowed_roles: list[str] | None = None,
    ocr_applied: bool = False,
    text_coverage_ratio: float = 0.0,
    low_text_pages: int = 0,
    ingestion_warnings: list[str] | None = None,
    is_demo: bool = False,
    demo_session_id: str = "",
    expires_at: int = 0,
) -> None:
    normalized_roles = sorted({str(role).strip().lower() for role in (allowed_roles or []) if str(role).strip()})
    normalized_visibility = visibility if visibility in {"private", "shared", "role"} else "shared"
    normalized_warnings = [str(item).strip() for item in (ingestion_warnings or []) if str(item).strip()]
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                file_hash, filename, chunk_count, document_id, embedding_model,
                indexed_at, parsing_method, upload_path, upload_status, vision_calls_used,
                owner_user_id, visibility, allowed_roles_json, ocr_applied, text_coverage_ratio,
                low_text_pages, ingestion_warnings_json, is_demo, demo_session_id, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                filename = excluded.filename,
                chunk_count = excluded.chunk_count,
                document_id = excluded.document_id,
                embedding_model = excluded.embedding_model,
                indexed_at = excluded.indexed_at,
                parsing_method = excluded.parsing_method,
                upload_path = excluded.upload_path,
                upload_status = excluded.upload_status,
                vision_calls_used = excluded.vision_calls_used,
                owner_user_id = CASE
                    WHEN excluded.owner_user_id != '' THEN excluded.owner_user_id
                    ELSE documents.owner_user_id
                END,
                visibility = excluded.visibility,
                allowed_roles_json = excluded.allowed_roles_json,
                ocr_applied = excluded.ocr_applied,
                text_coverage_ratio = excluded.text_coverage_ratio,
                low_text_pages = excluded.low_text_pages,
                ingestion_warnings_json = excluded.ingestion_warnings_json,
                is_demo = excluded.is_demo,
                demo_session_id = excluded.demo_session_id,
                expires_at = excluded.expires_at
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
                owner_user_id,
                normalized_visibility,
                json.dumps(normalized_roles),
                1 if ocr_applied else 0,
                float(text_coverage_ratio or 0.0),
                int(low_text_pages or 0),
                json.dumps(normalized_warnings),
                1 if is_demo else 0,
                demo_session_id,
                int(expires_at or 0),
            ),
        )


def get_document(file_hash: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
    return _decode_document_row(row) if row else None


def document_exists(file_hash: str) -> bool:
    return get_document(file_hash) is not None


def list_documents() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY lower(filename), indexed_at DESC"
        ).fetchall()
    return [_decode_document_row(row) for row in rows]


def list_documents_for_user(user: dict[str, Any]) -> list[dict[str, Any]]:
    documents = list_documents()
    return [doc for doc in documents if can_user_access_document(user, doc)]


def count_documents_for_owner(owner_user_id: str, *, include_expired: bool = False) -> int:
    now = int(time.time())
    with _connect() as conn:
        if include_expired:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM documents WHERE owner_user_id = ?",
                (owner_user_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM documents
                WHERE owner_user_id = ?
                  AND (expires_at = 0 OR expires_at > ?)
                """,
                (owner_user_id, now),
            ).fetchone()
    return int(row["count"] if row else 0)


def list_expired_demo_documents(now: int | None = None) -> list[dict[str, Any]]:
    cutoff = int(now if now is not None else time.time())
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE is_demo = 1
              AND expires_at > 0
              AND expires_at <= ?
            """,
            (cutoff,),
        ).fetchall()
    return [_decode_document_row(row) for row in rows]


def delete_document(file_hash: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM documents WHERE file_hash = ?", (file_hash,))
    return _decode_document_row(row)


def clear_documents() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents")


def update_document_permissions(
    file_hash: str,
    *,
    visibility: str,
    allowed_roles: list[str] | None = None,
) -> dict[str, Any] | None:
    normalized_visibility = visibility if visibility in {"private", "shared", "role"} else "shared"
    normalized_roles = sorted({str(role).strip().lower() for role in (allowed_roles or []) if str(role).strip()})
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        if row is None:
            return None
        conn.execute(
            """
            UPDATE documents
            SET visibility = ?, allowed_roles_json = ?
            WHERE file_hash = ?
            """,
            (normalized_visibility, json.dumps(normalized_roles), file_hash),
        )
    return get_document(file_hash)


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 210_000)
    return f"pbkdf2_sha256$210000${salt}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt, expected_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations_raw),
        )
        return hmac.compare_digest(digest.hex(), expected_hex)
    except Exception:
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def users_exist() -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
    return bool(row and int(row["count"]) > 0)


def create_user(
    *,
    user_id: str,
    email: str,
    password: str,
    display_name: str = "",
    role: str = "user",
) -> dict[str, Any]:
    now = int(time.time())
    normalized_email = email.strip().lower()
    normalized_role = role if role in {"admin", "user"} else "user"
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (
                id, email, display_name, role, password_hash, disabled, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                user_id,
                normalized_email,
                display_name.strip() or normalized_email,
                normalized_role,
                _hash_password(password),
                now,
                now,
            ),
        )
    return get_user(user_id) or {}


def get_user(user_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, email, display_name, role, disabled, created_at, updated_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    normalized_email = email.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, email, display_name, role, disabled, created_at, updated_at
            FROM users
            WHERE lower(email) = ?
            """,
            (normalized_email,),
        ).fetchone()
    return dict(row) if row else None


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    normalized_email = email.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE lower(email) = ?",
            (normalized_email,),
        ).fetchone()
    if not row:
        return None
    payload = dict(row)
    if int(payload.get("disabled", 0) or 0):
        return None
    if not _verify_password(password, str(payload.get("password_hash", ""))):
        return None
    payload.pop("password_hash", None)
    return payload


def create_auth_token(user_id: str, ttl_seconds: int) -> dict[str, Any]:
    now = int(time.time())
    token = secrets.token_urlsafe(32)
    expires_at = now + max(60, int(ttl_seconds))
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at, revoked_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (_hash_token(token), user_id, now, expires_at),
        )
    return {"access_token": token, "token_type": "bearer", "expires_at": expires_at}


def get_user_for_token(token: str) -> dict[str, Any] | None:
    token_hash = _hash_token(token)
    now = int(time.time())
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, u.display_name, u.role, u.disabled, u.created_at, u.updated_at
            FROM auth_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token_hash = ?
              AND t.revoked_at IS NULL
              AND t.expires_at > ?
              AND u.disabled = 0
            """,
            (token_hash, now),
        ).fetchone()
    return dict(row) if row else None


def revoke_auth_token(token: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE auth_tokens SET revoked_at = ? WHERE token_hash = ?",
            (int(time.time()), _hash_token(token)),
        )


def list_users() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, email, display_name, role, disabled, created_at, updated_at
            FROM users
            ORDER BY lower(email)
            """
        ).fetchall()
    return [dict(row) for row in rows]


def allowed_file_hashes_for_user(user: dict[str, Any]) -> list[str]:
    return [str(doc["file_hash"]) for doc in list_documents_for_user(user)]


def check_rate_limit(
    *,
    key: str,
    action: str,
    limit: int,
    window_seconds: int = 3600,
) -> dict[str, Any]:
    normalized_key = str(key or "").strip()
    normalized_action = str(action or "").strip()
    resolved_limit = max(1, int(limit))
    resolved_window = max(60, int(window_seconds))
    now = int(time.time())
    window_start = now - (now % resolved_window)

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT window_start, count
            FROM rate_limits
            WHERE key = ? AND action = ?
            """,
            (normalized_key, normalized_action),
        ).fetchone()

        current_count = 0
        if row and int(row["window_start"]) == window_start:
            current_count = int(row["count"])

        allowed = current_count < resolved_limit
        next_count = current_count + 1 if allowed else current_count
        conn.execute(
            """
            INSERT INTO rate_limits (key, action, window_start, count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key, action) DO UPDATE SET
                window_start = excluded.window_start,
                count = excluded.count
            """,
            (normalized_key, normalized_action, window_start, next_count),
        )
        conn.execute(
            "DELETE FROM rate_limits WHERE window_start < ?",
            (window_start - resolved_window,),
        )

    return {
        "allowed": allowed,
        "limit": resolved_limit,
        "remaining": max(0, resolved_limit - next_count),
        "reset_at": window_start + resolved_window,
    }


def record_audit_event(
    *,
    actor_user_id: str = "",
    actor_email: str = "",
    action: str,
    resource_type: str = "",
    resource_id: str = "",
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = int(time.time())
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO audit_events (
                created_at, actor_user_id, actor_email, action,
                resource_type, resource_id, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                actor_user_id,
                actor_email,
                action,
                resource_type,
                resource_id,
                json.dumps(detail or {}),
            ),
        )
        event_id = int(cursor.lastrowid)
    return {
        "id": event_id,
        "created_at": created_at,
        "actor_user_id": actor_user_id,
        "actor_email": actor_email,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "detail": detail or {},
    }


def list_audit_events(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM audit_events
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["detail"] = json.loads(item.pop("detail_json") or "{}")
        events.append(item)
    return events


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
    user_id: str = "",
) -> dict[str, Any]:
    created_at = int(time.time())
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback (
                created_at, question, answer, rating, reason, comment,
                confidence_score, sources_json, diagnostics_json, user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                user_id,
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


def record_query_diagnostic(
    *,
    user_id: str = "",
    question: str,
    answer: str,
    confidence_score: float | None = None,
    confidence_level: str = "",
    sources: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    created_at = int(time.time())
    diagnostics_payload = diagnostics or {}
    query_type = str(diagnostics_payload.get("query_type", "") or "")
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO query_diagnostics (
                created_at, user_id, question, answer_preview, confidence_score,
                confidence_level, query_type, sources_json, diagnostics_json, latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                user_id,
                question,
                answer[:500],
                confidence_score,
                confidence_level,
                query_type,
                json.dumps(sources or []),
                json.dumps(diagnostics_payload),
                float(latency_ms),
            ),
        )
        row_id = int(cursor.lastrowid)
    return {"id": row_id, "created_at": created_at, "query_type": query_type}


def list_query_diagnostics(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, user_id, question, answer_preview, confidence_score,
                   confidence_level, query_type, sources_json, diagnostics_json, latency_ms
            FROM query_diagnostics
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json") or "[]")
        item["diagnostics"] = json.loads(item.pop("diagnostics_json") or "{}")
        items.append(item)
    return items


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
        query_diag_row = conn.execute(
            "SELECT COUNT(*) AS count FROM query_diagnostics"
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
        users_row = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        audit_row = conn.execute("SELECT COUNT(*) AS count FROM audit_events").fetchone()

    return {
        "document_count": int(doc_row["count"] if doc_row else 0),
        "chunk_count": int(doc_row["chunks"] if doc_row else 0),
        "feedback_count": int(feedback_row["count"] if feedback_row else 0),
        "query_diagnostic_count": int(query_diag_row["count"] if query_diag_row else 0),
        "chat_session_count": int(sessions_row["count"] if sessions_row else 0),
        "eval_run_count": int(eval_row["count"] if eval_row else 0),
        "user_count": int(users_row["count"] if users_row else 0),
        "audit_event_count": int(audit_row["count"] if audit_row else 0),
        "recent_feedback": [dict(row) for row in recent_feedback],
        "metadata_db_path": str(_db_path()),
    }


def create_ingestion_job(job_id: str, total_files: int, created_by_user_id: str = "") -> dict[str, Any]:
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_jobs (
                id, created_at, updated_at, status, stage, message, total_files, created_by_user_id
            )
            VALUES (?, ?, ?, 'queued', 'queued', 'Queued for ingestion.', ?, ?)
            """,
            (job_id, now, now, int(total_files), created_by_user_id),
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


def create_chat_session(session_id: str, title: str, user_id: str = "") -> dict[str, Any]:
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at, user_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, now, now, user_id),
        )
    return get_chat_session(session_id) or {}


def get_chat_session(session_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_chat_sessions(limit: int = 30, user_id: str | None = None, include_all: bool = False) -> list[dict[str, Any]]:
    with _connect() as conn:
        if include_all or user_id is None:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, max(1, min(int(limit), 100))),
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
