"""
SQLite-backed chat history store.

Messages are stored in chat_history.db at the project root. Each row
includes the session_id so multiple conversations can coexist.

Usage:
    from src.agents.history import ChatHistory

    h = ChatHistory()
    h.save_message("user", "hello", agent_name=None)
    h.save_message("assistant", "hi!", agent_name="default")

    messages = h.load_messages()   # returns list[dict] compatible with Ollama/OpenAI

    h.clear()  # wipe this session's history
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_DB = Path("chat_history.db")
_DEFAULT_SESSION = "default"


class ChatHistory:
    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        session_id: str = _DEFAULT_SESSION,
    ) -> None:
        self.db_path = Path(db_path)
        self.session_id = session_id
        self._init_db()

    # ── Internal ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT    NOT NULL,
                    role       TEXT    NOT NULL,
                    content    TEXT    NOT NULL,
                    agent_name TEXT,
                    timestamp  TEXT    NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON messages (session_id)"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def save_message(
        self,
        role: str,
        content: str,
        agent_name: str | None = None,
    ) -> None:
        """Persist a single message to the database."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, agent_name, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.session_id, role, content, agent_name, ts),
            )

    def load_messages(self) -> list[dict]:
        """Return all messages for this session as a list of {role, content} dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, agent_name FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "agent_name": row["agent_name"]}
            for row in rows
        ]

    def clear(self) -> None:
        """Delete all messages for this session."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (self.session_id,)
            )
