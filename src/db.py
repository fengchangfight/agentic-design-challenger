import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "design_challenger.db"


def get_db():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            requirement TEXT,
            challenge_level TEXT,
            status TEXT DEFAULT 'pending',
            design_doc TEXT,
            conversation TEXT,
            token_usage INTEGER DEFAULT 0,
            total_rounds INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


class SessionDB:
    def __init__(self):
        init_db()

    def create_session(self, requirement: str, challenge_level: str, title: str = None) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = get_db()
        conn.execute(
            "INSERT INTO sessions (id, title, requirement, challenge_level, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, title or "Untitled", requirement, challenge_level, "running", now, now)
        )
        conn.commit()
        conn.close()
        return session_id

    def update_session(self, session_id: str, **kwargs):
        if not kwargs:
            return
        kwargs["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [session_id]
        conn = get_db()
        conn.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
        conn.commit()
        conn.close()

    def get_session(self, session_id: str) -> Optional[Dict]:
        conn = get_db()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        conn.close()
        if row:
            d = dict(row)
            if d.get("conversation") and isinstance(d["conversation"], str):
                try:
                    d["conversation"] = json.loads(d["conversation"])
                except json.JSONDecodeError:
                    d["conversation"] = []
            return d
        return None

    def list_sessions(self) -> List[Dict]:
        conn = get_db()
        rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str):
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
