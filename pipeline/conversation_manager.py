"""Conversation history manager.

Manages multi-turn conversation sessions with session-based history,
coreference resolution, and formatted history output for LLM prompts.
Uses SQLite for persistent storage.
"""

from __future__ import annotations

import re
import time
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    query: str
    response: str
    route_used: str
    timestamp: float
    latency_ms: float = 0.0
    router_latency_ms: float = 0.0

class ConversationManager:
    """Session-based conversation history manager using SQLite."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config_path = PROJECT_ROOT / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("conversation", {})

        self.max_history_turns: int = config.get("max_history_turns", 5)
        self.session_timeout: float = config.get("session_timeout_minutes", 30) * 60

        # Setup SQLite
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        self.db_path = data_dir / "chat_history.db"
        
        self._init_db()

        logger.info(
            "ConversationManager initialized (SQLite) | max_turns={} | timeout={}min",
            self.max_history_turns,
            self.session_timeout / 60,
        )

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    username TEXT DEFAULT 'default'
                )
            ''')
            # Check if username column exists, if not add it (for migration)
            try:
                cursor.execute('ALTER TABLE sessions ADD COLUMN username TEXT DEFAULT "default"')
            except sqlite3.OperationalError:
                pass
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    route_used TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    latency_ms REAL DEFAULT 0.0,
                    router_latency_ms REAL DEFAULT 0.0
                )
            ''')
            try:
                cursor.execute('ALTER TABLE turns ADD COLUMN latency_ms REAL DEFAULT 0.0')
                cursor.execute('ALTER TABLE turns ADD COLUMN router_latency_ms REAL DEFAULT 0.0')
            except sqlite3.OperationalError:
                pass
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON turns(session_id)')
            conn.commit()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)

    def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        route: str,
        username: str = "default",
        latency_ms: float = 0.0,
        router_latency_ms: float = 0.0
    ) -> None:
        
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure session exists
            title = query[:50] + "..." if len(query) > 50 else query
            cursor.execute('''
                INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at, username)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, title, now, now, username))
            
            # Update the updated_at time
            cursor.execute('''
                UPDATE sessions SET updated_at = ? WHERE session_id = ? AND username = ?
            ''', (now, session_id, username))
            
            cursor.execute('''
                INSERT INTO turns (session_id, query, response, route_used, timestamp, latency_ms, router_latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, query, response, route, now, latency_ms, router_latency_ms))
            conn.commit()

    def truncate_session(self, session_id: str, keep_turns: int, username: str = "default") -> None:
        """Keep only the first `keep_turns` turns in the session. Delete the rest."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM sessions WHERE session_id = ? AND username = ?',
                           (session_id, username))
            if username != "default" and not cursor.fetchone():
                return
            
            cursor.execute('''
                SELECT id FROM turns 
                WHERE session_id = ? 
                ORDER BY id ASC
            ''', (session_id,))
            rows = cursor.fetchall()
            
            if len(rows) > keep_turns:
                ids_to_delete = [r[0] for r in rows[keep_turns:]]
                cursor.execute(f'''
                    DELETE FROM turns 
                    WHERE id IN ({','.join('?' * len(ids_to_delete))})
                ''', ids_to_delete)
            conn.commit()

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Order by id ASC for deterministic ordering
            cursor.execute('''
                SELECT query, response, route_used, timestamp, latency_ms, router_latency_ms 
                FROM turns 
                WHERE session_id = ? 
                ORDER BY id ASC
            ''', (session_id,))
            rows = cursor.fetchall()
            
        turns = [
            ConversationTurn(
                query=r[0], 
                response=r[1], 
                route_used=r[2], 
                timestamp=r[3],
                latency_ms=r[4],
                router_latency_ms=r[5]
            )
            for r in rows
        ]
        return turns[-self.max_history_turns:] if len(turns) > self.max_history_turns else turns

    def get_history_string(
        self,
        session_id: str,
        max_turns: int | None = None,
    ) -> str:
        turns = self.get_history(session_id)
        if not turns:
            return ""

        limit = max_turns or self.max_history_turns
        recent = turns[-limit:]

        lines: list[str] = []
        for i, turn in enumerate(recent, 1):
            lines.append(f"Lượt {i}:")
            lines.append(f"  Người dùng: {turn.query}")
            resp = turn.response
            if len(resp) > 300:
                resp = resp[:300] + "..."
            lines.append(f"  Hệ thống: {resp}")
            lines.append("")

        return "\n".join(lines).strip()

    def get_all_sessions(self, username: str = "default") -> list[dict]:
        """Fetch all unique sessions sorted by latest activity for a specific user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.session_id, s.title, s.updated_at, COUNT(t.id) as turn_count
                FROM sessions s
                LEFT JOIN turns t ON s.session_id = t.session_id
                WHERE s.username = ?
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            ''', (username,))
            rows = cursor.fetchall()
            return [
                {
                    "session_id": r[0],
                    "title": r[1],
                    "last_updated": r[2],
                    "turn_count": r[3]
                }
                for r in rows
            ]

    def update_session_title(self, session_id: str, new_title: str, username: str = "default") -> None:
        """Update the title of a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET title = ?, updated_at = ?
                WHERE session_id = ? AND username = ?
            ''', (new_title, time.time(), session_id, username))
            conn.commit()

    def resolve_coreference(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> str:
        if not history:
            return query

        resolved_query = query
        last_turn = history[-1]
        
        # Anti-loop for Clarify options
        if last_turn.route_used == "clarify" and "[Option]" in last_turn.response:
            if query.strip().startswith("Tôi muốn") or query.strip().startswith("Option"):
                resolved_query = f"{query} đối với: {last_turn.query}"

        pronoun_patterns: list[tuple[str, re.Pattern[str], str]] = [
            ("person", re.compile(r"\bông\s+ấy\b", re.IGNORECASE), "PERSON"),
            ("person", re.compile(r"\bbà\s+ấy\b", re.IGNORECASE), "PERSON"),
            ("person", re.compile(r"\bngười\s+đó\b", re.IGNORECASE), "PERSON"),
            ("org", re.compile(r"\bcơ\s+quan\s+đó\b", re.IGNORECASE), "ORGANIZATION"),
            ("legal_article", re.compile(r"\b(?:điều|khoản)\s+(?:đó|này)\b", re.IGNORECASE), "LEGAL_ARTICLE"),
            ("legal_doc", re.compile(r"\bluật\s+(?:đó|này)\b", re.IGNORECASE), "LEGAL_DOC"),
            ("legal_doc", re.compile(r"\b(?:văn\s+bản|nghị\s+định|quy\s+định|thông\s+tư|quyết\s+định)\s+(?:đó|này|trên)\b", re.IGNORECASE), "LEGAL_DOC"),
            ("general", re.compile(r"\bviệc\s+đó\b", re.IGNORECASE), "GENERAL"),
        ]
        resolved_any = False

        for _category, pattern, entity_type in pronoun_patterns:
            if not pattern.search(resolved_query):
                continue

            referent = self._find_referent(history, entity_type)
            if referent:
                resolved_query = pattern.sub(referent, resolved_query)
                resolved_any = True

        if resolved_any:
            logger.info(
                "Coreference resolved | original='{}' → resolved='{}'",
                query[:80],
                resolved_query[:80],
            )

        return resolved_query

    def _find_referent(
        self,
        history: list[ConversationTurn],
        entity_type: str,
    ) -> str | None:
        for turn in reversed(history):
            for text in [turn.query, turn.response]:
                if entity_type == "PERSON":
                    persons = re.findall(
                        r"(?:ông|bà|anh|chị)\s+[A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÊÔƠƯ]"
                        r"[a-zđàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵâăêôơư]+(?:\s+[A-ZĐÀÁẢÃẠÈÉẺẼẸ]"
                        r"[a-zđàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵâăêôơư]+)*",
                        text,
                    )
                    if persons:
                        return persons[-1]
    
                elif entity_type == "ORGANIZATION":
                    orgs = re.findall(
                        r"(?:cơ quan|tổ chức|công ty|doanh nghiệp)\s+[^\n,.]{3,40}",
                        text,
                        re.IGNORECASE,
                    )
                    if orgs:
                        return orgs[-1]
    
                elif entity_type == "LEGAL_ARTICLE":
                    terms = re.findall(
                        r"(?:Điều|Khoản)\s+\d+[a-zđ]?",
                        text,
                        re.IGNORECASE,
                    )
                    if terms:
                        return terms[-1].strip()
    
                elif entity_type == "LEGAL_DOC":
                    terms = re.findall(
                        r"(?:Luật\s+[a-zA-ZđĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]{3,30}|(?:Nghị\s+định|Thông\s+tư|Quyết\s+định)\s+(?:số\s+)?\d+(?:/\d{4}/[a-zA-ZĐđ\-]+)?)",
                        text,
                        re.IGNORECASE,
                    )
                    if terms:
                        return terms[0].strip()
    
                elif entity_type == "LEGAL_TERM":
                    terms = re.findall(
                        r"(?:Điều\s+\d+[a-zđ]?|Luật\s+[a-zA-ZđĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]{3,30}|(?:Nghị\s+định|Thông\s+tư|Quyết\s+định)\s+(?:số\s+)?\d+(?:/\d{4}/[a-zA-ZĐđ\-]+)?)",
                        text,
                        re.IGNORECASE,
                    )
                    if terms:
                        return terms[0].strip()
    
                elif entity_type == "GENERAL":
                    sentences = text.split(".")
                    if sentences:
                        return sentences[0].strip()[:100]

        return None

    def _cleanup_expired(self, session_id: str) -> None:
        """Remove expired turns from a session."""
        cutoff_time = time.time() - self.session_timeout
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM turns WHERE session_id = ? AND timestamp < ?', (session_id, cutoff_time))
            conn.commit()

    def clear_session(self, session_id: str, username: str = "default") -> None:
        """Clear all history for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # First check if the session belongs to the user
            cursor.execute('SELECT 1 FROM sessions WHERE session_id = ? AND username = ?', (session_id, username))
            if cursor.fetchone():
                cursor.execute('DELETE FROM turns WHERE session_id = ?', (session_id,))
                cursor.execute('DELETE FROM sessions WHERE session_id = ? AND username = ?', (session_id, username))
                conn.commit()
                logger.info("Session {} cleared", session_id)

    def register_user(self, username: str, password_hash: str) -> bool:
        """Register a new user. Returns True if successful, False if username exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (username, password_hash, created_at)
                    VALUES (?, ?, ?)
                ''', (username, password_hash, time.time()))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def verify_user(self, username: str) -> str | None:
        """Get password hash for a user. Returns None if user not found."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return None
