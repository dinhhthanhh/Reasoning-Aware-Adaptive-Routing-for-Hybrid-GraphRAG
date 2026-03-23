"""Conversation history manager.

Manages multi-turn conversation sessions with session-based history,
coreference resolution, and formatted history output for LLM prompts.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


@dataclass
class ConversationTurn:
    """A single turn in a conversation.

    Attributes:
        query: User query text.
        response: System response text.
        route_used: Which route was used (vector/graph/clarify).
        timestamp: Unix timestamp of the turn.
    """

    query: str
    response: str
    route_used: str
    timestamp: float


class ConversationManager:
    """Session-based conversation history manager.

    Maintains per-session conversation history with configurable
    max turns and session timeout. Provides history formatting
    for LLM prompts and basic coreference resolution.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize conversation manager.

        Args:
            config: Conversation config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("conversation", {})

        self.max_history_turns: int = config.get("max_history_turns", 5)
        self.session_timeout: float = config.get("session_timeout_minutes", 30) * 60

        self._sessions: dict[str, list[ConversationTurn]] = defaultdict(list)

        logger.info(
            "ConversationManager initialized | max_turns={} | timeout={}min",
            self.max_history_turns,
            self.session_timeout / 60,
        )

    def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        route: str,
    ) -> None:
        """Add a conversation turn to a session.

        Args:
            session_id: Unique session identifier.
            query: User query text.
            response: System response text.
            route: Route used for this query.
        """
        self._cleanup_expired(session_id)

        turn = ConversationTurn(
            query=query,
            response=response,
            route_used=route,
            timestamp=time.time(),
        )
        self._sessions[session_id].append(turn)

        # Trim to max turns
        if len(self._sessions[session_id]) > self.max_history_turns:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history_turns:]

        logger.debug(
            "Added turn to session {} | total_turns={}",
            session_id,
            len(self._sessions[session_id]),
        )

    def get_history_string(
        self,
        session_id: str,
        max_turns: int | None = None,
    ) -> str:
        """Format conversation history as a string for LLM prompts.

        Args:
            session_id: Session to retrieve history for.
            max_turns: Override max turns to include.

        Returns:
            Formatted history string, or empty string if no history.
        """
        self._cleanup_expired(session_id)

        turns = self._sessions.get(session_id, [])
        if not turns:
            return ""

        limit = max_turns or self.max_history_turns
        recent = turns[-limit:]

        lines: list[str] = []
        for i, turn in enumerate(recent, 1):
            lines.append(f"Lượt {i}:")
            lines.append(f"  Người dùng: {turn.query}")
            # Truncate long responses
            resp = turn.response
            if len(resp) > 300:
                resp = resp[:300] + "..."
            lines.append(f"  Hệ thống: {resp}")
            lines.append("")

        return "\n".join(lines).strip()

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        """Get raw conversation history for a session.

        Args:
            session_id: Session to retrieve.

        Returns:
            List of ConversationTurn objects.
        """
        self._cleanup_expired(session_id)
        return list(self._sessions.get(session_id, []))

    def resolve_coreference(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> str:
        """Attempt to resolve pronouns in the query using conversation history.

        Replaces ambiguous pronouns (e.g., 'ông ấy', 'luật đó') with
        specific entities from the previous turns when possible.

        Args:
            query: Current user query.
            history: Previous conversation turns.

        Returns:
            Query enriched with resolved context, or original query if
            no resolution was possible.
        """
        if not history:
            return query

        # Patterns for pronouns that might need resolution
        pronoun_patterns: list[tuple[str, re.Pattern[str], str]] = [
            ("person", re.compile(r"\bông\s+ấy\b", re.IGNORECASE), "PERSON"),
            ("person", re.compile(r"\bbà\s+ấy\b", re.IGNORECASE), "PERSON"),
            ("person", re.compile(r"\bngười\s+đó\b", re.IGNORECASE), "PERSON"),
            ("org", re.compile(r"\bcơ\s+quan\s+đó\b", re.IGNORECASE), "ORGANIZATION"),
            ("legal", re.compile(r"\bđiều\s+đó\b", re.IGNORECASE), "LEGAL_TERM"),
            ("legal", re.compile(r"\bluật\s+đó\b", re.IGNORECASE), "LEGAL_TERM"),
            ("general", re.compile(r"\bviệc\s+đó\b", re.IGNORECASE), "GENERAL"),
        ]

        resolved_query = query
        resolved_any = False

        for _category, pattern, entity_type in pronoun_patterns:
            if not pattern.search(resolved_query):
                continue

            # Search backwards through history for the referenced entity
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
        """Find the most recent entity of a given type in history.

        Args:
            history: Conversation turns to search.
            entity_type: Type of entity to find.

        Returns:
            Entity text if found, None otherwise.
        """
        # Search recent turns first (reversed)
        for turn in reversed(history):
            text = f"{turn.query} {turn.response}"

            if entity_type == "PERSON":
                # Look for Vietnamese person name patterns
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

            elif entity_type == "LEGAL_TERM":
                terms = re.findall(
                    r"(?:Điều\s+\d+[a-zđ]?|Luật\s+[^\n,.]{3,40}|Nghị\s+định\s+[^\n,.]{3,40})",
                    text,
                    re.IGNORECASE,
                )
                if terms:
                    return terms[-1]

            elif entity_type == "GENERAL":
                # Return the main topic from the last response
                sentences = turn.response.split(".")
                if sentences:
                    return sentences[0].strip()[:100]

        return None

    def _cleanup_expired(self, session_id: str) -> None:
        """Remove expired turns from a session.

        Args:
            session_id: Session to clean up.
        """
        if session_id not in self._sessions:
            return

        now = time.time()
        self._sessions[session_id] = [
            turn for turn in self._sessions[session_id]
            if (now - turn.timestamp) < self.session_timeout
        ]

    def clear_session(self, session_id: str) -> None:
        """Clear all history for a session.

        Args:
            session_id: Session to clear.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session {} cleared", session_id)
