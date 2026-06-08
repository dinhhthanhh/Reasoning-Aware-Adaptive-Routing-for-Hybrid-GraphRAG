"""Ambiguity detection for Vietnamese legal queries.

Detects ambiguous queries using hybrid rule-based + lightweight ML
features. Identifies pronoun issues, vague references, missing entities,
and entity conflicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from router.history_resolver import HistoryResolutionResult, resolve_history_referents


@dataclass
class AmbiguityReport:
    """Report from ambiguity detection analysis.

    Attributes:
        score: Overall ambiguity score (0.0 to 1.0).
        is_ambiguous: Whether the query exceeds the ambiguity threshold.
        ambiguity_types: List of detected ambiguity types.
        missing_entity_type: Type of entity that is missing, if any.
        clarification_question: Suggested follow-up question for the user.
    """

    score: float = 0.0
    is_ambiguous: bool = False
    ambiguity_types: list[str] = field(default_factory=list)
    missing_entity_type: str | None = None
    detected_topics: list[str] = field(default_factory=list)
    clarification_question: str | None = None
    missing_entity: bool = False
    multi_interpretation: bool = False
    incomplete_context: bool = False
    pronoun_reference: bool = False
    semantic_ambiguity_score: float = 0.0
    contextual_reference_score: float = 0.0
    query_has_contextual_reference: bool = False
    history_resolution_status: str = "not_needed"
    history_resolution_confidence: float = 0.0
    resolved_referent: str | None = None
    candidate_referents: list[dict[str, object]] = field(default_factory=list)


class AmbiguityDetector:
    """Detect ambiguous queries in Vietnamese legal text.

    Uses a hybrid approach combining rule-based pattern matching
    with lightweight statistical features to identify queries that
    cannot be confidently answered without clarification.
    """

    # Pronoun patterns that indicate ambiguity
    PRONOUN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("ông ấy", re.compile(r"\bông\s+ấy\b", re.IGNORECASE)),
        ("bà ấy", re.compile(r"\bbà\s+ấy\b", re.IGNORECASE)),
        ("họ", re.compile(r"\bhọ\b(?!\s+(?:tên|và))", re.IGNORECASE)),
        ("nó", re.compile(r"\bnó\b", re.IGNORECASE)),
        ("người đó", re.compile(r"\bngười\s+đó\b", re.IGNORECASE)),
        ("công ty đó", re.compile(r"\bcông\s+ty\s+(?:đó|này)\b", re.IGNORECASE)),
        ("doanh nghiệp đó", re.compile(r"\bdoanh\s+nghiệp\s+(?:đó|này)\b", re.IGNORECASE)),
        ("tổ chức đó", re.compile(r"\btổ\s+chức\s+(?:đó|này)\b", re.IGNORECASE)),
        ("cơ quan đó", re.compile(r"\bcơ\s+quan\s+đó\b", re.IGNORECASE)),
        ("điều đó", re.compile(r"\bđiều\s+đó\b", re.IGNORECASE)),
        ("luật đó", re.compile(r"\bluật\s+đó\b", re.IGNORECASE)),
        ("bên đó", re.compile(r"\bbên\s+đó\b", re.IGNORECASE)),
        ("việc đó", re.compile(r"\bviệc\s+đó\b", re.IGNORECASE)),
        ("trường hợp đó", re.compile(r"\btrường\s+hợp\s+(?:đó|này)\b", re.IGNORECASE)),
    ]

    # Vague legal reference patterns
    VAGUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("quy_dinh", re.compile(r"\bquy\s+định\b(?!\s+(?:tại|của|về|số|theo|gì|như|cho|gồm|nào))", re.IGNORECASE)),
        ("luat", re.compile(r"\bluật\b(?!\s+(?:\w{2,}))", re.IGNORECASE)),
        ("dieu_khoan", re.compile(r"\bđiều\s+khoản\b(?!\s+\d)", re.IGNORECASE)),
        ("van_ban", re.compile(r"\bvăn\s+bản\b(?!\s+(?:số|quy))", re.IGNORECASE)),
        ("nhu_tren", re.compile(r"\bnhư\s+trên\b", re.IGNORECASE)),
        ("nhu_vay", re.compile(r"\bnhư\s+vậy\b", re.IGNORECASE)),
        ("da_noi", re.compile(r"\bđã\s+nói\b", re.IGNORECASE)),
        ("context_dependent_followup", re.compile(
            r"\b(?:vậy|nếu\s+không|trường\s+hợp\s+đó|trường\s+hợp\s+này|"
            r"có\s+hợp\s+pháp\s+không|có\s+bị\s+xử\s+phạt\s+không)\b",
            re.IGNORECASE,
        )),
    ]

    # Entity type patterns for missing entity detection
    ENTITY_CHECKS: list[tuple[str, re.Pattern[str]]] = [
        ("PERSON", re.compile(
            r"\b(?:ai|người|ông|bà|cá\s+nhân|công\s+dân)\b", re.IGNORECASE)
        ),
        ("ORGANIZATION", re.compile(
            r"\b(?:cơ\s+quan|tổ\s+chức|công\s+ty|doanh\s+nghiệp)\b", re.IGNORECASE)
        ),
        ("LEGAL_ARTICLE", re.compile(
            r"\b(?:Điều\s+\d+|Khoản\s+\d+|Luật\s+\S+)\b", re.IGNORECASE)
        ),
        ("TIME", re.compile(
            r"\b(?:năm\s+\d{4}|tháng\s+\d+|ngày\s+\d+)\b", re.IGNORECASE)
        ),
    ]

    # Clarification templates
    CLARIFICATION_TEMPLATES: dict[str, str] = {
        "pronoun": "Bạn có thể cho biết cụ thể {pronoun} là ai/tổ chức nào không?",
        "vague_reference": "Bạn đang đề cập đến quy định/luật cụ thể nào? "
                          "Vui lòng cung cấp tên hoặc số hiệu văn bản.",
        "missing_entity": "Câu hỏi của bạn thiếu thông tin về {entity_type}. "
                         "Vui lòng cung cấp thêm chi tiết.",
        "entity_conflict": "Câu hỏi đề cập đến nhiều đối tượng cùng loại. "
                          "Bạn muốn hỏi về đối tượng nào cụ thể?",
        "multi_interpretation": "Câu hỏi có thể được hiểu theo nhiều cách pháp lý. "
                                "Bạn vui lòng nêu lĩnh vực, hành vi hoặc văn bản cụ thể.",
        "incomplete_context": "Câu hỏi đang phụ thuộc vào ngữ cảnh trước đó. "
                              "Bạn vui lòng nêu rõ văn bản, thủ tục hoặc trường hợp cần hỏi.",
        "general": "Câu hỏi của bạn chưa đủ rõ ràng. "
                  "Vui lòng cung cấp thêm chi tiết để tôi có thể trả lời chính xác hơn.",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize ambiguity detector.

        Args:
            config: Ambiguity config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("ambiguity", {})

        self.score_threshold: float = config.get("score_threshold", 0.6)
        self.custom_pronouns: list[str] = config.get("pronoun_list", [])
        self.custom_vague_terms: list[str] = config.get("vague_terms", [])

        logger.info(
            "AmbiguityDetector initialized | threshold={}",
            self.score_threshold,
        )

    def detect(
        self,
        query: str,
        history: str | None = None,
        history_resolution: HistoryResolutionResult | None = None,
    ) -> AmbiguityReport:
        """Detect ambiguity in a Vietnamese legal query.

        Checks for:
        1. Pronoun resolution: pronouns without antecedent in history
        2. Vague legal reference: "điều đó", "luật trên", "quy định này"
        3. Missing subject/object: query lacks legal subject
        4. Entity conflict: 2+ entities of same type causing ambiguity

        Args:
            query: The Vietnamese legal query.
            history: Optional conversation history string.

        Returns:
            AmbiguityReport with score, types, and clarification suggestion.
        """
        scores: list[float] = []
        ambiguity_types: list[str] = []
        missing_entity_type: str | None = None
        primary_ambiguity: str = "general"
        pronoun_found: str = ""
        detected_topics: list[str] = []
        has_legal_intent: bool = False
        semantic_ambiguity_score = 0.0

        history_resolution = history_resolution or resolve_history_referents(query, history)
        history_status = history_resolution.resolution_status
        query_has_contextual_reference = history_resolution.query_has_contextual_reference

        # Extract potential topics for context (including unaccented for robustness)
        topic_keywords = {
            "ngoại tình": ["ngoại tình", "ngoai tinh", "bồ nhí", "bo nhi", "tuesday"],
            "ly hôn": ["ly hôn", "ly hon", "chia tay", "ly thân", "ly than", "li hon"],
            "tài sản": ["tài sản", "tai san", "nhà đất", "nha dat", "xe cộ", "xe co", "tiền bạc", "tien bac", "chia tai san"],
            "con cái": ["con cái", "con cai", "nuôi con", "nuoi con", "quyền nuôi con", "quyen nuoi con", "trợ cấp", "tro cap", "nuoi con"],
            "kết hôn": ["kết hôn", "ket hon", "đăng ký kết hôn", "dang ky ket hon", "lấy nhau", "lay nhau"],
            "lao động": ["lao động", "lao dong", "lương", "luong", "thử việc", "thu viec", "hợp đồng", "hop dong"],
        }
        for topic, keywords in topic_keywords.items():
            if any(k in query.lower() for k in keywords):
                detected_topics.append(topic)
                has_legal_intent = True

        # Check 1: Pronoun resolution
        pronoun_score, found_pronoun = self._check_pronouns(
            query,
            history,
            history_resolution=history_resolution,
        )
        if pronoun_score > 0:
            scores.append(pronoun_score)
            ambiguity_types.append("pronoun")
            ambiguity_types.append("pronoun_reference")
            primary_ambiguity = "pronoun"
            pronoun_found = found_pronoun

        # Check 2: Vague legal references
        vague_score = self._check_vague_references(query)
        if vague_score > 0:
            scores.append(vague_score)
            ambiguity_types.append("vague_reference")
            if not primary_ambiguity or primary_ambiguity == "general":
                primary_ambiguity = "vague_reference"

        # Step 2: Extract features (with ambiguity info)
        # Check missing entities with legal intent info
        missing_score, missing_type = self._check_missing_entities(query, has_legal_intent)
        if missing_score > 0:
            scores.append(missing_score)
            ambiguity_types.append("missing_entity")
            missing_entity_type = missing_type
            semantic_ambiguity_score = max(semantic_ambiguity_score, missing_score)
            if not primary_ambiguity or primary_ambiguity == "general":
                primary_ambiguity = "missing_entity"

        multi_score = self._check_multi_interpretation(query)
        if multi_score > 0:
            scores.append(multi_score)
            ambiguity_types.append("multi_interpretation")
            semantic_ambiguity_score = max(semantic_ambiguity_score, multi_score)
            if not primary_ambiguity or primary_ambiguity == "general":
                primary_ambiguity = "multi_interpretation"

        incomplete_score = self._check_incomplete_context(query, history_resolution)
        if incomplete_score > 0:
            scores.append(incomplete_score)
            ambiguity_types.append("incomplete_context")
            if not primary_ambiguity or primary_ambiguity == "general":
                primary_ambiguity = "incomplete_context"

        # Check 4: Entity conflict
        conflict_score = self._check_entity_conflict(query)
        if conflict_score > 0:
            scores.append(conflict_score)
            ambiguity_types.append("entity_conflict")

        # Calculate overall score
        overall_score = max(scores) if scores else 0.0
        ambiguity_types = list(dict.fromkeys(ambiguity_types))
        is_ambiguous = overall_score >= self.score_threshold

        # Generate clarification question
        clarification = None
        if is_ambiguous:
            clarification = self.generate_clarification(
                AmbiguityReport(
                    score=overall_score,
                    is_ambiguous=True,
                    ambiguity_types=ambiguity_types,
                    missing_entity_type=missing_entity_type,
                    detected_topics=detected_topics,
                    missing_entity="missing_entity" in ambiguity_types,
                    multi_interpretation="multi_interpretation" in ambiguity_types,
                    incomplete_context="incomplete_context" in ambiguity_types,
                    pronoun_reference="pronoun_reference" in ambiguity_types,
                    semantic_ambiguity_score=semantic_ambiguity_score,
                    contextual_reference_score=pronoun_score or incomplete_score,
                    query_has_contextual_reference=query_has_contextual_reference,
                    history_resolution_status=history_status,
                    history_resolution_confidence=history_resolution.history_resolution_confidence,
                    resolved_referent=history_resolution.resolved_referent,
                    candidate_referents=[candidate.to_dict() for candidate in history_resolution.candidate_referents],
                ),
                query,
                pronoun=pronoun_found,
            )

        report = AmbiguityReport(
            score=overall_score,
            is_ambiguous=is_ambiguous,
            ambiguity_types=ambiguity_types,
            missing_entity_type=missing_entity_type,
            detected_topics=detected_topics,
            clarification_question=clarification,
            missing_entity="missing_entity" in ambiguity_types,
            multi_interpretation="multi_interpretation" in ambiguity_types,
            incomplete_context="incomplete_context" in ambiguity_types,
            pronoun_reference="pronoun_reference" in ambiguity_types,
            semantic_ambiguity_score=semantic_ambiguity_score,
            contextual_reference_score=pronoun_score or incomplete_score,
            query_has_contextual_reference=query_has_contextual_reference,
            history_resolution_status=history_status,
            history_resolution_confidence=history_resolution.history_resolution_confidence,
            resolved_referent=history_resolution.resolved_referent,
            candidate_referents=[candidate.to_dict() for candidate in history_resolution.candidate_referents],
        )

        logger.debug(
            "Ambiguity detection | score={:.2f} | ambiguous={} | types={}",
            report.score,
            report.is_ambiguous,
            report.ambiguity_types,
        )

        return report

    def _check_pronouns(
        self,
        query: str,
        history: str | None,
        history_resolution: HistoryResolutionResult | None = None,
    ) -> tuple[float, str]:
        """Check for unresolved pronouns.

        Args:
            query: The query text.
            history: Conversation history, if any.

        Returns:
            Tuple of (ambiguity score, found pronoun text).
        """
        found_pronouns: list[str] = []

        for name, pattern in self.PRONOUN_PATTERNS:
            if pattern.search(query):
                found_pronouns.append(name)

        # Also check custom pronouns
        for pronoun in self.custom_pronouns:
            if pronoun.lower() in query.lower() and pronoun not in found_pronouns:
                found_pronouns.append(pronoun)

        if history_resolution and history_resolution.query_has_contextual_reference and not found_pronouns:
            found_pronouns.append("tham chiếu ngữ cảnh")

        if not found_pronouns:
            return 0.0, ""

        if history_resolution:
            if history_resolution.resolution_status == "resolved":
                return 0.0, found_pronouns[0]
            if history_resolution.resolution_status == "no_history":
                return 0.9, found_pronouns[0]
            if history_resolution.resolution_status == "conflicting_history":
                return 0.9, found_pronouns[0]
            if history_resolution.resolution_status == "irrelevant_history":
                return 0.85, found_pronouns[0]

        # No history → pronouns are definitely ambiguous
        return (0.9 if not history else 0.7), found_pronouns[0]

    def _check_vague_references(self, query: str) -> float:
        """Check for vague legal references.

        Args:
            query: The query text.

        Returns:
            Ambiguity score for vague references.
        """
        vague_count = 0
        for _name, pattern in self.VAGUE_PATTERNS:
            vague_count += len(pattern.findall(query))

        # Custom vague terms only count if query also lacks specific legal entities
        has_specific_entity = bool(re.search(
            r"(?:Điều\s+\d+|Khoản\s+\d+|Luật\s+\S{2,}|Nghị\s+định|Bộ\s+luật)",
            query, re.IGNORECASE
        ))
        if not has_specific_entity:
            for term in self.custom_vague_terms:
                if term.lower() in query.lower():
                    vague_count = vague_count + 1

        if vague_count == 0:
            return 0.0
        return min(1.0, vague_count * 0.3)

    def has_any_entity(self, query: str) -> bool:
        """Check if the query contains any recognized entity or legal reference."""
        for _type, pattern in self.ENTITY_CHECKS:
            if pattern.search(query):
                return True
        if re.search(
            r"(?<!\w)(?:Nghị\s+định|Thông\s+tư|Quyết\s+định|Luật|Bộ\s+luật|Điều\s+\d+|\d+/\d{4}/[A-ZĐ\-]+|\d+/QĐ-[A-ZĐ]+)",
            query,
            re.IGNORECASE,
        ):
            return True
        return False

    def _check_missing_entities(self, query: str, has_legal_intent: bool = False) -> tuple[float, str | None]:
        """Check for missing entities or vague references.
        
        Returns:
            Ambiguity score and type of missing entity.
        """
        words = query.lower().split()
        word_count = len(words)
        query_l = query.lower()

        generic_need = re.search(
            r"(?<!\w)(?:hồ\s+sơ|thủ\s+tục|mức\s+phạt|trách\s+nhiệm|cơ\s+quan|"
            r"giấy\s+phép|nghĩa\s+vụ|áp\s+dụng|xin\s+phép|luật\s+nào|"
            r"quy\s+định\s+cũ|chịu\s+trách\s+nhiệm)(?!\w)",
            query_l,
            re.IGNORECASE,
        )
        concrete_anchor = self.has_any_entity(query) or bool(re.search(
            r"(?<!\w)(?:xây\s+dựng|kết\s+hôn|đất\s+đai|giao\s+thông|bảo\s+hiểm|"
            r"môi\s+trường|y\s+tế|hôn\s+nhân|lao\s+động|thuế|ngân\s+sách|"
            r"chất\s+thải|đường\s+bộ|tàu\s+biển|hàng\s+không)(?!\w)",
            query_l,
            re.IGNORECASE,
        ))

        if generic_need and not concrete_anchor:
            return 0.82, "LEGAL_TARGET"

        # Simple rule: if no recognized legal entities and short query
        if not self.has_any_entity(query):
            if word_count < 5:
                return 0.7, "LEGAL_ARTICLE"
            elif word_count < 10:
                # If it has legal intent but no specific entity, don't flag as ambiguous
                if has_legal_intent:
                    return 0.1, "LEGAL_ARTICLE"
                return 0.3, "LEGAL_ARTICLE"
            else:
                if has_legal_intent:
                    return 0.05, "LEGAL_ARTICLE"
                return 0.2, "LEGAL_ARTICLE"
        
        return 0.0, None

    def _check_multi_interpretation(self, query: str) -> float:
        """Detect short broad legal questions with multiple plausible targets."""
        query_l = query.lower()
        broad_subject = re.search(
            r"(?<!\w)(?:doanh\s+nghiệp|cá\s+nhân|công\s+ty|cơ\s+quan\s+nhà\s+nước|"
            r"ubnd|ủy\s+ban\s+nhân\s+dân|tổ\s+chức\s+nước\s+ngoài|tổ\s+chức)(?!\w)",
            query_l,
            re.IGNORECASE,
        )
        broad_predicate = re.search(
            r"(?<!\w)(?:có\s+được|có\s+phải|có\s+bị|có\s+thẩm\s+quyền|"
            r"được\s+hoạt\s+động|bị\s+phạt|được\s+cấp\s+giấy\s+phép|"
            r"phải\s+lưu\s+trữ)(?!\w)",
            query_l,
            re.IGNORECASE,
        )
        domain_anchor = re.search(
            r"(?<!\w)(?:Nghị\s+định|Thông\s+tư|Quyết\s+định|Luật|Điều\s+\d+|"
            r"đất\s+đai|xây\s+dựng|môi\s+trường|giao\s+thông|y\s+tế|"
            r"bảo\s+hiểm|lao\s+động|thuế|ngân\s+sách|hôn\s+nhân)(?!\w)",
            query_l,
            re.IGNORECASE,
        )
        if broad_subject and broad_predicate and not domain_anchor:
            return 0.85
        if len(query_l.split()) <= 7 and broad_predicate and not domain_anchor:
            return 0.75
        return 0.0

    def _check_incomplete_context(
        self,
        query: str,
        history_resolution: HistoryResolutionResult,
    ) -> float:
        """Detect follow-up questions whose answer depends on missing context."""
        if not history_resolution.query_has_contextual_reference:
            return 0.0
        if history_resolution.resolution_status == "resolved":
            return 0.0
        if history_resolution.resolution_status in {"no_history", "conflicting_history"}:
            return 0.9
        if history_resolution.resolution_status == "irrelevant_history":
            return 0.85
        return 0.0

    def _check_entity_conflict(self, query: str) -> float:
        """Check for conflicting entities of the same type.

        Args:
            query: The query text.

        Returns:
            Ambiguity score for entity conflicts.
        """
        from collections import Counter

        type_counts: Counter = Counter()
        for entity_type, pattern in self.ENTITY_CHECKS:
            matches = pattern.findall(query)
            if len(matches) > 1:
                type_counts[entity_type] += len(matches)

        if not type_counts:
            return 0.0

        # Multiple entities of same type might be intentional (comparison)
        # Only flag if there's no comparison keyword
        if re.search(r"\b(?:so\s+sánh|khác\s+nhau|giữa)\b", query, re.IGNORECASE):
            return 0.0

        return min(0.5, sum(type_counts.values()) * 0.15)

    def generate_clarification(
        self,
        report: AmbiguityReport,
        query: str,
        pronoun: str = "",
    ) -> str:
        """Generate a clarification question based on the ambiguity report.

        Args:
            report: AmbiguityReport from detect().
            query: Original query.
            pronoun: The specific pronoun found, if any.

        Returns:
            A Vietnamese clarification question string.
        """
        if "pronoun" in report.ambiguity_types and pronoun:
            return self.CLARIFICATION_TEMPLATES["pronoun"].format(pronoun=pronoun)
        elif "vague_reference" in report.ambiguity_types:
            return self.CLARIFICATION_TEMPLATES["vague_reference"]
        elif "incomplete_context" in report.ambiguity_types:
            return self.CLARIFICATION_TEMPLATES["incomplete_context"]
        elif "missing_entity" in report.ambiguity_types and report.missing_entity_type:
            type_names = {
                "PERSON": "người/cá nhân",
                "ORGANIZATION": "tổ chức/cơ quan",
                "LEGAL_ARTICLE": "điều luật/văn bản pháp luật",
                "TIME": "thời gian/năm",
            }
            entity_name = type_names.get(report.missing_entity_type, report.missing_entity_type)
            return self.CLARIFICATION_TEMPLATES["missing_entity"].format(
                entity_type=entity_name
            )
        elif "entity_conflict" in report.ambiguity_types:
            return self.CLARIFICATION_TEMPLATES["entity_conflict"]
        elif "multi_interpretation" in report.ambiguity_types:
            return self.CLARIFICATION_TEMPLATES["multi_interpretation"]
        else:
            return self.CLARIFICATION_TEMPLATES["general"]
