"""Query feature extraction for Stage 1 routing.

Extracts statistical, syntactic, multi-hop, graph-specific, ambiguity,
and conversation context features from Vietnamese legal queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ner.vi_ner import ViNER, Entity


@dataclass
class QueryFeatures:
    """Feature vector for a Vietnamese legal query.

    Used as input to the Stage 1 XGBoost router classifier.
    Captures statistical, syntactic, multi-hop, graph-specific,
    ambiguity, and conversation context signals.
    """

    # Statistical features
    query_length: int = 0
    entity_count: int = 0
    named_entity_types: list[str] = field(default_factory=list)

    # Syntactic features
    question_word: str = ""
    has_comparison: bool = False
    has_causality: bool = False
    has_temporal: bool = False

    # Multi-hop signals
    multi_hop_score: float = 0.0
    relation_chain_length: int = 0
    cross_doc_signals: bool = False

    # Graph-specific keywords
    graph_keyword_count: int = 0
    legal_reference_count: int = 0

    # Ambiguity signals
    ambiguity_score: float = 0.0
    has_pronoun: bool = False
    missing_entity_type: str | None = None

    # Conversation context
    history_length: int = 0
    history_resolves_ambiguity: bool = False

    def to_vector(self) -> list[float]:
        """Convert features to a flat numeric vector for ML models.

        Returns:
            List of floats representing all features.
        """
        return [
            float(self.query_length),
            float(self.entity_count),
            float(len(self.named_entity_types)),
            1.0 if self.has_comparison else 0.0,
            1.0 if self.has_causality else 0.0,
            1.0 if self.has_temporal else 0.0,
            self.multi_hop_score,
            float(self.relation_chain_length),
            1.0 if self.cross_doc_signals else 0.0,
            float(self.graph_keyword_count),
            float(self.legal_reference_count),
            self.ambiguity_score,
            1.0 if self.has_pronoun else 0.0,
            1.0 if self.missing_entity_type is not None else 0.0,
            float(self.history_length),
            1.0 if self.history_resolves_ambiguity else 0.0,
            # Question word encoding
            self._encode_question_word(),
        ]

    def _encode_question_word(self) -> float:
        """Encode question word as a numeric value.

        Returns:
            Float encoding: ai=1, gì=2, khi_nào=3, tại_sao=4, thế_nào=5, other=0
        """
        mapping = {
            "ai": 1.0,
            "gì": 2.0,
            "khi nào": 3.0,
            "tại sao": 4.0,
            "như thế nào": 5.0,
            "thế nào": 5.0,
            "bao nhiêu": 6.0,
            "ở đâu": 7.0,
        }
        return mapping.get(self.question_word, 0.0)

    @staticmethod
    def feature_names() -> list[str]:
        """Return the names of features in the vector.

        Returns:
            List of feature name strings.
        """
        return [
            "query_length",
            "entity_count",
            "entity_type_count",
            "has_comparison",
            "has_causality",
            "has_temporal",
            "multi_hop_score",
            "relation_chain_length",
            "cross_doc_signals",
            "graph_keyword_count",
            "legal_reference_count",
            "ambiguity_score",
            "has_pronoun",
            "has_missing_entity",
            "history_length",
            "history_resolves_ambiguity",
            "question_word_encoded",
        ]


# Vietnamese question word patterns
QUESTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ai", re.compile(r"\bai\b", re.IGNORECASE)),
    ("gì", re.compile(r"\bgì\b", re.IGNORECASE)),
    ("khi nào", re.compile(r"\bkhi\s+nào\b", re.IGNORECASE)),
    ("tại sao", re.compile(r"\b(?:tại\s+sao|vì\s+sao)\b", re.IGNORECASE)),
    ("như thế nào", re.compile(r"\b(?:như\s+thế\s+nào|thế\s+nào|ra\s+sao)\b", re.IGNORECASE)),
    ("bao nhiêu", re.compile(r"\bbao\s+nhiêu\b", re.IGNORECASE)),
    ("ở đâu", re.compile(r"\bở\s+đâu\b", re.IGNORECASE)),
]

# Comparison keywords
COMPARISON_PATTERNS = re.compile(
    r"\b(?:so\s+sánh|khác\s+nhau|giống\s+nhau|khác\s+biệt|hơn|kém|"
    r"giữa\s+.*?và\s+|so\s+với|đối\s+chiếu)\b",
    re.IGNORECASE,
)

# Causality keywords
CAUSALITY_PATTERNS = re.compile(
    r"\b(?:vì|do|tại|bởi\s+vì|nguyên\s+nhân|hậu\s+quả|dẫn\s+đến|"
    r"kết\s+quả|nên|cho\s+nên)\b",
    re.IGNORECASE,
)

# Temporal keywords
TEMPORAL_PATTERNS = re.compile(
    r"\b(?:khi|lúc|năm|tháng|ngày|thời\s+điểm|thời\s+hạn|hiệu\s+lực|"
    r"trước|sau|từ\s+ngày|đến\s+ngày)\b",
    re.IGNORECASE,
)

# Multi-hop signal keywords
MULTI_HOP_KEYWORDS = re.compile(
    r"\b(?:liên\s+quan|dẫn\s+chiếu|tham\s+chiếu|căn\s+cứ|theo\s+quy\s+định|"
    r"kết\s+hợp|đồng\s+thời|ngoài\s+ra|bên\s+cạnh|phối\s+hợp)\b",
    re.IGNORECASE,
)

# Graph-specific keywords
GRAPH_KEYWORDS = re.compile(
    r"\b(?:theo\s+quy\s+định|căn\s+cứ\s+vào|liên\s+quan\s+đến|"
    r"được\s+quy\s+định\s+tại|phù\s+hợp\s+với|tuân\s+theo|"
    r"quy\s+định\s+tại|áp\s+dụng\s+theo)\b",
    re.IGNORECASE,
)

# Legal article reference pattern
LEGAL_REF_PATTERN = re.compile(
    r"(?:Điều\s+\d+|Khoản\s+\d+|Luật\s+\S+|Nghị\s+định\s+\S+)",
    re.IGNORECASE,
)

# Cross-document signal patterns
CROSS_DOC_PATTERNS = re.compile(
    r"\b(?:nhiều\s+văn\s+bản|các\s+luật|giữa\s+.*?luật|cả\s+hai|"
    r"trong\s+.*?và\s+.*?luật)\b",
    re.IGNORECASE,
)

# Pronoun patterns (Vietnamese)
PRONOUN_PATTERNS = re.compile(
    r"\b(?:ông\s+ấy|bà\s+ấy|họ|nó|người\s+đó|cơ\s+quan\s+đó|"
    r"điều\s+đó|luật\s+đó|việc\s+đó|bên\s+đó)\b",
    re.IGNORECASE,
)


class FeatureExtractor:
    """Extract features from Vietnamese legal queries for Stage 1 routing.

    Combines NER, regex pattern matching, and heuristics to produce
    a comprehensive feature vector.
    """

    def __init__(self, ner_model: ViNER | None = None, config: dict[str, Any] | None = None) -> None:
        """Initialize feature extractor.

        Args:
            ner_model: Pre-initialized ViNER model. If None, creates one.
            config: Ambiguity config dict for pronoun/vague term lists.
        """
        self.ner = ner_model or ViNER()
        self.pronoun_list: list[str] = []
        self.vague_terms: list[str] = []

        if config and "ambiguity" in config:
            self.pronoun_list = config["ambiguity"].get("pronoun_list", [])
            self.vague_terms = config["ambiguity"].get("vague_terms", [])

    def extract(
        self,
        query: str,
        history: str | None = None,
        ambiguity_score: float = 0.0,
        has_pronoun: bool = False,
        missing_entity_type: str | None = None,
    ) -> QueryFeatures:
        """Extract all features from a query.

        Args:
            query: The Vietnamese legal query.
            history: Optional conversation history string.
            ambiguity_score: Pre-computed ambiguity score (from AmbiguityDetector).
            has_pronoun: Pre-computed pronoun detection flag.
            missing_entity_type: Pre-computed missing entity type.

        Returns:
            QueryFeatures dataclass with all features populated.
        """
        # Tokenize (simple whitespace for Vietnamese)
        tokens = query.split()

        # NER extraction
        entities_list = self.ner.extract([query])
        entities = entities_list[0] if entities_list else []
        entity_types = list(set(e.label for e in entities))

        # Question word detection
        question_word = ""
        for qw, pattern in QUESTION_PATTERNS:
            if pattern.search(query):
                question_word = qw
                break

        # Multi-hop score calculation
        multi_hop_hits = len(MULTI_HOP_KEYWORDS.findall(query))
        legal_refs = len(LEGAL_REF_PATTERN.findall(query))
        comparison = bool(COMPARISON_PATTERNS.search(query))
        multi_hop_score = min(
            1.0,
            (multi_hop_hits * 0.2 + (1.0 if legal_refs > 1 else 0.0) * 0.3 +
             (0.3 if comparison else 0.0)),
        )

        # Relation chain length estimate
        relation_chain = legal_refs + multi_hop_hits

        # History analysis
        history_length = 0
        history_resolves = False
        if history:
            history_length = len(history.strip().split("\n"))
            # Check if pronouns in query are resolved by history
            if has_pronoun and history:
                history_resolves = any(
                    p not in history.lower() for p in ["ông ấy", "bà ấy", "người đó"]
                )

        features = QueryFeatures(
            query_length=len(tokens),
            entity_count=len(entities),
            named_entity_types=entity_types,
            question_word=question_word,
            has_comparison=comparison,
            has_causality=bool(CAUSALITY_PATTERNS.search(query)),
            has_temporal=bool(TEMPORAL_PATTERNS.search(query)),
            multi_hop_score=multi_hop_score,
            relation_chain_length=relation_chain,
            cross_doc_signals=bool(CROSS_DOC_PATTERNS.search(query)),
            graph_keyword_count=len(GRAPH_KEYWORDS.findall(query)),
            legal_reference_count=legal_refs,
            ambiguity_score=ambiguity_score,
            has_pronoun=has_pronoun or bool(PRONOUN_PATTERNS.search(query)),
            missing_entity_type=missing_entity_type,
            history_length=history_length,
            history_resolves_ambiguity=history_resolves,
        )

        logger.debug(
            "Features extracted | length={} | entities={} | multi_hop={:.2f} | graph_kw={}",
            features.query_length,
            features.entity_count,
            features.multi_hop_score,
            features.graph_keyword_count,
        )

        return features
