"""Query feature extraction for Stage 1 routing.

Extracts statistical, syntactic, multi-hop, graph-specific, ambiguity,
and conversation context features from Vietnamese legal queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ner.factory import get_ner_model
from ner.vi_ner import Entity
from router.query_complexity import QueryComplexityAnalyzer, ComplexityFeatures


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
    history_resolution_status: str = "not_needed"
    history_resolution_confidence: float = 0.0
    resolved_referent: str | None = None
    candidate_referents: list[dict[str, object]] = field(default_factory=list)
    query_has_contextual_reference: bool = False

    # Ambiguity metadata used by Stage 2 policy, intentionally not included in
    # the Stage 1 vector so the existing XGBoost checkpoint remains compatible.
    missing_entity: bool = False
    multi_interpretation: bool = False
    incomplete_context: bool = False
    pronoun_reference: bool = False
    semantic_ambiguity_score: float = 0.0
    contextual_reference_score: float = 0.0

    # --- Advanced complexity features (Adaptive-RAG) ---
    complexity_level: int = 1          # 1=simple, 2=multi-hop, 3=cross-doc
    sub_question_count: int = 0        # Number of decomposable sub-questions
    entity_density: float = 0.0        # entity count / token count
    law_specificity: int = 0           # 2=specific article, 1=law name, 0=none
    conditional_depth: int = 0         # Nesting depth of conditionals
    is_factoid: bool = False           # Direct definition lookup
    multi_hop_verb_count: int = 0      # Count of multi-hop verbs
    comparative_depth: int = 0         # Count of comparative expressions
    authority_chain_count: int = 0     # Authority / responsibility chain signals
    legal_effect_count: int = 0        # Legal effect / amendment / repeal signals
    procedural_count: int = 0          # Ordered procedure / workflow signals
    multi_entity_relation_count: int = 0  # Relation among multiple legal actors

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
            # --- Adaptive-RAG complexity features ---
            float(self.complexity_level),
            float(self.sub_question_count),
            self.entity_density,
            float(self.law_specificity),
            float(self.conditional_depth),
            1.0 if self.is_factoid else 0.0,
            float(self.multi_hop_verb_count),
            float(self.comparative_depth),
            float(self.authority_chain_count),
            float(self.legal_effect_count),
            float(self.procedural_count),
            float(self.multi_entity_relation_count),
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
            # Adaptive-RAG complexity features
            "complexity_level",
            "sub_question_count",
            "entity_density",
            "law_specificity",
            "conditional_depth",
            "is_factoid",
            "multi_hop_verb_count",
            "comparative_depth",
            "authority_chain_count",
            "legal_effect_count",
            "procedural_count",
            "multi_entity_relation_count",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Vietnamese Legal Reference Sub-patterns (Problem A fix)
# ─────────────────────────────────────────────────────────────────────────────

# 1. Article/clause/point references: "Điều 1", "khoản 22", "điểm c",
#    "Điểm a Khoản 2 Điều 5", etc.
#    Uses (?<!\w) instead of \b to avoid Unicode boundary issues.
ARTICLE_REF_PATTERN = re.compile(
    r"(?<!\w)(?:Điều|Khoản|Điểm)\s+\w+",
    re.IGNORECASE,
)

# 2. Full document number pattern: "77/2017/TT-BTC", "320/2025/NĐ-CP",
#    "25/2026/QĐ-UBND", "01/2026/NQ-HĐND".
#    Matches the standard VN format: number/year/suffix-abbreviation.
LEGAL_DOC_NUMBER_PATTERN = re.compile(
    r"\d+\s*/\s*\d{4}\s*/\s*[A-Za-zĐđÀ-ỹ][A-Za-zĐđÀ-ỹ\-]*",
)

# 3. Document type + name/number: "Thông tư số 02/2026/TT-BYT",
#    "Quyết định 846/QĐ-BNNMT", etc. Excludes Luật which is handled below.
LEGAL_DOC_TYPE_PATTERN = re.compile(
    r"(?<!\w)(?:Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết|"
    r"Chỉ\s+thị|Pháp\s+lệnh|Văn\s+bản(?:\s+hợp\s+nhất)?)"
    r"(?:\s+số)?\s+\S+",
    re.IGNORECASE,
)

# 3b. Named laws: "Luật Kinh doanh Bảo hiểm", "Bộ luật Hình sự"
#     Stops matching at common question words or punctuation.
NAMED_LAW_PATTERN = re.compile(
    r"(?<!\w)(?:Luật|Hiến\s+pháp|Bộ\s+luật)\s+.*?(?=\s*(?:quy\s+định|áp\s+dụng|có|không|là|về|\?|\.|\,|$))",
    re.IGNORECASE,
)

# 4. VBHN (Văn bản hợp nhất) patterns: "VBHN-BXD 12", "14/VBHN-BXD",
#    "47/VBHN-VPQH", "Văn bản hợp nhất 12/VBHN-BXD".
VBHN_PATTERN = re.compile(
    r"(?:"
    r"VBHN[- ][A-ZĐ]{2,10}(?:\s+\d+)?"
    r"|\d+\s*/\s*VBHN[- ][A-ZĐ]{2,10}"
    r")",
    re.IGNORECASE,
)

# 5. Legal anchor patterns: "Thông tư này", "Quyết định này", "Nghị định này",
#    "văn bản này", "quy định này". These contribute to graph_keyword_count
#    when paired with relational signals.
LEGAL_ANCHOR_PATTERN = re.compile(
    r"(?<!\w)(?:Thông\s+tư|Quyết\s+định|Nghị\s+định|Nghị\s+quyết|"
    r"Luật|Pháp\s+lệnh|văn\s+bản|quy\s+định)\s+này(?!\w)",
    re.IGNORECASE,
)

# English legal reference pattern (for HotpotQA fallback)
LEGAL_REF_PATTERN_EN = re.compile(
    r"(?:Article\s+\d+|Clause\s+\d+|Section\s+\d+|Law\s+\S+|Decree\s+\S+)",
    re.IGNORECASE,
)

# Combined legacy pattern kept for backward compatibility in non-extract paths
LEGAL_REF_PATTERN = re.compile(
    r"(?:"
    r"(?:Điều|Khoản|Điểm)\s+\w+|"
    r"(?:Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết|Chỉ\s+thị|Hiến\s+pháp)(?:\s+số)?\s+[A-Za-z0-9\/\-]+|"
    r"VBHN(?:\-[A-ZĐ]+)?\s*\d*|"
    r"\d+\/\d+\/[A-Za-zĐđÀ-ỹ\-]+|"
    r"Article\s+\d+|Clause\s+\d+|Law\s+\S+|Decree\s+\S+"
    r")",
    re.IGNORECASE,
)


def _count_legal_references(query: str, language: str = "vi") -> int:
    """Count distinct legal references in a query.

    Uses multiple sub-patterns and deduplicates by character span to avoid
    double-counting overlapping matches (e.g. "Nghị định số 50/2026/NĐ-CP"
    should not count both the doc-type match and the number match separately
    when they overlap).

    Returns:
        Non-negative integer count of distinct legal references.
    """
    if language == "en":
        return len(LEGAL_REF_PATTERN_EN.findall(query))

    # Collect all match spans from sub-patterns
    spans: list[tuple[int, int]] = []
    for pattern in (
        ARTICLE_REF_PATTERN,
        LEGAL_DOC_NUMBER_PATTERN,
        LEGAL_DOC_TYPE_PATTERN,
        NAMED_LAW_PATTERN,
        VBHN_PATTERN,
    ):
        for m in pattern.finditer(query):
            spans.append((m.start(), m.end()))

    if not spans:
        return 0

    # Merge overlapping spans to avoid double-counting
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping or adjacent: merge
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return len(merged)


# ─────────────────────────────────────────────────────────────────────────────
# Vietnamese patterns (Legacy + Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_PATTERNS_VI: list[tuple[str, re.Pattern[str]]] = [
    ("ai", re.compile(r"\bai\b", re.IGNORECASE)),
    ("gì", re.compile(r"\bgì\b", re.IGNORECASE)),
    ("khi nào", re.compile(r"\bkhi\s+nào\b", re.IGNORECASE)),
    ("tại sao", re.compile(r"\b(?:tại\s+sao|vì\s+sao)\b", re.IGNORECASE)),
    ("như thế nào", re.compile(r"\b(?:như\s+thế\s+nào|thế\s+nào|ra\s+sao)\b", re.IGNORECASE)),
    ("bao nhiêu", re.compile(r"\bbao\s+nhiêu\b", re.IGNORECASE)),
    ("ở đâu", re.compile(r"\bở\s+đâu\b", re.IGNORECASE)),
]

COMPARISON_PATTERNS_VI = re.compile(
    r"\b(?:so\s+sánh|khác\s+nhau|giống\s+nhau|khác\s+biệt|hơn|kém|"
    r"giữa\s+.*?và\s+|so\s+với|đối\s+chiếu)\b",
    re.IGNORECASE,
)

MULTI_HOP_KEYWORDS_VI = re.compile(
    r"\b(?:liên\s+quan|dẫn\s+chiếu|tham\s+chiếu|căn\s+cứ|theo\s+quy\s+định|"
    r"kết\s+hợp|đồng\s+thời|ngoài\s+ra|bên\s+cạnh|phối\s+hợp|"
    r"trình\s+tự|quy\s+trình|thủ\s+tục|thẩm\s+quyền|trách\s+nhiệm|"
    r"bãi\s+bỏ|sửa\s+đổi|thay\s+thế|chuyển\s+tiếp)\b",
    re.IGNORECASE,
)

# Graph keywords: relational signals indicating graph traversal is needed.
# Enhanced to include anchor-aware patterns like "theo Điều", "theo Thông tư này",
# "cơ quan ban hành", and standalone legal effect / authority keywords.
# Uses (?<!\w)...(?!\w) instead of \b for Vietnamese Unicode safety.
GRAPH_KEYWORDS_VI = re.compile(
    r"(?<!\w)(?:"
    # Relational phrases
    r"theo\s+quy\s+định|căn\s+cứ\s+vào|căn\s+cứ|liên\s+quan\s+đến|"
    r"được\s+quy\s+định\s+tại|phù\s+hợp\s+với|tuân\s+theo|"
    r"quy\s+định\s+tại|áp\s+dụng\s+theo|cơ\s+sở\s+pháp\s+lý|"
    r"căn\s+cứ\s+pháp\s+lý|"
    # Authority / responsibility
    r"thẩm\s+quyền|cơ\s+quan\s+có\s+thẩm\s+quyền|"
    r"chịu\s+trách\s+nhiệm|trách\s+nhiệm|"
    r"cơ\s+quan\s+ban\s+hành|"
    # Procedure
    r"quy\s+trình|trình\s+tự|thủ\s+tục|phân\s+cấp|"
    # Legal effect
    r"bãi\s+bỏ|sửa\s+đổi|bổ\s+sung|thay\s+thế|chuyển\s+tiếp|"
    r"hiệu\s+lực|hết\s+hiệu\s+lực|còn\s+hiệu\s+lực|có\s+hiệu\s+lực|"
    r"ban\s+hành\s+kèm\s+theo|ban\s+hành|"
    r"áp\s+dụng\s+trong\s+trường\s+hợp|"
    # Anchor-aware phrases: "theo Điều X", "theo Thông tư này", etc.
    r"theo\s+Điều|theo\s+Thông\s+tư\s+này|theo\s+Quyết\s+định\s+này|"
    r"theo\s+Nghị\s+định\s+này|theo\s+Luật\s+này"
    r")(?!\w)",
    re.IGNORECASE,
)

CROSS_DOC_PATTERNS_VI = re.compile(
    r"\b(?:"
    r"nhiều\s+văn\s+bản|các\s+luật|giữa\s+.*?luật|cả\s+hai|"
    r"chồng\s+chéo|mâu\s+thuẫn|xung\s+đột|không\s+nhất\s+quán|"
    r"văn\s+bản.*?bãi\s+bỏ|bãi\s+bỏ.*?văn\s+bản|hết\s+hiệu\s+lực|"
    r"thay\s+thế.*?quy\s+định|quy\s+định.*?cũ.*?mới"
    r")\b",
    re.IGNORECASE,
)

PRONOUN_PATTERNS_VI = re.compile(
    r"\b(?:ông\s+ấy|bà\s+ấy|họ|nó|người\s+đó|cơ\s+quan\s+đó|"
    r"điều\s+đó|luật\s+đó|việc\s+đó|bên\s+đó)\b",
    re.IGNORECASE,
)

# English patterns (HotpotQA)
QUESTION_PATTERNS_EN: list[tuple[str, re.Pattern[str]]] = [
    ("who", re.compile(r"\bwho\b", re.IGNORECASE)),
    ("what", re.compile(r"\bwhat\b", re.IGNORECASE)),
    ("which", re.compile(r"\bwhich\b", re.IGNORECASE)),
    ("where", re.compile(r"\bwhere\b", re.IGNORECASE)),
    ("when", re.compile(r"\bwhen\b", re.IGNORECASE)),
    ("how", re.compile(r"\bhow\b", re.IGNORECASE)),
    ("are/were/is", re.compile(r"\b(?:are|were|is|was|do|did|does)\b", re.IGNORECASE)),
]

COMPARISON_PATTERNS_EN = re.compile(
    r"\b(?:same|different|both|compare|versus|vs\.?|"
    r"older|newer|earlier|later|higher|lower|"
    r"more|less|better|worse|between\s+\w+\s+and)\b",
    re.IGNORECASE,
)

MULTI_HOP_KEYWORDS_EN = re.compile(
    r"\b(?:who\s+(?:was|is|were)|which\s+(?:was|is)|"
    r"what\s+(?:did|was|is)|born\s+in|founded\s+by|"
    r"located\s+in|same\s+(?:as|nationality|country)|"
    r"both|also|and\s+(?:also|additionally)|"
    r"first\s+(?:to|who|that)|prior\s+to|before|after)\b",
    re.IGNORECASE,
)

GRAPH_KEYWORDS_EN = re.compile(
    r"\b(?:according\s+to|based\s+on|related\s+to|specified\s+in|"
    r"consistent\s+with|follows|under\s+the\s+provisions\s+of|"
    r"applied\s+per)\b",
    re.IGNORECASE,
)

CROSS_DOC_PATTERNS_EN = re.compile(
    r"\b(?:"
    r"multiple\s+documents|various\s+laws|between\s+.*?and\s+.*?laws|"
    r"both\s+of|conflict|contradiction|overlap|superseded|amended|"
    r"replaced|prior\s+to|after\s+the|based\s+on\s+.*?and\s+.*?"
    r")\b",
    re.IGNORECASE,
)

PRONOUN_PATTERNS_EN = re.compile(
    r"\b(?:he|she|they|it|that\s+person|that\s+place|that\s+document|"
    r"this\s+regulation)\b",
    re.IGNORECASE,
)

# Universal patterns
CAUSALITY_PATTERNS = re.compile(
    r"\b(?:vì|do|tại|bởi\s+vì|nguyên\s+nhân|hậu\s+quả|dẫn\s+đến|"
    r"because|due\s+to|cause|reason|result|lead\s+to|consequently)\b",
    re.IGNORECASE,
)

TEMPORAL_PATTERNS = re.compile(
    r"\b(?:khi|lúc|năm|tháng|ngày|thời\s+điểm|thời\s+hạn|hiệu\s+lực|"
    r"when|year|month|day|time|period|effective|before|after|since|until)\b",
    re.IGNORECASE,
)


class FeatureExtractor:
    """Extract features from Vietnamese legal queries for Stage 1 routing.

    Combines NER, regex pattern matching, and heuristics to produce
    a comprehensive feature vector.
    """

    def __init__(self, ner_model: Any | None = None, config: dict[str, Any] | None = None) -> None:
        """Initialize feature extractor.

        Args:
            ner_model: Pre-initialized NER model. If None, creates one based on config.
            config: Full config dict.
        """
        self._config = config or {}
        self.ner = ner_model or get_ner_model(self._config.get("ner"))
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.pronoun_list: list[str] = []
        self.vague_terms: list[str] = []

        # Configurable scoring parameters (for ablation study)
        scoring_cfg = self._config.get("router", {}).get("scoring", {})
        self.multi_hop_norm_divisor: float = scoring_cfg.get("multi_hop_norm_divisor", 3.0)
        self.comparison_boost: float = scoring_cfg.get("comparison_boost", 0.7)
        self.cross_doc_boost: float = scoring_cfg.get("cross_doc_boost", 0.9)

        if "ambiguity" in self._config:
            self.pronoun_list = self._config["ambiguity"].get("pronoun_list", [])
            self.vague_terms = self._config["ambiguity"].get("vague_terms", [])

    def extract(
        self,
        query: str,
        history: str | None = None,
        ambiguity_score: float = 0.0,
        has_pronoun: bool = False,
        missing_entity_type: str | None = None,
        history_resolution_status: str = "not_needed",
        history_resolution_confidence: float = 0.0,
        resolved_referent: str | None = None,
        candidate_referents: list[dict[str, object]] | None = None,
        query_has_contextual_reference: bool = False,
        missing_entity: bool = False,
        multi_interpretation: bool = False,
        incomplete_context: bool = False,
        pronoun_reference: bool = False,
        semantic_ambiguity_score: float = 0.0,
        contextual_reference_score: float = 0.0,
    ) -> QueryFeatures:
        """Extract all features from a query.

        Args:
            query: The legal query.
            history: Optional conversation history string.
            ambiguity_score: Pre-computed ambiguity score.
            has_pronoun: Pre-computed pronoun detection flag.
            missing_entity_type: Pre-computed missing entity type.

        Returns:
            QueryFeatures dataclass with all features populated.
        """
        tokens = query.split()
        lang = self._config.get("language", "vi")

        # NER extraction
        entities_list = self.ner.extract([query])
        entities = entities_list[0] if entities_list else []
        entity_types = list(set(e.label for e in entities))

        # Select patterns based on language
        if lang == "en":
            q_patterns = QUESTION_PATTERNS_EN
            comp_pattern = COMPARISON_PATTERNS_EN
            mh_pattern = MULTI_HOP_KEYWORDS_EN
            graph_pattern = GRAPH_KEYWORDS_EN
            cross_doc_pattern = CROSS_DOC_PATTERNS_EN
            pronoun_pattern = PRONOUN_PATTERNS_EN
        else:
            q_patterns = QUESTION_PATTERNS_VI
            comp_pattern = COMPARISON_PATTERNS_VI
            mh_pattern = MULTI_HOP_KEYWORDS_VI
            graph_pattern = GRAPH_KEYWORDS_VI
            cross_doc_pattern = CROSS_DOC_PATTERNS_VI
            pronoun_pattern = PRONOUN_PATTERNS_VI

        # Question word extraction
        question_word = ""
        for qw, pattern in q_patterns:
            if pattern.search(query):
                question_word = qw
                break

        # Keyword matching
        comparison = bool(comp_pattern.search(query))
        multi_hop_hits = len(mh_pattern.findall(query))
        graph_kw_hits = len(graph_pattern.findall(query))

        # Legal reference counting (Problem A fix: use sub-pattern helper)
        legal_refs = _count_legal_references(query, language=lang)

        is_cross_doc = bool(cross_doc_pattern.search(query))
        
        # History analysis
        history_length = 0
        history_resolves = False
        if history:
            history_length = len(history.strip().split("\n"))
            if history_resolution_status != "not_needed":
                history_resolves = history_resolution_status == "resolved"
            elif has_pronoun:
                # Check if pronouns in query are resolved by history
                res_terms = ["ông ấy", "bà ấy", "người đó"] if lang == "vi" else ["he", "she", "they", "it"]
                history_resolves = any(p in history.lower() for p in res_terms)

        # Final feature flags
        has_pronoun = has_pronoun or bool(pronoun_pattern.search(query))

        # Multi-hop scoring — parameters configurable for ablation study
        if lang == "en":
            multi_hop_score = min(1.0, max(0.0, multi_hop_hits - 1) / self.multi_hop_norm_divisor)
        else:
            multi_hop_score = min(1.0, multi_hop_hits / self.multi_hop_norm_divisor)
            
        if comparison:
            multi_hop_score = max(multi_hop_score, self.comparison_boost)
        if is_cross_doc:
            multi_hop_score = max(multi_hop_score, self.cross_doc_boost)

        # Relation chain length estimate
        relation_chain = legal_refs + multi_hop_hits

        # Adaptive-RAG complexity features
        complexity_feats = self.complexity_analyzer.analyze(
            query=query,
            entity_count=len(entities),
            language=lang,
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
            cross_doc_signals=is_cross_doc,
            graph_keyword_count=graph_kw_hits,
            legal_reference_count=legal_refs,
            ambiguity_score=ambiguity_score,
            has_pronoun=has_pronoun,
            missing_entity_type=missing_entity_type,
            history_length=history_length,
            history_resolves_ambiguity=history_resolves,
            history_resolution_status=history_resolution_status,
            history_resolution_confidence=history_resolution_confidence,
            resolved_referent=resolved_referent,
            candidate_referents=candidate_referents or [],
            query_has_contextual_reference=query_has_contextual_reference,
            missing_entity=missing_entity,
            multi_interpretation=multi_interpretation,
            incomplete_context=incomplete_context,
            pronoun_reference=pronoun_reference,
            semantic_ambiguity_score=semantic_ambiguity_score,
            contextual_reference_score=contextual_reference_score,
            complexity_level=complexity_feats.complexity_level,
            sub_question_count=complexity_feats.sub_question_count,
            entity_density=complexity_feats.entity_density,
            law_specificity=complexity_feats.law_specificity,
            conditional_depth=complexity_feats.conditional_depth,
            is_factoid=complexity_feats.is_factoid,
            multi_hop_verb_count=complexity_feats.multi_hop_verb_count,
            comparative_depth=complexity_feats.comparative_depth,
            authority_chain_count=complexity_feats.authority_chain_count,
            legal_effect_count=complexity_feats.legal_effect_count,
            procedural_count=complexity_feats.procedural_count,
            multi_entity_relation_count=complexity_feats.multi_entity_relation_count,
        )

        logger.debug(
            "Features extracted | length={} | entities={} | multi_hop={:.2f} | "
            "graph_kw={} | legal_ref={} | cross_doc={} | complexity_lvl={}",
            features.query_length,
            features.entity_count,
            features.multi_hop_score,
            features.graph_keyword_count,
            features.legal_reference_count,
            features.cross_doc_signals,
            features.complexity_level,
        )

        return features
