"""Query Complexity Analyzer for Adaptive Routing.

Implements the 3-level complexity classification from Adaptive-RAG (Jeong et al., 2024):
  Level 1 (Simple)    → dense_retrieval   (factoid, single law article lookup)
  Level 2 (Multi-hop) → graph_traversal   (multi-step intra-document reasoning)
  Level 3 (Complex)   → hybrid_reasoning  (cross-document synthesis)

Features extracted here augment the XGBoost router's feature vector and
directly address the low recall (30.4%) of the graph_traversal class.

Reference:
  Jeong et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented
  Large Language Models through Question Complexity. arXiv:2403.14403.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Bank — Vietnamese Legal Domain + English benchmark support
# ─────────────────────────────────────────────────────────────────────────────

# Sub-question decomposition signals (suggests multi-hop)
SUB_QUESTION_PATTERNS = re.compile(
    r"(?:"
    r"thứ nhất|thứ hai|thứ ba|một là|hai là|ba là|"
    r"đồng thời|ngoài ra|bên cạnh đó"
    r")",
    re.IGNORECASE,
)

# Conditional complexity markers (suggests cross-document reasoning)
CONDITIONAL_PATTERNS = re.compile(
    r"(?:"
    r"nếu|trong trường hợp|khi|giả sử|trừ khi|trừ trường hợp|"
    r"điều kiện|theo điều kiện|tùy thuộc|tùy theo|phụ thuộc"
    r")",
    re.IGNORECASE,
)

# Specific article reference (Điều X → likely single-hop dense retrieval)
SPECIFIC_ARTICLE_PATTERN = re.compile(
    r"Điều\s+\d+[a-zA-Z]?",
    re.IGNORECASE,
)

# Law name only (no article number → likely hybrid)
LAW_NAME_ONLY_PATTERN = re.compile(
    r"(?:Luật|Nghị định|Thông tư|Quyết định)\s+\S+",
    re.IGNORECASE,
)

# Explicit multi-hop verbs (suggests graph traversal)
MULTI_HOP_VERBS = re.compile(
    r"(?:"
    r"dẫn chiếu|tham chiếu|căn cứ.*và|liên quan.*đến|ảnh hưởng.*đến|"
    r"kết hợp.*với|phối hợp.*với|theo.*và.*theo"
    r")",
    re.IGNORECASE,
)

# Comparative patterns (feature for graph_traversal)
DEEP_COMPARATIVE = re.compile(
    r"(?:"
    r"so sánh|khác nhau.*như thế nào|giữa.*và.*khác|"
    r"ưu điểm|nhược điểm|điểm khác biệt|mức độ.*hơn"
    r")",
    re.IGNORECASE,
)

# Aggregation/summary patterns (factoid, → dense_retrieval)
FACTOID_PATTERNS = re.compile(
    r"(?:"
    r"bao\s+nhiêu|mức\s+phạt|mức\s+xử\s+phạt|phạt\s+bao\s+nhiêu|bao\s+nhiêu\s+tiền|"
    r"là\s+gì|như\s+thế\s+nào|gồm\s+những\s+gì|bao\s+gồm|"
    r"thời\s+hạn|thời\s+gian|khi\s+nào|ngày\s+nào|tối\s+đa|tối\s+thiểu|bao\s+lâu|"
    r"có\s+được\s+không|ai\s+được|tại\s+sao|"
    r"khái\s+niệm|thuật\s+ngữ"
    r")",
    re.IGNORECASE,
)

# Authority-chain signals: legal authority, responsibility, and hierarchy.
AUTHORITY_CHAIN_PATTERNS = re.compile(
    r"(?:"
    r"thẩm\s+quyền|có\s+quyền|được\s+quyền|chịu\s+trách\s+nhiệm|"
    r"cấp\s+trên|cấp\s+dưới|cơ\s+quan\s+chủ\s+quản|cơ\s+quan\s+có\s+thẩm\s+quyền|"
    r"người\s+có\s+thẩm\s+quyền|ai\s+được\s+phép|ai\s+có\s+quyền"
    r")",
    re.IGNORECASE,
)

# Legal-effect signals: amendment, repeal, replacement, and transitional logic.
LEGAL_EFFECT_PATTERNS = re.compile(
    r"(?:"
    r"bãi\s+bỏ|thay\s+thế|sửa\s+đổi|bổ\s+sung|không\s+còn\s+hiệu\s+lực|"
    r"hết\s+hiệu\s+lực|có\s+hiệu\s+lực|được\s+áp\s+dụng\s+thay|"
    r"quy\s+định\s+cũ|quy\s+định\s+mới|chuyển\s+tiếp|điều\s+khoản\s+chuyển\s+tiếp"
    r")",
    re.IGNORECASE,
)

# Strong legal-effect signals are safer graph indicators than plain "có hiệu lực".
STRONG_LEGAL_EFFECT_PATTERNS = re.compile(
    r"(?:"
    r"bãi\s+bỏ|thay\s+thế|sửa\s+đổi|bổ\s+sung|không\s+còn\s+hiệu\s+lực|"
    r"hết\s+hiệu\s+lực|được\s+áp\s+dụng\s+thay|quy\s+định\s+cũ|"
    r"quy\s+định\s+mới|chuyển\s+tiếp|điều\s+khoản\s+chuyển\s+tiếp"
    r")",
    re.IGNORECASE,
)

# Procedural sequence signals: ordered steps and administrative workflows.
PROCEDURAL_SEQUENCE_PATTERNS = re.compile(
    r"(?:"
    r"quy\s+trình|thủ\s+tục|các\s+bước|bước\s+thứ|trình\s+tự|"
    r"nộp\s+hồ\s+sơ|xét\s+duyệt|phê\s+duyệt|ban\s+hành|"
    r"sau\s+khi|trước\s+khi.*thì|phải.*mới\s+được|điều\s+kiện\s+để\s+được"
    r")",
    re.IGNORECASE,
)

# Multi-entity relation signals: relations between legal actors or obligations.
MULTI_ENTITY_RELATION_PATTERNS = re.compile(
    r"(?:"
    r"giữa\s+\S+\s+và\s+\S+|mối\s+quan\s+hệ|quyền\s+và\s+nghĩa\s+vụ|"
    r"trách\s+nhiệm\s+của.*và.*của|cả.*lẫn|vừa.*vừa"
    r")",
    re.IGNORECASE,
)

# Organization / institutional structure signals.
# These queries often require graph traversal over LegalDoc -> agency/function/unit relations.
ORGANIZATION_STRUCTURE_PATTERNS = re.compile(
    r"(?:"
    r"tư\s+cách\s+pháp\s+nhân|đơn\s+vị\s+trực\s+thuộc|"
    r"thành\s+lập|tổ\s+chức\s+lại|giải\s+thể|"
    r"cơ\s+cấu\s+tổ\s+chức|chức\s+năng\s+nhiệm\s+vụ|"
    r"vị\s+trí\s+pháp\s+lý|nhiệm\s+vụ\s+và\s+quyền\s+hạn"
    r")",
    re.IGNORECASE,
)

# Quantitative legal norm signals.
# These cases often require selecting the correct norm table/threshold
# before answering, not merely semantic lookup.
QUANTITATIVE_NORM_PATTERNS = re.compile(
    r"(?:"
    r"định\s+mức|mức\s+sử\s+dụng|số\s+lượng|"
    r"bao\s+nhiêu|cần\s+bao\s+nhiêu|"
    r"diện\s+tích|tỷ\s+lệ|tỉ\s+lệ|ha\b|1/\d+|"
    r"ngưỡng|mức\s+tối\s+đa|mức\s+tối\s+thiểu"
    r")",
    re.IGNORECASE,
)

# Public finance / plan / project signals.
# These often require checking eligibility or authority under budget/project rules.
PUBLIC_FINANCE_PLAN_PATTERNS = re.compile(
    r"(?:"
    r"vốn\s+ODA|vốn\s+vay\s+ưu\s+đãi|nguồn\s+vốn|"
    r"sử\s+dụng\s+vốn|dự\s+án|kế\s+hoạch\s+chuyển\s+đổi\s+số|"
    r"đề\s+án|kinh\s+phí|ngân\s+sách|"
    r"được\s+phép\s+.*vốn|có\s+được\s+phép\s+.*(?:vốn|kinh\s+phí|ngân\s+sách)"
    r")",
    re.IGNORECASE,
)

SIMPLE_EFFECTIVE_DATE_PATTERNS = re.compile(
    r"(?:"
    r"khi\s+nào.*(?:có\s+hiệu\s+lực|hiệu\s+lực\s+thi\s+hành)|"
    r"(?:có\s+hiệu\s+lực|hiệu\s+lực\s+thi\s+hành).*(?:từ\s+ngày\s+nào|khi\s+nào)"
    r")",
    re.IGNORECASE,
)

SUB_QUESTION_PATTERNS_EN = re.compile(
    r"(?:\band\b|\bboth\b|\balso\b|\bfirst\b|\bsecond\b|\bthen\b|\bafter\b|\bbefore\b)",
    re.IGNORECASE,
)

CONDITIONAL_PATTERNS_EN = re.compile(
    r"(?:\bif\b|\bwhen\b|\bunless\b|\bprovided that\b|\bdepending on\b)",
    re.IGNORECASE,
)

SPECIFIC_ARTICLE_PATTERN_EN = re.compile(
    r"(?:Article\s+\d+[a-zA-Z]?|Section\s+\d+[a-zA-Z]?)",
    re.IGNORECASE,
)

LAW_NAME_ONLY_PATTERN_EN = re.compile(
    r"(?:Law|Act|Code|Regulation|Decree)\s+[A-Z][A-Za-z0-9\- ]{2,80}",
    re.IGNORECASE,
)

MULTI_HOP_VERBS_EN = re.compile(
    r"(?:"
    r"born in|located in|founded by|directed by|written by|starring|"
    r"same nationality|same country|same place|part of|member of|"
    r"father of|mother of|spouse of|capital of|held by|portrayed by"
    r")",
    re.IGNORECASE,
)

DEEP_COMPARATIVE_EN = re.compile(
    r"(?:\bsame\b|\bdifferent\b|\bcompare\b|\bversus\b|\bvs\.?\b|\bolder\b|\byounger\b|\blarger\b|\bsmaller\b)",
    re.IGNORECASE,
)

FACTOID_PATTERNS_EN = re.compile(
    r"(?:\bwhat is\b|\bwho is\b|\bwhen was\b|\bwhere is\b|\bhow many\b|\bhow long\b)",
    re.IGNORECASE,
)

AUTHORITY_CHAIN_PATTERNS_EN = re.compile(
    r"(?:authority|jurisdiction|responsible for|authorized to|"
    r"who has the right|who is allowed|chain of command)",
    re.IGNORECASE,
)

LEGAL_EFFECT_PATTERNS_EN = re.compile(
    r"(?:repeal|replace|amend|supersede|no longer in effect|"
    r"come into force|transitional provision|prior law)",
    re.IGNORECASE,
)

STRONG_LEGAL_EFFECT_PATTERNS_EN = re.compile(
    r"(?:repeal|replace|amend|supersede|no longer in effect|"
    r"transitional provision|prior law)",
    re.IGNORECASE,
)

PROCEDURAL_SEQUENCE_PATTERNS_EN = re.compile(
    r"(?:procedure|process|steps|sequence|before.*can|"
    r"required to.*before|submit.*then|after approval)",
    re.IGNORECASE,
)

MULTI_ENTITY_RELATION_PATTERNS_EN = re.compile(
    r"(?:between.*and|relationship between|rights and obligations|"
    r"both.*and|responsibility of.*and)",
    re.IGNORECASE,
)

SIMPLE_EFFECTIVE_DATE_PATTERNS_EN = re.compile(
    r"(?:when.*(?:come into force|effective)|(?:come into force|effective).*when)",
    re.IGNORECASE,
)

# Entity density: legal entities pattern (high density → graph)
LEGAL_ENTITY_PATTERN = re.compile(
    r"(?:công ty|doanh nghiệp|cơ quan|tổ chức|cá nhân|người|bên|chủ thể)",
    re.IGNORECASE,
)


@dataclass
class ComplexityFeatures:
    """Features derived from query complexity analysis.

    These complement the basic QueryFeatures from features.py
    with higher-level semantic analysis.
    """

    # Core complexity level (1=Simple, 2=Multi-hop, 3=Complex)
    complexity_level: int = 1

    # Sub-question decomposability
    sub_question_count: int = 0

    # Entity density (entity count / token count)
    entity_density: float = 0.0

    # Law specificity signal
    # 2 = Specific article cited (dense_retrieval signal)
    # 1 = Only law name (mid-specificity)
    # 0 = No legal reference (need more context)
    law_specificity: int = 0

    # Conditional nesting depth
    conditional_depth: int = 0

    # Is factoid (direct definition lookup)
    is_factoid: bool = False

    # Multi-hop verb count
    multi_hop_verb_count: int = 0

    # Deep comparative count
    comparative_depth: int = 0

    # Graph-signal features for legal routing
    authority_chain_count: int = 0
    legal_effect_count: int = 0
    procedural_count: int = 0
    multi_entity_relation_count: int = 0

    def to_vector(self) -> list[float]:
        """Convert to flat numeric vector for XGBoost."""
        return [
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

    @staticmethod
    def feature_names() -> list[str]:
        return [
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


class QueryComplexityAnalyzer:
    """Analyzes query complexity for adaptive routing.

    Implements the 3-level complexity classification from Adaptive-RAG paper.
    Uses heuristic patterns specifically designed for Vietnamese legal text.

    Complexity levels follow Adaptive-RAG taxonomy:
      1 = Simple factoid (Điều X quy định gì?)
      2 = Multi-hop within document (so sánh Điều X và Điều Y cùng luật)
      3 = Cross-document (kết hợp Luật A và Nghị định B)
    """

    def analyze(self, query: str, entity_count: int = 0, language: str = "vi") -> ComplexityFeatures:
        """Analyze query complexity.

        Args:
            query: Vietnamese legal query text.
            entity_count: Number of named entities (from NER, optional).

        Returns:
            ComplexityFeatures with all complexity signals.
        """
        tokens = query.split()
        token_count = max(len(tokens), 1)

        # --- Feature extraction ---

        if language == "en":
            sub_question_pattern = SUB_QUESTION_PATTERNS_EN
            conditional_pattern = CONDITIONAL_PATTERNS_EN
            specific_article_pattern = SPECIFIC_ARTICLE_PATTERN_EN
            law_name_pattern = LAW_NAME_ONLY_PATTERN_EN
            multi_hop_pattern = MULTI_HOP_VERBS_EN
            comparative_pattern = DEEP_COMPARATIVE_EN
            factoid_pattern = FACTOID_PATTERNS_EN
            authority_chain_pattern = AUTHORITY_CHAIN_PATTERNS_EN
            legal_effect_pattern = LEGAL_EFFECT_PATTERNS_EN
            strong_legal_effect_pattern = STRONG_LEGAL_EFFECT_PATTERNS_EN
            procedural_pattern = PROCEDURAL_SEQUENCE_PATTERNS_EN
            multi_entity_relation_pattern = MULTI_ENTITY_RELATION_PATTERNS_EN
            simple_effective_date_pattern = SIMPLE_EFFECTIVE_DATE_PATTERNS_EN
        else:
            sub_question_pattern = SUB_QUESTION_PATTERNS
            conditional_pattern = CONDITIONAL_PATTERNS
            specific_article_pattern = SPECIFIC_ARTICLE_PATTERN
            law_name_pattern = LAW_NAME_ONLY_PATTERN
            multi_hop_pattern = MULTI_HOP_VERBS
            comparative_pattern = DEEP_COMPARATIVE
            factoid_pattern = FACTOID_PATTERNS
            authority_chain_pattern = AUTHORITY_CHAIN_PATTERNS
            legal_effect_pattern = LEGAL_EFFECT_PATTERNS
            strong_legal_effect_pattern = STRONG_LEGAL_EFFECT_PATTERNS
            procedural_pattern = PROCEDURAL_SEQUENCE_PATTERNS
            multi_entity_relation_pattern = MULTI_ENTITY_RELATION_PATTERNS
            simple_effective_date_pattern = SIMPLE_EFFECTIVE_DATE_PATTERNS

        # Sub-question count: count conjunctions that may split into sub-questions
        sub_q = len(sub_question_pattern.findall(query))

        # Entity density
        legal_entities_in_query = len(LEGAL_ENTITY_PATTERN.findall(query))
        total_entities = max(entity_count, legal_entities_in_query)
        entity_density = total_entities / token_count

        # Law specificity
        specific_articles = len(specific_article_pattern.findall(query))
        law_names = len(law_name_pattern.findall(query))
        law_specificity = 0
        if specific_article_pattern.search(query):
            law_specificity = 2
        elif law_name_pattern.search(query):
            law_specificity = 1

        conditional_depth = len(conditional_pattern.findall(query))
        
        # Modify graph keyword logic: conditionals in very short/informal queries might be less indicative of graph traversal
        # We can use a proxy like the lack of law_specificity to downweight it slightly if needed, but since we don't have access to the full feature extractor here we'll just check if it's highly specific.
        if law_specificity == 0 and conditional_depth > 0:
            # We don't have legal_reference_count directly here, so we use law_specificity as a proxy.
            # If no specific law is mentioned, informal conditionals shouldn't dominate.
            pass # Keep it for now, but rely on the TwoStageRouter override to catch the zero-reference cases.

        factoid = factoid_pattern.search(query) is not None
        is_factoid = bool(factoid_pattern.search(query))

        # Multi-hop verbs
        multi_hop_verbs = len(multi_hop_pattern.findall(query))

        # Comparative depth
        comp_depth = len(comparative_pattern.findall(query))

        # Domain-specific graph signals
        authority_chain = len(authority_chain_pattern.findall(query))
        legal_effect = len(legal_effect_pattern.findall(query))
        strong_legal_effect = len(strong_legal_effect_pattern.findall(query))
        procedural = len(procedural_pattern.findall(query))
        multi_entity_rel = len(multi_entity_relation_pattern.findall(query))
        simple_effective_date = bool(simple_effective_date_pattern.search(query))

        if language != "en":
            organization_structure = len(ORGANIZATION_STRUCTURE_PATTERNS.findall(query))
            quantitative_norm = len(QUANTITATIVE_NORM_PATTERNS.findall(query))
            public_finance_plan = len(PUBLIC_FINANCE_PLAN_PATTERNS.findall(query))
            
            # Organization structure behaves like authority/institutional relation.
            authority_chain += organization_structure
            multi_entity_rel += 1 if organization_structure >= 2 else 0
            
            # Quantitative norm queries often require threshold/table selection.
            if quantitative_norm >= 2:
                sub_q = max(sub_q, 1)
                multi_entity_rel += 1
            
            # Public finance / project plan queries behave like procedural/eligibility reasoning.
            procedural += public_finance_plan
            if public_finance_plan >= 2 and conditional_depth >= 1:
                sub_q = max(sub_q, 1)

        # --- Complexity level classification (Adaptive-RAG 3-level) ---
        complexity_level = self._classify_complexity(
            specific_articles=specific_articles,
            law_names=law_names,
            sub_q=sub_q,
            multi_hop_verbs=multi_hop_verbs,
            cond_depth=conditional_depth,
            is_factoid=is_factoid,
            comp_depth=comp_depth,
            token_count=token_count,
            authority_chain=authority_chain,
            legal_effect=legal_effect,
            strong_legal_effect=strong_legal_effect,
            procedural=procedural,
            multi_entity_rel=multi_entity_rel,
            simple_effective_date=simple_effective_date,
        )

        features = ComplexityFeatures(
            complexity_level=complexity_level,
            sub_question_count=sub_q,
            entity_density=round(entity_density, 4),
            law_specificity=law_specificity,
            conditional_depth=conditional_depth,
            is_factoid=is_factoid,
            multi_hop_verb_count=multi_hop_verbs,
            comparative_depth=comp_depth,
            authority_chain_count=authority_chain,
            legal_effect_count=legal_effect,
            procedural_count=procedural,
            multi_entity_relation_count=multi_entity_rel,
        )

        logger.debug(
            "Complexity | level={} | sub_q={} | law_spec={} | factoid={} | cond={} | "
            "auth={} | effect={} | proc={} | rel={}",
            complexity_level,
            sub_q,
            law_specificity,
            is_factoid,
            conditional_depth,
            authority_chain,
            legal_effect,
            procedural,
            multi_entity_rel,
        )

        return features

    def _classify_complexity(
        self,
        specific_articles: int,
        law_names: int,
        sub_q: int,
        multi_hop_verbs: int,
        cond_depth: int,
        is_factoid: bool,
        comp_depth: int,
        token_count: int,
        authority_chain: int,
        legal_effect: int,
        strong_legal_effect: int,
        procedural: int,
        multi_entity_rel: int,
        simple_effective_date: bool,
    ) -> int:
        """Classify into levels 1, 2, 3 following Adaptive-RAG taxonomy.

        Decision logic is deliberately rule-based so it is interpretable
        for the thesis write-up.

        Returns:
            1, 2, or 3.
        """
        # Level 3: Cross-document complexity
        # Signals: multiple law names, multi-hop verbs, deep conditionals
        cross_doc_signals = (
            law_names >= 2
            or multi_hop_verbs >= 2
            or strong_legal_effect >= 2
            or (cond_depth >= 2 and law_names >= 1)
            or (sub_q >= 3 and specific_articles == 0)
            or (multi_entity_rel >= 1 and law_names >= 2)
        )
        if cross_doc_signals:
            return 3

        if (
            simple_effective_date
            and strong_legal_effect == 0
            and authority_chain == 0
            and multi_entity_rel == 0
            and sub_q == 0
            and procedural <= 1
        ):
            return 1

        # Level 2: Multi-hop within same document
        # Signals: sub-questions + article refs, comparative, multi-hop verbs
        # Plain effective-date questions ("Nghị định X có hiệu lực khi nào?")
        # remain Level 1 unless paired with transition/amendment/procedure signals.
        multi_hop_signals = (
            (specific_articles >= 2 and sub_q >= 1)
            or multi_hop_verbs >= 1
            or comp_depth >= 1
            or (token_count > 20 and specific_articles >= 1 and sub_q >= 1)
            or (authority_chain >= 1 and (sub_q >= 1 or token_count > 16 or specific_articles >= 1))
            or strong_legal_effect >= 1
            or (legal_effect >= 1 and (sub_q >= 1 or cond_depth >= 1 or specific_articles >= 1))
            or procedural >= 2
            or (procedural >= 1 and (sub_q >= 1 or authority_chain >= 1 or token_count > 20))
            or multi_entity_rel >= 1
            or (authority_chain >= 1 and token_count > 12)
            or (procedural >= 1 and cond_depth >= 1)
            or (multi_entity_rel >= 1 and cond_depth >= 1)
        )
        if multi_hop_signals:
            return 2

        # Level 1: Simple factoid lookup
        return 1

    @staticmethod
    def level_to_route_hint(level: int) -> str:
        """Map complexity level to a routing hint string.

        Not a hard routing decision — just a signal fed into features.

        Args:
            level: Complexity level 1, 2, or 3.

        Returns:
            Route hint string.
        """
        mapping = {
            1: "dense_retrieval",
            2: "graph_traversal",
            3: "hybrid_reasoning",
        }
        return mapping.get(level, "dense_retrieval")
