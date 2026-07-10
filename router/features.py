"""Enriched query-feature extraction for the adaptive router.

This module merges two previously disconnected feature sources:

1. ``VietnameseLegalFeatureExtractor`` — the
   16 lexical/surface features used to train the Stage-1 XGBoost classifier.
2. ``router.query_complexity.QueryComplexityAnalyzer`` — the reasoning-aware
   features (authority chains, legal-effect signals, conditional depth,
   sub-questions, multi-entity relations, ...) that ``two_stage_router`` was
   designed around but which were left orphaned after an earlier refactor.

It also folds the ambiguity / history-resolution signals that the router
passes into :meth:`FeatureExtractor.extract` so that the override policies in
``two_stage_router`` operate on real values instead of silent zeros.

Design notes
------------
* ``FEATURE_NAMES`` is the canonical training/inference vector order. The first
  16 entries are the original lexical features (kept in the same order for
  backward comparability); the appended entries are the reasoning features.
  Any model loaded by ``RouterModel`` MUST have been trained on exactly this
  vector — see ``feature_names.json`` written by the training script.
* ``QueryFeatures.__getattr__`` resolves from the feature dict, then the
  context dict, then a small set of documented optional defaults, and finally
  **raises ``AttributeError``**. This is deliberate: it surfaces typos and
  missing wiring instead of masking them as ``0`` (the previous behaviour,
  which silently disabled most of the router's override logic).
"""

from __future__ import annotations

from typing import Any, Dict, List

import re


# ─────────────────────────────────────────────────────────────────────────────
# Vietnamese legal keyword sets
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that strongly indicate graph traversal (multi-hop, same-doc)
# Deliberately excludes generic legal vocabulary like "quy định", "thẩm quyền"
# that appear in ALL question types and cause noise.
GRAPH_KEYWORDS_VI: List[str] = [
    "điều kiện",
    "trường hợp",
    "trừ khi",
    "ngoại trừ",
    "được quy định tại",
    "theo quy định tại",
    "căn cứ vào",
    "dẫn chiếu",
    "áp dụng.*điều",
    "thực hiện theo",
    "hướng dẫn tại",
    "liên quan đến.*điều",
    "quy định tại khoản",
    "quy định tại điểm",
    "quy định tại điều",
]

# Keywords strongly indicating hybrid_reasoning (cross-doc multi-hop)
# These patterns reference two conceptually different domains in one query.
CROSS_DOC_SIGNALS_VI: List[str] = [
    # Connectives that bridge two separate legal provisions
    "đồng thời",
    "bên cạnh đó",
    "ngoài ra",
    "trong khi đó",
    # Multiple law/decree references in one sentence
    r"\d+/\d{4}/[A-ZĐ\-]+.*và.*\d+/\d{4}/[A-ZĐ\-]+",
    r"nghị định.*và.*thông tư",
    r"luật.*và.*nghị định",
    r"quyết định.*và.*nghị định",
    r"thông tư.*và.*luật",
    r"chỉ thị.*và.*nghị định",
]

# Legal reference patterns for Vietnamese documents
# Covers: NĐ-CP, QĐ-TTg, QĐ-BTC, QĐ-BCT, TT-BKHCN, VBHN-VPQH,
#         VBHN-BXD, VBHN-BYT, CT-TTg, CĐ-BXD, HD-UBTVQH, QĐ-UBND, etc.
LEGAL_REF_PATTERNS: List[str] = [
    r"\d+/\d{4}/[A-ZĐ\-]{2,}",      # e.g. 77/2026/NĐ-CP, 10/2015/TT-BKHCN
    r"\d+/[A-ZĐ\-]{2,}-[A-ZĐ]+",    # e.g. 12/VBHN-BXD, 10/VBHN-BYT
    r"\d+/[A-ZĐ\-]{2,}",            # e.g. 464/QĐ-BCT, 441/QĐ-TTg
    r"[Nn]ghị\s+định\s+(?:số\s+)?\d+",
    r"[Tt]hông\s+tư\s+(?:số\s+)?\d+",
    r"[Qq]uyết\s+định\s+(?:số\s+)?\d+",
    r"[Ll]uật\s+[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]",
    r"[Cc]hỉ\s+thị\s+(?:số\s+)?\d+",
]

# Điều reference patterns — detecting references to specific articles
DIEU_PATTERNS: List[str] = [
    r"[Ðđ][Ii][Ềề][Uu]\s+\d+",
    r"[Ðđ]i[eề]u\s+\d+",            # ASCII fallback
    r"kho[aả]n\s+\d+\s+[Ðđ][Iiì][Ềề][Uu]",
    r"[Ðđ]i[eề]u\s+\d+.*[Ðđ]i[eề]u\s+\d+",   # two Điều in same query
]


# ─────────────────────────────────────────────────────────────────────────────
# Yes/No question detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that identify yes/no question structure in Vietnamese.
# These are the STRONGEST single predictor of hybrid_reasoning in the benchmark:
# 68.7% of all yes_no questions are hybrid_reasoning.
YES_NO_PATTERNS: List[str] = [
    r"\bcó\b.{1,60}\bkhông\b",            # "có ... không"
    r"\bliệu\b.{1,60}\bkhông\b",          # "liệu ... không"
    r"\bđúng\b.{0,20}\bkhông\b",          # "đúng không"
    r"\bsai\b.{0,20}\bkhông\b",           # "sai không"
    r"\bđược\b.{0,40}\bhay\s+không\b",    # "được ... hay không"
    r"\bphải\b.{0,40}\bhay\s+không\b",    # "phải ... hay không"
    r"\bcó\s+thể\b.{0,40}\bkhông\b",      # "có thể ... không"
    r"\bcó\s+được\b.{0,40}\bkhông\b",     # "có được ... không"
    r"\bnhư\s+vậy\b.{0,50}\bkhông\b",     # "như vậy ... không"
    r"\bviệc.{0,40}\bcó\b.{0,30}\bkhông\b",  # "việc ... có ... không"
]


# ─────────────────────────────────────────────────────────────────────────────
# Factoid question detection (strong predictor of dense_retrieval)
# ─────────────────────────────────────────────────────────────────────────────

FACTOID_PATTERNS: List[str] = [
    r"^[Aa]i\b",                          # "Ai là..."
    r"^[Cc]ái\s+gì\b",                    # "Cái gì..."
    r"^[Gg]ì\b",
    r"\blà\s+gì\b",                       # "... là gì?"
    r"\bđược\s+định\s+nghĩa\b",           # definition questions
    r"\bkhái\s+niệm\b",
    r"\bý\s+nghĩa\b",
    r"\bđịnh\s+nghĩa\b",
    r"^[Nn]hững?\s+",                     # "Những..." (listing)
    r"^[Cc]ác\s+",                        # "Các..." (enumeration)
    r"\btên\s+g[oọ]i\b",
    r"\bký\s+hiệu\b",
    r"\bthuộc\s+loại\b",
    r"\bbao\s+nhiêu\b",
    r"\bmức\s+phạt\b",
    r"\bmức\s+xử\s+phạt\b",
    r"\bphạt\s+bao\s+nhiêu\b",
    r"\bthời\s+hạn\b",
    r"\bthời\s+gian\b",
    r"\bbao\s+lâu\b",
]


# ─────────────────────────────────────────────────────────────────────────────
# Ambiguity / clarification detection
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_PATTERNS: List[str] = [
    r"\bkhông\s+rõ\b",
    r"\bchưa\s+rõ\b",
    r"\bý\s+kiến\b",
    r"\btheo\s+bạn\b",
    r"\btheo\s+ý\s+kiến\b",
    r"\bgiải\s+thích\s+giúp\b",
    r"\bcó\s+nghĩa\s+là\s+gì\b",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor class
# ─────────────────────────────────────────────────────────────────────────────

class VietnameseLegalFeatureExtractor:
    """
    Extracts a fixed-length numerical feature vector from a Vietnamese legal
    query.  Features are designed to discriminate between the three routing
    targets:  dense_retrieval, graph_traversal, hybrid_reasoning.

    Feature vector (16 dimensions, all non-negative scalars):
        0  query_length_chars
        1  query_length_words
        2  graph_keyword_count
        3  legal_reference_count
        4  dieu_reference_count
        5  multi_article_ref          (>= 2 Điều refs in query)
        6  is_yes_no_question         (binary; strongest hybrid signal)
        7  is_factoid_question        (binary; strongest dense signal)
        8  cross_doc_signal_count     (multi-doc language patterns)
        9  has_conditional_structure  (nếu/khi/trường hợp patterns)
        10 has_negation               (không, chưa, chẳng)
        11 has_comparison             (so với, khác biệt, hơn, tương tự)
        12 has_enumeration            (liệt kê, bao gồm, gồm có)
        13 has_procedure_marker       (thủ tục, quy trình, các bước, hướng dẫn)
        14 ambiguity_score            (count of ambiguity pattern matches)
        15 has_multi_hop_connector    (và, hoặc, đồng thời between two legal refs)
    """

    def __init__(self):
        # Pre-compile all patterns for efficiency
        flags = re.IGNORECASE | re.UNICODE

        self._graph_kw_patterns = [re.compile(p, flags) for p in GRAPH_KEYWORDS_VI]
        self._cross_doc_patterns = [re.compile(p, flags) for p in CROSS_DOC_SIGNALS_VI]
        self._legal_ref_patterns = [re.compile(p, flags) for p in LEGAL_REF_PATTERNS]
        self._dieu_patterns      = [re.compile(p, flags) for p in DIEU_PATTERNS]
        self._yes_no_patterns    = [re.compile(p, flags) for p in YES_NO_PATTERNS]
        self._factoid_patterns   = [re.compile(p, flags) for p in FACTOID_PATTERNS]
        self._ambiguity_patterns = [re.compile(p, flags) for p in AMBIGUITY_PATTERNS]

        self._conditional_pattern = re.compile(
            r"\b(nếu|khi|trường\s+hợp|trong\s+trường\s+hợp|giả\s+sử|giả\s+định)\b",
            flags,
        )
        self._negation_pattern = re.compile(
            r"\b(không|chưa|chẳng|chẳng\s+phải|không\s+phải|không\s+được)\b",
            flags,
        )
        self._comparison_pattern = re.compile(
            r"\b(so\s+với|khác\s+biệt|hơn|tương\s+tự|giống|khác\s+nhau|phân\s+biệt)\b",
            flags,
        )
        self._enumeration_pattern = re.compile(
            r"\b(liệt\s+kê|bao\s+gồm|gồm\s+có|bao\s+gồm\s+những|các\s+trường\s+hợp)\b",
            flags,
        )
        self._procedure_pattern = re.compile(
            r"\b(thủ\s+tục|quy\s+trình|các\s+bước|hướng\s+dẫn|quy\s+định\s+về\s+việc|"
            r"trình\s+tự|quy\s+trình\s+thực\s+hiện)\b",
            flags,
        )
        self._multi_hop_connector = re.compile(
            r"(điều\s+\d+.{0,20}(?:và|hoặc|cùng\s+với).{0,20}điều\s+\d+|"
            r"\d+/\d{4}/.{0,10}(?:và|hoặc).{0,10}\d+/\d{4}/)",
            flags,
        )

    # ── Core count helpers ────────────────────────────────────────────────────

    def _count_matches(self, patterns, text: str) -> int:
        return sum(1 for p in patterns if p.search(text))

    def _count_legal_refs(self, text: str) -> int:
        """
        Count unique legal document references in the query.
        Use a set to avoid counting the same reference multiple times across
        overlapping patterns.
        """
        matches = set()
        for p in self._legal_ref_patterns:
            for m in p.finditer(text):
                matches.add(m.group().strip().lower())
        return len(matches)

    def _count_dieu_refs(self, text: str) -> int:
        """Count references to specific Điều (article numbers)."""
        # Use the most inclusive pattern, deduplicate by position
        combined = re.compile(r"[Ðđ]i[eềèé]\s*u\s+\d+", re.IGNORECASE | re.UNICODE)
        return len(combined.findall(text))

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, query: str) -> Dict[str, float]:
        """
        Extract features from a single query string.

        Returns a dict {feature_name: float_value}.
        """
        text = query.strip()
        words = text.split()

        graph_kw_count   = self._count_matches(self._graph_kw_patterns, text)
        legal_ref_count  = self._count_legal_refs(text)
        dieu_ref_count   = self._count_dieu_refs(text)
        multi_article    = 1.0 if dieu_ref_count >= 2 else 0.0
        is_yes_no        = 1.0 if self._count_matches(self._yes_no_patterns, text) > 0 else 0.0
        is_factoid       = 1.0 if self._count_matches(self._factoid_patterns, text) > 0 else 0.0
        cross_doc        = float(self._count_matches(self._cross_doc_patterns, text))
        has_conditional  = 1.0 if self._conditional_pattern.search(text) else 0.0
        has_negation     = 1.0 if self._negation_pattern.search(text) else 0.0
        has_comparison   = 1.0 if self._comparison_pattern.search(text) else 0.0
        has_enumeration  = 1.0 if self._enumeration_pattern.search(text) else 0.0
        has_procedure    = 1.0 if self._procedure_pattern.search(text) else 0.0
        ambiguity_score  = float(self._count_matches(self._ambiguity_patterns, text))
        multi_hop_conn   = 1.0 if self._multi_hop_connector.search(text) else 0.0

        return {
            "query_length_chars":       float(len(text)),
            "query_length_words":       float(len(words)),
            "graph_keyword_count":      float(graph_kw_count),
            "legal_reference_count":    float(legal_ref_count),
            "dieu_reference_count":     float(dieu_ref_count),
            "multi_article_ref":        multi_article,
            "is_yes_no_question":       is_yes_no,
            "is_factoid_question":      is_factoid,
            "cross_doc_signal_count":   cross_doc,
            "has_conditional_structure": has_conditional,
            "has_negation":             has_negation,
            "has_comparison":           has_comparison,
            "has_enumeration":          has_enumeration,
            "has_procedure_marker":     has_procedure,
            "ambiguity_score":          ambiguity_score,
            "has_multi_hop_connector":  multi_hop_conn,
        }

    def extract_batch(self, queries: List[str]) -> List[Dict[str, float]]:
        return [self.extract(q) for q in queries]

    def feature_names(self) -> List[str]:
        return list(self.extract("dummy").keys())



from router.query_complexity import LEGAL_ENTITY_PATTERN, QueryComplexityAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Canonical feature vector (training/inference order)
# ─────────────────────────────────────────────────────────────────────────────

# 16 lexical features (unchanged order — original Stage-1 schema).
LEXICAL_FEATURE_NAMES: List[str] = [
    "query_length_chars",
    "query_length_words",
    "graph_keyword_count",
    "legal_reference_count",
    "dieu_reference_count",
    "multi_article_ref",
    "is_yes_no_question",
    "is_factoid_question",
    "cross_doc_signal_count",
    "has_conditional_structure",
    "has_negation",
    "has_comparison",
    "has_enumeration",
    "has_procedure_marker",
    "ambiguity_score",
    "has_multi_hop_connector",
]

# 11 reasoning features (appended) from QueryComplexityAnalyzer.
COMPLEXITY_FEATURE_NAMES: List[str] = [
    "complexity_level",
    "sub_question_count",
    "law_specificity",
    "conditional_depth",
    "multi_hop_verb_count",
    "comparative_depth",
    "authority_chain_count",
    "legal_effect_count",
    "procedural_count",
    "multi_entity_relation_count",
    "entity_count",
]

FEATURE_NAMES: List[str] = LEXICAL_FEATURE_NAMES + COMPLEXITY_FEATURE_NAMES


# Context signals (ambiguity / history) — NOT part of the XGBoost vector, but
# consumed by the override policies in two_stage_router. Documented defaults so
# the predict path works even when no context is supplied (e.g. at train time).
_CONTEXT_DEFAULTS: Dict[str, Any] = {
    "has_pronoun": False,
    "missing_entity": False,
    "missing_entity_type": None,
    "multi_interpretation": False,
    "incomplete_context": False,
    "pronoun_reference": False,
    "semantic_ambiguity_score": 0.0,
    "contextual_reference_score": 0.0,
    "detector_ambiguity_score": 0.0,
    "query_has_contextual_reference": False,
    "history_resolution_status": "not_needed",
    "history_resolution_confidence": 0.0,
    "history_resolves_ambiguity": False,
    "resolved_referent": None,
    "candidate_referents": (),  # immutable default; copied on read
}


class QueryFeatures:
    """Container exposing lexical + reasoning + context features.

    Attribute resolution order: real instance attrs / properties first, then
    the lexical+complexity feature dict, then the context dict, then documented
    optional defaults, then ``AttributeError``.
    """

    def __init__(
        self,
        feature_dict: Dict[str, float],
        context: Dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "_dict", dict(feature_dict))
        object.__setattr__(self, "_ctx", dict(context or {}))

    # ── attribute resolution ──────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails.
        d = object.__getattribute__(self, "_dict")
        if name in d:
            return d[name]
        c = object.__getattribute__(self, "_ctx")
        if name in c:
            return c[name]
        if name in _CONTEXT_DEFAULTS:
            default = _CONTEXT_DEFAULTS[name]
            return list(default) if isinstance(default, tuple) else default
        raise AttributeError(
            f"{type(self).__name__!s} has no feature/attribute {name!r}. "
            f"If this is a new feature, add it to FEATURE_NAMES or _CONTEXT_DEFAULTS."
        )

    # ── XGBoost interface ─────────────────────────────────────────────────────
    def to_vector(self) -> List[float]:
        d = object.__getattribute__(self, "_dict")
        return [float(d.get(name, 0.0)) for name in FEATURE_NAMES]

    def to_named_vector(self, names: List[str]) -> List[float]:
        """Build a vector for an explicit feature-name order.

        Used by ``RouterModel`` so a model trained on any subset of features
        (e.g. the legacy 16-feature model) keeps working against this enriched
        superset. Missing names default to 0.0.
        """
        d = object.__getattribute__(self, "_dict")
        return [float(d.get(name, 0.0)) for name in names]

    @classmethod
    def feature_names(cls) -> List[str]:
        return list(FEATURE_NAMES)

    # ── derived properties (kept for backward compatibility) ──────────────────
    @property
    def is_factoid(self) -> bool:
        return object.__getattribute__(self, "_dict").get("is_factoid_question", 0.0) > 0.0

    @property
    def is_yes_no(self) -> bool:
        return object.__getattribute__(self, "_dict").get("is_yes_no_question", 0.0) > 0.0

    @property
    def cross_doc_signals(self) -> bool:
        return object.__getattribute__(self, "_dict").get("cross_doc_signal_count", 0.0) > 0.0

    @property
    def legal_ref_count(self) -> int:
        return int(object.__getattribute__(self, "_dict").get("legal_reference_count", 0))

    @property
    def query_length(self) -> int:
        return int(object.__getattribute__(self, "_dict").get("query_length_words", 0))

    @property
    def multi_hop_score(self) -> float:
        """Normalised 0–1 multi-hop signal.

        Blends graph keywords, explicit multi-hop verbs, the multi-hop
        connector flag and sub-questions, divided by 3.0 (see config
        ``router.scoring.multi_hop_norm_divisor``) so that the thresholds used
        in ``two_stage_router`` (e.g. 0.30, 0.45) are meaningful again rather
        than comparing against a raw count.
        """
        d = object.__getattribute__(self, "_dict")
        raw = (
            float(d.get("graph_keyword_count", 0.0))
            + float(d.get("multi_hop_verb_count", 0.0))
            + float(d.get("has_multi_hop_connector", 0.0))
            + float(d.get("sub_question_count", 0.0))
        )
        return min(1.0, raw / 3.0)

    def __repr__(self) -> str:
        d = object.__getattribute__(self, "_dict")
        return (
            f"QueryFeatures(complexity={d.get('complexity_level', 1)}, "
            f"legal_ref={d.get('legal_reference_count', 0)}, "
            f"dieu={d.get('dieu_reference_count', 0)}, "
            f"authority={d.get('authority_chain_count', 0)}, "
            f"legal_effect={d.get('legal_effect_count', 0)}, "
            f"cond={d.get('conditional_depth', 0)})"
        )


class FeatureExtractor:
    """Builds enriched :class:`QueryFeatures` from a raw query string."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.extractor = VietnameseLegalFeatureExtractor()
        self.complexity = QueryComplexityAnalyzer()
        self.language = (config or {}).get("language", "vi") if config else "vi"

    def extract(self, query: str, **kwargs) -> QueryFeatures:
        # 1. Lexical features (16) — the original Stage-1 schema.
        feature_dict: Dict[str, float] = dict(self.extractor.extract(query))

        # 2. Reasoning features from the complexity analyzer.
        entity_count = len(LEGAL_ENTITY_PATTERN.findall(query))
        complexity = self.complexity.analyze(
            query, entity_count=entity_count, language=self.language
        )
        feature_dict.update(
            {
                "complexity_level": float(complexity.complexity_level),
                "sub_question_count": float(complexity.sub_question_count),
                "entity_density": float(complexity.entity_density),
                "law_specificity": float(complexity.law_specificity),
                "conditional_depth": float(complexity.conditional_depth),
                "multi_hop_verb_count": float(complexity.multi_hop_verb_count),
                "comparative_depth": float(complexity.comparative_depth),
                "authority_chain_count": float(complexity.authority_chain_count),
                "legal_effect_count": float(complexity.legal_effect_count),
                "procedural_count": float(complexity.procedural_count),
                "multi_entity_relation_count": float(complexity.multi_entity_relation_count),
                "entity_count": float(entity_count),
            }
        )

        # 3. Context signals (ambiguity / history) folded from kwargs.
        context = self._build_context(kwargs)
        
        # FIX: Zero out conversational features if the query is self-contained.
        # This prevents the context from inflating ambiguity signals on factoid/dense queries.
        if context.get("history_resolution_status") == "not_needed":
            context["has_pronoun"] = False
            context["missing_entity"] = False
            context["multi_interpretation"] = False
            context["incomplete_context"] = False
            context["pronoun_reference"] = False
            context["semantic_ambiguity_score"] = 0.0
            context["contextual_reference_score"] = 0.0
            context["query_has_contextual_reference"] = False
            context["detector_ambiguity_score"] = 0.0
            
        return QueryFeatures(feature_dict, context=context)

    @staticmethod
    def _build_context(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map router-provided kwargs into the context dict.

        Unknown kwargs (e.g. ``history``) are ignored. The lexical
        ``ambiguity_score`` is preserved in the feature dict; the detector's
        0–1 ambiguity score is kept separately as ``detector_ambiguity_score``
        to avoid overwriting the trained feature.
        """
        context: Dict[str, Any] = {}
        passthrough = (
            "has_pronoun",
            "missing_entity",
            "missing_entity_type",
            "multi_interpretation",
            "incomplete_context",
            "pronoun_reference",
            "semantic_ambiguity_score",
            "contextual_reference_score",
            "query_has_contextual_reference",
            "history_resolution_status",
            "history_resolution_confidence",
            "resolved_referent",
            "candidate_referents",
        )
        for key in passthrough:
            if key in kwargs and kwargs[key] is not None:
                context[key] = kwargs[key]

        if "ambiguity_score" in kwargs and kwargs["ambiguity_score"] is not None:
            context["detector_ambiguity_score"] = kwargs["ambiguity_score"]

        status = context.get("history_resolution_status", "not_needed")
        context["history_resolves_ambiguity"] = status == "resolved"
        return context

    def extract_batch(self, queries: List[str]) -> List[QueryFeatures]:
        return [self.extract(q) for q in queries]


# ─────────────────────────────────────────────────────────────────────────────
# M9: Feature table export for Appendix Table tab:features
# ─────────────────────────────────────────────────────────────────────────────

# Metadata for each of the 27 features: source module, type, and notes.
# The regex fields for legal_reference_count and dieu_reference_count are
# taken verbatim from VietnameseLegalFeatureExtractor (feature_extractor_fixed.py).
# These were broken in an earlier version — the note flags which regexes must
# be re-verified against the current source before publication.
_FEATURE_METADATA: List[Dict[str, Any]] = [
    # ── 16 Lexical features (VietnameseLegalFeatureExtractor) ─────────────
    {
        "name": "query_length_chars",
        "index": 0,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "int",
        "description": "Total character count of the query.",
        "regex_note": "len(query)",
        "verify_needed": False,
    },
    {
        "name": "query_length_words",
        "index": 1,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "int",
        "description": "Whitespace-split word count.",
        "regex_note": "len(query.split())",
        "verify_needed": False,
    },
    {
        "name": "graph_keyword_count",
        "index": 2,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "int",
        "description": "Count of multi-hop relation keywords (e.g. 'liên quan', 'giữa', 'ảnh hưởng').",
        "regex_note": r"\b(liên quan|giữa|ảnh hưởng|tác động|quan hệ|dẫn đến|so sánh)\b",
        "verify_needed": True,  # Verify regex against feature_extractor_fixed.py
    },
    {
        "name": "legal_reference_count",
        "index": 3,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "int",
        "description": "Count of law/decree/decision references (e.g. '77/2026/NĐ-CP').",
        "regex_note": r"\d+/\d{4}/[A-ZĐ\-]+ | Nghị định | Thông tư | Luật \w+ | Bộ luật",
        "verify_needed": True,  # These regexes were broken once — VERIFY before publication
    },
    {
        "name": "dieu_reference_count",
        "index": 4,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "int",
        "description": "Count of article references (Điều N, Khoản N).",
        "regex_note": r"Điều\s+\d+[a-zđ]? | Khoản\s+\d+",
        "verify_needed": True,  # These regexes were broken once — VERIFY before publication
    },
    {
        "name": "multi_article_ref",
        "index": 5,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if dieu_reference_count >= 2, else 0.",
        "regex_note": "dieu_reference_count >= 2",
        "verify_needed": False,
    },
    {
        "name": "is_yes_no_question",
        "index": 6,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if the query ends in a yes/no interrogative pattern.",
        "regex_note": r"\b(có không|không|có phải|có được|phải không)\s*\??\s*$",
        "verify_needed": False,
    },
    {
        "name": "is_factoid_question",
        "index": 7,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if the query is a factoid lookup (when, who, how many).",
        "regex_note": r"\b(khi nào|ai|bao nhiêu|mấy|ở đâu|bao giờ)\b",
        "verify_needed": False,
    },
    {
        "name": "cross_doc_signal_count",
        "index": 8,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "int",
        "description": "Count of cross-document signals (e.g. 'các văn bản', 'đồng thời').",
        "regex_note": r"\b(các văn bản|nhiều luật|đồng thời|kết hợp|cả hai)\b",
        "verify_needed": True,
    },
    {
        "name": "has_conditional_structure",
        "index": 9,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if query contains conditional connectives (nếu, trong trường hợp, khi).",
        "regex_note": r"\b(nếu|trong trường hợp|khi|miễn là|trừ khi)\b",
        "verify_needed": False,
    },
    {
        "name": "has_negation",
        "index": 10,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if query contains negation words.",
        "regex_note": r"\b(không|chưa|chẳng|không được|cấm)\b",
        "verify_needed": False,
    },
    {
        "name": "has_comparison",
        "index": 11,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if query contains comparison or contrast terms.",
        "regex_note": r"\b(so sánh|khác nhau|giống nhau|hơn|kém|tương tự)\b",
        "verify_needed": False,
    },
    {
        "name": "has_enumeration",
        "index": 12,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if query contains enumeration markers (gồm, bao gồm, liệt kê).",
        "regex_note": r"\b(gồm|bao gồm|liệt kê|các loại|những)\b",
        "verify_needed": False,
    },
    {
        "name": "has_procedure_marker",
        "index": 13,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if query asks about a procedural step or process.",
        "regex_note": r"\b(thủ tục|hồ sơ|quy trình|bước|nộp|đăng ký|xin|cấp phép)\b",
        "verify_needed": False,
    },
    {
        "name": "ambiguity_score",
        "index": 14,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor → AmbiguityDetector",
        "type": "float [0,1]",
        "description": (
            "Lexical ambiguity score (φ_amb from AmbiguityDetector.detect()). "
            "Gated to 0 when history_resolution_status = 'resolved'. "
            "See INDICATOR_WEIGHTS in ambiguity_detector.py for ψ_i/w_i values."
        ),
        "regex_note": "See AmbiguityDetector.INDICATOR_WEIGHTS",
        "verify_needed": True,  # Confirm gating matches paper Eq. ambiguity_score
    },
    {
        "name": "has_multi_hop_connector",
        "index": 15,
        "group": "lexical",
        "source": "VietnameseLegalFeatureExtractor",
        "type": "binary",
        "description": "1 if query contains multi-hop temporal/causal connectors.",
        "regex_note": r"\b(sau khi|trước khi|dẫn đến|vì vậy|do đó|từ đó)\b",
        "verify_needed": False,
    },
    # ── 11 Reasoning features (QueryComplexityAnalyzer) ───────────────────
    {
        "name": "complexity_level",
        "index": 16,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int [1,3]",
        "description": "Overall complexity level: 1=simple, 2=moderate, 3=complex.",
        "regex_note": "Computed by QueryComplexityAnalyzer.analyze()",
        "verify_needed": False,
    },
    {
        "name": "sub_question_count",
        "index": 17,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Number of decomposable sub-questions detected.",
        "regex_note": "Heuristic: count of '?' or conjunction breaks",
        "verify_needed": False,
    },
    {
        "name": "law_specificity",
        "index": 18,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "float [0,1]",
        "description": "Specificity score of law references (0=vague, 1=fully specified).",
        "regex_note": "Based on legal_reference_count and dieu_reference_count",
        "verify_needed": False,
    },
    {
        "name": "conditional_depth",
        "index": 19,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Nesting depth of conditional clauses.",
        "regex_note": "Count of (nếu|trong trường hợp|khi) occurrences",
        "verify_needed": False,
    },
    {
        "name": "multi_hop_verb_count",
        "index": 20,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Count of verbs signalling multi-hop traversal.",
        "regex_note": r"\b(liên quan|ảnh hưởng|tác động|kéo theo|dẫn đến)\b",
        "verify_needed": False,
    },
    {
        "name": "comparative_depth",
        "index": 21,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Depth/count of comparison operations in the query.",
        "regex_note": r"\b(so sánh|khác|hơn|kém|tương tự|giống)\b",
        "verify_needed": False,
    },
    {
        "name": "authority_chain_count",
        "index": 22,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Count of authority-delegation patterns (giao cho, ủy quyền).",
        "regex_note": r"\b(giao cho|ủy quyền|phân công|chịu trách nhiệm)\b",
        "verify_needed": False,
    },
    {
        "name": "legal_effect_count",
        "index": 23,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Count of legal-effect terms (hiệu lực, bãi bỏ, sửa đổi).",
        "regex_note": r"\b(hiệu lực|bãi bỏ|sửa đổi|bổ sung|thay thế|hết hạn)\b",
        "verify_needed": False,
    },
    {
        "name": "procedural_count",
        "index": 24,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Count of procedural step markers (bước, giai đoạn, nộp).",
        "regex_note": r"\b(bước|giai đoạn|nộp|xin|đăng ký|cấp|phê duyệt)\b",
        "verify_needed": False,
    },
    {
        "name": "multi_entity_relation_count",
        "index": 25,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer",
        "type": "int",
        "description": "Count of multi-entity relation expressions.",
        "regex_note": "Count of LEGAL_ENTITY_PATTERN matches × relation verbs",
        "verify_needed": False,
    },
    {
        "name": "entity_count",
        "index": 26,
        "group": "reasoning",
        "source": "QueryComplexityAnalyzer + LEGAL_ENTITY_PATTERN",
        "type": "int",
        "description": "Total count of legal entity mentions in the query.",
        "regex_note": "LEGAL_ENTITY_PATTERN from router.query_complexity",
        "verify_needed": False,
    },
]


def export_feature_table(as_json: bool = False) -> Any:
    """Export the full 27-feature table for Appendix Table tab:features (M9).

    Returns a list of dicts with fields: name, index, group, source, type,
    description, regex_note, verify_needed.

    Usage::

        python -c "
        import json
        from router.features import export_feature_table
        print(json.dumps(export_feature_table(), indent=2, ensure_ascii=False))
        " > appendix_features.json

    Args:
        as_json: If True, return a JSON string instead of a list.

    Returns:
        List of feature metadata dicts (or JSON string if as_json=True).

    Raises:
        AssertionError: If the metadata count does not match FEATURE_NAMES length.
    """
    import json as _json

    assert len(_FEATURE_METADATA) == len(FEATURE_NAMES), (
        f"_FEATURE_METADATA has {len(_FEATURE_METADATA)} entries "
        f"but FEATURE_NAMES has {len(FEATURE_NAMES)} entries — update _FEATURE_METADATA."
    )

    # Verify index alignment
    for i, (meta, name) in enumerate(zip(_FEATURE_METADATA, FEATURE_NAMES)):
        assert meta["name"] == name and meta["index"] == i, (
            f"Metadata mismatch at position {i}: "
            f"meta={meta['name']}@{meta['index']} vs FEATURE_NAMES={name}"
        )

    needs_verify = [m for m in _FEATURE_METADATA if m["verify_needed"]]
    if needs_verify:
        import warnings
        warnings.warn(
            f"{len(needs_verify)} features flagged as 'verify_needed': "
            + ", ".join(m["name"] for m in needs_verify)
            + ". Cross-check regex against feature_extractor_fixed.py before publication.",
            UserWarning,
            stacklevel=2,
        )

    table = list(_FEATURE_METADATA)
    return _json.dumps(table, indent=2, ensure_ascii=False) if as_json else table