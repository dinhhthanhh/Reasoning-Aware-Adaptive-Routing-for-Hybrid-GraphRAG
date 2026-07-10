"""Enriched query-feature extraction for the adaptive router.

This module merges two previously disconnected feature sources:

1. ``scripts.feature_extractor_fixed.VietnameseLegalFeatureExtractor`` — the
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

import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.feature_extractor_fixed import VietnameseLegalFeatureExtractor
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