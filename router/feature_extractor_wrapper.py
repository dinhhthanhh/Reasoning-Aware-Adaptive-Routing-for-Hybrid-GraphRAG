"""
feature_extractor_wrapper.py
=============================
Backward-compatibility wrapper around VietnameseLegalFeatureExtractor.

Purpose
-------
The rest of the pipeline (router_model/two_stage_router.py, etc.) was written
against an older FeatureExtractor API that returned objects with properties
like .complexity_level, .is_factoid, .multi_hop_score, etc.

This module provides:
  - QueryFeatures  — a dict-backed object that exposes both the new 16-feature
                     dict AND the old named properties needed by router_model.py
  - FeatureExtractor — drop-in replacement for the old extractor class

Usage
-----
  from feature_extractor_wrapper import FeatureExtractor

  extractor = FeatureExtractor()
  features  = extractor.extract("Ai có thẩm quyền ký quyết định?")

  # New 16-feature dict (for XGBoost)
  x_vec = features.to_vector()

  # Old named property access (for router_model.py backward compat)
  level = features.complexity_level   # 1 | 2 | 3
  yn    = features.is_yes_no          # bool
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional

# ── Locate feature_extractor_fixed ───────────────────────────────────────────
try:
    from feature_extractor_fixed import VietnameseLegalFeatureExtractor
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.feature_extractor_fixed import VietnameseLegalFeatureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# ORDERED FEATURE NAMES (must match training order exactly)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
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


# ─────────────────────────────────────────────────────────────────────────────
# QueryFeatures
# ─────────────────────────────────────────────────────────────────────────────

class QueryFeatures:
    """
    Wraps a 16-feature dict and exposes backward-compatible named properties.

    Attribute lookup order
    ----------------------
    1. Instance attributes set by __init__ (the 16 feature floats + _dict)
    2. @property descriptors defined on the class (compat properties)
    3. __getattr__ fallback → returns 0 for any unknown old attribute name,
       so legacy code never gets AttributeError even for removed features.

    Note on __getattr__
    -------------------
    __getattr__ is ONLY called when normal attribute lookup fails.
    It is NOT called for attributes that exist (unlike __getattribute__).
    Therefore there is no infinite-recursion risk here.
    """

    def __init__(self, feature_dict: Dict[str, float]) -> None:
        # Store raw dict FIRST (before setattr loop, so @property fallbacks work)
        object.__setattr__(self, "_dict", feature_dict)
        # Expose each feature directly as an attribute for convenient dot-access
        for k, v in feature_dict.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name: str) -> Any:
        """
        Fallback for any attribute not found by normal lookup.
        Returns 0 (falsy int) so legacy code never raises AttributeError.
        """
        return 0

    # ── XGBoost interface ─────────────────────────────────────────────────────

    def to_vector(self) -> List[float]:
        """Return features in canonical training order for XGBoost predict."""
        d = object.__getattribute__(self, "_dict")
        return [d.get(name, 0.0) for name in FEATURE_NAMES]

    @classmethod
    def feature_names(cls) -> List[str]:
        return list(FEATURE_NAMES)

    # ── Backward-compat properties (old router_model.py API) ─────────────────

    @property
    def complexity_level(self) -> int:
        """
        Legacy 3-tier complexity signal.
          3 = cross-doc multi-hop  (→ hybrid_reasoning)
          2 = same-doc multi-hop   (→ graph_traversal)
          1 = single-hop           (→ dense_retrieval)

        Heuristic notes
        ---------------
        - cross_doc_signal_count > 0 is the only reliable level-3 indicator
          because it requires explicit references to two different legal documents.
        - has_multi_hop_connector (e.g. "Điều 18 và Điều 35") is a same-doc
          multi-hop signal → level 2, NOT level 3.
        - When cross_doc_signal_count = 0, XGBoost still classifies hybrid
          queries correctly via query_length (~60 words) + conditional structure.
          This heuristic is a best-effort fallback for legacy code only.
        """
        d = object.__getattribute__(self, "_dict")
        # Level 3: explicit references to two or more legal documents in the query
        if d.get("cross_doc_signal_count", 0) > 0:
            return 3
        # Level 2: same-doc multi-hop (multiple Điều refs, graph keywords,
        # or multi-hop connector "Điều X và Điều Y")
        if (d.get("graph_keyword_count", 0) > 0
                or d.get("multi_article_ref", 0) > 0
                or d.get("has_multi_hop_connector", 0) > 0):
            return 2
        return 1

    @property
    def is_factoid(self) -> bool:
        """Legacy name → is_factoid_question."""
        d = object.__getattribute__(self, "_dict")
        return d.get("is_factoid_question", 0.0) > 0.0

    @property
    def is_yes_no(self) -> bool:
        """Legacy name → is_yes_no_question."""
        d = object.__getattribute__(self, "_dict")
        return d.get("is_yes_no_question", 0.0) > 0.0

    @property
    def cross_doc_signals(self) -> bool:
        """Legacy name → cross_doc_signal_count > 0."""
        d = object.__getattribute__(self, "_dict")
        return d.get("cross_doc_signal_count", 0.0) > 0.0

    @property
    def multi_hop_score(self) -> float:
        """Legacy name → graph_keyword_count (continuous score)."""
        d = object.__getattribute__(self, "_dict")
        return d.get("graph_keyword_count", 0.0)

    @property
    def legal_ref_count(self) -> int:
        """Legacy name → legal_reference_count."""
        d = object.__getattribute__(self, "_dict")
        return int(d.get("legal_reference_count", 0))

    # Properties that had no equivalent in the old extractor → always False/0
    # These are kept so legacy code that references them doesn't break.
    @property
    def has_pronoun(self) -> bool:
        return False

    @property
    def missing_entity(self) -> bool:
        return False

    @property
    def multi_interpretation(self) -> bool:
        return False

    @property
    def incomplete_context(self) -> bool:
        return False

    @property
    def pronoun_reference(self) -> bool:
        return False

    def __repr__(self) -> str:
        d = object.__getattribute__(self, "_dict")
        top = sorted(d.items(), key=lambda x: -x[1])[:4]
        top_str = ", ".join(f"{k}={v:.1f}" for k, v in top)
        return f"QueryFeatures(complexity={self.complexity_level}, {top_str})"


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Drop-in replacement for the old FeatureExtractor class.

    Parameters
    ----------
    config : dict, optional
        Ignored — kept for API compatibility with code that passes config dicts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._extractor = VietnameseLegalFeatureExtractor()

    def extract(self, query: str, **kwargs) -> QueryFeatures:
        """
        Extract features from a query string.

        Parameters
        ----------
        query   : the raw query text
        **kwargs: ignored (kept for compat with callers that pass context, etc.)

        Returns
        -------
        QueryFeatures object
        """
        feat_dict = self._extractor.extract(query)
        return QueryFeatures(feat_dict)

    def extract_batch(self, queries: List[str]) -> List[QueryFeatures]:
        return [self.extract(q) for q in queries]

    @staticmethod
    def feature_names() -> List[str]:
        return list(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = FeatureExtractor()

    tests = [
        ("Ai có thẩm quyền quyết định thành lập đơn vị trực thuộc Cục Hoá chất?",
         "dense_retrieval", 1),
        ("Trong trường hợp địa phương chưa bố trí được nhân sự, có được gia hạn không?",
         "hybrid_reasoning", 3),
        ("Điều 18 và Điều 35 của văn bản 12/VBHN-BXD quy định gì?",
         "graph_traversal", 2),
    ]

    print(f"{'Query':<55} {'cmplx':>5} {'y_no':>4} {'fact':>4} {'xdoc':>4} | Expected")
    print("-" * 90)
    for query, expected_label, expected_level in tests:
        f = extractor.extract(query)
        match = "✓" if f.complexity_level == expected_level else "✗"
        print(
            f"  {query[:53]:<53} "
            f"{f.complexity_level:>5} "
            f"{int(f.is_yes_no):>4} "
            f"{int(f.is_factoid):>4} "
            f"{int(f.cross_doc_signals):>4} "
            f"| {expected_label} {match}"
        )

    # Verify to_vector order matches FEATURE_NAMES
    f = extractor.extract("test query")
    vec = f.to_vector()
    assert len(vec) == len(FEATURE_NAMES), f"Vector length {len(vec)} != {len(FEATURE_NAMES)}"
    print(f"\n  to_vector() length: {len(vec)} ✓")
    print(f"  __repr__: {f!r}")
    print(f"  __getattr__ fallback for unknown attr: f.old_unknown_attr = {f.old_unknown_attr}")
