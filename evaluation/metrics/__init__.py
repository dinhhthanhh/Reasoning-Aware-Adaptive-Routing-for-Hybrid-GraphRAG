"""Evaluation metrics package for the Vietnamese Legal QA system.

Public surface:

* Real metrics (use these for new work):
    - :func:`compute_token_f1`           - Vietnamese token-level P/R/F1/EM
    - :func:`compute_corpus_token_f1`    - macro-averaged token F1
    - :func:`normalize_legal_id`         - canonical legal-article IDs
    - :func:`compute_hit_at_k`, :func:`compute_mrr` - retrieval metrics
    - :func:`compute_bertscore`          - single canonical BERTScore
    - :func:`evaluate_prediction`        - (em, f1, contains) used by benchmark scripts

* Backward-compatible legacy API (kept for old scripts):
    - ``MetricResults``, ``compute_routing_accuracy``, ``compute_ambiguity_f1``,
      ``compute_answer_f1`` (DEPRECATED: returns keyword recall),
      ``compute_latency_stats``, ``compute_stage2_rates``, ``format_latex_table``
"""

from __future__ import annotations

# --- Legacy (backward compatible) -------------------------------------------
from evaluation.metrics.legacy import (
    MetricResults,
    compute_ambiguity_f1,
    compute_answer_f1,
    compute_latency_stats,
    compute_routing_accuracy,
    compute_stage2_rates,
    format_latex_table,
)

# --- New, correct metrics ----------------------------------------------------
from evaluation.metrics.token_f1 import (
    active_tokenizer,
    compute_corpus_token_f1,
    compute_token_f1,
    normalize_text,
    vi_tokenize,
)
from evaluation.metrics.id_normalizer import (
    CanonicalID,
    canonical_key,
    compute_hit_at_k,
    compute_mrr,
    normalize_gold_article,
    normalize_legal_id,
)
from evaluation.metrics.bertscore_eval import (
    BERTSCORE_LANG,
    BERTSCORE_MODEL,
    compute_bertscore,
)

__all__ = [
    # legacy
    "MetricResults",
    "compute_routing_accuracy",
    "compute_ambiguity_f1",
    "compute_answer_f1",
    "compute_latency_stats",
    "compute_stage2_rates",
    "format_latex_table",
    # token f1
    "compute_token_f1",
    "compute_corpus_token_f1",
    "vi_tokenize",
    "normalize_text",
    "active_tokenizer",
    # id normalization / retrieval
    "CanonicalID",
    "normalize_legal_id",
    "normalize_gold_article",
    "canonical_key",
    "compute_hit_at_k",
    "compute_mrr",
    # bertscore
    "compute_bertscore",
    "BERTSCORE_MODEL",
    "BERTSCORE_LANG",
    # convenience
    "evaluate_prediction",
]


def evaluate_prediction(ground_truth: str, answer: str) -> tuple[float, float, float]:
    """Score a single prediction the way benchmark scripts expect.

    This is the function ``scripts/run_benchmark_eval.py`` and
    ``scripts/run_oracle_eval.py`` try to import. Previously it was missing, so
    those scripts silently fell back to keyword recall. It now returns proper
    Vietnamese token-level metrics.

    Args:
        ground_truth: Reference answer text.
        answer: Generated answer text.

    Returns:
        Tuple ``(exact_match, token_f1, contains)`` where ``contains`` is 1.0 if
        the normalised ground truth is a substring of the normalised answer.
    """
    scores = compute_token_f1(answer or "", ground_truth or "")
    gt_norm = normalize_text(ground_truth or "")
    ans_norm = normalize_text(answer or "")
    contains = 1.0 if gt_norm and gt_norm in ans_norm else 0.0
    return float(scores["exact_match"]), float(scores["f1"]), contains
