"""Main evaluation script for the Vietnamese Legal QA system.

Runs evaluation on test_queries.json, computes all metrics,
prints results in readable and LaTeX-ready formats, and saves
results to evaluation/results.json.

Improvements over the original version (see review.md M1/M4/M5/M6):
  - Per-route-class F1 (groups F1 by gold route label) → tab:per_class_f1
  - Bootstrap 95% CI (B=1000) for F1/EM per system         → M5
  - Hit@k and MRR via id_normalizer (canonical law IDs)     → M3
  - BERTScore column via PhoBERT                             → M6
  - Latency decomposition per component (stage1/stage2/retrieval/generation)
  - Retrieved source-ID logging for debugging M3 ID mismatch
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tabulate import tabulate

from evaluation.metrics import (
    MetricResults,
    compute_ambiguity_f1,
    compute_answer_f1,
    compute_hit_at_k,
    compute_latency_stats,
    compute_mrr,
    compute_routing_accuracy,
    compute_stage2_rates,
    compute_token_f1,
    evaluate_prediction,
    format_latex_table,
    normalize_gold_article,
    normalize_legal_id,
)
from pipeline.hybrid_pipeline import HybridPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI (M5)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap CI for a list of per-query scores.

    Args:
        values: Per-query metric scores (e.g. F1 per query).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return mean, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Per-class F1 (M1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_class_f1(
    f1_scores: list[float],
    gold_route_labels: list[str],
) -> dict[str, dict[str, float]]:
    """Group F1 by gold route label, compute mean + bootstrap CI.

    Args:
        f1_scores: Per-query token F1 scores.
        gold_route_labels: Gold route label for each query.

    Returns:
        Dict mapping route label → {mean, ci_lo, ci_hi, n}.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for f1, label in zip(f1_scores, gold_route_labels):
        grouped[label].append(f1)

    result: dict[str, dict[str, float]] = {}
    for label, scores in grouped.items():
        mean, lo, hi = bootstrap_ci(scores)
        result[label] = {
            "mean": mean,
            "ci_lo_95": lo,
            "ci_hi_95": hi,
            "n": len(scores),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test_queries(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load test queries from JSON file.

    Args:
        path: Path to test_queries.json. Defaults to evaluation/test_queries.json.

    Returns:
        List of test query dicts.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "test_queries.json"

    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info("Loaded {} test queries from {}", len(queries), path)
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    config_path: str | Path | None = None,
    test_path: str | Path | None = None,
    verbose: bool = False,
    compute_bert: bool = False,
    n_bootstrap: int = 1000,
) -> MetricResults:
    """Run full evaluation on test queries.

    Args:
        config_path: Path to config.yaml.
        test_path: Path to test_queries.json.
        verbose: Print per-query details.
        compute_bert: Compute BERTScore (requires bert-score + torch).
        n_bootstrap: Number of bootstrap resamples for CI (M5).

    Returns:
        MetricResults with all computed metrics.
    """
    test_queries = load_test_queries(test_path)

    # Initialize pipeline
    logger.info("Initializing pipeline for evaluation...")
    pipeline = HybridPipeline(config_path)

    # Per-query accumulators
    predicted_routes: list[str] = []
    expected_routes: list[str] = []
    gold_route_labels: list[str] = []
    predicted_ambiguous: list[bool] = []
    expected_ambiguous: list[bool] = []

    # Answer metrics
    token_f1_scores: list[float] = []
    em_scores: list[float] = []
    answer_f1_scores: list[float] = []   # legacy keyword recall

    # Retrieval metrics (M3)
    hit_at1_scores: list[int] = []
    hit_at3_scores: list[int] = []
    hit_at5_scores: list[int] = []
    mrr_scores: list[float] = []

    # BERTScore accumulators (M6)
    predictions_for_bert: list[str] = []
    references_for_bert: list[str] = []

    # Latency decomposition (M5)
    latencies: list[float] = []
    stage1_latencies: list[float] = []
    stage2_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []

    # Stage-2 rates
    stage2_invoked_list: list[bool] = []
    stage2_override_list: list[bool] = []

    # Per-query log (for debugging M3, writing JSONL)
    per_query_log: list[dict[str, Any]] = []

    logger.info("Running evaluation on {} queries...", len(test_queries))

    for i, tq in enumerate(test_queries, 1):
        query = tq["query"]
        expected_route = tq.get("expected_route", "dense_retrieval")
        gold_route_label = tq.get("routing_label", expected_route)
        expected_keywords = tq.get("expected_answer_keywords", [])
        gold_answer = tq.get("answer", tq.get("gold_answer", ""))
        is_ambiguous_expected = tq.get("is_ambiguous", False)
        gold_sources = tq.get("relevant_articles", tq.get("gold_sources", []))

        try:
            response = pipeline.query(
                query=query,
                session_id=f"eval_{i}",
                verbose=verbose,
            )

            predicted_routes.append(response.route_used)
            expected_routes.append(expected_route)
            gold_route_labels.append(gold_route_label)
            predicted_ambiguous.append(response.is_ambiguous)
            expected_ambiguous.append(is_ambiguous_expected)

            # ── Answer metrics ────────────────────────────────────────────────
            answer_text = getattr(response, "answer", "") or ""
            em, tf1, _ = evaluate_prediction(gold_answer, answer_text)
            token_f1_scores.append(tf1)
            em_scores.append(float(em))

            # Legacy keyword recall (kept for backward compat)
            kw_f1 = compute_answer_f1(answer_text, expected_keywords)
            answer_f1_scores.append(kw_f1)

            # BERTScore data collection (M6) — only if gold answer exists
            if gold_answer:
                predictions_for_bert.append(answer_text)
                references_for_bert.append(gold_answer)

            # ── Retrieval metrics (M3) ────────────────────────────────────────
            retrieved_sources = getattr(response, "source_ids", []) or []
            hit1 = compute_hit_at_k(retrieved_sources, gold_sources, k=1, mode="strict")
            hit3 = compute_hit_at_k(retrieved_sources, gold_sources, k=3, mode="strict")
            hit5 = compute_hit_at_k(retrieved_sources, gold_sources, k=5, mode="strict")
            mrr = compute_mrr(retrieved_sources, gold_sources, mode="strict")
            hit_at1_scores.append(hit1)
            hit_at3_scores.append(hit3)
            hit_at5_scores.append(hit5)
            mrr_scores.append(mrr)

            # ── Latency decomposition (M5) ────────────────────────────────────
            total_ms = getattr(response, "latency_ms", 0.0) or 0.0
            latencies.append(total_ms)
            stage1_latencies.append(getattr(response, "stage1_latency_ms", 0.0) or 0.0)
            stage2_latencies.append(getattr(response, "stage2_latency_ms", 0.0) or 0.0)
            retrieval_latencies.append(getattr(response, "retrieval_latency_ms", 0.0) or 0.0)
            generation_latencies.append(getattr(response, "generation_latency_ms", 0.0) or 0.0)

            stage2_invoked_list.append(getattr(response, "stage2_invoked", False))
            stage2_override_list.append(False)

            # ── Per-query log entry ───────────────────────────────────────────
            log_entry: dict[str, Any] = {
                "query_id": i,
                "query": query,
                "predicted_route": response.route_used,
                "expected_route": expected_route,
                "gold_route_label": gold_route_label,
                "route_correct": response.route_used == expected_route,
                "token_f1": tf1,
                "exact_match": em,
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "hit_at_5": hit5,
                "mrr": mrr,
                "latency_ms": total_ms,
                "stage1_latency_ms": stage1_latencies[-1],
                "stage2_latency_ms": stage2_latencies[-1],
                "retrieval_latency_ms": retrieval_latencies[-1],
                "generation_latency_ms": generation_latencies[-1],
                "stage2_invoked": stage2_invoked_list[-1],
                "is_ambiguous_pred": response.is_ambiguous,
                "is_ambiguous_gold": is_ambiguous_expected,
                # ID debugging (M3)
                "retrieved_source_ids": [str(s) for s in retrieved_sources[:10]],
                "gold_source_ids": [str(g) for g in gold_sources[:10]],
            }
            per_query_log.append(log_entry)

            if verbose:
                status = "✓" if response.route_used == expected_route else "✗"
                logger.info(
                    "[{}/{}] {} route={}/{} tf1={:.3f} em={:.0f} hit@1={} mrr={:.3f} lat={:.0f}ms",
                    i, len(test_queries), status,
                    response.route_used, expected_route,
                    tf1, em, hit1, mrr, total_ms,
                )

        except Exception as exc:
            logger.error("Query {} failed: {}", i, exc)
            predicted_routes.append("error")
            expected_routes.append(expected_route)
            gold_route_labels.append(gold_route_label)
            predicted_ambiguous.append(False)
            expected_ambiguous.append(is_ambiguous_expected)
            latencies.append(0.0)
            token_f1_scores.append(0.0)
            em_scores.append(0.0)
            answer_f1_scores.append(0.0)
            hit_at1_scores.append(0)
            hit_at3_scores.append(0)
            hit_at5_scores.append(0)
            mrr_scores.append(0.0)
            stage1_latencies.append(0.0)
            stage2_latencies.append(0.0)
            retrieval_latencies.append(0.0)
            generation_latencies.append(0.0)
            stage2_invoked_list.append(False)
            stage2_override_list.append(False)
            per_query_log.append({
                "query_id": i, "query": query, "error": str(exc),
                "predicted_route": "error", "expected_route": expected_route,
                "gold_route_label": gold_route_label,
            })

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    routing_acc, per_route_acc = compute_routing_accuracy(predicted_routes, expected_routes)
    amb_precision, amb_recall, amb_f1 = compute_ambiguity_f1(
        predicted_ambiguous, expected_ambiguous
    )
    lat_mean, lat_p50, lat_p95 = compute_latency_stats(latencies)
    s2_trigger, s2_override = compute_stage2_rates(stage2_invoked_list, stage2_override_list)

    # Bootstrap CIs (M5)
    tf1_mean, tf1_lo, tf1_hi = bootstrap_ci(token_f1_scores, n_bootstrap=n_bootstrap)
    em_mean, em_lo, em_hi = bootstrap_ci(em_scores, n_bootstrap=n_bootstrap)
    hit1_mean, hit1_lo, hit1_hi = bootstrap_ci([float(h) for h in hit_at1_scores], n_bootstrap=n_bootstrap)
    mrr_mean, mrr_lo, mrr_hi = bootstrap_ci(mrr_scores, n_bootstrap=n_bootstrap)

    # Per-class F1 (M1)
    per_class_f1 = compute_per_class_f1(token_f1_scores, gold_route_labels)

    # Latency decomposition summary
    lat_decomp = {
        "stage1_mean_ms": float(np.mean(stage1_latencies)) if stage1_latencies else 0.0,
        "stage2_mean_ms": float(np.mean(stage2_latencies)) if stage2_latencies else 0.0,
        "retrieval_mean_ms": float(np.mean(retrieval_latencies)) if retrieval_latencies else 0.0,
        "generation_mean_ms": float(np.mean(generation_latencies)) if generation_latencies else 0.0,
    }

    # BERTScore (M6, optional — skip if not requested or not installed)
    bert_result: dict[str, Any] = {}
    if compute_bert and predictions_for_bert:
        try:
            from evaluation.metrics.bertscore_eval import compute_bertscore
            bert_result = compute_bertscore(
                predictions_for_bert,
                references_for_bert,
                reference_field="gold_answer",
            )
            logger.info("BERTScore F1 = {:.4f}", bert_result.get("f1", 0.0))
        except ImportError:
            logger.warning("bert-score not installed — BERTScore skipped. pip install bert-score")

    results = MetricResults(
        routing_accuracy=routing_acc,
        ambiguity_precision=amb_precision,
        ambiguity_recall=amb_recall,
        ambiguity_f1=amb_f1,
        answer_f1_mean=float(tf1_mean),
        latency_mean_ms=lat_mean,
        latency_p50_ms=lat_p50,
        latency_p95_ms=lat_p95,
        stage2_trigger_rate=s2_trigger,
        stage2_override_rate=s2_override,
        per_route_accuracy=per_route_acc,
        total_queries=len(test_queries),
    )

    # Print results
    _print_results(
        results,
        tf1_ci=(tf1_mean, tf1_lo, tf1_hi),
        em_ci=(em_mean, em_lo, em_hi),
        hit1_ci=(hit1_mean, hit1_lo, hit1_hi),
        mrr_ci=(mrr_mean, mrr_lo, mrr_hi),
        per_class_f1=per_class_f1,
        lat_decomp=lat_decomp,
        bert_result=bert_result,
    )

    # Save extended results
    _save_results(
        results,
        per_query_log=per_query_log,
        tf1_ci=(tf1_mean, tf1_lo, tf1_hi),
        em_ci=(em_mean, em_lo, em_hi),
        hit1_ci=(hit1_mean, hit1_lo, hit1_hi),
        mrr_ci=(mrr_mean, mrr_lo, mrr_hi),
        per_class_f1=per_class_f1,
        lat_decomp=lat_decomp,
        bert_result=bert_result,
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(
    results: MetricResults,
    tf1_ci: tuple[float, float, float] | None = None,
    em_ci: tuple[float, float, float] | None = None,
    hit1_ci: tuple[float, float, float] | None = None,
    mrr_ci: tuple[float, float, float] | None = None,
    per_class_f1: dict[str, dict[str, float]] | None = None,
    lat_decomp: dict[str, float] | None = None,
    bert_result: dict[str, Any] | None = None,
) -> None:
    """Print evaluation results in readable format."""
    print("\n" + "=" * 75)
    print("EVALUATION RESULTS")
    print("=" * 75)

    def _ci_str(ci: tuple[float, float, float] | None) -> str:
        if ci is None:
            return ""
        return f"[{ci[1]:.4f}, {ci[2]:.4f}]"

    table_data = [
        ["Routing Accuracy", f"{results.routing_accuracy:.4f}", ""],
        ["Ambiguity Precision", f"{results.ambiguity_precision:.4f}", ""],
        ["Ambiguity Recall", f"{results.ambiguity_recall:.4f}", ""],
        ["Ambiguity F1", f"{results.ambiguity_f1:.4f}", ""],
        ["Token F1 (Vietnamese)", f"{tf1_ci[0]:.4f}" if tf1_ci else f"{results.answer_f1_mean:.4f}",
         _ci_str(tf1_ci)],
        ["Exact Match", f"{em_ci[0]:.4f}" if em_ci else "—", _ci_str(em_ci)],
        ["Hit@1", f"{hit1_ci[0]:.4f}" if hit1_ci else "—", _ci_str(hit1_ci)],
        ["MRR", f"{mrr_ci[0]:.4f}" if mrr_ci else "—", _ci_str(mrr_ci)],
        ["Latency Mean (ms)", f"{results.latency_mean_ms:.1f}", ""],
        ["Latency P50 (ms)", f"{results.latency_p50_ms:.1f}", ""],
        ["Latency P95 (ms)", f"{results.latency_p95_ms:.1f}", ""],
        ["Stage 2 Trigger Rate", f"{results.stage2_trigger_rate:.4f}", ""],
        ["Stage 2 Override Rate", f"{results.stage2_override_rate:.4f}", ""],
        ["Total Queries", str(results.total_queries), ""],
    ]

    print(tabulate(table_data, headers=["Metric", "Value", "95% CI"], tablefmt="grid"))

    # Per-class F1 (M1 — tab:per_class_f1)
    if per_class_f1:
        print("\n--- Per-Route-Class F1 (tab:per_class_f1) ---")
        pc_data = [
            [label,
             f"{v['mean']:.4f}",
             f"[{v['ci_lo_95']:.4f}, {v['ci_hi_95']:.4f}]",
             int(v["n"])]
            for label, v in sorted(per_class_f1.items())
        ]
        print(tabulate(pc_data, headers=["Route Class", "Mean F1", "95% CI", "N"], tablefmt="grid"))

    # Latency decomposition
    if lat_decomp:
        print("\n--- Latency Decomposition (tab:latency_decomp) ---")
        ld_data = [
            ["Stage-1 routing", f"{lat_decomp.get('stage1_mean_ms', 0):.1f}"],
            ["Stage-2 reasoning", f"{lat_decomp.get('stage2_mean_ms', 0):.1f}"],
            ["Retrieval", f"{lat_decomp.get('retrieval_mean_ms', 0):.1f}"],
            ["Generation", f"{lat_decomp.get('generation_mean_ms', 0):.1f}"],
        ]
        print(tabulate(ld_data, headers=["Component", "Mean (ms)"], tablefmt="grid"))

    # BERTScore (M6)
    if bert_result and bert_result.get("n", 0) > 0:
        print(f"\n--- BERTScore (PhoBERT, n={bert_result['n']}) ---")
        print(f"  P = {bert_result['precision']:.4f} ± {bert_result['precision_std']:.4f}")
        print(f"  R = {bert_result['recall']:.4f} ± {bert_result['recall_std']:.4f}")
        print(f"  F1= {bert_result['f1']:.4f} ± {bert_result['f1_std']:.4f}")

    # Per-route accuracy
    if results.per_route_accuracy:
        print("\nPer-Route Accuracy:")
        route_data = [
            [route, f"{acc:.4f}"]
            for route, acc in sorted(results.per_route_accuracy.items())
        ]
        print(tabulate(route_data, headers=["Route", "Accuracy"], tablefmt="grid"))

    # LaTeX table
    print("\n--- LaTeX Table ---")
    print(format_latex_table(results))
    print("=" * 75)


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(
    results: MetricResults,
    per_query_log: list[dict[str, Any]] | None = None,
    tf1_ci: tuple[float, float, float] | None = None,
    em_ci: tuple[float, float, float] | None = None,
    hit1_ci: tuple[float, float, float] | None = None,
    mrr_ci: tuple[float, float, float] | None = None,
    per_class_f1: dict[str, dict[str, float]] | None = None,
    lat_decomp: dict[str, float] | None = None,
    bert_result: dict[str, Any] | None = None,
) -> None:
    """Save results to JSON file and per-query log to JSONL."""
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "results.json"
    log_path = output_dir / "per_query_log.jsonl"

    def _ci_dict(ci: tuple[float, float, float] | None) -> dict[str, float]:
        if ci is None:
            return {}
        return {"mean": ci[0], "ci_lo_95": ci[1], "ci_hi_95": ci[2]}

    results_dict: dict[str, Any] = {
        "routing_accuracy": results.routing_accuracy,
        "ambiguity_precision": results.ambiguity_precision,
        "ambiguity_recall": results.ambiguity_recall,
        "ambiguity_f1": results.ambiguity_f1,
        # Token F1 with bootstrap CI (M5)
        "token_f1": _ci_dict(tf1_ci),
        "exact_match": _ci_dict(em_ci),
        # Retrieval (M3)
        "hit_at_1": _ci_dict(hit1_ci),
        "mrr": _ci_dict(mrr_ci),
        # Per-class F1 (M1)
        "per_class_f1": per_class_f1 or {},
        # Legacy
        "answer_f1_mean": results.answer_f1_mean,
        "latency_mean_ms": results.latency_mean_ms,
        "latency_p50_ms": results.latency_p50_ms,
        "latency_p95_ms": results.latency_p95_ms,
        "latency_decomp": lat_decomp or {},
        "stage2_trigger_rate": results.stage2_trigger_rate,
        "stage2_override_rate": results.stage2_override_rate,
        "per_route_accuracy": results.per_route_accuracy,
        "total_queries": results.total_queries,
        # BERTScore (M6)
        "bertscore": bert_result or {},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logger.info("Aggregate results saved to {}", output_path)

    # Per-query JSONL log (for bootstrap re-use and debugging M3)
    if per_query_log:
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in per_query_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Per-query log saved to {} ({} entries)", log_path, len(per_query_log))
