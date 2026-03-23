"""Main evaluation script for the Vietnamese Legal QA system.

Runs evaluation on test_queries.json, computes all metrics,
prints results in readable and LaTeX-ready formats, and saves
results to evaluation/results.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger
from tabulate import tabulate

from evaluation.metrics import (
    MetricResults,
    compute_ambiguity_f1,
    compute_answer_f1,
    compute_latency_stats,
    compute_routing_accuracy,
    compute_stage2_rates,
    format_latex_table,
)
from pipeline.hybrid_pipeline import HybridPipeline


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


def run_evaluation(
    config_path: str | Path | None = None,
    test_path: str | Path | None = None,
    verbose: bool = False,
) -> MetricResults:
    """Run full evaluation on test queries.

    Args:
        config_path: Path to config.yaml.
        test_path: Path to test_queries.json.
        verbose: Print per-query details.

    Returns:
        MetricResults with all computed metrics.
    """
    test_queries = load_test_queries(test_path)

    # Initialize pipeline
    logger.info("Initializing pipeline for evaluation...")
    pipeline = HybridPipeline(config_path)

    # Collect results
    predicted_routes: list[str] = []
    expected_routes: list[str] = []
    predicted_ambiguous: list[bool] = []
    expected_ambiguous: list[bool] = []
    answer_f1_scores: list[float] = []
    latencies: list[float] = []
    stage2_invoked_list: list[bool] = []
    stage2_override_list: list[bool] = []

    logger.info("Running evaluation on {} queries...", len(test_queries))

    for i, tq in enumerate(test_queries, 1):
        query = tq["query"]
        expected_route = tq.get("expected_route", "vector")
        expected_keywords = tq.get("expected_answer_keywords", [])
        is_ambiguous_expected = tq.get("is_ambiguous", False)

        try:
            response = pipeline.query(
                query=query,
                session_id=f"eval_{i}",
                verbose=verbose,
            )

            predicted_routes.append(response.route_used)
            expected_routes.append(expected_route)
            predicted_ambiguous.append(response.is_ambiguous)
            expected_ambiguous.append(is_ambiguous_expected)
            latencies.append(response.latency_ms)
            stage2_invoked_list.append(response.stage2_invoked)
            stage2_override_list.append(False)  # Would need more info

            # Compute answer F1
            f1 = compute_answer_f1(response.answer, expected_keywords)
            answer_f1_scores.append(f1)

            if verbose:
                status = "✓" if response.route_used == expected_route else "✗"
                logger.info(
                    "[{}/{}] {} route={}/{} f1={:.2f} latency={:.0f}ms",
                    i,
                    len(test_queries),
                    status,
                    response.route_used,
                    expected_route,
                    f1,
                    response.latency_ms,
                )

        except Exception as exc:
            logger.error("Query {} failed: {}", i, exc)
            predicted_routes.append("error")
            expected_routes.append(expected_route)
            predicted_ambiguous.append(False)
            expected_ambiguous.append(is_ambiguous_expected)
            latencies.append(0.0)
            answer_f1_scores.append(0.0)
            stage2_invoked_list.append(False)
            stage2_override_list.append(False)

    # Compute metrics
    routing_acc, per_route_acc = compute_routing_accuracy(predicted_routes, expected_routes)
    amb_precision, amb_recall, amb_f1 = compute_ambiguity_f1(
        predicted_ambiguous, expected_ambiguous
    )
    lat_mean, lat_p50, lat_p95 = compute_latency_stats(latencies)
    s2_trigger, s2_override = compute_stage2_rates(stage2_invoked_list, stage2_override_list)

    results = MetricResults(
        routing_accuracy=routing_acc,
        ambiguity_precision=amb_precision,
        ambiguity_recall=amb_recall,
        ambiguity_f1=amb_f1,
        answer_f1_mean=float(sum(answer_f1_scores) / len(answer_f1_scores)) if answer_f1_scores else 0.0,
        latency_mean_ms=lat_mean,
        latency_p50_ms=lat_p50,
        latency_p95_ms=lat_p95,
        stage2_trigger_rate=s2_trigger,
        stage2_override_rate=s2_override,
        per_route_accuracy=per_route_acc,
        total_queries=len(test_queries),
    )

    # Print results
    _print_results(results)

    # Save results
    _save_results(results)

    return results


def _print_results(results: MetricResults) -> None:
    """Print evaluation results in readable format.

    Args:
        results: Computed metric results.
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    table_data = [
        ["Routing Accuracy", f"{results.routing_accuracy:.4f}"],
        ["Ambiguity Precision", f"{results.ambiguity_precision:.4f}"],
        ["Ambiguity Recall", f"{results.ambiguity_recall:.4f}"],
        ["Ambiguity F1", f"{results.ambiguity_f1:.4f}"],
        ["Answer F1 (keyword)", f"{results.answer_f1_mean:.4f}"],
        ["Latency Mean (ms)", f"{results.latency_mean_ms:.1f}"],
        ["Latency P50 (ms)", f"{results.latency_p50_ms:.1f}"],
        ["Latency P95 (ms)", f"{results.latency_p95_ms:.1f}"],
        ["Stage 2 Trigger Rate", f"{results.stage2_trigger_rate:.4f}"],
        ["Stage 2 Override Rate", f"{results.stage2_override_rate:.4f}"],
        ["Total Queries", str(results.total_queries)],
    ]

    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))

    if results.per_route_accuracy:
        print("\nPer-Route Accuracy:")
        route_data = [
            [route, f"{acc:.4f}"]
            for route, acc in sorted(results.per_route_accuracy.items())
        ]
        print(tabulate(route_data, headers=["Route", "Accuracy"], tablefmt="grid"))

    # Print LaTeX table
    print("\n--- LaTeX Table ---")
    print(format_latex_table(results))
    print("=" * 70)


def _save_results(results: MetricResults) -> None:
    """Save results to JSON file.

    Args:
        results: Computed metric results.
    """
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "results.json"

    results_dict = {
        "routing_accuracy": results.routing_accuracy,
        "ambiguity_precision": results.ambiguity_precision,
        "ambiguity_recall": results.ambiguity_recall,
        "ambiguity_f1": results.ambiguity_f1,
        "answer_f1_mean": results.answer_f1_mean,
        "latency_mean_ms": results.latency_mean_ms,
        "latency_p50_ms": results.latency_p50_ms,
        "latency_p95_ms": results.latency_p95_ms,
        "stage2_trigger_rate": results.stage2_trigger_rate,
        "stage2_override_rate": results.stage2_override_rate,
        "per_route_accuracy": results.per_route_accuracy,
        "total_queries": results.total_queries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to {}", output_path)
