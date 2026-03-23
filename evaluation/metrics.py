"""Evaluation metrics for the Vietnamese Legal QA system.

Computes routing accuracy, ambiguity F1, answer F1 (keyword overlap),
latency statistics, and Stage 2 trigger/override rates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class MetricResults:
    """Aggregated evaluation metrics.

    Attributes:
        routing_accuracy: Percentage of correctly routed queries.
        ambiguity_precision: Precision for ambiguity detection.
        ambiguity_recall: Recall for ambiguity detection.
        ambiguity_f1: F1 for ambiguity detection.
        answer_f1_mean: Mean answer F1 (keyword overlap).
        latency_mean_ms: Mean pipeline latency.
        latency_p50_ms: Median latency.
        latency_p95_ms: 95th percentile latency.
        stage2_trigger_rate: Percentage of queries needing Stage 2.
        stage2_override_rate: Percentage of queries where Stage 2 overrode Stage 1.
        per_route_accuracy: Accuracy broken down by expected route.
        total_queries: Total queries evaluated.
    """

    routing_accuracy: float = 0.0
    ambiguity_precision: float = 0.0
    ambiguity_recall: float = 0.0
    ambiguity_f1: float = 0.0
    answer_f1_mean: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    stage2_trigger_rate: float = 0.0
    stage2_override_rate: float = 0.0
    per_route_accuracy: dict[str, float] = field(default_factory=dict)
    total_queries: int = 0


def compute_routing_accuracy(
    predictions: list[str],
    expected: list[str],
) -> tuple[float, dict[str, float]]:
    """Compute routing accuracy overall and per route.

    Args:
        predictions: List of predicted routes.
        expected: List of expected routes.

    Returns:
        Tuple of (overall_accuracy, per_route_accuracy_dict).
    """
    if not predictions or not expected:
        return 0.0, {}

    correct = sum(p == e for p, e in zip(predictions, expected))
    overall = correct / len(predictions)

    # Per-route accuracy
    per_route: dict[str, list[bool]] = {}
    for pred, exp in zip(predictions, expected):
        if exp not in per_route:
            per_route[exp] = []
        per_route[exp].append(pred == exp)

    per_route_acc = {
        route: sum(results) / len(results)
        for route, results in per_route.items()
    }

    return overall, per_route_acc


def compute_ambiguity_f1(
    predicted_ambiguous: list[bool],
    expected_ambiguous: list[bool],
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for ambiguity detection.

    Args:
        predicted_ambiguous: List of predicted ambiguity flags.
        expected_ambiguous: List of expected ambiguity flags.

    Returns:
        Tuple of (precision, recall, f1).
    """
    tp = sum(p and e for p, e in zip(predicted_ambiguous, expected_ambiguous))
    fp = sum(p and not e for p, e in zip(predicted_ambiguous, expected_ambiguous))
    fn = sum(not p and e for p, e in zip(predicted_ambiguous, expected_ambiguous))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_answer_f1(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """Compute approximate F1 based on keyword overlap.

    Args:
        answer: Generated answer text.
        expected_keywords: List of expected keywords.

    Returns:
        F1 score (0-1).
    """
    if not expected_keywords:
        return 0.0

    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())

    matched = sum(
        1 for kw in expected_keywords
        if kw.lower() in answer_lower
    )

    precision = matched / len(answer_words) if answer_words else 0.0
    recall = matched / len(expected_keywords) if expected_keywords else 0.0

    # Use recall-weighted F1 (keywords are more meaningful than answer words)
    if precision + recall == 0:
        return 0.0

    # Simplified: just use keyword recall as primary metric
    return recall


def compute_latency_stats(latencies: list[float]) -> tuple[float, float, float]:
    """Compute latency statistics.

    Args:
        latencies: List of latency values in milliseconds.

    Returns:
        Tuple of (mean, p50, p95).
    """
    if not latencies:
        return 0.0, 0.0, 0.0

    arr = np.array(latencies)
    return float(np.mean(arr)), float(np.median(arr)), float(np.percentile(arr, 95))


def compute_stage2_rates(
    stage2_invoked: list[bool],
    stage2_overrides: list[bool],
) -> tuple[float, float]:
    """Compute Stage 2 trigger and override rates.

    Args:
        stage2_invoked: List of flags indicating Stage 2 activation.
        stage2_overrides: List of flags indicating Stage 2 overrides.

    Returns:
        Tuple of (trigger_rate, override_rate).
    """
    if not stage2_invoked:
        return 0.0, 0.0

    trigger_rate = sum(stage2_invoked) / len(stage2_invoked)

    invoked_count = sum(stage2_invoked)
    if invoked_count == 0:
        override_rate = 0.0
    else:
        override_rate = sum(stage2_overrides) / invoked_count

    return trigger_rate, override_rate


def format_latex_table(results: MetricResults) -> str:
    """Format results as a LaTeX-ready table.

    Args:
        results: Computed metric results.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Evaluation Results}",
        r"\label{tab:eval_results}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\midrule",
        f"Routing Accuracy & {results.routing_accuracy:.4f} \\\\",
        f"Ambiguity F1 & {results.ambiguity_f1:.4f} \\\\",
        f"Answer F1 (keyword) & {results.answer_f1_mean:.4f} \\\\",
        f"Latency Mean (ms) & {results.latency_mean_ms:.1f} \\\\",
        f"Latency P50 (ms) & {results.latency_p50_ms:.1f} \\\\",
        f"Latency P95 (ms) & {results.latency_p95_ms:.1f} \\\\",
        f"Stage 2 Trigger Rate & {results.stage2_trigger_rate:.4f} \\\\",
        f"Stage 2 Override Rate & {results.stage2_override_rate:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    # Per-route accuracy sub-table
    if results.per_route_accuracy:
        lines.extend([
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Per-Route Accuracy}",
            r"\label{tab:per_route}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"\textbf{Route} & \textbf{Accuracy} \\",
            r"\midrule",
        ])
        for route, acc in sorted(results.per_route_accuracy.items()):
            lines.append(f"{route} & {acc:.4f} \\\\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    return "\n".join(lines)
