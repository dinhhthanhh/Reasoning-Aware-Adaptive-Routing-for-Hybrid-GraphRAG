"""Evaluate conversation-aware ambiguity routing.

This evaluator is routing-only by default: it calls the current router with
query and history, records routing metadata, and does not run retrieval or
answer generation. Use --no-llm to disable Stage 2 for a cheap dry run.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.two_stage_router import TwoStageRouter


ROUTE_LABELS = ["dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarify"]


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("id"):
                cache[str(row["id"])] = row
    return cache


def _safe_div(num: int | float, denom: int | float) -> float | None:
    if not denom:
        return None
    return float(num) / float(denom)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_output_fields(output: Any) -> dict[str, Any]:
    features = getattr(output, "features", None)
    return {
        "predicted_route": getattr(output, "route", None),
        "confidence": getattr(output, "confidence", None),
        "stage1_route": getattr(output, "stage1_route", None),
        "stage1_confidence": getattr(output, "stage1_confidence", None),
        "stage2_invoked": getattr(output, "stage2_invoked", None),
        "stage2_override": getattr(output, "stage2_override", None),
        "stage2_trigger_reasons": getattr(output, "stage2_trigger_reasons", None),
        "stage2_override_reason": getattr(output, "stage2_override_reason", None),
        "stage2_override_policy_reason": getattr(output, "stage2_override_policy_reason", None),
        "stage2_ambiguity_flags": getattr(output, "stage2_ambiguity_flags", None),
        "stage2_reasoning_steps": getattr(output, "stage2_reasoning_steps", None),
        "stage2_parse_error": getattr(output, "stage2_parse_error", None),
        "clarify_question": getattr(output, "clarify_question", None),
        "reasoning": getattr(output, "reasoning", None),
        "latency_ms": getattr(output, "latency_ms", None),
        "is_ambiguous": getattr(output, "is_ambiguous", None),
        "ambiguity_score": getattr(features, "ambiguity_score", None),
        "has_pronoun": getattr(features, "has_pronoun", None),
        "missing_entity_type": getattr(features, "missing_entity_type", None),
        "history_length": getattr(features, "history_length", None),
        "history_resolves_ambiguity": getattr(features, "history_resolves_ambiguity", None),
        "complexity_level": getattr(features, "complexity_level", None),
        "multi_hop_score": getattr(features, "multi_hop_score", None),
        "graph_keyword_count": getattr(features, "graph_keyword_count", None),
        "legal_reference_count": getattr(features, "legal_reference_count", None),
        "resolved_referent": getattr(output, "resolved_referent", "not_available"),
        "resolved_query": getattr(output, "resolved_query", "not_available"),
    }


def _prediction_from_item(item: dict[str, Any], output_fields: dict[str, Any], mode: str) -> dict[str, Any]:
    predicted_route = output_fields.get("predicted_route")
    expected_route = item.get("expected_route", "")
    route_correct = predicted_route == expected_route
    return {
        "id": item.get("id", ""),
        "query": item.get("query", ""),
        "history": item.get("history", ""),
        "expected_route": expected_route,
        "predicted_route": predicted_route,
        "route_correct": route_correct,
        "ambiguity_type": item.get("ambiguity_type", ""),
        "expected_behavior": item.get("expected_behavior", ""),
        "gold_resolved_entity": item.get("gold_resolved_entity", ""),
        "gold_clarification_question": item.get("gold_clarification_question", ""),
        "notes": item.get("notes", ""),
        "mode": mode,
        **output_fields,
    }


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [label for label in ROUTE_LABELS if label in {
        *(row.get("expected_route") for row in rows),
        *(row.get("predicted_route") for row in rows),
    }]
    matrix = []
    for gold in labels:
        matrix.append([
            sum(1 for row in rows if row.get("expected_route") == gold and row.get("predicted_route") == pred)
            for pred in labels
        ])
    return {"labels": labels, "matrix": matrix}


def _summarize(rows: list[dict[str, Any]], *, eval_file: Path, mode: str, limit: int | None) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("route_correct"))
    predicted_clarify = [row for row in rows if row.get("predicted_route") == "clarify"]
    expected_clarify = [row for row in rows if row.get("expected_route") == "clarify"]
    true_clarify = [
        row for row in rows
        if row.get("expected_route") == "clarify" and row.get("predicted_route") == "clarify"
    ]
    precision = _safe_div(len(true_clarify), len(predicted_clarify)) or 0.0
    recall = _safe_div(len(true_clarify), len(expected_clarify)) or 0.0

    stage2_values = [row.get("stage2_invoked") for row in rows if row.get("stage2_invoked") is not None]
    stage2_rows = [row for row in rows if row.get("stage2_invoked") is True]
    override_values = [row.get("stage2_override") for row in rows if row.get("stage2_override") is not None]
    override_rows = [row for row in rows if row.get("stage2_override") is True]
    latencies = [
        float(row["latency_ms"]) for row in rows
        if isinstance(row.get("latency_ms"), (int, float))
    ]

    answerable_with_history = [row for row in rows if row.get("ambiguity_type") == "answerable_with_history"]
    resolved_available = [
        row for row in answerable_with_history
        if row.get("resolved_referent") not in (None, "", "not_available")
    ]
    non_clarify_history = [
        row for row in answerable_with_history
        if row.get("predicted_route") != "clarify"
    ]
    if resolved_available:
        history_resolution_accuracy: float | str | None = _safe_div(len(resolved_available), len(answerable_with_history))
    else:
        history_resolution_accuracy = "not_available"

    category_false_counts = {
        "false_clarification_on_answerable_with_history": sum(
            1 for row in rows
            if row.get("ambiguity_type") == "answerable_with_history"
            and row.get("predicted_route") == "clarify"
        ),
        "false_answer_on_clarify_without_history": sum(
            1 for row in rows
            if row.get("ambiguity_type") == "clarify_without_history"
            and row.get("predicted_route") != "clarify"
        ),
        "false_answer_on_irrelevant_history": sum(
            1 for row in rows
            if row.get("ambiguity_type") == "irrelevant_history"
            and row.get("predicted_route") != "clarify"
        ),
        "false_answer_on_conflicting_history": sum(
            1 for row in rows
            if row.get("ambiguity_type") == "conflicting_history"
            and row.get("predicted_route") != "clarify"
        ),
    }

    by_type: dict[str, dict[str, Any]] = {}
    for key in sorted({row.get("ambiguity_type", "") for row in rows}):
        group = [row for row in rows if row.get("ambiguity_type") == key]
        group_pred_clarify = [row for row in group if row.get("predicted_route") == "clarify"]
        group_expected_clarify = [row for row in group if row.get("expected_route") == "clarify"]
        group_true_clarify = [
            row for row in group
            if row.get("expected_route") == "clarify" and row.get("predicted_route") == "clarify"
        ]
        group_stage2_values = [row.get("stage2_invoked") for row in group if row.get("stage2_invoked") is not None]
        group_latencies = [
            float(row["latency_ms"]) for row in group
            if isinstance(row.get("latency_ms"), (int, float))
        ]
        group_precision = _safe_div(len(group_true_clarify), len(group_pred_clarify))
        group_recall = _safe_div(len(group_true_clarify), len(group_expected_clarify))
        by_type[key] = {
            "total": len(group),
            "route_accuracy": _safe_div(sum(1 for row in group if row.get("route_correct")), len(group)),
            "clarify_precision": group_precision,
            "clarify_recall": group_recall,
            "clarify_f1": _f1(group_precision, group_recall),
            "stage2_trigger_rate": (
                _safe_div(sum(1 for value in group_stage2_values if value is True), len(group_stage2_values))
                if group_stage2_values else "not_available"
            ),
            "avg_latency_ms": mean(group_latencies) if group_latencies else None,
            "prediction_distribution": dict(Counter(str(row.get("predicted_route")) for row in group)),
        }

    failures = [row for row in rows if not row.get("route_correct")]
    top_failures = [
        {
            "id": row.get("id"),
            "ambiguity_type": row.get("ambiguity_type"),
            "expected_route": row.get("expected_route"),
            "predicted_route": row.get("predicted_route"),
            "stage1_route": row.get("stage1_route"),
            "stage1_confidence": row.get("stage1_confidence"),
            "stage2_invoked": row.get("stage2_invoked"),
            "ambiguity_score": row.get("ambiguity_score"),
            "query": row.get("query"),
            "history": row.get("history"),
        }
        for row in failures[:15]
    ]

    return {
        "eval_file": str(eval_file),
        "mode": mode,
        "limit": limit,
        "total_samples": total,
        "route_accuracy": _safe_div(correct, total) or 0.0,
        "clarify_precision": precision,
        "clarify_recall": recall,
        "clarify_f1": _f1(precision, recall) or 0.0,
        "stage2_trigger_rate": (
            _safe_div(sum(1 for value in stage2_values if value is True), len(stage2_values))
            if stage2_values else "not_available"
        ),
        "stage2_override_rate": (
            _safe_div(len(override_rows), len(stage2_rows))
            if stage2_rows else (0.0 if override_values else "not_available")
        ),
        "avg_latency_ms": mean(latencies) if latencies else None,
        "false_counts": category_false_counts,
        "history_resolution_accuracy": history_resolution_accuracy,
        "history_non_clarify_rate_proxy": _safe_div(len(non_clarify_history), len(answerable_with_history)),
        "confusion_matrix": _confusion_matrix(rows),
        "route_distribution": dict(Counter(str(row.get("predicted_route")) for row in rows)),
        "category_distribution": dict(Counter(str(row.get("ambiguity_type")) for row in rows)),
        "by_ambiguity_type": by_type,
        "top_failure_examples": top_failures,
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Conversation Ambiguity Summary",
        "",
        f"- Eval file: `{summary['eval_file']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Limit: `{summary['limit']}`",
        f"- Total samples: `{summary['total_samples']}`",
        f"- Route accuracy: `{summary['route_accuracy']:.3f}`",
        f"- Clarify precision: `{summary['clarify_precision']:.3f}`",
        f"- Clarify recall: `{summary['clarify_recall']:.3f}`",
        f"- Clarify F1: `{summary['clarify_f1']:.3f}`",
        f"- Stage 2 trigger rate: `{summary['stage2_trigger_rate']}`",
        f"- Stage 2 override rate: `{summary['stage2_override_rate']}`",
        f"- Avg latency ms: `{summary['avg_latency_ms']}`",
        f"- History resolution accuracy: `{summary['history_resolution_accuracy']}`",
        f"- History non-clarify proxy: `{summary['history_non_clarify_rate_proxy']}`",
        "",
        "## Category Distribution",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key, value in summary["category_distribution"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Metrics by Type", "", "| Type | Total | Route Acc. | Clarify R | Stage2 rate | Pred routes |", "|---|---:|---:|---:|---:|---|"])
    for key, row in summary["by_ambiguity_type"].items():
        route_acc = row["route_accuracy"]
        clarify_recall = row["clarify_recall"]
        lines.append(
            f"| `{key}` | {row['total']} | "
            f"{0.0 if route_acc is None else route_acc:.3f} | "
            f"{'n/a' if clarify_recall is None else f'{clarify_recall:.3f}'} | "
            f"{row['stage2_trigger_rate']} | `{row['prediction_distribution']}` |"
        )

    lines.extend(["", "## False Counts", ""])
    for key, value in summary["false_counts"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Top Failure Examples", ""])
    for row in summary["top_failure_examples"]:
        lines.append(
            f"- `{row['id']}` `{row['ambiguity_type']}` expected `{row['expected_route']}`, "
            f"predicted `{row['predicted_route']}`: {row['query']}"
        )
    lines.append("")
    return "\n".join(lines)


def _run_router(config: dict[str, Any], items: list[dict[str, Any]], cache: dict[str, dict[str, Any]], use_cache: bool, mode: str) -> list[dict[str, Any]]:
    router = TwoStageRouter(config)
    rows: list[dict[str, Any]] = []
    for item in items:
        qid = str(item.get("id", ""))
        if use_cache and qid in cache:
            rows.append(cache[qid])
            continue
        output = router.route(
            query=str(item.get("query", "")),
            history=str(item.get("history") or "") or None,
            session_id=f"conversation_ambiguity_{qid}",
        )
        row = _prediction_from_item(item, _extract_output_fields(output), mode)
        rows.append(row)
        cache[qid] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conversation-aware ambiguity benchmark")
    parser.add_argument("--eval-file", default="evaluation/conversation_ambiguity_eval.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Disable Stage 2 LLM verifier")
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    output_dir = Path(args.output_dir)
    cache_file = Path(args.cache_file) if args.cache_file else output_dir / "conversation_ambiguity_cache.jsonl"

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.no_llm:
        config = copy.deepcopy(config)
        config.setdefault("router", {}).setdefault("stage2", {})["enabled"] = False

    items = _read_json(eval_file)
    if args.limit is not None:
        items = items[: args.limit]

    mode = "stage1_no_llm" if args.no_llm else "two_stage_current"
    cache = _read_jsonl_cache(cache_file) if args.use_cache else {}
    rows = _run_router(config, items, cache, args.use_cache, mode)
    summary = _summarize(rows, eval_file=eval_file, mode=mode, limit=args.limit)
    failures = [row for row in rows if not row.get("route_correct")]

    _write_jsonl(output_dir / "conversation_ambiguity_predictions.jsonl", rows)
    _write_json(output_dir / "conversation_ambiguity_summary.json", summary)
    _write_jsonl(output_dir / "conversation_ambiguity_failures.jsonl", failures)
    _write_jsonl(cache_file, list(cache.values()))
    (output_dir / "conversation_ambiguity_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
