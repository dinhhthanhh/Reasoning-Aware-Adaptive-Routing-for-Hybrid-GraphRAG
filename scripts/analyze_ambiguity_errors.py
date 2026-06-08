"""Analyze clarification and ambiguity errors from existing eval outputs.

The script reads the constructed ambiguity benchmark and a saved clarify
prediction CSV. It does not call the router or LLM; it only aggregates the
existing outputs into audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(num: int | float, denom: int | float) -> float | None:
    if not denom:
        return None
    return float(num) / float(denom)


def _benchmark_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items, start=1):
        item_id = item.get("id") or item.get("ID") or f"clarify_eval_{i:03d}"
        index[str(item_id)] = item
    return index


def _history_to_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                q = item.get("question") or item.get("q") or item.get("user")
                a = item.get("answer") or item.get("a") or item.get("assistant")
                if q:
                    parts.append(f"User: {q}")
                if a:
                    parts.append(f"Assistant: {a}")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def _sample_from_row(row: dict[str, str], benchmark: dict[str, Any]) -> dict[str, Any]:
    item = benchmark.get(row.get("ID", ""), {})
    return {
        "id": row.get("ID", ""),
        "query": row.get("Query") or item.get("query") or item.get("question") or "",
        "history": _history_to_text(item.get("history")),
        "expected_route": row.get("Expected_Route", ""),
        "predicted_route": row.get("Predicted_Route", ""),
        "ambiguity_type": row.get("Ambiguity_Type", ""),
        "expected_flag": row.get("Expected_Flag", ""),
        "predicted_flag_ok": _as_bool(row.get("Predicted_Flag_OK")),
        "stage1_route": row.get("Stage1_Route", ""),
        "stage1_confidence": _as_float(row.get("Stage1_Confidence")),
        "stage2_triggered": _as_bool(row.get("Stage2")),
        "stage2_override": _as_bool(row.get("Stage2_Override")),
        "stage2_override_reason": row.get("Stage2_Override_Reason", ""),
        "clarify_question": row.get("Clarify_Question", ""),
        "reasoning_steps": row.get("Reasoning_Steps", ""),
        "parse_error": row.get("Parse_Error", ""),
        "latency_ms": _as_float(row.get("Latency_ms")),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_summary(metrics: dict[str, Any], false_negative_counts: dict[str, int]) -> str:
    overall = metrics["overall"]
    lines = [
        "# Ambiguity Error Summary",
        "",
        "This report aggregates existing clarification evaluation outputs. "
        "No router, retrieval, or LLM generation was rerun.",
        "",
        "## Overall",
        "",
        f"- Total queries: `{overall['total']}`",
        f"- Expected clarify queries: `{overall['expected_clarify']}`",
        f"- Predicted clarify queries: `{overall['predicted_clarify']}`",
        f"- Clarify precision: `{overall['clarify_precision']:.3f}`",
        f"- Clarify recall: `{overall['clarify_recall']:.3f}`",
        f"- Clarify F1: `{overall['clarify_f1']:.3f}`",
        f"- Stage 2 trigger rate: `{overall['stage2_trigger_rate']:.3f}`",
        "",
        "## By Ambiguity Type",
        "",
        "| Type | Total | Expected clarify | Stage2 rate | Clarify recall | False negatives |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, row in metrics["by_type"].items():
        trigger = row["stage2_trigger_rate"]
        recall = row["clarify_recall"]
        lines.append(
            f"| `{key}` | {row['total']} | {row['expected_clarify']} | "
            f"{trigger:.3f} | {0.0 if recall is None else recall:.3f} | {row['false_negatives']} |"
        )

    lines.extend(
        [
            "",
            "## Main Failure Buckets",
            "",
            f"- Missing entity false negatives: `{false_negative_counts.get('missing_entity', 0)}`",
            f"- Multi-interpretation false negatives: `{false_negative_counts.get('multi_interpretation', 0)}`",
            "",
            "The two strongest failure buckets are semantic ambiguity cases. They do not contain the same "
            "surface cues as unresolved pronouns or phrases such as `quy định này`, so Stage 2 was not "
            "triggered in the saved run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved ambiguity-eval predictions.")
    parser.add_argument("--eval-file", default="evaluation/legal_clarify_eval.json")
    parser.add_argument("--predictions", default="eval_results/clarify_two_stage.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    if not eval_path.exists():
        raise FileNotFoundError(f"Ambiguity benchmark not found: {eval_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Clarify predictions CSV not found: {predictions_path}")

    benchmark_items = _load_json(eval_path)
    benchmark = _benchmark_index(benchmark_items)
    rows = _read_csv(predictions_path)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get("Ambiguity_Type", "").strip() or "answerable_control"
        groups[key].append(row)

    by_type: dict[str, Any] = {}
    all_expected_clarify = 0
    all_predicted_clarify = 0
    all_true_clarify = 0
    all_stage2 = 0
    false_negative_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for key in sorted(groups):
        group_rows = groups[key]
        expected_clarify = [r for r in group_rows if r.get("Expected_Route") == "clarify"]
        predicted_clarify = [r for r in group_rows if r.get("Predicted_Route") == "clarify"]
        true_clarify = [
            r for r in group_rows
            if r.get("Expected_Route") == "clarify" and r.get("Predicted_Route") == "clarify"
        ]
        false_negatives = [
            r for r in group_rows
            if r.get("Expected_Route") == "clarify" and r.get("Predicted_Route") != "clarify"
        ]
        false_positives = [
            r for r in group_rows
            if r.get("Expected_Route") != "clarify" and r.get("Predicted_Route") == "clarify"
        ]
        stage2 = [r for r in group_rows if _as_bool(r.get("Stage2"))]
        route_correct = [r for r in group_rows if _as_bool(r.get("Route_Correct"))]

        for row in false_negatives:
            false_negative_samples[key].append(_sample_from_row(row, benchmark))

        all_expected_clarify += len(expected_clarify)
        all_predicted_clarify += len(predicted_clarify)
        all_true_clarify += len(true_clarify)
        all_stage2 += len(stage2)

        by_type[key] = {
            "total": len(group_rows),
            "expected_clarify": len(expected_clarify),
            "predicted_clarify": len(predicted_clarify),
            "true_clarify": len(true_clarify),
            "false_negatives": len(false_negatives),
            "false_positives": len(false_positives),
            "stage2_triggered": len(stage2),
            "stage2_trigger_rate": _safe_div(len(stage2), len(group_rows)) or 0.0,
            "clarify_recall": _safe_div(len(true_clarify), len(expected_clarify)),
            "route_accuracy": _safe_div(len(route_correct), len(group_rows)) or 0.0,
        }

    precision = _safe_div(all_true_clarify, all_predicted_clarify) or 0.0
    recall = _safe_div(all_true_clarify, all_expected_clarify) or 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    metrics = {
        "source_files": {
            "eval_file": str(eval_path),
            "predictions": str(predictions_path),
        },
        "overall": {
            "total": len(rows),
            "expected_clarify": all_expected_clarify,
            "predicted_clarify": all_predicted_clarify,
            "true_clarify": all_true_clarify,
            "clarify_precision": precision,
            "clarify_recall": recall,
            "clarify_f1": f1,
            "stage2_triggered": all_stage2,
            "stage2_trigger_rate": _safe_div(all_stage2, len(rows)) or 0.0,
        },
        "by_type": by_type,
    }

    _write_json(output_dir / "ambiguity_type_metrics.json", metrics)
    for key in ("missing_entity", "multi_interpretation"):
        _write_jsonl(
            output_dir / f"ambiguity_false_negatives_{key}.jsonl",
            false_negative_samples.get(key, []),
        )

    (output_dir / "ambiguity_error_summary.md").write_text(
        _build_summary(
            metrics,
            {key: len(value) for key, value in false_negative_samples.items()},
        ),
        encoding="utf-8",
    )

    print(f"Ambiguity diagnostics written to {output_dir}")
    print(f"Clarify recall={recall:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
