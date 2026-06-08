"""Routing-only sanity check on the strict legal QA test set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.two_stage_router import TwoStageRouter


ROUTES = ["dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarify"]


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


def _safe_div(num: int | float, denom: int | float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _item_query(item: dict[str, Any]) -> str:
    return str(item.get("question") or item.get("query") or item.get("q") or "").strip()


def _item_label(item: dict[str, Any]) -> str:
    return str(item.get("routing_label") or item.get("expected_route") or item.get("route") or "").strip()


def _summarize(rows: list[dict[str, Any]], eval_file: Path) -> dict[str, Any]:
    total = len(rows)
    stage2_rows = [row for row in rows if row.get("stage2_invoked")]
    latencies = [float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    by_gold: dict[str, dict[str, Any]] = {}
    for label in sorted({row.get("expected_route") for row in rows}):
        group = [row for row in rows if row.get("expected_route") == label]
        by_gold[str(label)] = {
            "total": len(group),
            "route_accuracy": _safe_div(sum(1 for row in group if row.get("route_correct")), len(group)),
            "prediction_distribution": dict(Counter(str(row.get("predicted_route")) for row in group)),
            "clarify_false_positives": sum(1 for row in group if row.get("predicted_route") == "clarify"),
        }
    return {
        "eval_file": str(eval_file),
        "total_samples": total,
        "route_accuracy": _safe_div(sum(1 for row in rows if row.get("route_correct")), total),
        "clarify_false_positive_count": sum(
            1 for row in rows
            if row.get("predicted_route") == "clarify" and row.get("expected_route") != "clarify"
        ),
        "stage2_trigger_rate": _safe_div(len(stage2_rows), total),
        "stage2_override_rate": _safe_div(sum(1 for row in stage2_rows if row.get("stage2_override")), len(stage2_rows)),
        "avg_latency_ms": mean(latencies) if latencies else None,
        "route_distribution": dict(Counter(str(row.get("predicted_route")) for row in rows)),
        "gold_distribution": dict(Counter(str(row.get("expected_route")) for row in rows)),
        "by_gold_route": by_gold,
    }


def _summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Strict Routing Sanity Summary",
        "",
        f"- Eval file: `{summary['eval_file']}`",
        f"- Total samples: `{summary['total_samples']}`",
        f"- Route accuracy: `{summary['route_accuracy']:.4f}`",
        f"- Clarify false positives: `{summary['clarify_false_positive_count']}`",
        f"- Stage 2 trigger rate: `{summary['stage2_trigger_rate']:.4f}`",
        f"- Stage 2 override rate: `{summary['stage2_override_rate']:.4f}`",
        f"- Avg latency ms: `{summary['avg_latency_ms']}`",
        f"- Route distribution: `{summary['route_distribution']}`",
        "",
        "## By Gold Route",
        "",
        "| Gold route | Total | Route Acc. | Clarify FP | Pred routes |",
        "|---|---:|---:|---:|---|",
    ]
    for label, row in summary["by_gold_route"].items():
        lines.append(
            f"| `{label}` | {row['total']} | {row['route_accuracy']:.3f} | "
            f"{row['clarify_false_positives']} | `{row['prediction_distribution']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current router on strict test set without retrieval/generation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--test-file", default="qa_pipeline/data/legal_strict/test.json")
    parser.add_argument("--output-dir", default="results_phase3")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    items = _read_json(Path(args.test_file))
    if args.limit is not None:
        items = items[: args.limit]

    router = TwoStageRouter(config)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        query = _item_query(item)
        expected = _item_label(item)
        if not query or not expected:
            continue
        output = router.route(query=query, history=None, session_id=f"strict_routing_{idx}")
        features = getattr(output, "features", None)
        rows.append({
            "id": item.get("id") or item.get("_id") or idx,
            "query": query,
            "expected_route": expected,
            "predicted_route": output.route,
            "route_correct": output.route == expected,
            "stage1_route": output.stage1_route,
            "stage1_confidence": output.stage1_confidence,
            "stage2_invoked": output.stage2_invoked,
            "stage2_override": output.stage2_override,
            "stage2_override_policy_reason": output.stage2_override_policy_reason,
            "ambiguity_score": getattr(features, "ambiguity_score", None),
            "history_resolution_status": getattr(features, "history_resolution_status", None),
            "latency_ms": output.latency_ms,
            "reasoning": output.reasoning,
        })

    output_dir = Path(args.output_dir)
    summary = _summarize(rows, Path(args.test_file))
    _write_json(output_dir / "strict_routing_sanity_summary.json", summary)
    _write_jsonl(output_dir / "strict_routing_sanity_predictions.jsonl", rows)
    (output_dir / "strict_routing_sanity_summary.md").write_text(_summary_md(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
