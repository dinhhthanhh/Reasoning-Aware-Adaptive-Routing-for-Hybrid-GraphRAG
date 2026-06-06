#!/usr/bin/env python3
"""Evaluate clarify/ambiguity routing without running full RAG generation.

This benchmark focuses on the router behavior:
- Does an ambiguous query route to `clarify`?
- Does Stage 2 run and preserve reasoning metadata?
- Do Stage 2 ambiguity flags match the annotated ambiguity type?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.two_stage_router import TwoStageRouter


AMBIGUITY_TYPE_TO_FLAG = {
    "missing_entity": "missing_entity",
    "pronoun_reference": "pronoun_reference",
    "multi_interpretation": "multi_interpretation",
    "incomplete_context": "incomplete_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clarify routing benchmark")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--eval-file", default="evaluation/legal_clarify_eval.json")
    parser.add_argument("--output", default="eval_results/legal_clarify_eval_summary.json")
    parser.add_argument("--csv-output", default="eval_results/legal_clarify_eval_results.csv")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items")
    parser.add_argument(
        "--disable-stage2",
        action="store_true",
        help="Evaluate Stage 1 + rule overrides only",
    )
    parser.add_argument(
        "--disable-ambiguity-rule",
        action="store_true",
        help="Disable hard ambiguity-to-clarify overrides for Stage 1-only ablation",
    )
    parser.add_argument(
        "--router-model-path",
        default=None,
        help="Override router.stage1.model_path, useful for strict no-leakage experiments.",
    )
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def history_to_text(history: Any) -> str | None:
    if not history:
        return None
    if isinstance(history, str):
        return history
    if isinstance(history, list):
        return "\n".join(str(turn) for turn in history if turn)
    return str(history)


def evaluate(config: dict[str, Any], items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    router = TwoStageRouter(config)
    rows: list[dict[str, Any]] = []

    for item in items:
        qid = item.get("id") or item.get("_id") or item.get("qid") or f"item_{len(rows)}"
        query = item.get("query") or item.get("question") or ""
        expected_route = item.get("expected_route") or item.get("routing_label") or item.get("route") or ""
        ambiguity_type = item.get("ambiguity_type", "")
        expected_complexity = item.get("expected_complexity", "")
        history = history_to_text(item.get("history"))

        output = router.route(query=query, history=history, session_id=f"clarify_eval_{qid}")
        expected_flag = AMBIGUITY_TYPE_TO_FLAG.get(ambiguity_type)
        predicted_flag_ok = (
            bool(output.stage2_ambiguity_flags.get(expected_flag, False))
            if expected_flag and output.stage2_ambiguity_flags
            else False
        )

        rows.append({
            "ID": qid,
            "Query": query,
            "Expected_Route": expected_route,
            "Predicted_Route": output.route,
            "Route_Correct": output.route == expected_route,
            "Expected_Complexity": expected_complexity,
            "Predicted_Complexity": output.stage2_complexity_level,
            "Ambiguity_Type": ambiguity_type,
            "Expected_Flag": expected_flag or "",
            "Predicted_Flag_OK": predicted_flag_ok,
            "Stage1_Route": output.stage1_route,
            "Stage1_Confidence": round(output.stage1_confidence, 4),
            "Stage2": output.stage2_invoked,
            "Stage2_Override": output.stage2_override,
            "Stage2_Override_Reason": output.stage2_override_reason or "",
            "Clarify_Question": output.clarify_question or "",
            "Reasoning_Steps": " | ".join(output.stage2_reasoning_steps),
            "Parse_Error": output.stage2_parse_error or "",
            "Latency_ms": round(output.latency_ms, 2),
        })

    summary = summarize(rows)
    return rows, summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "route_accuracy": 0.0,
            "clarify_precision": 0.0,
            "clarify_recall": 0.0,
            "clarify_f1": 0.0,
            "stage2_trigger_rate": 0.0,
            "stage2_override_rate": 0.0,
            "flag_accuracy_on_stage2": 0.0,
            "route_distribution": {},
            "clarify_false_positives": 0,
            "clarify_false_negatives": 0,
            "by_ambiguity_type": {},
        }

    correct = sum(bool(row["Route_Correct"]) for row in rows)
    predicted_clarify = [row for row in rows if row["Predicted_Route"] == "clarify"]
    expected_clarify = [row for row in rows if row["Expected_Route"] == "clarify"]
    true_positive = sum(
        row["Predicted_Route"] == "clarify" and row["Expected_Route"] == "clarify"
        for row in rows
    )
    precision = true_positive / len(predicted_clarify) if predicted_clarify else 0.0
    recall = true_positive / len(expected_clarify) if expected_clarify else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    stage2_rows = [row for row in rows if row["Stage2"]]
    override_rows = [row for row in rows if row["Stage2_Override"]]
    flag_total = sum(1 for row in stage2_rows if row["Expected_Flag"])
    flag_correct = sum(1 for row in stage2_rows if row["Expected_Flag"] and row["Predicted_Flag_OK"])
    route_distribution: dict[str, int] = {}
    for row in rows:
        route = str(row["Predicted_Route"])
        route_distribution[route] = route_distribution.get(route, 0) + 1
    clarify_false_positives = sum(
        row["Predicted_Route"] == "clarify" and row["Expected_Route"] != "clarify"
        for row in rows
    )
    clarify_false_negatives = sum(
        row["Predicted_Route"] != "clarify" and row["Expected_Route"] == "clarify"
        for row in rows
    )

    by_type: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["Ambiguity_Type"] or "unknown"
        entry = by_type.setdefault(key, {"total": 0, "route_correct": 0, "stage2": 0, "flag_correct": 0})
        entry["total"] += 1
        entry["route_correct"] += int(bool(row["Route_Correct"]))
        entry["stage2"] += int(bool(row["Stage2"]))
        entry["flag_correct"] += int(bool(row["Predicted_Flag_OK"]))

    for entry in by_type.values():
        total_for_type = entry["total"]
        entry["route_accuracy"] = entry["route_correct"] / total_for_type if total_for_type else 0.0
        entry["stage2_trigger_rate"] = entry["stage2"] / total_for_type if total_for_type else 0.0
        entry["flag_accuracy"] = entry["flag_correct"] / total_for_type if total_for_type else 0.0

    return {
        "total": total,
        "route_accuracy": correct / total,
        "clarify_precision": precision,
        "clarify_recall": recall,
        "clarify_f1": f1,
        "stage2_trigger_rate": len(stage2_rows) / total,
        "stage2_override_rate": len(override_rows) / len(stage2_rows) if stage2_rows else 0.0,
        "flag_accuracy_on_stage2": flag_correct / flag_total if flag_total else 0.0,
        "route_distribution": route_distribution,
        "clarify_false_positives": clarify_false_positives,
        "clarify_false_negatives": clarify_false_negatives,
        "by_ambiguity_type": by_type,
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["ID"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.router_model_path:
        config.setdefault("router", {}).setdefault("stage1", {})["model_path"] = args.router_model_path

    if args.disable_stage2:
        config.setdefault("router", {}).setdefault("stage2", {})["enabled"] = False
    if args.disable_ambiguity_rule:
        override_cfg = config.setdefault("router", {}).setdefault("override_rules", {})
        override_cfg["ambiguity_clarify_threshold"] = 2.0
        override_cfg["ambiguity_force_stage2_threshold"] = 2.0
        override_cfg["reasoning_force_stage2_enabled"] = False

    items = load_items(Path(args.eval_file))
    if args.limit is not None:
        items = items[: args.limit]

    rows, summary = evaluate(config, items)
    write_outputs(rows, summary, Path(args.output), Path(args.csv_output))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
