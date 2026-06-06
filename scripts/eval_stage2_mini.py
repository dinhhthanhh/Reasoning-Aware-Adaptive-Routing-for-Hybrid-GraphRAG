#!/usr/bin/env python3
"""Route-only Stage 2 mini audit.

Reads a selected mini-eval JSON file and routes each case through the
two-stage router. Optionally joins an existing benchmark result CSV to add
answer F1/accuracy without re-running answer generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.two_stage_router import TwoStageRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 2 routing on a mini eval sample.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample", default="eval_results/mini_eval_benchmark_input.json")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--output-prefix", default="eval_results/mini_eval_routing_audit_current")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def load_answer_metrics(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("ID") or row.get("id")
            if not qid:
                continue
            try:
                answer_f1 = float(row.get("F1", ""))
            except ValueError:
                answer_f1 = None
            try:
                answer_correct = bool(int(float(row.get("Acc", "0"))))
            except ValueError:
                answer_correct = None
            context_chars = int(float(row.get("Context_Chars", "0") or 0))
            metrics[qid] = {
                "answer_correct": answer_correct,
                "answer_f1": answer_f1,
                "expected_retriever_hit": context_chars > 0 or bool(row.get("Sources")),
            }
    return metrics


def classify_error(gold: str, stage1: str, final: str) -> str:
    if stage1 != gold and final == gold:
        return "stage1_wrong_stage2_fixed"
    if stage1 == gold and final != gold:
        return "stage1_correct_stage2_broke"
    if final == "clarify" and gold != "clarify":
        return "clarify_false_positive"
    if final == "dense_retrieval" and gold in {"graph_traversal", "hybrid_reasoning"}:
        return "dense_false_negative_for_graph_hybrid"
    if final == "hybrid_reasoning" and gold != "hybrid_reasoning":
        return "hybrid_overuse"
    if final == gold:
        return "correct"
    return "other_misroute"


def case_type(query: str, gold: str) -> str:
    q = query.lower()
    if any(term in q for term in ("bãi bỏ", "sửa đổi", "thay thế", "hiệu lực", "chuyển tiếp")):
        return "legal_effect"
    if any(term in q for term in ("thẩm quyền", "trách nhiệm", "chịu trách nhiệm", "cơ sở pháp lý")):
        return "authority_responsibility"
    if any(term in q for term in ("thông tư này", "quyết định này", "nghị định này", "văn bản này")):
        return "document_anchor"
    if any(term in q for term in ("nếu", "trường hợp", "khi")):
        return "conditional"
    if gold == "hybrid_reasoning":
        return "hybrid"
    if gold == "graph_traversal":
        return "graph"
    return "simple_factoid"


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cases = load_cases(Path(args.sample))
    answer_metrics = load_answer_metrics(Path(args.results_csv) if args.results_csv else None)
    router = TwoStageRouter(config)

    rows: list[dict[str, Any]] = []
    for item in tqdm(cases, desc="Auditing routes"):
        qid = str(item.get("id", ""))
        query = str(item.get("query", item.get("question", "")))
        gold = str(item.get("expected_route", item.get("gold_label", item.get("routing_label", ""))))
        output = router.route(query=query, history=None, session_id=f"audit_{qid}")
        stage1 = output.stage1_route
        final = output.route
        metrics = answer_metrics.get(qid, {})
        stage2_route = output.stage2_raw_route if output.stage2_invoked else ""
        if not stage2_route and output.stage2_invoked:
            stage2_route = final

        row = {
            "id": qid,
            "case_type": case_type(query, gold),
            "query": query,
            "gold_route": gold,
            "stage1_route": stage1,
            "base_route": output.base_route,
            "base_confidence": output.base_confidence,
            "ambiguity_candidate_route": output.ambiguity_candidate_route,
            "ambiguity_override_allowed": output.ambiguity_override_allowed,
            "ambiguity_override_reason": output.ambiguity_override_reason,
            "stage2_route": stage2_route,
            "final_route": final,
            "changed_by_stage2": output.stage2_override,
            "route_changed_from_stage1": final != stage1,
            "correct_before_stage2": stage1 == gold,
            "correct_after_stage2": final == gold,
            "error_type": classify_error(gold, stage1, final),
            "expected_retriever_hit": metrics.get("expected_retriever_hit"),
            "answer_correct": metrics.get("answer_correct"),
            "answer_f1": metrics.get("answer_f1"),
            "stage2_invoked": output.stage2_invoked,
            "stage2_override": output.stage2_override,
            "stage2_raw_route": output.stage2_raw_route,
            "stage2_override_allowed": output.stage2_override_allowed,
            "stage2_override_policy_reason": output.stage2_override_policy_reason,
            "guardrail_applied": output.stage2_guardrail_applied,
            "guardrail_reason": output.stage2_guardrail_reason,
            "stage2_parse_error": output.stage2_parse_error,
            "stage2_trigger_reasons": output.stage2_trigger_reasons,
            "feature_legal_reference_count": output.features.legal_reference_count,
            "feature_graph_keyword_count": output.features.graph_keyword_count,
            "feature_law_specificity": output.features.law_specificity,
            "feature_complexity_level": output.features.complexity_level,
            "feature_sub_question_count": output.features.sub_question_count,
            "feature_conditional_depth": output.features.conditional_depth,
            "feature_authority_chain_count": output.features.authority_chain_count,
            "feature_legal_effect_count": output.features.legal_effect_count,
            "feature_procedural_count": output.features.procedural_count,
            "feature_multi_entity_relation_count": output.features.multi_entity_relation_count,
            "feature_cross_doc_signals": output.features.cross_doc_signals,
            "feature_multi_hop_score": output.features.multi_hop_score,
            "feature_has_comparison": output.features.has_comparison,
            "feature_is_factoid": output.features.is_factoid,
        }
        rows.append(row)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_keys = [
        "stage1_wrong_stage2_fixed",
        "stage1_correct_stage2_broke",
        "clarify_false_positive",
        "dense_false_negative_for_graph_hybrid",
        "hybrid_overuse",
    ]
    counts = {key: sum(row["error_type"] == key for row in rows) for key in summary_keys}
    counts["total"] = len(rows)
    counts["routing_accuracy_before_stage2"] = (
        sum(row["correct_before_stage2"] for row in rows) / len(rows) if rows else 0.0
    )
    counts["routing_accuracy_after_stage2"] = (
        sum(row["correct_after_stage2"] for row in rows) / len(rows) if rows else 0.0
    )
    counts["stage2_trigger_rate"] = (
        sum(row["stage2_invoked"] for row in rows) / len(rows) if rows else 0.0
    )
    counts["guardrail_applied"] = sum(row["guardrail_applied"] for row in rows)
    counts["ambiguity_override_allowed"] = sum(row["ambiguity_override_allowed"] for row in rows)
    counts["stage2_override_allowed"] = sum(row["stage2_override_allowed"] for row in rows)
    counts["stage2_override_blocked"] = sum(
        row["stage2_invoked"]
        and row["stage2_route"]
        and row["stage2_route"] != row["stage1_route"]
        and not row["stage2_override_allowed"]
        for row in rows
    )

    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
