"""Aggregate paper-ready metrics for the revised routing paper.

This script is intentionally conservative. It reuses completed full
generation runs when available, runs only lightweight routing baselines
from the strict split, and marks unsupported experiments as not run in the
generated summary. It does not call Neo4j, the vector store, or an LLM.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
STRICT_TEST = ROOT / "qa_pipeline" / "data" / "legal_strict" / "test.json"
FULL_SUMMARY = ROOT / "eval_results" / "legal_strict_full_summary.json"
STRICT_CSVS = {
    "Pure Vector": ROOT / "eval_results" / "legal_strict_pure_vector_results.csv",
    "Pure Graph": ROOT / "eval_results" / "legal_strict_pure_graph_results.csv",
    "Single-stage Router": ROOT / "eval_results" / "legal_strict_single_stage_router_results.csv",
    "Two-stage Hybrid": ROOT / "eval_results" / "legal_strict_two_stage_hybrid_results.csv",
    "Always Hybrid": ROOT / "eval_results" / "legal_strict_always_hybrid_results.csv",
    "Graph + Text Fallback": ROOT / "eval_results" / "legal_strict_graph_text_fallback_results.csv",
    "LLM-only Router": ROOT / "eval_results" / "legal_strict_llm_only_router_results.csv",
    "w/o ambiguity features": ROOT / "eval_results" / "legal_strict_w_o ambiguity features_results.csv",
    "w/o relation features": ROOT / "eval_results" / "legal_strict_w_o relation features_results.csv",
    "w/o history resolver": ROOT / "eval_results" / "legal_strict_w_o history resolver_results.csv",
    "w/o severe-ambiguity override": ROOT / "eval_results" / "legal_strict_w_o severe-ambiguity override_results.csv",
    "w/o clarification sanity check": ROOT / "eval_results" / "legal_strict_w_o clarification sanity check_results.csv",
    "w/o fallback guard": ROOT / "eval_results" / "legal_strict_w_o fallback guard_results.csv",
    "Stage 2 always-on": ROOT / "eval_results" / "legal_strict_Stage 2 always-on_results.csv",
}
CLARIFY_FULL = ROOT / "results_phase3" / "clarify_eval_summary.json"
CLARIFY_STAGE1 = ROOT / "results" / "research_paper_clarify_stage1_only.json"
CONV_FULL = ROOT / "results_phase3" / "conversation_ambiguity_summary.json"
STRICT_ROUTING_SANITY = ROOT / "results_phase3" / "strict_routing_sanity_summary.json"

LABELS = ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "TODO", "Not run", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def prf(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    weighted = 0.0
    macro_values: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_values.append(f1)
        weighted += f1 * support
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(macro_values) / len(macro_values) if macro_values else 0.0,
        "weighted_f1": weighted / total if total else 0.0,
        "per_class": per_class,
        "support": total,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def summarize_full_csv(name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    y_true = [row.get("Expected_Route", "") for row in rows if row.get("Expected_Route")]
    y_pred = [row.get("Route", "") for row in rows if row.get("Expected_Route")]
    route_metrics = prf(y_true, y_pred, LABELS)
    latencies = [v for row in rows if (v := safe_float(row.get("Time_ms"))) is not None]
    stage2_rows = [row for row in rows if str(row.get("Stage2", "")).lower() == "true"]
    override_rows = [row for row in stage2_rows if str(row.get("Stage2_Override", "")).lower() == "true"]
    f1s = [v for row in rows if (v := safe_float(row.get("F1"))) is not None]
    ems = [v for row in rows if (v := safe_float(row.get("EM"))) is not None]
    return {
        "system": name,
        "n": len(rows),
        "em": sum(ems) / len(ems) if ems else None,
        "answer_f1": sum(f1s) / len(f1s) if f1s else None,
        "routing_accuracy": route_metrics["accuracy"],
        "macro_f1": route_metrics["macro_f1"],
        "weighted_f1": route_metrics["weighted_f1"],
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "stage2_trigger_rate": len(stage2_rows) / len(rows) if rows else 0.0,
        "stage2_override_rate": len(override_rows) / len(stage2_rows) if stage2_rows else 0.0,
    }


def normalize(text: str) -> str:
    return " ".join(str(text).lower().replace("\n", " ").split())


RELATION_PAT = re.compile(
    r"sửa đổi|bãi bỏ|thay thế|hiệu lực|hết hiệu lực|căn cứ|thẩm quyền|"
    r"trách nhiệm|ban hành|áp dụng đồng thời|liên quan|quy định tại|dẫn chiếu",
    re.I,
)
HYBRID_PAT = re.compile(r"đồng thời|so sánh|khác nhau|nhiều văn bản|liên văn bản|cross", re.I)
CLARIFY_PAT = re.compile(r"\b(đó|này|trên|ấy|văn bản đó|quy định đó|nội dung đó|trường hợp này)\b", re.I)
LEGAL_ID_PAT = re.compile(r"\d+/(?:\d{4}/)?[A-ZĐ-]+|Điều\s+\d+", re.I)


def rule_based_route(question: str) -> str:
    q = normalize(question)
    has_legal_id = bool(LEGAL_ID_PAT.search(question))
    if CLARIFY_PAT.search(q) and not has_legal_id:
        return "clarify"
    if HYBRID_PAT.search(q):
        return "hybrid_reasoning"
    relation_hits = len(RELATION_PAT.findall(q))
    if relation_hits >= 2 or (relation_hits >= 1 and has_legal_id):
        return "graph_traversal"
    if relation_hits >= 1 and ("và" in q or len(q.split()) > 28):
        return "hybrid_reasoning"
    return "dense_retrieval"


def adaptive_rag_style_route(question: str) -> str:
    q = normalize(question)
    if CLARIFY_PAT.search(q) and not LEGAL_ID_PAT.search(question):
        return "clarify"
    complexity = 0
    complexity += 2 if HYBRID_PAT.search(q) else 0
    complexity += 1 if RELATION_PAT.search(q) else 0
    complexity += 1 if q.count(" và ") >= 2 else 0
    complexity += 1 if len(q.split()) >= 32 else 0
    return "hybrid_reasoning" if complexity >= 2 else "dense_retrieval"


def fixed_route_metrics(strict_items: list[dict[str, Any]], route: str) -> dict[str, Any]:
    y_true = [item["routing_label"] for item in strict_items]
    y_pred = [route for _ in strict_items]
    return prf(y_true, y_pred, LABELS)


def lightweight_baselines(strict_items: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [item["routing_label"] for item in strict_items]
    outputs = lightweight_predictions(strict_items)
    return {name: prf(y_true, preds, LABELS) for name, preds in outputs.items()}


def lightweight_predictions(strict_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "Always Hybrid": ["hybrid_reasoning" for _ in strict_items],
        "Rule-based Router": [rule_based_route(item["question"]) for item in strict_items],
        "Adaptive-RAG-style Router": [adaptive_rag_style_route(item["question"]) for item in strict_items],
    }


def gold_units(item: dict[str, Any]) -> set[str]:
    units: set[str] = set()
    for art in item.get("relevant_articles") or []:
        law = normalize(art.get("law_id", ""))
        article = normalize(art.get("article_id", ""))
        if law:
            units.add(law)
        if law and article:
            units.add(f"{law}::{article}")
        if article:
            units.add(article)
    if item.get("doc_number"):
        units.add(normalize(item["doc_number"]))
    return {u for u in units if u}


def source_relevant(source: str, units: set[str]) -> bool:
    s = normalize(source)
    if not s:
        return False
    for unit in units:
        if unit and unit in s:
            return True
        if "::" in unit:
            law, article = unit.split("::", 1)
            if law in s and article in s:
                return True
    return False


def retrieval_metrics(strict_items: list[dict[str, Any]], rows: list[dict[str, str]]) -> dict[str, Any]:
    by_id = {item["id"]: item for item in strict_items}
    recalls = {1: [], 5: [], 10: []}
    rr_values: list[float] = []
    ndcg_values: list[float] = []
    precision_values: list[float] = []
    citation_hits: list[float] = []
    evaluated = 0
    for row in rows:
        item = by_id.get(row.get("ID", ""))
        if not item:
            continue
        units = gold_units(item)
        sources = [s.strip() for s in row.get("Sources", "").split(";") if s.strip()]
        if not units or not sources:
            continue
        relevance = [1 if source_relevant(src, units) else 0 for src in sources[:10]]
        evaluated += 1
        for k in (1, 5, 10):
            recalls[k].append(1.0 if any(relevance[:k]) else 0.0)
        if any(relevance):
            first = relevance.index(1) + 1
            rr_values.append(1.0 / first)
        else:
            rr_values.append(0.0)
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))
        ideal_rels = sorted(relevance, reverse=True)
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
        ndcg_values.append(dcg / idcg if idcg else 0.0)
        precision_values.append(sum(relevance) / len(relevance) if relevance else 0.0)
        citation_hits.append(1.0 if any(relevance) else 0.0)
    return {
        "evaluated": evaluated,
        "recall_at_1": sum(recalls[1]) / len(recalls[1]) if recalls[1] else None,
        "recall_at_5": sum(recalls[5]) / len(recalls[5]) if recalls[5] else None,
        "recall_at_10": sum(recalls[10]) / len(recalls[10]) if recalls[10] else None,
        "mrr": sum(rr_values) / len(rr_values) if rr_values else None,
        "ndcg_at_10": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
        "evidence_precision": sum(precision_values) / len(precision_values) if precision_values else None,
        "citation_accuracy": sum(citation_hits) / len(citation_hits) if citation_hits else None,
        "note": "source-id proxy against gold law/article identifiers",
    }


def shorten(text: str, limit: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def first_or_note(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No representative example in logs."
    row = rows[0]
    return shorten(row.get("Query") or row.get("query") or "")


def manual_error_analysis(strict_items: list[dict[str, Any]], two_stage_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    clarify_rows = read_csv(ROOT / "results_phase3" / "clarify_eval_results.csv")
    conv_failures = []
    conv_path = ROOT / "results_phase3" / "conversation_ambiguity_failures.jsonl"
    if conv_path.exists():
        with open(conv_path, "r", encoding="utf-8") as f:
            conv_failures = [json.loads(line) for line in f if line.strip()]
    by_id = {item["id"]: item for item in strict_items}

    def source_match(row: dict[str, str]) -> bool:
        item = by_id.get(row.get("ID", ""))
        return bool(item and any(source_relevant(src, gold_units(item)) for src in row.get("Sources", "").split(";")))

    categories = [
        (
            "Dense predicted but graph needed",
            [r for r in two_stage_rows if r.get("Expected_Route") == "graph_traversal" and r.get("Route") == "dense_retrieval"],
            "Relation signal was not strong enough or graph-specific identifier was missed.",
            "Improve relation features and doc/article identifier linking.",
        ),
        (
            "Graph predicted but dense sufficient",
            [r for r in two_stage_rows if r.get("Expected_Route") == "dense_retrieval" and r.get("Route") == "graph_traversal"],
            "Legal keywords triggered graph traversal although direct text lookup was enough.",
            "Calibrate graph escalation and add dense fast-path guards.",
        ),
        (
            "Hybrid confused with graph",
            [r for r in two_stage_rows if {r.get("Expected_Route"), r.get("Route")} == {"hybrid_reasoning", "graph_traversal"}],
            "The route boundary between relation-only and relation-plus-text evidence is soft.",
            "Add labels that distinguish graph paths from graph-plus-article synthesis.",
        ),
        (
            "Hybrid confused with dense",
            [r for r in two_stage_rows if {r.get("Expected_Route"), r.get("Route")} == {"hybrid_reasoning", "dense_retrieval"}],
            "Multi-hop evidence need is not visible from surface query features.",
            "Use entity-linking uncertainty and cross-document candidate counts.",
        ),
        (
            "False clarification",
            [r for r in two_stage_rows if r.get("Expected_Route") != "clarify" and r.get("Route") == "clarify"],
            "Ambiguity threshold or Stage 2 verifier was too conservative.",
            "Tune clarification threshold jointly with strict routing data.",
        ),
        (
            "Missed clarification",
            [r for r in clarify_rows if r.get("Expected_Route") == "clarify" and r.get("Predicted_Route") != "clarify"],
            "Semantic ambiguity can look answerable without an explicit pronoun.",
            "Generate candidate interpretations and ask clarification when they conflict.",
        ),
        (
            "History resolution error",
            [r for r in conv_failures if r.get("ambiguity_type") in {"answerable_with_history", "irrelevant_history", "conflicting_history"}],
            "History contains no usable referent or multiple competing referents.",
            "Improve referent ranking and expose candidate conflict to Stage 2.",
        ),
        (
            "Incorrect or weak citation",
            [r for r in two_stage_rows if r.get("Route") == r.get("Expected_Route") and not source_match(r)],
            "Answer route is correct but serialized sources do not match gold article identifiers.",
            "Normalize citation IDs and preserve article-level provenance.",
        ),
        (
            "Retrieval succeeded but generation failed",
            [
                r for r in two_stage_rows
                if r.get("Route") == r.get("Expected_Route")
                and source_match(r)
                and (safe_float(r.get("F1")) or 0.0) < 0.20
            ],
            "Evidence appears available, but generation omits or distorts key legal facts.",
            "Use stricter answer synthesis prompts and citation-constrained decoding.",
        ),
        (
            "Gold label may be ambiguous/noisy",
            [
                r for r in two_stage_rows
                if r.get("Route") != r.get("Expected_Route")
                and (safe_float(r.get("F1")) or 0.0) >= 0.55
            ],
            "Metadata-derived route label may not uniquely identify the best execution route.",
            "Manually review high-F1 misroutes and refine label policy.",
        ),
    ]

    output = []
    for error_type, rows, cause, fix in categories:
        output.append({
            "error_type": error_type,
            "count": len(rows),
            "representative_example": first_or_note(rows),
            "likely_cause": cause,
            "possible_fix": fix,
        })
    return output


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def result_slug(name: str) -> str:
    return (
        name.lower()
        .replace("~", "")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def full_prediction_rows(name: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("ID", ""),
            "system": name,
            "query": row.get("Query", ""),
            "expected_route": row.get("Expected_Route", ""),
            "predicted_route": row.get("Route", ""),
            "actual_route": row.get("Actual_Route", ""),
            "em": row.get("EM", ""),
            "answer_f1": row.get("F1", ""),
            "latency_ms": row.get("Time_ms", ""),
            "stage2_invoked": row.get("Stage2", ""),
            "stage2_override": row.get("Stage2_Override", ""),
            "sources": row.get("Sources", ""),
        }
        for row in rows
    ]


def lightweight_prediction_rows(
    name: str, strict_items: list[dict[str, Any]], preds: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id", ""),
            "system": name,
            "query": item.get("question", ""),
            "expected_route": item.get("routing_label", ""),
            "predicted_route": pred,
            "status": "routing_only",
        }
        for item, pred in zip(strict_items, preds)
    ]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    strict_items = read_json(STRICT_TEST, [])
    full_summary = read_json(FULL_SUMMARY, {})
    full_rows = {name: read_csv(path) for name, path in STRICT_CSVS.items()}
    full_metrics = {name: summarize_full_csv(name, rows) for name, rows in full_rows.items() if rows}
    light_preds = lightweight_predictions(strict_items)
    light = lightweight_baselines(strict_items)
    retrieval = {
        name: retrieval_metrics(strict_items, rows)
        for name, rows in full_rows.items()
        if name in {"Pure Vector", "Pure Graph", "Single-stage Router", "Two-stage Hybrid"} and rows
    }
    errors = manual_error_analysis(strict_items, full_rows.get("Two-stage Hybrid", []))
    clarify_full = read_json(CLARIFY_FULL, {})
    clarify_stage1 = read_json(CLARIFY_STAGE1, {})
    conv_full = read_json(CONV_FULL, {})
    strict_sanity = read_json(STRICT_ROUTING_SANITY, {})

    not_run = {
        "Dense + Reranker": "Not run: no reranker module, endpoint, or checkpoint was found in the repository.",
        "Stage 2 always-on": "Not run: no configuration flag forces Stage 2 verification for every query.",
        "True citation accuracy": "N/A: current logs expose serialized source identifiers, not final answer citation spans.",
        "Token cost": "N/A: Stage 2 input/output token usage is not logged in router outputs.",
    }

    payload = {
        "project_structure": {
            "latex_source": "docs/AI(PM)_ver 2.3.tex",
            "bibliography": "docs/biblio.bib",
            "strict_test": str(STRICT_TEST.relative_to(ROOT)),
            "router_checkpoint": "data/router_training/legal_strict/router_model.pkl",
            "dense_retrieval": "rag/vector_rag.py, vector_store/chroma_store.py",
            "graph_retrieval": "rag/graph_rag_adapter.py, graph/neo4j_client.py",
            "pipeline": "pipeline/hybrid_pipeline.py",
            "router": "router/two_stage_router.py",
            "main_eval_script": "scripts/run_benchmark_eval.py",
            "clarify_eval_script": "scripts/run_clarify_eval.py",
            "conversation_eval_script": "scripts/evaluate_conversation_ambiguity.py",
        },
        "full_metrics": full_metrics,
        "existing_full_summary": full_summary,
        "lightweight_routing_baselines": light,
        "retrieval_proxy_metrics": retrieval,
        "clarify_full": clarify_full,
        "clarify_stage1_no_stage2": clarify_stage1,
        "conversation_full": conv_full,
        "strict_routing_sanity": strict_sanity,
        "manual_error_analysis": errors,
        "not_run_experiments": not_run,
    }
    write_json(RESULTS_DIR / "research_paper_metrics.json", payload)

    for name, rows in full_rows.items():
        if rows:
            write_jsonl(RESULTS_DIR / f"baselines_{result_slug(name)}.jsonl", full_prediction_rows(name, rows))
    for name, preds in light_preds.items():
        write_jsonl(
            RESULTS_DIR / f"baselines_{result_slug(name)}.jsonl",
            lightweight_prediction_rows(name, strict_items, preds),
        )

    baseline_rows: list[dict[str, Any]] = []
    for name in ["Pure Vector", "Pure Graph", "Always Hybrid", "Dense + Reranker", "Graph + Text Fallback", "Rule-based Router", "LLM-only Router", "Adaptive-RAG-style Router", "Single-stage Router", "Two-stage Hybrid"]:
        if name in full_metrics:
            m = full_metrics[name]
            baseline_rows.append({
                "system": name,
                "status": "Completed end-to-end",
                "em": fmt(m["em"]),
                "answer_f1": fmt(m["answer_f1"]),
                "routing_accuracy": fmt(m["routing_accuracy"]),
                "macro_f1": fmt(m["macro_f1"]),
                "weighted_f1": fmt(m["weighted_f1"]),
                "avg_latency_ms": fmt(m["avg_latency_ms"], 1),
                "median_latency_ms": fmt(m["median_latency_ms"], 1),
                "p95_latency_ms": fmt(m["p95_latency_ms"], 1),
                "notes": "reused completed strict 600-query log",
            })
        elif name in light:
            m = light[name]
            baseline_rows.append({
                "system": name,
                "status": "Completed routing-only",
                "em": "N/A",
                "answer_f1": "N/A",
                "routing_accuracy": fmt(m["accuracy"]),
                "macro_f1": fmt(m["macro_f1"]),
                "weighted_f1": fmt(m["weighted_f1"]),
                "avg_latency_ms": "N/A",
                "median_latency_ms": "N/A",
                "p95_latency_ms": "N/A",
                "notes": "routing-only baseline computed from strict test questions",
            })
        else:
            reason_key = name if name in not_run else f"{name} end-to-end"
            baseline_rows.append({
                "system": name,
                "status": "Not run",
                "em": "N/A",
                "answer_f1": "N/A",
                "routing_accuracy": "N/A",
                "macro_f1": "N/A",
                "weighted_f1": "N/A",
                "avg_latency_ms": "N/A",
                "median_latency_ms": "N/A",
                "p95_latency_ms": "N/A",
                "notes": not_run.get(reason_key, "Not run: no supported benchmark entry point was found."),
            })
    write_csv(RESULTS_DIR / "main_baseline_results.csv", baseline_rows)
    write_json(RESULTS_DIR / "main_baseline_results.json", baseline_rows)

    retrieval_rows: list[dict[str, Any]] = []
    for name in ["Pure Vector", "Pure Graph", "Single-stage Router", "Two-stage Hybrid", "Always Hybrid"]:
        m = retrieval.get(name, {})
        if m:
            retrieval_rows.append({
                "system": name,
                "status": "Completed",
                "recall_at_1": fmt(m.get("recall_at_1")),
                "recall_at_5": fmt(m.get("recall_at_5")),
                "recall_at_10": fmt(m.get("recall_at_10")),
                "mrr": fmt(m.get("mrr")),
                "ndcg_at_10": fmt(m.get("ndcg_at_10")),
                "evidence_precision": fmt(m.get("evidence_precision")),
                "citation_accuracy": fmt(m.get("citation_accuracy")),
                "notes": m.get("note", "source-id proxy against gold law/article identifiers"),
            })
        else:
            retrieval_rows.append({
                "system": name,
                "status": "Not run",
                "recall_at_1": "N/A",
                "recall_at_5": "N/A",
                "recall_at_10": "N/A",
                "mrr": "N/A",
                "ndcg_at_10": "N/A",
                "evidence_precision": "N/A",
                "citation_accuracy": "N/A",
                "notes": not_run["Always Hybrid end-to-end"],
            })
    write_csv(RESULTS_DIR / "retrieval_metrics.csv", retrieval_rows)
    write_json(RESULTS_DIR / "retrieval_metrics.json", retrieval_rows)

    cost_rows: list[dict[str, Any]] = []
    for name in ["Pure Vector", "Pure Graph", "Single-stage Router", "Two-stage Hybrid", "Always Hybrid", "Stage 2 always-on", "LLM-only Router"]:
        m = full_metrics.get(name)
        if m:
            cost_rows.append({
                "system": name,
                "status": "Completed",
                "answer_f1": fmt(m["answer_f1"]),
                "avg_latency_ms": fmt(m["avg_latency_ms"], 1),
                "median_latency_ms": fmt(m["median_latency_ms"], 1),
                "p95_latency_ms": fmt(m["p95_latency_ms"], 1),
                "stage2_trigger_rate": fmt(m["stage2_trigger_rate"]),
                "stage2_override_rate": fmt(m["stage2_override_rate"]),
                "token_cost": "N/A: token usage not logged",
                "notes": "measured full strict run",
            })
        else:
            reason = not_run.get(name, not_run.get(f"{name} end-to-end", "Not run: no supported benchmark entry point was found."))
            cost_rows.append({
                "system": name,
                "status": "Not run",
                "answer_f1": "N/A",
                "avg_latency_ms": "N/A",
                "median_latency_ms": "N/A",
                "p95_latency_ms": "N/A",
                "stage2_trigger_rate": "N/A",
                "stage2_override_rate": "N/A",
                "token_cost": "N/A: token usage not logged",
                "notes": reason,
            })
    write_csv(RESULTS_DIR / "cost_quality.csv", cost_rows)
    write_json(RESULTS_DIR / "cost_quality.json", cost_rows)

    full = full_metrics.get("Two-stage Hybrid", {})
    single = full_metrics.get("Single-stage Router", {})
    ablation_rows: list[dict[str, Any]] = [
        {
            "variant": "Full System",
            "status": "Completed",
            "answer_f1": fmt(full.get("answer_f1")),
            "routing_accuracy": fmt(full.get("routing_accuracy")),
            "macro_f1": fmt(full.get("macro_f1")),
            "clarify_precision": fmt(clarify_full.get("clarify_precision")),
            "clarify_recall": fmt(clarify_full.get("clarify_recall")),
            "clarify_f1": fmt(clarify_full.get("clarify_f1")),
            "avg_latency_ms": fmt(full.get("avg_latency_ms"), 1),
            "stage2_trigger_rate": fmt(full.get("stage2_trigger_rate")),
            "stage2_override_rate": fmt(full.get("stage2_override_rate")),
            "notes": "selective Stage 2 verifier enabled",
        },
        {
            "variant": "w/o Stage 2",
            "status": "Completed",
            "answer_f1": fmt(single.get("answer_f1")),
            "routing_accuracy": fmt(single.get("routing_accuracy")),
            "macro_f1": fmt(single.get("macro_f1")),
            "clarify_precision": fmt(clarify_stage1.get("clarify_precision")),
            "clarify_recall": fmt(clarify_stage1.get("clarify_recall")),
            "clarify_f1": fmt(clarify_stage1.get("clarify_f1")),
            "avg_latency_ms": fmt(single.get("avg_latency_ms"), 1),
            "stage2_trigger_rate": "0.0000",
            "stage2_override_rate": "0.0000",
            "notes": "equivalent to Stage 2 never-on for this strict run",
        },
        {
            "variant": "Stage 2 never-on",
            "status": "Completed",
            "answer_f1": fmt(single.get("answer_f1")),
            "routing_accuracy": fmt(single.get("routing_accuracy")),
            "macro_f1": fmt(single.get("macro_f1")),
    ablation_rows: list[dict[str, Any]] = []
    ablation_names = [
        "w/o ambiguity features",
        "w/o relation features",
        "w/o history resolver",
        "w/o severe-ambiguity override",
        "w/o clarification sanity check",
        "w/o fallback guard",
        "Stage 2 always-on",
    ]
    for name in ablation_names:
        if name in full_metrics:
            m = full_metrics[name]
            ablation_rows.append({
                "ablation": name,
                "status": "Completed end-to-end",
                "answer_f1": fmt(m["answer_f1"]),
                "routing_accuracy": fmt(m["routing_accuracy"]),
                "macro_f1": fmt(m["macro_f1"]),
                "avg_latency_ms": fmt(m["avg_latency_ms"], 1),
                "notes": "reused completed ablation run",
            })
        else:
            ablation_rows.append({
                "ablation": name,
                "status": "Not run",
                "answer_f1": "N/A",
                "routing_accuracy": "N/A",
                "macro_f1": "N/A",
                "avg_latency_ms": "N/A",
                "notes": not_run.get(name, not_run.get("Feature-removal ablations", "Not run")),
            })
    write_csv(RESULTS_DIR / "ablation_results.csv", ablation_rows)
    write_json(RESULTS_DIR / "ablation_results.json", ablation_rows)

    write_csv(RESULTS_DIR / "error_analysis.csv", errors)
    write_json(RESULTS_DIR / "error_analysis.json", errors)
    examples = ["# Error Examples", ""]
    for row in errors:
        examples.append(f"## {row['error_type']} ({row['count']})")
        examples.append("")
        examples.append(f"- Representative example: {row['representative_example']}")
        examples.append(f"- Likely cause: {row['likely_cause']}")
        examples.append(f"- Possible fix: {row['possible_fix']}")
        examples.append("")
    (RESULTS_DIR / "error_examples.md").write_text("\n".join(examples), encoding="utf-8")

    lines = [
        "# Experiment Summary for Revised Routing Paper",
        "",
        "This summary was generated from existing strict benchmark logs and lightweight routing-only baselines. It does not call Neo4j, Chroma, or an LLM.",
        "",
        "## Discovered Project Structure",
        "",
    ]
    for key, value in payload["project_structure"].items():
        lines.append(f"- **{key}**: `{value}`")
    lines.extend([
        "",
        "## Main Comparison with Stronger Baselines",
        "",
        markdown_table(
            ["System", "Status", "EM", "Answer F1", "Routing Acc.", "Macro-F1", "Avg. Latency", "Notes"],
            [[r["system"], r["status"], r["em"], r["answer_f1"], r["routing_accuracy"], r["macro_f1"], r["avg_latency_ms"], r["notes"]] for r in baseline_rows],
        ),
        "",
        "## Retrieval and Evidence Quality",
        "",
        "Metrics are source-id proxy metrics computed from `Sources` against gold law/article identifiers.",
        "",
        markdown_table(
            ["System", "Status", "R@1", "R@5", "R@10", "MRR", "nDCG@10", "Evidence Precision", "Citation Acc."],
            [[r["system"], r["status"], r["recall_at_1"], r["recall_at_5"], r["recall_at_10"], r["mrr"], r["ndcg_at_10"], r["evidence_precision"], r["citation_accuracy"]] for r in retrieval_rows],
        ),
        "",
        "## Cost-quality Trade-off",
        "",
        markdown_table(
            ["System", "Status", "Answer F1", "Avg. Latency", "Median", "p95", "Stage 2 Trigger", "Override", "Cost Note"],
            [[r["system"], r["status"], r["answer_f1"], r["avg_latency_ms"], r["median_latency_ms"], r["p95_latency_ms"], r["stage2_trigger_rate"], r["stage2_override_rate"], r["token_cost"]] for r in cost_rows],
        ),
        "",
        "## Ablation Study",
        "",
        markdown_table(
            ["Variant", "Status", "Answer F1", "Routing Acc.", "Clarify P", "Clarify R", "Clarify F1", "Latency", "Notes"],
            [[r["variant"], r["status"], r["answer_f1"], r["routing_accuracy"], r["clarify_precision"], r["clarify_recall"], r["clarify_f1"], r["avg_latency_ms"], r["notes"]] for r in ablation_rows],
        ),
        "",
        "## Clarification Metrics",
        "",
        f"- Full Stage 2 ambiguity F1: `{fmt(clarify_full.get('clarify_f1'))}`",
        f"- Stage 2 disabled ambiguity F1: `{fmt(clarify_stage1.get('clarify_f1'))}`",
        f"- Conversation clarify F1: `{fmt(conv_full.get('clarify_f1'))}`",
        "",
        "## Manual Error Analysis",
        "",
        markdown_table(["Error Type", "Count", "Representative Example", "Likely Cause", "Possible Fix"], [
            [row["error_type"], row["count"], row["representative_example"], row["likely_cause"], row["possible_fix"]]
            for row in errors
        ]),
        "",
        "## Experiments Not Run",
        "",
    ])
    for name, reason in not_run.items():
        lines.append(f"- **{name}**: {reason}")
    lines.append("")
    (RESULTS_DIR / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "metrics_json": "results/research_paper_metrics.json",
        "summary_md": "results/experiment_summary.md",
        "main_baseline_results": "results/main_baseline_results.csv",
        "ablation_results": "results/ablation_results.csv",
        "retrieval_metrics": "results/retrieval_metrics.csv",
        "cost_quality": "results/cost_quality.csv",
        "error_analysis": "results/error_analysis.csv",
    }, indent=2))


if __name__ == "__main__":
    main()
