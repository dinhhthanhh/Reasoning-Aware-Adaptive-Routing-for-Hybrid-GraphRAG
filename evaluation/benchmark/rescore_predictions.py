"""Re-score stored prediction files with the corrected metric implementations.

Why this exists
---------------
The previously reported "F1" was keyword recall (see
``evaluation/metrics/legacy.py``) and retrieval Hit@k used mismatched ID
schemes. This tool recomputes, OFFLINE, from already-stored model predictions:

    * routing accuracy        (predicted_route vs gold routing_label)
    * answer token F1 / EM    (Vietnamese-aware, vs gold ``answer`` and ``gold_context``)
    * retrieval Hit@k / MRR   (canonical-ID, strict + article-only modes)
    * unresolvable-ID rate    (diagnostic for the citation-ID mismatch)

It does NOT call the LLM, Neo4j or Chroma — it only re-scores stored outputs, so
the corrected numbers are reproducible without the full serving stack.

Limitations (documented honestly):
    * Stored prediction files carry no system label and no Stage-2 metadata, so
      this tool reports per-FILE metrics, not a Single-stage-vs-Two-stage split.
      Producing that split requires re-running the pipeline (see
      ``results/final/README.md``).

Usage
-----
Score one file::

    python -m evaluation.benchmark.rescore_predictions \
        --predictions results_final_unified/e2e_benchmark/predictions_v2.json \
        --label router_v2 --output results/final/rescore_router_v2.json

Build the consolidated official metrics file::

    python -m evaluation.benchmark.rescore_predictions --build-official
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from evaluation.metrics.id_normalizer import (
    compute_hit_at_k,
    compute_mrr,
    normalize_legal_id,
)
from evaluation.metrics.token_f1 import (
    active_tokenizer,
    compute_corpus_token_f1,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOLD = PROJECT_ROOT / "qa_pipeline" / "data" / "legal_strict" / "test.json"
OFFICIAL_DIR = PROJECT_ROOT / "results" / "final"

# Prediction files that can be re-scored offline, with what they represent.
# (Stored files are retrieval/answer dumps; router files have no SS/TS label.)
KNOWN_PREDICTIONS: dict[str, dict[str, str]] = {
    "router_run_a": {
        "path": "results_final_unified/e2e_benchmark/predictions.json",
        "note": "Router run with honest test-time routing (~0.857 accuracy).",
    },
    "router_run_b_LEAKED": {
        "path": "results_final_unified/e2e_benchmark/predictions_v2.json",
        "note": (
            "REJECTED: routing accuracy ~0.9667 here is the data-leakage figure "
            "(re-routed/evaluated on training data). Do NOT cite this routing "
            "number; use stage1_routing_cv instead. Kept only for transparency."
        ),
    },
    "pure_graph": {
        "path": "results_final_unified/e2e_benchmark/predictions_pure_graph.json",
        "note": "Pure GraphRAG baseline (always graph_traversal).",
    },
    "pure_vector_retrieval": {
        "path": "results_final_unified/e2e_benchmark/predictions_temp_vector.json",
        "note": "Pure vector retrieval dump (no predicted_route, retrieval only).",
    },
}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _answer_text(pred: dict[str, Any]) -> str:
    return str(pred.get("generated_answer") or pred.get("answer") or "")


def _gold_articles(record: dict[str, Any]) -> list[Any]:
    rel = record.get("relevant_articles")
    if rel:
        return rel
    key = record.get("article_key")
    return [key] if key else []


def score_predictions(
    predictions_path: Path,
    gold_by_id: dict[str, dict[str, Any]],
    label: str,
    note: str = "",
) -> dict[str, Any]:
    """Re-score a single predictions file against the gold test set."""
    preds = _load_json(predictions_path)

    matched = 0
    has_route = 0
    route_correct = 0
    pred_answers: list[str] = []
    gold_answers: list[str] = []
    gold_contexts: list[str] = []

    hit1 = hit3 = hit5 = 0
    hit5_article = 0
    mrr_sum = 0.0
    retrieval_evaluated = 0
    total_retrieved = 0
    unresolvable_retrieved = 0

    for pred in preds:
        rec = gold_by_id.get(pred.get("id"))
        if rec is None:
            continue
        matched += 1

        route = pred.get("predicted_route")
        if route:
            has_route += 1
            if route == rec.get("routing_label"):
                route_correct += 1

        ans = _answer_text(pred)
        pred_answers.append(ans)
        gold_answers.append(str(rec.get("answer") or ""))
        gold_contexts.append(str(rec.get("gold_context") or rec.get("evidence") or ""))

        retrieved = pred.get("retrieved_articles") or []
        gold_arts = _gold_articles(rec)
        if gold_arts:
            retrieval_evaluated += 1
            hit1 += compute_hit_at_k(retrieved, gold_arts, 1, mode="strict")
            hit3 += compute_hit_at_k(retrieved, gold_arts, 3, mode="strict")
            hit5 += compute_hit_at_k(retrieved, gold_arts, 5, mode="strict")
            hit5_article += compute_hit_at_k(retrieved, gold_arts, 5, mode="article")
            mrr_sum += compute_mrr(retrieved, gold_arts, mode="strict")
        for item in retrieved:
            total_retrieved += 1
            if not normalize_legal_id(str(item)).is_resolvable:
                unresolvable_retrieved += 1

    f1_vs_answer = compute_corpus_token_f1(pred_answers, gold_answers)
    f1_vs_context = compute_corpus_token_f1(pred_answers, gold_contexts)

    def _rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    return {
        "label": label,
        "note": note,
        "predictions_file": str(predictions_path.relative_to(PROJECT_ROOT)),
        "n_predictions": len(preds),
        "n_matched_to_gold": matched,
        "routing": {
            "n_with_predicted_route": has_route,
            "accuracy": _rate(route_correct, has_route),
        },
        "answer_quality": {
            "token_f1_vs_answer": f1_vs_answer["f1"],
            "exact_match_vs_answer": f1_vs_answer["exact_match"],
            "token_f1_vs_gold_context": f1_vs_context["f1"],
            "n": f1_vs_answer["n"],
            "tokenizer": active_tokenizer(),
            "metric": "Vietnamese token-level F1 (SQuAD-style, multiset overlap)",
        },
        "retrieval": {
            "n_evaluated": retrieval_evaluated,
            "hit_at_1_strict": _rate(hit1, retrieval_evaluated),
            "hit_at_3_strict": _rate(hit3, retrieval_evaluated),
            "hit_at_5_strict": _rate(hit5, retrieval_evaluated),
            "hit_at_5_article_only": _rate(hit5_article, retrieval_evaluated),
            "mrr_strict": _rate(mrr_sum, retrieval_evaluated),
            "unresolvable_retrieved_id_rate": _rate(
                unresolvable_retrieved, total_retrieved
            ),
            "note": (
                "strict = document code AND article number must match; "
                "article_only = article number match ignoring document (upper "
                "bound recoverable given Pháp Điển vs VBPL ID schemes)."
            ),
        },
    }


def build_official(
    gold_path: Path = DEFAULT_GOLD,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Assemble the single canonical metrics file from offline re-scoring."""
    gold = _load_json(gold_path)
    exclude_ids = exclude_ids or set()
    gold_by_id = {r["id"]: r for r in gold if r["id"] not in exclude_ids}

    per_file: list[dict[str, Any]] = []
    for label, spec in KNOWN_PREDICTIONS.items():
        path = PROJECT_ROOT / spec["path"]
        if path.exists():
            per_file.append(
                score_predictions(path, gold_by_id, label, note=spec.get("note", ""))
            )

    # Authoritative Stage-1 routing number: 5-fold CV (NOT end-to-end on test,
    # NOT the rejected 96.67% leakage figure). Sourced from training_report.json.
    training_report = PROJECT_ROOT / "router_model" / "training_report.json"
    stage1_cv: dict[str, Any] = {}
    if training_report.exists():
        tr = _load_json(training_report)
        cv = tr.get("cross_validation", {})
        stage1_cv = {
            "cv_accuracy_mean": cv.get("cv_accuracy_mean"),
            "cv_accuracy_std": cv.get("cv_accuracy_std"),
            "cv_macro_f1_mean": cv.get("cv_macro_f1_mean"),
            "cv_macro_f1_std": cv.get("cv_macro_f1_std"),
            "source": "router_model/training_report.json",
            "note": "Authoritative routing metric. Do NOT use the 96.67% leakage figure.",
        }

    return {
        "_meta": {
            "description": (
                "Single canonical metrics file. Answer/retrieval numbers were "
                "RE-SCORED offline from stored predictions using corrected "
                "metrics (Vietnamese token F1 + canonical-ID retrieval). They "
                "supersede all keyword-recall 'F1' numbers in archived files."
            ),
            "git_commit": _git_commit(),
            "gold_dataset": str(gold_path.relative_to(PROJECT_ROOT)),
            "n_gold": len(gold),
            "metric_implementation": "evaluation/metrics (token_f1 + id_normalizer)",
            "tokenizer": active_tokenizer(),
            "n_gold_scored": len(gold_by_id),
            "n_gold_excluded": len(gold) - len(gold_by_id),
            "limitations": [
                "Stored predictions carry no system label / Stage-2 metadata, "
                "so per-file metrics are reported instead of a Single-stage vs "
                "Two-stage split.",
                "BERTScore is not recomputed here (heavy model); use "
                "evaluation/metrics/bertscore_eval.py on a live/stored run.",
                "A full Single-stage vs Two-stage comparison requires re-running "
                "the pipeline with the serving stack (Neo4j + Chroma + LLM).",
            ],
        },
        "stage1_routing_cv": stage1_cv,
        "rescored_predictions": per_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=str, help="Path to a predictions JSON file")
    parser.add_argument("--gold", type=str, default=str(DEFAULT_GOLD))
    parser.add_argument("--label", type=str, default="custom")
    parser.add_argument("--output", type=str, help="Where to write the JSON result")
    parser.add_argument(
        "--build-official",
        action="store_true",
        help="Re-score all known prediction files into results/final/official_metrics.json",
    )
    parser.add_argument(
        "--exclude-ids",
        type=str,
        help="JSON file with a list of gold ids to exclude (e.g. cleaned-out QA records)",
    )
    args = parser.parse_args()

    exclude_ids: set[str] = set()
    if args.exclude_ids:
        exclude_ids = set(_load_json(Path(args.exclude_ids)))

    if args.build_official:
        result = build_official(Path(args.gold), exclude_ids=exclude_ids)
        OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OFFICIAL_DIR / "official_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")
        for entry in result["rescored_predictions"]:
            aq = entry["answer_quality"]
            rt = entry["retrieval"]
            print(
                f"  {entry['label']:24s} "
                f"route_acc={entry['routing']['accuracy']:.4f} "
                f"tokenF1={aq['token_f1_vs_answer']:.4f} "
                f"hit@5={rt['hit_at_5_strict']:.4f} "
                f"hit@5(art)={rt['hit_at_5_article_only']:.4f}"
            )
        return

    if not args.predictions:
        parser.error("provide --predictions or --build-official")

    gold = _load_json(Path(args.gold))
    gold_by_id = {r["id"]: r for r in gold}
    result = score_predictions(Path(args.predictions), gold_by_id, args.label)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
