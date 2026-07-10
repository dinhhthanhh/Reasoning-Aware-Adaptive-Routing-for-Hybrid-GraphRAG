#!/usr/bin/env python3
"""
Oracle routing evaluation — fixed to use evaluate_prediction with
gold_context (field `answer` in test.json), IDENTICAL to run_benchmark_eval.py.

Run:
    python scripts/run_oracle_eval.py \
        --test_path qa_pipeline/data/legal_strict/test.json \
        --output_path results/oracle_eval_v3.json

Quick test (10 samples):
    python scripts/run_oracle_eval.py --max_samples 10 \
        --output_path results/oracle_eval_v3_test10.json
"""

import argparse
import json
import time
from pathlib import Path

# ── Same metric as run_benchmark_eval.py ─────────────────────────────────────
try:
    from evaluation.metrics import evaluate_prediction
except ImportError:
    # Fallback identical to run_benchmark_eval.py fallback (lines 38-46)
    def evaluate_prediction(ground_truth: str, answer: str):
        from evaluation.metrics import compute_answer_f1
        keywords = [k.strip() for k in ground_truth.split() if k.strip()]
        f1 = compute_answer_f1(answer, keywords)
        em = 1.0 if ground_truth.lower() == answer.lower() else 0.0
        acc = 1.0 if ground_truth.lower() in answer.lower() else 0.0
        return em, f1, acc

# ── BERTScore (optional) ──────────────────────────────────────────────────────
try:
    import torch
    from bert_score import score as _bert_score
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    print("[WARN] bert_score not installed — BERTScore will be skipped (None)")

from pipeline.hybrid_pipeline import HybridPipeline


# ── Label normalisation ───────────────────────────────────────────────────────
LABEL_MAP = {
    "dense":            "dense_retrieval",
    "dense_retrieval":  "dense_retrieval",
    "graph":            "graph_traversal",
    "graph_traversal":  "graph_traversal",
    "hybrid":           "hybrid_reasoning",
    "hybrid_reasoning": "hybrid_reasoning",
    "clarify":          "clarify",
}


def get_gold_route(sample: dict) -> str:
    """Read gold route — test.json uses 'routing_label' field."""
    raw = (
        sample.get("routing_label")
        or sample.get("route_label")
        or sample.get("gold_route")
        or sample.get("label")
        or ""
    ).lower().strip()
    return LABEL_MAP.get(raw, "dense_retrieval")


def get_ground_truth(sample: dict) -> str:
    """
    Get gold reference — mirrors _normalize_eval_item in run_benchmark_eval.py:
      ground_truth = item.get("ground_truth") or item.get("answer") or ...
    In legal_strict/test.json the field is `answer` (= gold_context text).
    """
    return (
        sample.get("ground_truth")
        or sample.get("answer")
        or sample.get("gold_context")
        or ""
    )


def compute_bertscore_batch(
    predictions: list[str],
    references: list[str],
    model_type: str = "xlm-roberta-large",
):
    if not _BERTSCORE_AVAILABLE:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, F1 = _bert_score(
            predictions,
            references,
            model_type=model_type,
            lang="vi",
            verbose=True,
            device=device,
            batch_size=32,
        )
        return float(F1.mean())
    except Exception as exc:
        print(f"[BERTScore ERROR] {exc}")
        return None


def run_oracle_eval(
    test_path: str,
    output_path: str,
    max_samples: int | None = None,
) -> dict:

    test_data = json.loads(Path(test_path).read_text(encoding="utf-8"))
    if max_samples:
        test_data = test_data[:max_samples]

    pipeline = HybridPipeline()

    results   = []
    latencies = []
    em_scores, f1_scores, acc_scores = [], [], []
    bertscore_preds, bertscore_refs = [], []

    for idx, sample in enumerate(test_data):
        question     = sample.get("question") or sample.get("query", "")
        gold_route   = get_gold_route(sample)
        ground_truth = get_ground_truth(sample)   # same reference as other systems

        t0 = time.perf_counter()
        try:
            response       = pipeline.query(
                query=question,
                session_id=f"oracle_{idx}",
                force_route=gold_route,
            )
            answer         = response.answer
            executed_route = response.route_used
            retrieval_ok   = True
        except Exception as exc:
            answer, executed_route, retrieval_ok = "", gold_route, False
            print(f"[{idx}] ERROR: {exc}")

        latency_ms = (time.perf_counter() - t0) * 1000

        # ── SAME metric as run_benchmark_eval.py ─────────────────────────────
        em, f1, acc = evaluate_prediction(ground_truth, answer)

        em_scores.append(em)
        f1_scores.append(f1)
        acc_scores.append(acc)
        latencies.append(latency_ms)
        bertscore_preds.append(answer)
        bertscore_refs.append(ground_truth)

        results.append({
            "idx":            idx,
            "question":       question,
            "gold_route":     gold_route,
            "executed_route": executed_route,
            "answer":         answer,
            "ground_truth":   ground_truth,
            "em":             em,
            "f1":             f1,
            "acc":            acc,
            "latency_ms":     latency_ms,
            "retrieval_ok":   retrieval_ok,
        })

        if (idx + 1) % 50 == 0:
            print(
                f"  [{idx+1}/{len(test_data)}] "
                f"avg F1={sum(f1_scores)/len(f1_scores):.4f}  "
                f"avg Latency={sum(latencies)/len(latencies):.1f}ms"
            )
            
        # --- INCREMENTAL SAVE ---
        partial_summary = {
            "n_samples":      len(results),
            "routing_acc":    1.0,
            "avg_em":         sum(em_scores)  / len(em_scores) if em_scores else 0,
            "avg_f1":         sum(f1_scores)  / len(f1_scores) if f1_scores else 0,
            "avg_acc":        sum(acc_scores) / len(acc_scores) if acc_scores else 0,
            "avg_latency_ms": sum(latencies)  / len(latencies) if latencies else 0,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps({"summary": partial_summary, "per_sample": results}, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )

    # ── BERTScore (batch) ─────────────────────────────────────────────────────
    print("\nComputing BERTScore (xlm-roberta-large)...")
    bertscore_f1 = compute_bertscore_batch(bertscore_preds, bertscore_refs)

    summary = {
        "n_samples":      len(results),
        "routing_acc":    1.0,
        "avg_em":         sum(em_scores)  / len(em_scores),
        "avg_f1":         sum(f1_scores)  / len(f1_scores),
        "avg_acc":        sum(acc_scores) / len(acc_scores),
        "bertscore_f1":   bertscore_f1,   # None if bert_score not installed
        "avg_latency_ms": sum(latencies)  / len(latencies),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "metric_note": (
            "avg_f1 uses evaluate_prediction(ground_truth, answer) identical to "
            "run_benchmark_eval.py. ground_truth = sample['answer'] = gold_context. "
            "Directly comparable to Pure Vector/Graph/Single-stage/Two-stage rows in Table 3."
        ),
    }

    output = {"summary": summary, "per_sample": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== ORACLE EVALUATION SUMMARY ===")
    for k, v in summary.items():
        if k == "metric_note":
            continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nMetric note: {summary['metric_note']}")
    print(f"\nResults saved → {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path",   default="qa_pipeline/data/legal_strict/test.json")
    parser.add_argument("--output_path", default="results/oracle_eval_v3.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for quick test, e.g. --max_samples 10")
    args = parser.parse_args()

    run_oracle_eval(
        test_path=args.test_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )
