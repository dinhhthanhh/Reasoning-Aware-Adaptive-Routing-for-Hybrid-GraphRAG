import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Stage-2 routing labels that the benchmark does not define.
# Map them to the nearest semantically equivalent benchmark label.
LABEL_REMAP: Dict[str, str] = {
    "dense_retrieval->graph_traversal": "hybrid_reasoning",
    "clarify": "dense_retrieval",  # conservative fallback
}

# Benchmark routing labels (ground-truth space)
VALID_LABELS = {"dense_retrieval", "graph_traversal", "hybrid_reasoning"}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 HELPERS — ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def remap_predicted_label(label: str) -> str:
    """Collapse Stage-2 routing labels into benchmark-compatible labels."""
    return LABEL_REMAP.get(label, label)


def evaluate_routing(
    predictions: List[Dict],
    benchmark: List[Dict],
) -> Dict:
    """
    Compute per-class and macro routing metrics.
    """
    assert len(predictions) == len(benchmark), (
        f"Length mismatch: {len(predictions)} predictions vs {len(benchmark)} benchmark entries"
    )

    y_true, y_pred = [], []
    for pred, gt in zip(predictions, benchmark):
        raw_pred = pred.get("predicted_route", pred.get("route", ""))
        remapped  = remap_predicted_label(raw_pred)
        y_true.append(gt["routing_label"])
        y_pred.append(remapped)

    # Per-class stats
    labels = sorted(VALID_LABELS)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    per_class = {}
    for label in labels:
        n = y_true.count(label)
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        recall    = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {"P": precision, "R": recall, "F1": f1, "N": n}

    accuracy    = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    macro_p     = sum(v["P"] for v in per_class.values()) / len(per_class)
    macro_r     = sum(v["R"] for v in per_class.values()) / len(per_class)
    macro_f1    = sum(v["F1"] for v in per_class.values()) / len(per_class)

    # Count residual out-of-vocabulary predictions (not in VALID_LABELS after remap)
    oov_preds = Counter(p for p in y_pred if p not in VALID_LABELS)

    return {
        "accuracy":        accuracy,
        "macro_precision": macro_p,
        "macro_recall":    macro_r,
        "macro_f1":        macro_f1,
        "per_class":       per_class,
        "oov_predictions": dict(oov_preds),
        "n_total":         len(y_true),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 HELPERS — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def normalize_article_key(law_id: str, article_id: str) -> str:
    """
    Canonical form: 'law_id::article_id'
    - collapse any internal whitespace (incl. embedded newlines) to single space
    - strip leading/trailing whitespace from both parts
    """
    art = re.sub(r"\s+", " ", str(article_id)).strip()
    lid = re.sub(r"\s+", " ", str(law_id)).strip()
    
    # Strip sub-path suffixes: keep only 'Điều \d+[a-zA-Z]*' if present
    match = re.match(r"(Điều\s+\d+[a-zA-Z]*)", art, re.IGNORECASE)
    if match:
        art = match.group(1)
        
    return f"{lid}::{art}"


def gold_facts_for(entry: Dict) -> Tuple[set, bool]:
    """
    Returns (canonical_fact_set, has_toan_van).

    'Toàn văn' entries mean the ground-truth is the full document, not a
    specific article chunk.  Such queries are excluded from Hit@K evaluation
    because no article-level retrieval system can produce a fair match.
    """
    gold = set()
    has_toan_van = False
    for sf in entry.get("supporting_facts", []):
        key = normalize_article_key(sf["law_id"], sf["article_id"])
        gold.add(key)
        if "Toàn văn" in key:
            has_toan_van = True
    return gold, has_toan_van


def parse_retrieved_articles(pred: Dict) -> List[str]:
    """
    Extract the list of retrieved article keys from a prediction dict.
    Accepts several field name conventions the pipeline might use.
    """
    candidates = (
        pred.get("retrieved_articles")
        or pred.get("sources")
        or pred.get("retrieved_ids")
        or []
    )
    result = []
    for c in candidates:
        if "::" in str(c):
            law_id, art_id = str(c).split("::", 1)
            result.append(normalize_article_key(law_id, art_id))
        else:
            # fallback: already-normalized or unknown format — use as-is
            result.append(normalize_article_key("", str(c)))
    return result


def dcg_at_k(relevant_ranks: List[int], k: int) -> float:
    """Compute DCG@k given ranks (1-indexed) of relevant items."""
    dcg = 0.0
    for rank in relevant_ranks:
        if rank <= k:
            import math
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(relevant_ranks: List[int], n_relevant: int, k: int) -> float:
    """Compute NDCG@k."""
    import math
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevant_ranks, k) / ideal_dcg


def evaluate_retrieval(
    predictions: List[Dict],
    benchmark: List[Dict],
) -> Dict:
    """
    Article-level retrieval evaluation.
    """
    assert len(predictions) == len(benchmark)

    h1, h3, h5, rr_list, ndcg_list = [], [], [], [], []
    skipped_toan_van = 0
    skipped_no_retrieval = 0

    for pred, gt in zip(predictions, benchmark):
        gold, has_toan_van = gold_facts_for(gt)

        # Exclude "Toàn văn" queries from article-level evaluation
        if has_toan_van:
            skipped_toan_van += 1
            continue

        retrieved = parse_retrieved_articles(pred)

        if not retrieved:
            skipped_no_retrieval += 1
            # Treat as zero-hit but still count in denominator
            h1.append(0.0); h3.append(0.0); h5.append(0.0)
            rr_list.append(0.0); ndcg_list.append(0.0)
            continue

        hit_1 = any(r in gold for r in retrieved[:1])
        hit_3 = any(r in gold for r in retrieved[:3])
        hit_5 = any(r in gold for r in retrieved[:5])
        h1.append(float(hit_1))
        h3.append(float(hit_3))
        h5.append(float(hit_5))

        # MRR + NDCG
        relevant_ranks = [i + 1 for i, r in enumerate(retrieved[:10]) if r in gold]
        rr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        rr_list.append(rr)
        ndcg_list.append(ndcg_at_k(relevant_ranks, len(gold), k=5))

    n_eval = len(h1)

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "n_evaluated":         n_eval,
        "n_skipped_toan_van":  skipped_toan_van,
        "n_skipped_no_retrieval": skipped_no_retrieval,
        "hit_at_1":            mean(h1),
        "hit_at_3":            mean(h3),
        "hit_at_5":            mean(h5),
        "mrr":                 mean(rr_list),
        "ndcg_at_5":           mean(ndcg_list),
        "evaluation_note": (
            f"Denominator = {n_eval} (excluded {skipped_toan_van} 'Toàn văn' queries "
            f"from {len(predictions)} total). "
            "Article-level exact match after whitespace normalization."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 HELPERS — ANSWER QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text_vi(text: str) -> str:
    """
    Basic Vietnamese text normalization for token F1.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Remove punctuation (keep Vietnamese characters + alphanumeric + space)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1_em(prediction: str, ground_truth: str) -> Tuple[float, float]:
    """
    Compute token-level F1 and Exact Match between prediction and ground truth.
    """
    pred_norm = normalize_text_vi(prediction)
    gold_norm = normalize_text_vi(ground_truth)

    # EM
    em = 1.0 if pred_norm == gold_norm else 0.0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    if not pred_tokens or not gold_tokens:
        return 0.0, em

    pred_count = Counter(pred_tokens)
    gold_count = Counter(gold_tokens)
    common = sum((pred_count & gold_count).values())

    if common == 0:
        return 0.0, em

    precision = common / sum(pred_count.values())
    recall    = common / sum(gold_count.values())
    f1 = 2 * precision * recall / (precision + recall)
    return f1, em


def evaluate_answer_quality(
    predictions: List[Dict],
    benchmark: List[Dict],
    use_bert_score: bool = True,
) -> Dict:
    """
    Layer 3: answer quality evaluation.
    """
    assert len(predictions) == len(benchmark)

    answers_pred   = [pred.get("answer", pred.get("generated_answer", "")) for pred in predictions]
    answers_concise = [gt.get("concise_answer", gt.get("answer", "")) for gt in benchmark]
    answers_gold    = [gt.get("gold_context", "") for gt in benchmark]

    # ── 3b & 3c: Token F1 / EM ──────────────────────────────────────────────
    f1_concise_list, em_concise_list = [], []
    f1_gold_list, em_gold_list = [], []

    for pred_ans, concise, gold in zip(answers_pred, answers_concise, answers_gold):
        f1_c, em_c = token_f1_em(pred_ans, concise)
        f1_g, em_g = token_f1_em(pred_ans, gold)
        f1_concise_list.append(f1_c)
        em_concise_list.append(em_c)
        f1_gold_list.append(f1_g)
        em_gold_list.append(em_g)

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "n_evaluated": len(predictions),
        "3b_token_f1_vs_concise": mean(f1_concise_list),
        "3b_em_vs_concise":       mean(em_concise_list),
        "3c_token_f1_vs_gold":    mean(f1_gold_list),
        "3c_em_vs_gold":          mean(em_gold_list),
        "3c_note": "gold_context is the FULL article text; Token F1 vs gold is an inflated upper-bound baseline.",
    }

    # ── 3a: BERTScore ────────────────────────────────────────────────────────
    if use_bert_score:
        try:
            from bert_score import score as bert_score_fn
            P, R, F = bert_score_fn(
                answers_pred,
                answers_concise,
                lang="vi",
                model_type="xlm-roberta-large",
                batch_size=16,
                verbose=False,
            )
            results["3a_bertscore_precision"] = P.mean().item()
            results["3a_bertscore_recall"]    = R.mean().item()
            results["3a_bertscore_f1"]        = F.mean().item()
            results["3a_bertscore_model"]     = "xlm-roberta-large"
        except ImportError:
            results["3a_bertscore_error"] = (
                "bert-score not installed. Run: pip install bert-score"
            )
        except Exception as e:
            results["3a_bertscore_error"] = str(e)
    else:
        results["3a_bertscore_skipped"] = True

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def print_routing_report(metrics: Dict) -> None:
    print("\n" + "=" * 62)
    print("LAYER 1 — ROUTING EVALUATION")
    print("=" * 62)
    print(f"  Accuracy:        {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  N total:         {metrics['n_total']}")
    if metrics["oov_predictions"]:
        print(f"  OOV labels (after remap): {metrics['oov_predictions']}")
    print("\n  Per-class breakdown:")
    print(f"  {'Label':<35} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print("  " + "-" * 58)
    for label, s in sorted(metrics["per_class"].items()):
        print(f"  {label:<35} {s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>6.3f} {s['N']:>6}")


def print_retrieval_report(metrics: Dict) -> None:
    print("\n" + "=" * 62)
    print("LAYER 2 — RETRIEVAL EVALUATION  (article-level)")
    print("=" * 62)
    print(f"  N evaluated (excl. Toàn văn): {metrics['n_evaluated']}")
    print(f"  N skipped (Toàn văn):         {metrics['n_skipped_toan_van']}")
    print(f"  N skipped (no retrieval):     {metrics['n_skipped_no_retrieval']}")
    print(f"  Hit@1:    {metrics['hit_at_1']:.4f}")
    print(f"  Hit@3:    {metrics['hit_at_3']:.4f}")
    print(f"  Hit@5:    {metrics['hit_at_5']:.4f}")
    print(f"  MRR:      {metrics['mrr']:.4f}")
    print(f"  NDCG@5:   {metrics['ndcg_at_5']:.4f}")
    print(f"\n  Note: {metrics['evaluation_note']}")


def print_answer_quality_report(metrics: Dict) -> None:
    print("\n" + "=" * 62)
    print("LAYER 3 — ANSWER QUALITY")
    print("=" * 62)
    print(f"  N evaluated: {metrics['n_evaluated']}")
    if "3a_bertscore_f1" in metrics:
        print(f"\n  3a. BERTScore ({metrics.get('3a_bertscore_model', 'unknown')}):")
        print(f"      P={metrics['3a_bertscore_precision']:.4f}  R={metrics['3a_bertscore_recall']:.4f}  F1={metrics['3a_bertscore_f1']:.4f}")
    elif "3a_bertscore_error" in metrics:
        print(f"\n  3a. BERTScore ERROR: {metrics['3a_bertscore_error']}")
    else:
        print("\n  3a. BERTScore: skipped")
    print(f"\n  3b. Token F1 vs concise_answer:  F1={metrics['3b_token_f1_vs_concise']:.4f}  EM={metrics['3b_em_vs_concise']:.4f}")
    print(f"\n  3c. Token F1 vs gold_context [UPPER BOUND BASELINE]:")
    print(f"      F1={metrics['3c_token_f1_vs_gold']:.4f}  EM={metrics['3c_em_vs_gold']:.4f}")
    print(f"      ({metrics['3c_note']})")


def main():
    parser = argparse.ArgumentParser(description="3-Layer Evaluation Pipeline (fixed)")
    parser.add_argument("--benchmark",   required=True, help="Path to test_benchmark_v2.json")
    parser.add_argument("--predictions", required=True, help="Path to predictions.json")
    parser.add_argument("--output",      default="eval_results/final_metrics.json")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore computation")
    args = parser.parse_args()

    print(f"Loading benchmark: {args.benchmark}")
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    print(f"Loading predictions: {args.predictions}")
    with open(args.predictions, encoding="utf-8") as f:
        raw_preds = json.load(f)

    # Normalize predictions to list aligned with benchmark
    if isinstance(raw_preds, dict):
        # Handle both {id: pred_dict} and {predictions: [...]} formats
        if "predictions" in raw_preds:
            predictions = raw_preds["predictions"]
        else:
            # dict keyed by query id — align to benchmark order
            predictions = [raw_preds.get(gt["id"], {}) for gt in benchmark]
    elif isinstance(raw_preds, list):
        predictions = raw_preds
    else:
        raise ValueError(f"Unexpected predictions format: {type(raw_preds)}")

    if len(predictions) != len(benchmark):
        print(f"WARNING: {len(predictions)} predictions vs {len(benchmark)} benchmark entries. Truncating to min.")
        n = min(len(predictions), len(benchmark))
        predictions = predictions[:n]
        benchmark   = benchmark[:n]

    # ── Run all three layers ────────────────────────────────────────────────
    print("\nEvaluating routing...")
    routing_metrics = evaluate_routing(predictions, benchmark)
    print_routing_report(routing_metrics)

    print("\nEvaluating retrieval...")
    retrieval_metrics = evaluate_retrieval(predictions, benchmark)
    print_retrieval_report(retrieval_metrics)

    print("\nEvaluating answer quality...")
    qa_metrics = evaluate_answer_quality(
        predictions, benchmark,
        use_bert_score=not args.no_bertscore
    )
    print_answer_quality_report(qa_metrics)

    # ── Save combined results ───────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined = {
        "layer1_routing":   routing_metrics,
        "layer2_retrieval": retrieval_metrics,
        "layer3_qa":        qa_metrics,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
