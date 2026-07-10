import json
import argparse
import os
import string
from collections import Counter, defaultdict
from typing import Optional

ROUTE_LABELS = ["dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarification"]

def evaluate_routing(benchmark: list[dict], predictions: list[dict]) -> dict:
    y_true = []
    y_pred = []
    
    for r, p in zip(benchmark, predictions):
        yt = r.get("routing_label", "")
        yp = p.get("predicted_route", "MISSING")
        
        # Label remapping for Stage 2 leakage
        if "->" in yp:
            yp = "hybrid_reasoning"
        elif yp == "clarify":
            yp = "dense_retrieval" # Map clarify to dense as default fallback
            
        y_true.append(yt)
        y_pred.append(yp)

    labels = sorted(set(y_true + y_pred))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    for yt, yp in zip(y_true, y_pred):
        confusion[yt][yp] += 1
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    per_class = {}
    for label in labels:
        support = tp[label] + fn[label]
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {"precision": p, "recall": r, "f1": f, "support": support}

    valid_labels = [l for l in labels if per_class[l]["support"] > 0]
    macro_p = sum(per_class[l]["precision"] for l in valid_labels) / len(valid_labels) if valid_labels else 0
    macro_r = sum(per_class[l]["recall"] for l in valid_labels) / len(valid_labels) if valid_labels else 0
    macro_f = sum(per_class[l]["f1"] for l in valid_labels) / len(valid_labels) if valid_labels else 0

    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true) if len(y_true) > 0 else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "n_total": len(y_true),
    }

def _ground_truth_article_ids(record: dict) -> list[str]:
    ids = []
    for sf in record.get("supporting_facts", []):
        law = sf.get("law_id", "").strip()
        art = sf.get("article_id", "").strip()
        if law and art:
            ids.append(f"{law}::{art}")
    return ids

def _reciprocal_rank(retrieved: list[str], relevant: set) -> float:
    for i, art in enumerate(retrieved):
        if any(_fuzzy_match(art, r_gt) for r_gt in relevant):
            return 1.0 / (i + 1)
    return 0.0

def _dcg(retrieved: list[str], relevant: set, k: int) -> float:
    import math
    dcg = 0.0
    for i, art in enumerate(retrieved[:k]):
        if any(_fuzzy_match(art, r_gt) for r_gt in relevant):
            dcg += 1.0 / math.log2(i + 2)
    return dcg

def _idcg(n_relevant: int, k: int) -> float:
    import math
    idcg = 0.0
    for i in range(min(n_relevant, k)):
        idcg += 1.0 / math.log2(i + 2)
    return idcg

import re

def _fuzzy_match(retrieved_id: str, gold_id: str) -> bool:
    if gold_id in retrieved_id: return True
    
    # Extract core parts from gold_id
    gold_parts = [p.upper() for p in re.findall(r'[A-ZĐÀ-Ỹa-zđà-ỹ]+|\d+', gold_id)]
    retrieved_parts = set(p.upper() for p in re.findall(r'[A-ZĐÀ-Ỹa-zđà-ỹ]+|\d+', retrieved_id))
    
    match_count = 0
    required = 0
    # Stop before 'ĐIỀU' if we just want document-level match
    # Or just require 2 out of 3 core law parts (number, year, type)
    law_parts = []
    for p in gold_parts:
        if p in ["ĐIỀU", "KHOẢN", "ĐIỂM"]: break
        if p not in ["CP", "UBND", "BKHCN", "BCT"]:
            law_parts.append(p)
            
    for p in law_parts:
        required += 1
        if p.isdigit() and len(p) == 4 and (p in retrieved_parts or p[-2:] in retrieved_parts):
            match_count += 1
        elif p in retrieved_parts:
            match_count += 1
            
    return required > 0 and (match_count / required) >= 0.66

def evaluate_retrieval(benchmark: list[dict], predictions: list[dict], k_values: list[int] = [1, 3, 5]) -> dict:
    hit_counts = {k: 0 for k in k_values}
    mrr_total = 0.0
    ndcg5_total = 0.0
    n_evaluated = 0
    missing = 0

    for record, pred in zip(benchmark, predictions):
        gt_ids = _ground_truth_article_ids(record)
        if not gt_ids:
            missing += 1
            continue

        retrieved = pred.get("retrieved_articles", [])
        relevant = set(gt_ids)

        for k in k_values:
            # Fuzzy match any retrieved chunk against any relevant ID
            if any(any(_fuzzy_match(art, r_gt) for r_gt in relevant) for art in retrieved[:k]):
                hit_counts[k] += 1

        mrr_total += _reciprocal_rank(retrieved, relevant)
        idcg = _idcg(len(relevant), k=5)
        if idcg > 0:
            ndcg5_total += _dcg(retrieved, relevant, k=5) / idcg

        n_evaluated += 1

    if n_evaluated == 0:
        return {"error": "No records with ground truth articles found."}

    result = {
        "n_evaluated": n_evaluated,
        "n_missing_ground_truth": missing,
        "MRR": mrr_total / n_evaluated,
        "NDCG@5": ndcg5_total / n_evaluated,
    }
    for k in k_values:
        result[f"Hit@{k}"] = hit_counts[k] / n_evaluated

    return result

def evaluate_bertscore(benchmark: list[dict], predictions: list[dict], lang: str = "vi", batch_size: int = 32, verbose: bool = True) -> dict:
    try:
        from bert_score import score as bertscore
    except ImportError:
        return {"error": "bert-score not installed. Run: pip install bert-score"}

    hypotheses = []
    references = []

    for i, (record, pred) in enumerate(zip(benchmark, predictions)):
        hyp = pred.get("generated_answer", "").strip()
        ref = record.get("gold_context", record.get("evidence", "")).strip()
        if hyp and ref:
            hypotheses.append(hyp)
            references.append(ref)

    if not hypotheses:
        return {"error": "No valid (answer, reference) pairs found."}

    model_type = "vinai/phobert-base-v2"
    if verbose:
        print(f"[BERTScore] Computing for {len(hypotheses)} samples using {model_type}...")

    try:
        P, R, F1 = bertscore(hypotheses, references, lang=lang, model_type=model_type, batch_size=batch_size, verbose=False)
    except Exception:
        if verbose:
            print(f"  PhoBERT failed, falling back to multilingual BERT...")
        P, R, F1 = bertscore(hypotheses, references, lang=lang, batch_size=batch_size, verbose=False)

    P = P.numpy()
    R = R.numpy()
    F1 = F1.numpy()

    return {
        "n_evaluated": len(hypotheses),
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
        "bertscore_f1_std": float(F1.std()),
    }

def _normalize_text(text: str) -> str:
    text = text.lower()
    table = str.maketrans("", "", string.punctuation + "–—""''")
    text = text.translate(table)
    text = " ".join(text.split())
    return text

def _token_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(ground_truth).split()

    if not pred_tokens and not truth_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    common = sum((pred_counter & truth_counter).values())
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall

def evaluate_token_f1(benchmark: list[dict], predictions: list[dict], answer_field: str = "concise_answer") -> dict:
    f1_scores, precision_scores, recall_scores, em_scores = [], [], [], []
    skipped = 0

    for record, pred in zip(benchmark, predictions):
        gold = record.get(answer_field, "")
        if not gold and answer_field == "gold_context":
            gold = record.get("evidence", "")
        hyp = pred.get("generated_answer", "")

        if not gold or not hyp:
            skipped += 1
            continue

        f1, prec, rec = _token_f1(hyp, gold)
        em = float(_normalize_text(hyp) == _normalize_text(gold))

        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)
        em_scores.append(em)

    if not f1_scores:
        return {"error": f"No valid samples. answer_field='{answer_field}' may be missing."}

    return {
        "reference_field": answer_field,
        "n_evaluated": len(f1_scores),
        "n_skipped": skipped,
        "token_f1": sum(f1_scores) / len(f1_scores),
        "token_precision": sum(precision_scores) / len(precision_scores),
        "token_recall": sum(recall_scores) / len(recall_scores),
        "exact_match": sum(em_scores) / len(em_scores),
    }

def format_routing_report(routing: dict) -> str:
    lines = [
        "=" * 60,
        "LAYER 1 — ROUTING EVALUATION",
        "=" * 60,
        f"  Accuracy:        {routing['accuracy']:.4f} ({routing['accuracy']*100:.2f}%)",
        f"  Macro F1:        {routing['macro_f1']:.4f}",
        f"  Macro Precision: {routing['macro_precision']:.4f}",
        f"  Macro Recall:    {routing['macro_recall']:.4f}",
        f"  N total:         {routing['n_total']}",
        "",
        "  Per-class breakdown:",
        f"  {'Label':<22} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}",
        "  " + "-" * 50,
    ]
    for label, m in routing["per_class"].items():
        lines.append(f"  {label:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}")
    return "\n".join(lines)

def format_retrieval_report(ret: dict) -> str:
    lines = ["=" * 60, "LAYER 2 — RETRIEVAL EVALUATION", "=" * 60, f"  N evaluated: {ret.get('n_evaluated', 'N/A')}"]
    for key in ["Hit@1", "Hit@3", "Hit@5", "MRR", "NDCG@5"]:
        if key in ret: lines.append(f"  {key:<12}: {ret[key]:.4f}")
    return "\n".join(lines)

def format_answer_report(bs: dict, tf_concise: Optional[dict], tf_gold: Optional[dict]) -> str:
    lines = ["=" * 60, "LAYER 3 — ANSWER QUALITY", "=" * 60]
    lines.append("  3a. BERTScore vs gold_context:")
    if not bs: lines.append("      SKIPPED")
    elif "error" in bs: lines.append(f"      ERROR: {bs['error']}")
    else: lines.append(f"      F1={bs['bertscore_f1']:.4f}  P={bs['bertscore_precision']:.4f}  R={bs['bertscore_recall']:.4f}")

    lines.append("\n  3b. Token F1 vs concise_answer:")
    if tf_concise is None: lines.append("      SKIPPED")
    elif "error" in tf_concise: lines.append(f"      ERROR: {tf_concise['error']}")
    else: lines.append(f"      F1={tf_concise['token_f1']:.4f}  EM={tf_concise['exact_match']:.4f}")

    lines.append("\n  3c. Token F1 vs gold_context [BROKEN BASELINE]:")
    if tf_gold is None: lines.append("      SKIPPED")
    elif "error" in tf_gold: lines.append(f"      ERROR: {tf_gold['error']}")
    else: lines.append(f"      F1={tf_gold['token_f1']:.4f}  EM={tf_gold['exact_match']:.4f}")

    return "\n".join(lines)

def run(args):
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    results = {}
    print("\nEvaluating routing...")
    routing_results = evaluate_routing(benchmark, predictions)
    results["routing"] = routing_results
    print(format_routing_report(routing_results))

    print("\nEvaluating retrieval...")
    retrieval_results = evaluate_retrieval(benchmark, predictions)
    results["retrieval"] = retrieval_results
    print(format_retrieval_report(retrieval_results))

    bs_results = {}
    if not args.skip_bertscore:
        print("\nEvaluating BERTScore...")
        bs_results = evaluate_bertscore(benchmark, predictions, lang=args.bertscore_lang)
    results["bertscore"] = bs_results

    has_concise = any("concise_answer" in r for r in benchmark)
    tf_concise = evaluate_token_f1(benchmark, predictions, "concise_answer") if has_concise else None
    results["token_f1_concise"] = tf_concise

    tf_gold = evaluate_token_f1(benchmark, predictions, "gold_context")
    results["token_f1_gold_context"] = tf_gold

    print(format_answer_report(bs_results, tf_concise, tf_gold))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--bertscore_lang", default="vi")
    p.add_argument("--skip_bertscore", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
