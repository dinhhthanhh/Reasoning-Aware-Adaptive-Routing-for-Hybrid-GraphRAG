"""
re_evaluate_gold_context.py (v2)
=================================
So với bản trước (do người dùng tự sửa), bản v2 này thêm:

  1. PHÁT HIỆN VÀ LOẠI prediction là THÔNG BÁO LỖI API (rate-limit 429,
     "OpenAI generation failed after 3 attempts"...) khỏi các số liệu
     trung bình — đồng thời BÁO CÁO RIÊNG tỷ lệ lỗi này (api_error_rate).
     Lần chạy gần nhất có ~80/600 (~13%) câu rơi vào trường hợp này, làm
     méo toàn bộ kết quả tổng hợp. Không nên đưa số liệu khi
     api_error_rate > 0 vào paper.

  2. Báo cáo CẢ Token F1 VÀ Token Recall vs gold_context (bản trước chỉ
     trả Recall nhưng đặt tên là "f1", dễ gây nhầm khi trích vào paper).

  3. Tự phát hiện nếu trường concise_answer KHÔNG có trong test file
     (toàn bộ rỗng). Nếu vậy, KHÔNG báo cáo "vs concise_answer" (bản
     trước fallback dùng gold_context khi concise_answer rỗng -> hai cột
     trùng hệt nhau, như thấy trong gold_context_details.jsonl lần trước).

  4. --exclude-ids-file: nhận output của find_failed_predictions.py để
     loại trừ thêm các id "fallback lặp lại" (không có marker lỗi rõ
     ràng nhưng cùng một đoạn text xuất hiện ở nhiều id khác nhau), để
     tính "clean metrics" trên phần dữ liệu chắc chắn hợp lệ trong khi
     chờ chạy lại phần còn thiếu.

Output:
  <output_dir>/gold_context_metrics.json
  <output_dir>/gold_context_details.jsonl
  <output_dir>/gold_context_report.txt

Chạy:
  python scripts/re_evaluate_gold_context.py \\
      --pred-file results_final_unified/e2e_benchmark/two_stage_hybrid_preds.jsonl \\
      --test-file qa_pipeline/data/legal_strict/test.json \\
      --output-dir results_final_unified/gold_context_eval \\
      --exclude-ids-file failed_query_ids.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# -----------------------------------------------------------------------
# Token-level metrics
# -----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def token_scores(pred: str, gold: str) -> tuple[float, float, float]:
    """Trả về (f1, precision, recall)."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


# -----------------------------------------------------------------------
# Phát hiện prediction = thông báo lỗi (rate-limit / API failure)
# Đồng bộ với pipeline/llm_retry_utils.ERROR_PREDICTION_MARKERS
# -----------------------------------------------------------------------
ERROR_PREDICTION_MARKERS = (
    "Lỗi khi tạo câu trả lời:",
    "Lỗi khi tổng hợp câu trả lời",
    "OpenAI generation failed",
    "HTTP 429",
    "Too Many Requests",
)


def is_error_prediction(text: str) -> bool:
    if not text:
        return False
    return any(marker in text for marker in ERROR_PREDICTION_MARKERS)


# -----------------------------------------------------------------------
# BERTScore (optional)
# -----------------------------------------------------------------------
def compute_bertscore(preds: list[str], golds: list[str],
                      model: str = "xlm-roberta-large") -> dict | None:
    try:
        from bert_score import score as bs_score  # noqa: PLC0415
        P, R, F1 = bs_score(
            preds, golds, model_type=model, lang="vi",
            verbose=False, batch_size=8,
        )
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
            "model": model,
        }
    except ImportError:
        print("[WARN] bert_score chưa được cài. pip install bert-score")
        return None


# -----------------------------------------------------------------------
# Load helpers
# -----------------------------------------------------------------------
def load_predictions(pred_file: str) -> tuple[dict[str, str], dict[str, str]]:
    path = Path(pred_file)
    if not path.exists():
        print(f"[ERROR] Prediction file not found: {pred_file}")
        return {}, {}

    preds_by_id: dict[str, str] = {}
    preds_by_q: dict[str, str] = {}
    content = path.read_text(encoding="utf-8").strip()

    def _extract(obj: dict) -> tuple[str, str, str]:
        qid = str(obj.get("id", obj.get("question_id", "")))
        q = str(obj.get("question", obj.get("query", "")))
        pred = str(obj.get("prediction", obj.get("answer", obj.get("generated", ""))))
        return qid, q, pred

    if content.startswith("{"):
        for line in content.splitlines():
            try:
                qid, q, pred = _extract(json.loads(line))
                if qid:
                    preds_by_id[qid] = pred
                if q:
                    preds_by_q[q] = pred
            except (json.JSONDecodeError, KeyError):
                continue
    else:
        try:
            data = json.loads(content)
            items = data if isinstance(data, list) else data.get("results", [])
            for obj in items:
                qid, q, pred = _extract(obj)
                if qid:
                    preds_by_id[qid] = pred
                if q:
                    preds_by_q[q] = pred
        except json.JSONDecodeError:
            pass

    print(
        f"[LOAD] {len(preds_by_id)} predictions (by id), "
        f"{len(preds_by_q)} (by question) from {pred_file}"
    )
    return preds_by_id, preds_by_q


def load_test_set(test_file: str) -> list[dict]:
    path = Path(test_file)
    content = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print(f"[LOAD] {len(data)} test samples from {test_file}")
            return data
    except json.JSONDecodeError:
        pass

    samples: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    print(f"[LOAD] {len(samples)} test samples from {test_file}")
    return samples


def load_exclude_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        print(f"[WARN] --exclude-ids-file không tồn tại: {path}")
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    ids = data.get("failed_ids", data if isinstance(data, list) else [])
    return {str(i) for i in ids}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def evaluate(
    pred_file: str,
    test_file: str,
    output_dir: str,
    run_bertscore: bool = False,
    exclude_ids_file: str | None = None,
) -> dict | None:

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_by_id, preds_by_q = load_predictions(pred_file)
    test_samples = load_test_set(test_file)
    extra_exclude = load_exclude_ids(exclude_ids_file)

    if not preds_by_id and not preds_by_q:
        print("[ERROR] Không load được predictions.")
        return None

    matched: list[dict] = []
    skipped_no_ctx = 0
    skipped_no_pred = 0

    for gold_obj in test_samples:
        qid = str(gold_obj.get("id", ""))
        question = str(gold_obj.get("question", ""))
        gold_context = gold_obj.get(
            "gold_context", gold_obj.get("answer", gold_obj.get("context", ""))
        )
        concise_ans = str(gold_obj.get("concise_answer", "")).strip()

        pred = preds_by_id.get(qid) or preds_by_q.get(question) or ""

        if not gold_context:
            skipped_no_ctx += 1
            continue
        if not pred:
            skipped_no_pred += 1
            continue

        matched.append({
            "id": qid,
            "question": question,
            "prediction": pred,
            "gold_context": gold_context,
            "concise_answer": concise_ans,
            "route": gold_obj.get("routing_label", "unknown"),
        })

    print(
        f"\n[EVAL] Matched: {len(matched)} | "
        f"Skipped (no ctx): {skipped_no_ctx} | "
        f"Skipped (no pred): {skipped_no_pred}"
    )
    if not matched:
        print("[ERROR] Không có sample nào matched.")
        return None

    # --- Phân loại valid / api_error / excluded ---
    n_error = 0
    n_excluded_extra = 0
    valid_items: list[dict] = []

    for item in matched:
        if item["id"] in extra_exclude:
            n_excluded_extra += 1
            continue
        if is_error_prediction(item["prediction"]):
            n_error += 1
            continue
        valid_items.append(item)

    n_total = len(matched)
    n_valid = len(valid_items)

    # --- concise_answer có khả dụng không? ---
    concise_available = any(it["concise_answer"] for it in valid_items)

    # --- Tính metric trên valid_items ---
    by_route_f1: defaultdict[str, list[float]] = defaultdict(list)
    by_route_recall: defaultdict[str, list[float]] = defaultdict(list)
    all_f1: list[float] = []
    all_precision: list[float] = []
    all_recall: list[float] = []
    all_em: list[float] = []
    all_f1_ans: list[float] = []
    all_recall_ans: list[float] = []
    details: list[dict] = []

    for item in valid_items:
        f1c, pc, rc = token_scores(item["prediction"], item["gold_context"])
        emc = exact_match(item["prediction"], item["gold_context"])

        all_f1.append(f1c)
        all_precision.append(pc)
        all_recall.append(rc)
        all_em.append(emc)
        by_route_f1[item["route"]].append(f1c)
        by_route_recall[item["route"]].append(rc)

        row = {
            "id": item["id"],
            "question": item["question"][:80],
            "route": item["route"],
            "f1_vs_gold_context": round(f1c, 4),
            "precision_vs_gold_context": round(pc, 4),
            "recall_vs_gold_context": round(rc, 4),
            "em_vs_gold_context": int(emc),
            "prediction_preview": item["prediction"][:120],
            "gold_context_preview": item["gold_context"][:120],
        }

        if concise_available and item["concise_answer"]:
            f1a, _, ra = token_scores(item["prediction"], item["concise_answer"])
            all_f1_ans.append(f1a)
            all_recall_ans.append(ra)
            row["f1_vs_concise_answer"] = round(f1a, 4)
            row["recall_vs_concise_answer"] = round(ra, 4)

        details.append(row)

    # Ghi nhận các id bị loại (để truy ngược dễ dàng), không tính vào trung bình
    for item in matched:
        if item["id"] in extra_exclude:
            details.append({
                "id": item["id"], "question": item["question"][:80],
                "route": item["route"], "status": "EXCLUDED_BY_FILE",
                "prediction_preview": item["prediction"][:120],
            })
        elif is_error_prediction(item["prediction"]):
            details.append({
                "id": item["id"], "question": item["question"][:80],
                "route": item["route"], "status": "EXCLUDED_API_ERROR",
                "prediction_preview": item["prediction"][:120],
            })

    bertscore_result = None
    if run_bertscore and valid_items:
        print("\n[BERT] Computing BERTScore vs gold_context...")
        bertscore_result = compute_bertscore(
            [m["prediction"] for m in valid_items],
            [m["gold_context"] for m in valid_items],
        )

    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    metrics = {
        "n_total_matched": n_total,
        "n_excluded_extra": n_excluded_extra,
        "n_api_error": n_error,
        "n_valid": n_valid,
        "api_error_rate": round(n_error / n_total, 4) if n_total else 0.0,
        "token_f1_vs_gold_context": _avg(all_f1),
        "token_precision_vs_gold_context": _avg(all_precision),
        "token_recall_vs_gold_context": _avg(all_recall),
        "em_vs_gold_context": _avg(all_em),
        "concise_answer_available": concise_available,
        "token_f1_vs_concise_answer": _avg(all_f1_ans) if concise_available else None,
        "token_recall_vs_concise_answer": _avg(all_recall_ans) if concise_available else None,
        "bertscore_vs_gold_context": bertscore_result,
        "per_route_f1": {r: _avg(v) for r, v in by_route_f1.items()},
        "per_route_recall": {r: _avg(v) for r, v in by_route_recall.items()},
    }

    metrics_path = out_dir / "gold_context_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    details_path = out_dir / "gold_context_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Report ---
    sep = "=" * 60
    lines = [sep, "  GOLD CONTEXT EVALUATION REPORT (v2)", sep, ""]

    if n_error:
        lines += [
            f"  \u26a0\ufe0f  CẢNH BÁO: {n_error}/{n_total} câu "
            f"({metrics['api_error_rate']:.1%}) có prediction là THÔNG BÁO "
            f"LỖI API (rate-limit/429) -- đã LOẠI khỏi số liệu dưới đây.",
            "  \u26a0\ufe0f  KHÔNG đưa số liệu này vào paper cho tới khi chạy lại "
            "các câu lỗi (xem scripts/find_failed_predictions.py +",
            "      pipeline/llm_retry_utils.py) và re-aggregate.",
            "",
        ]
    if n_excluded_extra:
        lines.append(
            f"  (Đã loại thêm {n_excluded_extra} câu theo --exclude-ids-file)"
        )
        lines.append("")

    lines += [
        f"  N (valid) = {n_valid} / {n_total}",
        "",
        f"  Token F1        vs gold_context : {metrics['token_f1_vs_gold_context']}",
        f"  Token Precision vs gold_context : {metrics['token_precision_vs_gold_context']}",
        f"  Token Recall    vs gold_context : {metrics['token_recall_vs_gold_context']}",
        f"  EM              vs gold_context : {metrics['em_vs_gold_context']}",
        "",
    ]

    if concise_available:
        lines += [
            f"  Token F1     vs concise_answer  : {metrics['token_f1_vs_concise_answer']}",
            f"  Token Recall vs concise_answer  : {metrics['token_recall_vs_concise_answer']}",
            "",
        ]
    else:
        lines += [
            "  (concise_answer KHÔNG có trong test file -> bỏ qua cột này)",
            "",
        ]

    lines.append("  Token F1 vs gold_context BY ROUTE:")
    for r, v in (metrics["per_route_f1"] or {}).items():
        lines.append(f"    {r:<35} F1={v}  Recall={metrics['per_route_recall'].get(r)}")

    if bertscore_result:
        lines += [
            "",
            "  BERTScore vs gold_context:",
            f"    Precision : {bertscore_result['precision']:.4f}",
            f"    Recall    : {bertscore_result['recall']:.4f}",
            f"    F1        : {bertscore_result['f1']:.4f}",
            f"    Model     : {bertscore_result['model']}",
        ]

    lines += [
        "",
        "  GIẢI THÍCH:",
        "  - Token F1/Recall vs gold_context cao → hệ thống đang trích nguyên văn (faithfulness ↑)",
        "  - Token F1 vs concise_answer thấp → bình thường (verbatim vs. tóm tắt)",
        "  - Dùng token_f1_vs_gold_context + token_recall_vs_gold_context trong paper",
        "  - api_error_rate > 0 → CẦN chạy lại phần thiếu trước khi đưa vào paper",
        "",
        f"  Saved: {metrics_path}",
        f"         {details_path}",
        sep,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "gold_context_report.txt").write_text(report, encoding="utf-8")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Re-evaluate predictions against gold_context (faithfulness metric) — v2"
    )
    ap.add_argument("--pred-file", required=True,
                    help="Prediction JSONL/JSON file")
    ap.add_argument("--test-file", default="qa_pipeline/data/legal_strict/test.json")
    ap.add_argument("--output-dir", default="results_final_unified/gold_context_eval")
    ap.add_argument("--bertscore", action="store_true",
                    help="Also compute BERTScore vs gold_context (slow, needs GPU)")
    ap.add_argument("--exclude-ids-file", default=None,
                    help="Path to failed_query_ids.json from find_failed_predictions.py "
                         "to additionally exclude from averages")
    args = ap.parse_args()

    evaluate(
        pred_file=args.pred_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        run_bertscore=args.bertscore,
        exclude_ids_file=args.exclude_ids_file,
    )
