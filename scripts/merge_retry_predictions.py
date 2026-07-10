"""
merge_retry_predictions.py
===========================
Merge kết quả retry (77 câu) vào file predictions sạch (600 câu),
ghi đè những câu bị lỗi/kẹt-fallback bằng kết quả mới.

Chạy NGAY SAU KHI chạy lại pipeline cho test_retry_77.json:
    python scripts/merge_retry_predictions.py \\
        --base  results_final_unified/e2e_benchmark/two_stage_hybrid_preds_clean.jsonl \\
        --retry <output_của_eval_runner_cho_77_câu>.jsonl \\
        --out   results_final_unified/e2e_benchmark/two_stage_hybrid_preds_final.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ERROR_MARKERS = (
    "Lỗi khi tạo câu trả lời:",
    "Lỗi khi tổng hợp câu trả lời",
    "OpenAI generation failed",
    "HTTP 429",
    "Too Many Requests",
)
DIEU85_MARKER = "Dựa trên ngữ cảnh:\n\n[Điều 85"


def is_bad_prediction(text: str) -> bool:
    if any(m in text for m in ERROR_MARKERS):
        return True
    if text.strip().startswith(DIEU85_MARKER.strip()):
        return True
    return False


def load_jsonl(path: str) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
        default="results_final_unified/e2e_benchmark/two_stage_hybrid_preds_clean.jsonl",
        help="File predictions sạch (600 câu, output của clean_and_rerun_failed.py)")
    ap.add_argument("--retry", required=True,
        help="File predictions mới cho 77 câu retry")
    ap.add_argument("--out",
        default="results_final_unified/e2e_benchmark/two_stage_hybrid_preds_final.jsonl",
        help="Output file cuối cùng")
    args = ap.parse_args()

    base = load_jsonl(args.base)
    retry = load_jsonl(args.retry)
    print(f"[LOAD] base={len(base)} dòng, retry={len(retry)} dòng")

    # Index retry records by id
    retry_by_id: dict[str, dict] = {}
    for r in retry:
        qid = str(r.get("id", ""))
        pred = str(r.get("prediction", r.get("answer", "")))
        if qid and not is_bad_prediction(pred):
            retry_by_id[qid] = r
        elif qid:
            print(f"  WARN: retry ID={qid} vẫn bị lỗi/fallback, bỏ qua")

    print(f"[RETRY] {len(retry_by_id)} predictions hợp lệ từ file retry")

    # Merge: ưu tiên retry nếu có, giữ base nếu không
    merged = []
    replaced = 0
    kept_bad = 0

    for r in base:
        qid = str(r.get("id", ""))
        if qid in retry_by_id:
            merged.append(retry_by_id[qid])
            replaced += 1
        else:
            pred = str(r.get("prediction", r.get("answer", "")))
            if is_bad_prediction(pred):
                kept_bad += 1
                print(f"  INFO: ID={qid} vẫn bị lỗi trong base và không có trong retry")
            merged.append(r)

    # Kiểm tra coverage
    retry_not_in_base = {k for k in retry_by_id if k not in {str(r.get('id','')) for r in base}}
    if retry_not_in_base:
        print(f"  WARN: {len(retry_not_in_base)} retry IDs không có trong base: {sorted(retry_not_in_base)[:5]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[MERGE SUMMARY]")
    print(f"  Total records: {len(merged)}")
    print(f"  Replaced by retry: {replaced}")
    print(f"  Kept bad (no valid retry): {kept_bad}")
    print(f"\n[SAVE] -> {out}")
    print(f"\n[NEXT] Chạy re-evaluate (không cần --exclude-ids-file):")
    print(f"  python scripts/re_evaluate_gold_context.py \\")
    print(f"    --pred-file {out} \\")
    print(f"    --test-file qa_pipeline/data/legal_strict/test.json \\")
    print(f"    --output-dir results_final_unified/gold_context_eval_final")


if __name__ == "__main__":
    main()
