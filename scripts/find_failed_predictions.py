"""
find_failed_predictions.py
===========================
Quét file predictions (JSONL) hoặc gold_context_details.jsonl để:

  1. Đếm số câu bị lỗi do rate-limit (chuỗi "Lỗi khi tạo câu trả lời:
     OpenAI generation failed after 3 attempts: HTTP 429 Too Many
     Requests...").
  2. Phát hiện các "fallback" GIỐNG NHAU lặp lại giữa nhiều id khác nhau
     (dấu hiệu generation thất bại nhưng bị catch âm thầm và trả về một
     đoạn context/placeholder cố định — vd nhiều câu hoàn toàn khác route
     và gold_context vẫn nhận đúng cùng một đoạn
     "Dựa trên ngữ cảnh:\\n\\n[Điều 85 — 49/VBHN-VPQH]...").
  3. Xuất danh sách id cần chạy lại -> failed_query_ids.json, để bạn chỉ
     re-run đúng các câu này (vd 80/600 ~ 13%) thay vì cả 600 câu.

Usage:
  # Trên gold_context_details.jsonl (có sẵn field prediction_preview):
  python scripts/find_failed_predictions.py \\
      --input results_final_unified/gold_context_eval/gold_context_details.jsonl \\
      --output failed_query_ids.json

  # Trên file predictions gốc (field "prediction" đầy đủ):
  python scripts/find_failed_predictions.py \\
      --input results_final_unified/e2e_benchmark/two_stage_hybrid_preds.jsonl \\
      --pred-field prediction \\
      --output failed_query_ids.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Trùng với pipeline/llm_retry_utils.py — khai báo lại tại đây để script
# này CHẠY ĐỘC LẬP, không phụ thuộc import path của project.
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


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def get_pred_text(item: dict, pred_field: str | None) -> str:
    """Lấy text prediction, ưu tiên pred_field nếu chỉ định, sau đó thử
    các tên field phổ biến (prediction_preview, prediction, answer)."""
    if pred_field:
        return str(item.get(pred_field, ""))
    return str(
        item.get("prediction_preview")
        or item.get("prediction")
        or item.get("answer")
        or ""
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Scan predictions JSONL for failed/corrupted entries "
            "(API error strings or repeated fallback text) and output "
            "their IDs to a JSON file for selective re-running."
        )
    )
    ap.add_argument("--input", required=True,
                    help="Predictions JSONL file (predictions or gold_context_details)")
    ap.add_argument("--pred-field", default=None,
                    help="Name of the prediction field. Leave blank to auto-detect "
                         "(tries prediction_preview -> prediction -> answer).")
    ap.add_argument("--output", default="failed_query_ids.json")
    ap.add_argument("--dup-threshold", type=int, default=3,
                    help="Minimum number of times a prediction must repeat across "
                         "different IDs to be considered a suspicious fallback "
                         "(default: 3)")
    ap.add_argument("--dup-prefix-len", type=int, default=150,
                    help="Compare duplicates by first N characters of prediction "
                         "(default: 150)")
    args = ap.parse_args()

    items = load_jsonl(args.input)
    print(f"[LOAD] {len(items)} dòng từ {args.input}")

    error_entries: list[dict] = []
    by_route_total: Counter = Counter()
    by_route_error: Counter = Counter()
    pred_to_ids: defaultdict[str, list[str]] = defaultdict(list)

    for item in items:
        qid = str(item.get("id", ""))
        route = str(item.get("route", "unknown"))
        pred = get_pred_text(item, args.pred_field)
        by_route_total[route] += 1

        if is_error_prediction(pred):
            error_entries.append({
                "id": qid, "route": route, "reason": "api_error",
                "preview": pred[:100],
            })
            by_route_error[route] += 1
            continue

        key = pred.strip()[: args.dup_prefix_len]
        if key:
            pred_to_ids[key].append(qid)

    # Phát hiện fallback giống nhau lặp lại nhiều lần
    dup_entries: list[dict] = []
    for pred_text, ids in pred_to_ids.items():
        if len(ids) >= args.dup_threshold:
            for qid in ids:
                dup_entries.append({
                    "id": qid, "reason": "duplicate_fallback",
                    "repeat_count": len(ids), "preview": pred_text[:100],
                })

    all_failed = error_entries + dup_entries
    failed_ids_set = sorted({d["id"] for d in all_failed if d["id"]})
    total = sum(by_route_total.values())

    print("\n[KẾT QUẢ]")
    print(f"  Tổng số câu                 : {total}")
    print(f"  Lỗi API (429/timeout/...)   : {len(error_entries)}")
    print(f"  Fallback lặp lại (>= {args.dup_threshold} lần)  : {len(dup_entries)}")
    pct = len(failed_ids_set) / total if total else 0.0
    print(f"  TỔNG CẦN CHẠY LẠI           : {len(failed_ids_set)} ({pct:.1%})")

    print("\n  Theo route:")
    for route, total_r in by_route_total.items():
        err = by_route_error.get(route, 0)
        print(f"    {route:<20} {err}/{total_r} lỗi API")

    if dup_entries:
        print("\n  Ví dụ fallback lặp lại (preview):")
        seen_preview: set[str] = set()
        for d in dup_entries:
            if d["preview"] not in seen_preview:
                seen_preview.add(d["preview"])
                print(f"    (x{d['repeat_count']}) {d['preview']}")

    out = {
        "input_file": args.input,
        "total": total,
        "n_api_error": len(error_entries),
        "n_duplicate_fallback": len(dup_entries),
        "n_failed": len(failed_ids_set),
        "failed_ids": failed_ids_set,
        "details": all_failed,
    }
    Path(args.output).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[SAVE] {args.output}")

    if failed_ids_set:
        print(
            "\n[NEXT] Chạy lại CHỈ các id trong failed_ids (vd lọc test.json "
            "theo failed_ids rồi chạy benchmark trên tập con đó), sau đó "
            "merge kết quả mới vào file predictions cũ trước khi tổng hợp "
            "lại. Xem llm_retry_utils.load_done_ids()/append_result() để "
            "resume tự động."
        )


if __name__ == "__main__":
    main()
