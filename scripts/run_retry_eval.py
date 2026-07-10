"""
run_retry_eval.py
==================
Chạy lại đúng 77 câu bị lỗi (rate-limit 429 + Điều-85 fallback) với:
  - Single-threaded (không ThreadPoolExecutor) để tránh blast rate limit
  - Pipeline đã fix (llm_retry_utils.RateLimiter + call_llm_with_backoff)
  - Checkpoint: bỏ qua câu đã có prediction hợp lệ trong output file
  - Không ghi chuỗi lỗi vào predictions — bỏ qua câu nếu exception sau retry

Output: eval_results/retry_two_stage_hybrid_preds.jsonl
Format: {"id":..., "question":..., "prediction":..., "route":..., "latency_ms":...}

Chạy:
    python scripts/run_retry_eval.py
    python scripts/run_retry_eval.py --retry-file qa_pipeline/data/legal_strict/test_retry_77.json
    python scripts/run_retry_eval.py --system two_stage_hybrid --retry-file ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from loguru import logger

from pipeline.llm_retry_utils import load_done_ids, append_result, is_error_prediction

ERROR_MARKERS = (
    "Lỗi khi tạo câu trả lời:",
    "Lỗi khi tổng hợp câu trả lời",
    "OpenAI generation failed",
    "HTTP 429",
    "Too Many Requests",
)
DIEU85_MARKER = "Dựa trên ngữ cảnh:\n\n[Điều 85"


def is_bad_prediction(text: str) -> bool:
    if not text:
        return True
    if any(m in text for m in ERROR_MARKERS):
        return True
    if text.strip().startswith(DIEU85_MARKER.strip()):
        return True
    return False


def run_retry(
    config_path: str,
    retry_file: str,
    output_file: str,
    system: str,
) -> None:
    # ── Load config ──────────────────────────────────────────────────────
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Force eval-answer-style settings (match original run_benchmark_eval.py flags)
    config["concise_answer"] = True
    if config.get("task_type") == "legal_citation":
        config["task_type"] = "legal_eval"
    config.setdefault("openai", {})["max_tokens"] = min(
        int(config.get("openai", {}).get("max_tokens", 1024)), 384
    )
    for store_key in ("faiss", "chroma"):
        if store_key in config:
            config[store_key]["top_k"] = max(int(config[store_key].get("top_k", 8)), 12)
    config.setdefault("rag", {})["max_context_chars"] = max(
        int(config.get("rag", {}).get("max_context_chars", 12000)), 18000
    )
    config["rag"]["max_chunk_chars"] = max(
        int(config["rag"].get("max_chunk_chars", 1500)), 2200
    )
    config["rag"]["hybrid_vector_top_k"] = max(
        int(config["rag"].get("hybrid_vector_top_k", 5)), 5
    )
    config["rag"]["hybrid_vector_candidate_k"] = max(
        int(config["rag"].get("hybrid_vector_candidate_k", 15)), 15
    )
    config["rag"]["retrieval_candidate_multiplier"] = max(
        int(config["rag"].get("retrieval_candidate_multiplier", 3)), 3
    )
    config["rag"]["hybrid_vector_chunk_chars"] = max(
        int(config["rag"].get("hybrid_vector_chunk_chars", 1000)), 1200
    )
    config["rag"]["hybrid_graph_top_k"] = max(
        int(config["rag"].get("hybrid_graph_top_k", 3)), 4
    )
    config["rag"]["hybrid_graph_context_chars"] = max(
        int(config["rag"].get("hybrid_graph_context_chars", 2500)), 3500
    )

    # ── Load retry queries ───────────────────────────────────────────────
    retry_data = json.loads(Path(retry_file).read_text(encoding="utf-8"))
    if isinstance(retry_data, dict):
        retry_items = retry_data.get("data", retry_data.get("items", []))
    else:
        retry_items = retry_data

    print(f"[LOAD] {len(retry_items)} câu cần retry từ {retry_file}")

    # ── Checkpoint: bỏ qua câu đã có prediction hợp lệ ─────────────────
    done_ids = load_done_ids(output_file)
    print(f"[CHECKPOINT] {len(done_ids)} câu đã có prediction hợp lệ trong {output_file}")

    remaining = [
        item for item in retry_items
        if str(item.get("id", "")) not in done_ids
    ]
    print(f"[TODO] {len(remaining)} câu cần chạy\n")

    if not remaining:
        print("Tất cả đã xong! Không cần chạy thêm.")
        return

    # ── Init pipeline ───────────────────────────────────────────────────
    print(f"[INIT] Loading pipeline: {system}...")
    from pipeline.hybrid_pipeline import HybridPipeline
    from rag.vector_rag import VectorRAG
    from rag.graph_rag_adapter import GraphRAGAdapter
    import copy

    if system == "pure_vector":
        pipeline_obj = VectorRAG(config)
        pipeline_obj.load_index()
        pipeline_obj.retriever.embedder._load_model()
    elif system == "pure_graph":
        pipeline_obj = GraphRAGAdapter(config)
    elif system == "single_stage_router":
        cfg = copy.deepcopy(config)
        cfg.setdefault("router", {}).setdefault("stage2", {})["enabled"] = False
        pipeline_obj = HybridPipeline.from_config(cfg)
        try:
            _ = pipeline_obj.vector_rag
            pipeline_obj.vector_rag.retriever.embedder._load_model()
        except Exception as e:
            logger.warning("Preload vector_rag: {}", e)
        try:
            _ = pipeline_obj.graph_rag
        except Exception as e:
            logger.warning("Preload graph_rag: {}", e)
    else:  # two_stage_hybrid (default)
        pipeline_obj = HybridPipeline.from_config(config)
        try:
            _ = pipeline_obj.vector_rag
            pipeline_obj.vector_rag.retriever.embedder._load_model()
        except Exception as e:
            logger.warning("Preload vector_rag: {}", e)
        try:
            _ = pipeline_obj.graph_rag
        except Exception as e:
            logger.warning("Preload graph_rag: {}", e)

    print(f"[INIT] Done. Starting single-threaded retry run...\n")

    # ── Main loop — single-threaded, with progress ───────────────────────
    ok = 0
    skipped_err = 0

    for i, item in enumerate(remaining, 1):
        qid = str(item.get("id", f"item_{i}"))
        query = str(item.get("question", item.get("query", "")))
        routing_label = item.get("routing_label", "")

        print(f"[{i:3d}/{len(remaining)}] {qid} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        try:
            if system == "pure_vector":
                result = pipeline_obj.answer(query, history=None)
                prediction = result.answer
                route = "dense_retrieval"
                latency_ms = result.latency_ms
            elif system == "pure_graph":
                result = pipeline_obj.answer(query, history=None)
                prediction = result.get("answer", "")
                route = "graph_traversal"
                latency_ms = result.get("latency_ms", (time.perf_counter() - t0) * 1000)
            else:
                response = pipeline_obj.query(query, session_id=f"retry_{qid}", verbose=False)
                prediction = response.answer
                route = response.route_used
                latency_ms = response.latency_ms

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("FAILED {}: {} (after {:.0f}ms)", qid, exc, latency_ms)
            print(f"ERROR ({latency_ms:.0f}ms) — bỏ qua, không ghi vào output")
            skipped_err += 1
            continue

        # Kiểm tra prediction có hợp lệ không
        if is_bad_prediction(prediction):
            print(f"BAD_PRED ({latency_ms:.0f}ms) — vẫn bị lỗi/fallback, bỏ qua")
            skipped_err += 1
            continue

        # Ghi checkpoint
        record = {
            "id": qid,
            "question": query,
            "prediction": prediction,
            "route": route,
            "latency_ms": round(latency_ms, 1),
            "routing_label": routing_label,
        }
        append_result(output_file, record)
        ok += 1
        print(f"OK ({latency_ms:.0f}ms) route={route}")

    # ── Tổng kết ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RETRY COMPLETE")
    print(f"  OK (ghi thành công) : {ok}")
    print(f"  Bỏ qua (vẫn lỗi)   : {skipped_err}")
    print(f"  Output              : {output_file}")
    print(f"{'='*60}")

    if skipped_err > 0:
        print(f"\n[WARN] {skipped_err} câu vẫn thất bại sau retry.")
        print("  → Chạy lại script này một lần nữa (checkpoint tự skip các câu đã OK)")
        print("  → Hoặc tăng min_request_interval_sec trong config.yaml (hiện 1.5s)")

    print(f"\n[NEXT] Merge + re-evaluate:")
    print(f"  python scripts/merge_retry_predictions.py \\")
    print(f"    --base  results_final_unified/e2e_benchmark/two_stage_hybrid_preds_clean.jsonl \\")
    print(f"    --retry {output_file} \\")
    print(f"    --out   results_final_unified/e2e_benchmark/two_stage_hybrid_preds_final.jsonl")
    print(f"  python scripts/re_evaluate_gold_context.py \\")
    print(f"    --pred-file results_final_unified/e2e_benchmark/two_stage_hybrid_preds_final.jsonl \\")
    print(f"    --output-dir results_final_unified/gold_context_eval_final")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-run 77 failed queries single-threaded with rate-limiting"
    )
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument(
        "--retry-file",
        default="qa_pipeline/data/legal_strict/test_retry_77.json",
        help="JSON file chứa các câu cần chạy lại (output của clean_and_rerun_failed.py)"
    )
    ap.add_argument(
        "--output",
        default="eval_results/retry_two_stage_hybrid_preds.jsonl",
        help="File JSONL output (append-mode, checkpoint-safe)"
    )
    ap.add_argument(
        "--system",
        default="two_stage_hybrid",
        choices=["pure_vector", "pure_graph", "single_stage_router", "two_stage_hybrid"],
        help="Hệ thống cần chạy lại"
    )
    args = ap.parse_args()

    run_retry(
        config_path=args.config,
        retry_file=args.retry_file,
        output_file=args.output,
        system=args.system,
    )


if __name__ == "__main__":
    main()
