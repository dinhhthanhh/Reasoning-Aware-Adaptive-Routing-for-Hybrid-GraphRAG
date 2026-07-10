"""End-to-end comparison harness — the core 'Evaluation' table for the thesis.

Runs the SAME test set through five configurations and reports answer quality,
retrieval quality, latency and routing behaviour side by side:

    pure_vector   : every query forced to dense_retrieval
    pure_graph    : every query forced to graph_traversal
    pure_hybrid   : every query forced to hybrid_reasoning
    router        : the proposed two-stage adaptive router (no force)
    oracle        : each query forced to its gold routing_label (upper bound)

It reuses the project's existing metrics so the numbers are consistent with
other scripts:
    * evaluation.metrics.evaluate_prediction  -> (EM, token-F1, contains)
    * evaluation.metrics.id_normalizer        -> source/citation Hit@k vs gold

Why this is the most important experiment
-----------------------------------------
Stage-1 routing accuracy is inflated by label leakage (routing_label is derived
from hop_count/is_cross_doc). This harness sidesteps that: it measures whether
the *router as a whole* delivers a better quality-vs-latency trade-off than any
single fixed strategy — which is the actual contribution. The router should
approach pure_hybrid/oracle answer quality while staying much closer to
pure_vector latency, and the oracle row bounds how much the router still leaves
on the table.

Usage (run in your local venv with the full stack up — Neo4j, LLM, vector store):
    python scripts/run_comparison_eval.py \
        --test-path qa_pipeline/data/phapdien_strict/test.json \
        --configs pure_vector,pure_graph,pure_hybrid,router,oracle \
        --max-samples 120 --stratified \
        --output-dir eval_results/comparison

Tip: start with a small --max-samples (e.g. 60) to validate the pipeline end to
end, then run the full 600 for the final thesis table.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.hybrid_pipeline import HybridPipeline
from evaluation.metrics import evaluate_prediction
from evaluation.metrics.id_normalizer import compute_hit_at_k, compute_mrr

CONFIG_ROUTES = {
    "pure_vector": "dense_retrieval",
    "pure_graph": "graph_traversal",
    "pure_hybrid": "hybrid_reasoning",
    "single_stage": None,  # Router with Stage 2 disabled
    "router": None,        # real adaptive router
    "oracle": "__gold__",  # force per-sample gold route (upper bound)
}


def get_gold_route(sample: dict) -> str:
    return str(
        sample.get("routing_label")
        or sample.get("route_label")
        or sample.get("gold_route")
        or ""
    ).lower().strip()


def get_ground_truth(sample: dict) -> str:
    return str(
        sample.get("ground_truth")
        or sample.get("answer")
        or sample.get("gold_context")
        or ""
    )


def get_gold_articles(sample: dict) -> list:
    """Gold article references for source/citation Hit@k."""
    gold = sample.get("relevant_articles") or []
    if not gold and sample.get("article_key"):
        gold = [sample["article_key"]]
    if not gold and sample.get("canonical_id"):
        gold = [sample["canonical_id"]]
    return gold


def stratified_subset(data: list, n: int, seed: int = 42) -> list:
    """Sample n items preserving routing_label proportions."""
    if n >= len(data):
        return data
    rng = random.Random(seed)
    by_label: dict[str, list] = defaultdict(list)
    for d in data:
        by_label[get_gold_route(d)].append(d)
    out: list = []
    for label, items in by_label.items():
        take = max(1, round(n * len(items) / len(data)))
        out.extend(rng.sample(items, min(take, len(items))))
    rng.shuffle(out)
    return out[:n]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def run_config(pipeline: HybridPipeline, config: str, data: list, out_path: Path) -> dict:
    forced = CONFIG_ROUTES[config]
    per_sample: list[dict] = []
    em_s, f1_s, contains_s = [], [], []
    hit1_s, hit3_s, mrr_s = [], [], []
    latencies: list[float] = []
    stage2_flags, fallback_flags = [], []
    route_match, executed_routes = [], []

    for idx, sample in enumerate(data):
        question = sample.get("question") or sample.get("query", "")
        gold_route = get_gold_route(sample)
        ground_truth = get_ground_truth(sample)
        gold_articles = get_gold_articles(sample)

        force = gold_route if forced == "__gold__" else forced

        t0 = time.perf_counter()
        try:
            resp = pipeline.query(
                query=question,
                session_id=f"{config}_{idx}",
                force_route=force,
            )
            answer = resp.answer or ""
            sources = list(getattr(resp, "sources", []) or [])
            executed = getattr(resp, "route_used", force or "")
            actual_pipeline = getattr(resp, "actual_pipeline_used", "") or ""
            stage2 = bool(getattr(resp, "stage2_invoked", False))
            ok = True
        except Exception as exc:  # noqa: BLE001 — record, never crash the sweep
            answer, sources, executed, actual_pipeline, stage2, ok = "", [], force or "", "", False, False
            print(f"  [{config}][{idx}] ERROR: {exc}")

        latency_ms = (time.perf_counter() - t0) * 1000
        em, f1, contains = evaluate_prediction(ground_truth, answer)
        hit1 = compute_hit_at_k(sources, gold_articles, k=1, mode="article")
        hit3 = compute_hit_at_k(sources, gold_articles, k=3, mode="article")
        mrr = compute_mrr(sources, gold_articles, mode="article")

        em_s.append(em); f1_s.append(f1); contains_s.append(contains)
        hit1_s.append(hit1); hit3_s.append(hit3); mrr_s.append(mrr)
        latencies.append(latency_ms)
        stage2_flags.append(1 if stage2 else 0)
        fallback_flags.append(1 if "->" in actual_pipeline else 0)
        executed_routes.append(executed)
        route_match.append(1 if executed == gold_route else 0)

        per_sample.append({
            "idx": idx, "question": question, "gold_route": gold_route,
            "executed_route": executed, "actual_pipeline": actual_pipeline,
            "stage2_invoked": stage2, "em": em, "f1": f1, "contains": contains,
            "hit@1": hit1, "hit@3": hit3, "latency_ms": round(latency_ms, 1),
            "sources": sources[:5], "retrieval_ok": ok,
        })

        if (idx + 1) % 25 == 0:
            print(f"  [{config}] {idx+1}/{len(data)}  "
                  f"F1={sum(f1_s)/len(f1_s):.3f}  Hit@1={sum(hit1_s)/len(hit1_s):.3f}  "
                  f"lat={sum(latencies)/len(latencies):.0f}ms")
            out_path.write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")

    n = max(1, len(data))
    summary = {
        "config": config, "n": len(data),
        "avg_em": sum(em_s) / n, "avg_f1": sum(f1_s) / n, "avg_contains": sum(contains_s) / n,
        "hit@1": sum(hit1_s) / n, "hit@3": sum(hit3_s) / n, "mrr": sum(mrr_s) / n,
        "latency_mean_ms": sum(latencies) / n,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "stage2_rate": sum(stage2_flags) / n,
        "fallback_rate": sum(fallback_flags) / n,
        "route_match_vs_gold": sum(route_match) / n,
        "route_distribution": dict(Counter(executed_routes)),
    }
    out_path.write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_markdown(summaries: list[dict], path: Path) -> None:
    cols = [
        ("config", "Config"), ("avg_f1", "F1"), ("avg_em", "EM"),
        ("avg_contains", "Contains"), ("hit@1", "Hit@1"), ("hit@3", "Hit@3"),
        ("latency_mean_ms", "Lat mean"), ("latency_p95_ms", "Lat p95"),
        ("stage2_rate", "Stage2%"), ("fallback_rate", "Fallback%"),
    ]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for s in summaries:
        row = []
        for key, _ in cols:
            v = s.get(key, "")
            if key == "config":
                row.append(str(v))
            elif "latency" in key:
                row.append(f"{v:.0f}")
            elif key in ("stage2_rate", "fallback_rate"):
                row.append(f"{v*100:.1f}%")
            else:
                row.append(f"{v:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-path", default="qa_pipeline/data/phapdien_strict/test.json")
    ap.add_argument("--configs", default="pure_vector,pure_graph,pure_hybrid,router,oracle")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--stratified", action="store_true", help="Preserve routing_label proportions when sampling")
    ap.add_argument("--output-dir", default="eval_results/comparison")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIG_ROUTES:
            raise SystemExit(f"Unknown config {c!r}; choose from {list(CONFIG_ROUTES)}")

    data = json.loads(Path(args.test_path).read_text(encoding="utf-8"))
    if args.max_samples:
        data = stratified_subset(data, args.max_samples) if args.stratified else data[:args.max_samples]
    print(f"Evaluating {len(data)} samples | configs={configs}")
    print(f"Label distribution: {dict(Counter(get_gold_route(d) for d in data))}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path("configs/build_kg_no_ner.yaml")
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    
    summaries = []
    for config in configs:
        print(f"\n===== {config} =====")
        
        # Toggle fallbacks: enabled only for router family
        is_router_family = config in ("router", "single_stage")
        if "rag" not in base_config:
            base_config["rag"] = {}
        base_config["rag"]["vector_fallback_to_graph"] = is_router_family
        base_config["rag"]["graph_fallback_to_vector"] = is_router_family
        base_config["rag"]["hybrid_fallback_to_vector"] = is_router_family
        
        if "router" not in base_config:
            base_config["router"] = {}
        if "stage2" not in base_config["router"]:
            base_config["router"]["stage2"] = {}
        base_config["router"]["stage2"]["enabled"] = (config != "single_stage")
        
        temp_config_path = Path("configs/temp_eval_config.yaml")
        temp_config_path.write_text(yaml.dump(base_config, allow_unicode=True), encoding="utf-8")
        
        pipeline = HybridPipeline(config_path=temp_config_path)
        
        t0 = time.perf_counter()
        summ = run_config(pipeline, config, data, out_dir / f"per_sample_{config}.json")
        summ["wall_clock_s"] = round(time.perf_counter() - t0, 1)
        summaries.append(summ)
        print(f"  -> F1={summ['avg_f1']:.3f}  Hit@1={summ['hit@1']:.3f}  "
              f"lat_mean={summ['latency_mean_ms']:.0f}ms  p95={summ['latency_p95_ms']:.0f}ms  "
              f"stage2={summ['stage2_rate']*100:.1f}%  fallback={summ['fallback_rate']*100:.1f}%")
        (out_dir / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(summaries, out_dir / "comparison_table.md")

    print(f"\nDone. Wrote {out_dir/'summary.json'} and {out_dir/'comparison_table.md'}")
    print("\n=== COMPARISON TABLE ===")
    print((out_dir / "comparison_table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
