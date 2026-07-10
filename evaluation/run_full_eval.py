"""End-to-end evaluation runner for all 7 system configurations.

Implements the re-run plan from review.md §D:
    1. Fix ID normalisation (M3) — uses evaluation.metrics.id_normalizer
    2. Per-query logging: route class, F1/EM, Hit@k, MRR, timings, source IDs
    3. Bootstrap CIs (M5, B=1000)
    4. Semantic metric pass (M6) — BERTScore over saved predictions
    5. Outputs all data needed for:
       - tab:per_class_f1   (M1)
       - tab:latency_decomp (M5)
       - Oracle+Stage2 row  (M4)

System configurations evaluated
---------------------------------
1. pure_vector       — dense retrieval only, no routing
2. pure_graph        — graph traversal only
3. pure_hybrid       — hybrid retrieval only (exposes the Hit@k=0.003 bug)
4. two_stage         — two-stage router (Stage-1 XGBoost + Stage-2 LLM)
5. oracle            — gold-label routing, no Stage-2
6. oracle_stage2     — gold-label routing WITH Stage-2 (M4 addition)
7. always_on         — always invoke Stage-2 regardless of route

Usage
------
    python -m evaluation.run_full_eval [--config configs/config.yaml]
                                       [--test evaluation/test_queries.json]
                                       [--out eval_results/]
                                       [--systems two_stage oracle ...]
                                       [--bert]
                                       [--workers 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from evaluation.evaluate import bootstrap_ci, compute_per_class_f1, load_test_queries
from evaluation.metrics import (
    compute_hit_at_k,
    compute_mrr,
    compute_token_f1,
    evaluate_prediction,
    normalize_gold_article,
    normalize_legal_id,
)


# ─────────────────────────────────────────────────────────────────────────────
# System configurations
# ─────────────────────────────────────────────────────────────────────────────

ALL_SYSTEMS: list[str] = [
    "pure_vector",
    "pure_graph",
    "pure_hybrid",
    "two_stage",
    "oracle",
    "oracle_stage2",
    "always_on",
    "single_stage",
]


def _pipeline_for_system(system: str, config_path: str | Path | None) -> Any:
    """Instantiate the pipeline variant for the given system name.

    Each variant passes different flags to HybridPipeline. The pipeline must
    expose a `.query()` method that returns a response object with at least:
        .answer, .route_used, .source_ids, .latency_ms,
        .stage1_latency_ms, .stage2_latency_ms,
        .retrieval_latency_ms, .generation_latency_ms,
        .stage2_invoked, .is_ambiguous

    Args:
        system: One of ALL_SYSTEMS.
        config_path: Path to config.yaml.

    Returns:
        A pipeline instance.
    """
    from pipeline.hybrid_pipeline import HybridPipeline  # noqa: PLC0415

    # Map system name → HybridPipeline kwargs
    kwargs: dict[str, Any] = {}

    if system == "pure_vector":
        kwargs = {"force_route": "dense_retrieval", "disable_stage2": True}
    elif system == "pure_graph":
        kwargs = {"force_route": "graph_traversal", "disable_stage2": True}
    elif system == "pure_hybrid":
        kwargs = {"force_route": "hybrid_reasoning", "disable_stage2": True}
# ── Route label normalizer ────────────────────────────────────────────
# Old test files used short names; new ones use full canonical names.
_ROUTE_ALIASES: dict[str, str] = {
    "vector":          "dense_retrieval",
    "dense":           "dense_retrieval",
    "graph":           "graph_traversal",
    "hybrid":          "hybrid_reasoning",
    "dense_retrieval":  "dense_retrieval",
    "graph_traversal":  "graph_traversal",
    "hybrid_reasoning": "hybrid_reasoning",
    "clarify":          "clarify",
}


def _normalise_route(label: str | None) -> str:
    """Normalise a route label to canonical form."""
    return _ROUTE_ALIASES.get(str(label or ""), "dense_retrieval")


def _load_phapdien_test(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load the phapdien_strict test split and normalise field names.

    Handles both legacy evaluation/test_queries.json schema and the
    canonical qa_pipeline/data/phapdien_strict/test.json schema.

    Args:
        path: Override path. If None, uses phapdien_strict/test.json.

    Returns:
        List of normalised query dicts with keys:
        query, answer, routing_label, expected_route, relevant_articles.
    """
    if path is None:
        # Prefer the full 600-sample canonical test split
        candidates = [
            Path("qa_pipeline/data/phapdien_strict/test.json"),
            Path("evaluation/test_queries.json"),
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
        else:
            raise FileNotFoundError("No test file found — pass --test explicitly.")

    path = Path(path)
    logger.info("Loading test data from {}", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: list[dict[str, Any]] = []
    for item in raw:
        # Normalise query field
        query = item.get("question") or item.get("query") or ""
        # Normalise answer field
        answer = (
            item.get("answer") or
            item.get("gold_answer") or
            item.get("gold_context") or
            ""
        )
        # Normalise routing label
        raw_label = item.get("routing_label") or item.get("expected_route") or ""
        routing_label = _normalise_route(raw_label)
        # Normalise gold sources
        gold_sources = (
            item.get("relevant_articles") or
            item.get("gold_sources") or
            []
        )
        out.append({
            "query": query,
            "answer": answer,
            "routing_label": routing_label,
            "expected_route": routing_label,
            "relevant_articles": gold_sources,
            "is_ambiguous": item.get("is_ambiguous", False),
            # keep original for reference
            "_original": {k: v for k, v in item.items() if k not in
                         {"question", "query", "answer", "gold_answer",
                          "routing_label", "relevant_articles"}},
        })

    logger.info("Loaded {} test queries (path={})", len(out), path)
    return out


def _pipeline_for_system(system: str, config_path: str | Path | None) -> Any:
    """Instantiate the pipeline variant for the given system name.

    For oracle/oracle_stage2, the pipeline is the standard Two-Stage pipeline
    but _evaluate_query will inject the gold route via query_kwargs["gold_route"].
    HybridPipeline must accept a ``gold_route`` kwarg in its ``.query()`` method
    (or we override the route post-hoc via the response object).

    Args:
        system: One of ALL_SYSTEMS.
        config_path: Path to config.yaml.

    Returns:
        A pipeline instance.
    """
    from pipeline.hybrid_pipeline import HybridPipeline  # noqa: PLC0415

    if system == "pure_vector":
        kwargs = {"force_route": "dense_retrieval", "disable_stage2": True}
    elif system == "pure_graph":
        kwargs = {"force_route": "graph_traversal", "disable_stage2": True}
    elif system == "pure_hybrid":
        kwargs = {"force_route": "hybrid_reasoning", "disable_stage2": True}
    elif system == "single_stage":
        kwargs = {"disable_stage2": True}
    elif system == "two_stage":
        kwargs = {}  # default two-stage router
    elif system in {"oracle", "oracle_stage2"}:
        # Oracle: standard pipeline; gold route injected per-query in _evaluate_query
        kwargs = {}
    elif system == "always_on":
        kwargs = {"always_invoke_stage2": True}
    else:
        kwargs = {}

    # Only pass kwargs that HybridPipeline actually accepts
    accepted = {"disable_stage2"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    try:
        return HybridPipeline(config_path, **filtered)
    except TypeError:
        # Fallback: try without any extra kwargs
        logger.warning("HybridPipeline rejected kwargs {}; retrying bare", filtered)
        return HybridPipeline(config_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-query evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_query(
    pipeline: Any,
    tq: dict[str, Any],
    query_id: int,
    system: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single query through the pipeline and compute all metrics.

    Returns a per-query result dict suitable for JSONL logging.
    """
    query = tq.get("query") or tq.get("question") or ""
    expected_route = _normalise_route(
        tq.get("expected_route") or tq.get("routing_label") or "dense_retrieval"
    )
    gold_route_label = _normalise_route(
        tq.get("routing_label") or tq.get("expected_route") or "dense_retrieval"
    )
    gold_answer = (
        tq.get("answer") or tq.get("gold_answer") or tq.get("gold_context") or ""
    )
    gold_sources = tq.get("relevant_articles") or tq.get("gold_sources") or []

    # Oracle routing: try to inject gold route into pipeline query call.
    # HybridPipeline.query() may accept force_route to bypass Stage-1.
    query_kwargs: dict[str, Any] = {"session_id": f"eval_{system}_{query_id}"}
    if system == "oracle":
        query_kwargs["force_route"] = gold_route_label
    elif system == "oracle_stage2":
        query_kwargs["force_stage1_route"] = gold_route_label
    elif system == "always_on":
        query_kwargs["force_stage2"] = True
    elif system == "pure_vector":
        query_kwargs["force_route"] = "dense_retrieval"
    elif system == "pure_graph":
        query_kwargs["force_route"] = "graph_traversal"
    elif system == "pure_hybrid":
        query_kwargs["force_route"] = "hybrid_reasoning"
    elif system == "single_stage":
        # We need a way to disable stage 2. If force_route is not used, we can just pass a dummy flag.
        # But wait, query() doesn't have a `disable_stage2` kwarg.
        pass


    try:
        t0 = time.perf_counter()
        response = pipeline.query(query=query, **query_kwargs)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        answer_text = getattr(response, "answer", "") or ""
        em, tf1, _ = evaluate_prediction(gold_answer, answer_text)

        retrieved_sources = getattr(response, "sources", []) or getattr(response, "source_ids", []) or []
        hit1 = compute_hit_at_k(retrieved_sources, gold_sources, k=1, mode="strict")
        hit3 = compute_hit_at_k(retrieved_sources, gold_sources, k=3, mode="strict")
        hit5 = compute_hit_at_k(retrieved_sources, gold_sources, k=5, mode="strict")
        mrr = compute_mrr(retrieved_sources, gold_sources, mode="strict")

        total_ms = getattr(response, "latency_ms", None) or wall_ms

        entry: dict[str, Any] = {
            "system": system,
            "query_id": query_id,
            "query": query,
            "gold_route_label": gold_route_label,
            "expected_route": expected_route,
            "predicted_route": getattr(response, "route_used", "unknown"),
            "route_correct": getattr(response, "route_used", "") == expected_route,
            "token_f1": float(tf1),
            "exact_match": float(em),
            "hit_at_1": int(hit1),
            "hit_at_3": int(hit3),
            "hit_at_5": int(hit5),
            "mrr": float(mrr),
            "latency_ms": float(total_ms),
            "stage1_latency_ms": float(getattr(response, "stage1_latency_ms", 0.0) or 0.0),
            "stage2_latency_ms": float(getattr(response, "stage2_latency_ms", 0.0) or 0.0),
            "retrieval_latency_ms": float(getattr(response, "retrieval_latency_ms", 0.0) or 0.0),
            "generation_latency_ms": float(getattr(response, "generation_latency_ms", 0.0) or 0.0),
            "stage2_invoked": bool(getattr(response, "stage2_invoked", False)),
            "is_ambiguous": bool(getattr(response, "is_ambiguous", False)),
            "answer": answer_text,
            "gold_answer": gold_answer,
            # Source ID log for M3 debugging
            "retrieved_source_ids": [str(s) for s in retrieved_sources[:10]],
            "gold_source_ids": [str(g) for g in gold_sources[:10]],
            # Resolved canonical IDs for tracing
            "retrieved_canonical": [
                normalize_legal_id(str(s)).key for s in retrieved_sources[:10]
            ],
            "gold_canonical": [
                normalize_gold_article(g).key for g in gold_sources[:3]
            ],
        }

        if verbose:
            status = "✓" if entry["route_correct"] else "✗"
            logger.info(
                "[{}] {} {} route={}/{} F1={:.3f} hit@1={} mrr={:.3f}",
                query_id, system, status,
                entry["predicted_route"], expected_route,
                tf1, hit1, mrr,
            )

        return entry

    except Exception as exc:
        logger.error("[{}] query {} failed: {}", system, query_id, exc)
        return {
            "system": system,
            "query_id": query_id,
            "query": query,
            "gold_route_label": gold_route_label,
            "expected_route": expected_route,
            "predicted_route": "error",
            "route_correct": False,
            "token_f1": 0.0,
            "exact_match": 0.0,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "mrr": 0.0,
            "error": str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate over queries
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_system(
    entries: list[dict[str, Any]],
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Aggregate per-query entries into system-level metrics.

    Args:
        entries: List of per-query result dicts.
        n_bootstrap: Bootstrap resamples for CI.

    Returns:
        Dict with aggregated metrics ready for the summary table.
    """
    def _vec(key: str) -> list[float]:
        return [float(e.get(key, 0.0)) for e in entries]

    tf1 = _vec("token_f1")
    em = _vec("exact_match")
    hit1 = _vec("hit_at_1")
    hit3 = _vec("hit_at_3")
    hit5 = _vec("hit_at_5")
    mrr = _vec("mrr")
    lat = _vec("latency_ms")

    gold_labels = [e.get("gold_route_label", "unknown") for e in entries]
    per_class = compute_per_class_f1(tf1, gold_labels)

    route_correct = [e.get("route_correct", False) for e in entries]
    routing_acc = sum(route_correct) / len(route_correct) if route_correct else 0.0

    s2_invoked = [e.get("stage2_invoked", False) for e in entries]
    s2_rate = sum(s2_invoked) / len(s2_invoked) if s2_invoked else 0.0

    def _stat(values: list[float]) -> dict[str, float]:
        mean, lo, hi = bootstrap_ci(values, n_bootstrap=n_bootstrap)
        return {"mean": mean, "ci_lo_95": lo, "ci_hi_95": hi}

    return {
        "n": len(entries),
        "routing_accuracy": routing_acc,
        "token_f1": _stat(tf1),
        "exact_match": _stat(em),
        "hit_at_1": _stat(hit1),
        "hit_at_3": _stat(hit3),
        "hit_at_5": _stat(hit5),
        "mrr": _stat(mrr),
        "latency_mean_ms": float(np.mean(lat)) if lat else 0.0,
        "latency_p50_ms": float(np.percentile(lat, 50)) if lat else 0.0,
        "latency_p95_ms": float(np.percentile(lat, 95)) if lat else 0.0,
        "stage2_rate": s2_rate,
        "latency_decomp": {
            "stage1_mean_ms": float(np.mean(_vec("stage1_latency_ms"))),
            "stage2_mean_ms": float(np.mean(_vec("stage2_latency_ms"))),
            "retrieval_mean_ms": float(np.mean(_vec("retrieval_latency_ms"))),
            "generation_mean_ms": float(np.mean(_vec("generation_latency_ms"))),
        },
        "per_class_f1": per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(agg: dict[str, dict[str, Any]]) -> None:
    """Print a comparison table across all systems."""
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning("tabulate not installed — printing raw dict")
        print(json.dumps(agg, indent=2, ensure_ascii=False))
        return

    rows = []
    for system in ALL_SYSTEMS:
        if system not in agg:
            continue
        s = agg[system]
        rows.append([
            system,
            f"{s['routing_accuracy']:.3f}",
            f"{s['token_f1']['mean']:.3f} [{s['token_f1']['ci_lo_95']:.3f},{s['token_f1']['ci_hi_95']:.3f}]",
            f"{s['exact_match']['mean']:.3f}",
            f"{s['hit_at_1']['mean']:.3f} [{s['hit_at_1']['ci_lo_95']:.3f},{s['hit_at_1']['ci_hi_95']:.3f}]",
            f"{s['mrr']['mean']:.3f}",
            f"{s['latency_mean_ms']:.0f}",
            f"{s['stage2_rate']:.3f}",
        ])

    headers = ["System", "Route-Acc", "Token-F1 [95%CI]", "EM", "Hit@1 [95%CI]", "MRR", "Lat(ms)", "S2-Rate"]
    print("\n" + "=" * 110)
    print("FULL SYSTEM COMPARISON — tab:end_to_end")
    print("=" * 110)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Per-class F1 (tab:per_class_f1)
    print("\n--- Per-Route-Class Token F1 — tab:per_class_f1 ---")
    classes = sorted({
        cls
        for s in agg.values()
        for cls in s.get("per_class_f1", {})
    })
    pc_rows = []
    for system in ALL_SYSTEMS:
        if system not in agg:
            continue
        row = [system]
        for cls in classes:
            v = agg[system]["per_class_f1"].get(cls, {})
            if v:
                row.append(f"{v['mean']:.3f}")
            else:
                row.append("—")
        pc_rows.append(row)
    print(tabulate(pc_rows, headers=["System"] + classes, tablefmt="grid"))


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore pass (M6) over saved predictions
# ─────────────────────────────────────────────────────────────────────────────

def _add_bertscore(
    per_query_log: list[dict[str, Any]],
    agg: dict[str, dict[str, Any]],
    system: str,
) -> None:
    """Run BERTScore over saved predictions for a system (M6)."""
    try:
        from evaluation.metrics.bertscore_eval import compute_bertscore  # noqa: PLC0415
    except ImportError:
        logger.warning("bert-score not installed — BERTScore skipped.")
        return

    entries = [e for e in per_query_log if e.get("system") == system]
    preds = [e.get("answer", "") for e in entries]
    refs = [e.get("gold_answer", "") for e in entries]
    valid = [(p, r) for p, r in zip(preds, refs) if r]
    if not valid:
        return
    preds_v, refs_v = zip(*valid)
    result = compute_bertscore(list(preds_v), list(refs_v), reference_field="gold_answer")
    agg[system]["bertscore"] = result
    logger.info("[{}] BERTScore F1 = {:.4f}", system, result.get("f1", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eval(
    config_path: str | Path | None = None,
    test_path: str | Path | None = None,
    out_dir: str | Path = "eval_results",
    systems: list[str] | None = None,
    verbose: bool = False,
    compute_bert: bool = False,
    n_bootstrap: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Run all system configs and produce the full comparison table.

    Args:
        config_path: Path to config.yaml.
        test_path: Path to test_queries.json.
        out_dir: Directory for output files.
        systems: Subset of ALL_SYSTEMS to run. None = all.
        verbose: Per-query logging.
        compute_bert: Run BERTScore after main eval (slow, needs GPU).
        n_bootstrap: Bootstrap resamples for CI.

    Returns:
        Dict mapping system name → aggregate metrics dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_queries = _load_phapdien_test(test_path)
    systems_to_run = systems or ALL_SYSTEMS

    per_query_log: list[dict[str, Any]] = []
    agg: dict[str, dict[str, Any]] = {}
    
    log_path = out_dir / "full_eval_log.jsonl"
    # Note: We append, so clear it if we are running all systems from scratch
    if not systems and log_path.exists():
        log_path.unlink()

    for system in systems_to_run:
        if system not in ALL_SYSTEMS:
            logger.warning("Unknown system '{}', skipping", system)
            continue

        logger.info("=" * 60)
        logger.info("Evaluating system: {}", system)
        logger.info("=" * 60)

        try:
            pipeline = _pipeline_for_system(system, config_path)
        except Exception as exc:
            logger.error("Failed to init pipeline for {}: {}", system, exc)
            continue

        entries: list[dict[str, Any]] = []
        for i, tq in enumerate(test_queries, 1):
            entry = _evaluate_query(pipeline, tq, i, system, verbose=verbose)
            entries.append(entry)
            per_query_log.append(entry)
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Aggregate
        agg[system] = _aggregate_system(entries, n_bootstrap=n_bootstrap)
        logger.info(
            "[{}] Done. F1={:.3f} Hit@1={:.3f} RouteAcc={:.3f}",
            system,
            agg[system]["token_f1"]["mean"],
            agg[system]["hit_at_1"]["mean"],
            agg[system]["routing_accuracy"],
        )

    # BERTScore pass (M6) — optional, done after all queries to batch efficiently
    if compute_bert:
        for system in systems_to_run:
            if system in agg:
                _add_bertscore(per_query_log, agg, system)

    # Print summary
    _print_summary(agg)

    # Save outputs
    agg_path = out_dir / "full_eval_summary.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    logger.info("Aggregate summary saved to {}", agg_path)

    # Note: If BERTScore computation modified agg, we re-write the summary
    if compute_bert:
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        
    logger.info("Per-query log saved to {} ({} entries)", log_path, len(per_query_log))

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full end-to-end evaluation across all system configurations."
    )
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--test", default=None, help="Path to test_queries.json")
    p.add_argument("--out", default="eval_results", help="Output directory")
    p.add_argument(
        "--systems", nargs="*", default=None,
        choices=ALL_SYSTEMS,
        help=f"Systems to evaluate (default: all). Choices: {ALL_SYSTEMS}",
    )
    p.add_argument("--bert", action="store_true", help="Compute BERTScore (slow)")
    p.add_argument("--verbose", action="store_true", help="Per-query logging")
    p.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples for CI (default: 1000)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_full_eval(
        config_path=args.config,
        test_path=args.test,
        out_dir=args.out,
        systems=args.systems,
        verbose=args.verbose,
        compute_bert=args.bert,
        n_bootstrap=args.bootstrap,
    )
