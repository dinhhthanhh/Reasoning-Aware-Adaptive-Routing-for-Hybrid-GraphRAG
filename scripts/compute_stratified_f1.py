"""Compute stratified token F1 and latency-by-route from full-corpus benchmark CSVs.

The full-corpus benchmark scores clarification responses as F1=0 against gold
answers. This script isolates retrieval-only F1 (excluding clarify routes) so
answer quality on answerable queries can be compared fairly.

Usage:
    python scripts/compute_stratified_f1.py
    python scripts/compute_stratified_f1.py --exclude-ids results/final/excluded_test_ids.json

Outputs:
    results/final/stratified_f1.json
    results/final/latency_by_route.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.metrics.token_f1 import compute_token_f1
from evaluation.significance.bootstrap_test import paired_bootstrap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = PROJECT_ROOT / "qa_pipeline" / "data" / "legal_strict" / "test.json"
DEFAULT_EXCLUDE = PROJECT_ROOT / "results" / "final" / "excluded_test_ids.json"
OUT_STRATIFIED = PROJECT_ROOT / "results" / "final" / "stratified_f1.json"
OUT_LATENCY = PROJECT_ROOT / "results" / "final" / "latency_by_route.json"

SYSTEMS: dict[str, str] = {
    "pure_vector": "eval_results/legal_strict_pure_vector_results.csv",
    "pure_graph": "eval_results/legal_strict_pure_graph_results.csv",
    "single_stage": "eval_results/legal_strict_single_stage_router_results.csv",
    "two_stage": "eval_results/legal_strict_two_stage_hybrid_results.csv",
}

RETRIEVAL_ROUTES = frozenset({"dense_retrieval", "graph_traversal", "hybrid_reasoning"})


def _load_gold(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}


def _load_exclude(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    return set(data.get("excluded_ids", data.get("ids", [])))


def _route(row: dict) -> str:
    return str(row.get("Route") or row.get("Actual_Route") or "").strip()


def _analyze_system(
    csv_path: Path,
    gold: dict[str, dict],
    exclude: set[str],
) -> tuple[dict, list[dict]]:
    """Return summary dict and per-query records for latency breakdown."""
    rows_out: list[dict] = []
    overall_f1s: list[float] = []
    retrieval_f1s: list[float] = []
    clarify_count = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("ID") or row.get("id")
            if not rid or rid in exclude or rid not in gold:
                continue
            route = _route(row)
            answer = row.get("Generated_Answer", "")
            gt = str(gold[rid].get("answer") or "")
            f1 = float(compute_token_f1(answer, gt)["f1"])
            try:
                latency = float(row.get("Time_ms") or 0)
            except (TypeError, ValueError):
                latency = 0.0
            stage2 = str(row.get("Stage2", "")).lower() in {"true", "1", "yes"}
            stage2_override = str(row.get("Stage2_Override", "")).lower() in {
                "true",
                "1",
                "yes",
            }

            overall_f1s.append(f1)
            is_clarify = route == "clarify"
            if is_clarify:
                clarify_count += 1
            else:
                retrieval_f1s.append(f1)

            rows_out.append(
                {
                    "id": rid,
                    "route": route,
                    "token_f1": f1,
                    "latency_ms": latency,
                    "stage2": stage2,
                    "stage2_override": stage2_override,
                }
            )

    n = len(overall_f1s)
    summary = {
        "n": n,
        "overall_f1": round(sum(overall_f1s) / n, 4) if n else 0.0,
        "retrieval_only_f1": round(sum(retrieval_f1s) / len(retrieval_f1s), 4)
        if retrieval_f1s
        else 0.0,
        "retrieval_only_n": len(retrieval_f1s),
        "clarify_count": clarify_count,
        "clarify_f1_mean": round(
            sum(r["token_f1"] for r in rows_out if r["route"] == "clarify")
            / clarify_count,
            4,
        )
        if clarify_count
        else None,
    }
    return summary, rows_out


def _latency_breakdown(label: str, records: list[dict]) -> dict:
    by_route: dict[str, list[float]] = {}
    stage2_lat: list[float] = []
    no_stage2_lat: list[float] = []
    for rec in records:
        route = rec["route"] or "unknown"
        by_route.setdefault(route, []).append(rec["latency_ms"])
        if rec["stage2"]:
            stage2_lat.append(rec["latency_ms"])
        else:
            no_stage2_lat.append(rec["latency_ms"])

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 1) if xs else None

    route_stats = {
        route: {"count": len(lats), "avg_latency_ms": _mean(lats)}
        for route, lats in sorted(by_route.items())
    }
    return {
        "system": label,
        "by_route": route_stats,
        "stage2_triggered": {
            "count": len(stage2_lat),
            "avg_latency_ms": _mean(stage2_lat),
        },
        "stage2_not_triggered": {
            "count": len(no_stage2_lat),
            "avg_latency_ms": _mean(no_stage2_lat),
        },
        "stage2_latency_delta_ms": round(_mean(stage2_lat) - _mean(no_stage2_lat), 1)
        if stage2_lat and no_stage2_lat
        else None,
        "overall_avg_latency_ms": _mean([r["latency_ms"] for r in records]),
    }


def _retrieval_only_significance(
    single_records: list[dict],
    two_stage_records: list[dict],
) -> dict:
    """Bootstrap two_stage vs single_stage on retrieval-only paired F1."""
    single_by_id = {r["id"]: r for r in single_records if r["route"] != "clarify"}
    two_by_id = {r["id"]: r for r in two_stage_records if r["route"] != "clarify"}
    common = sorted(set(single_by_id) & set(two_by_id))
    sa = [single_by_id[i]["token_f1"] for i in common]
    sb = [two_by_id[i]["token_f1"] for i in common]
    result = paired_bootstrap(sa, sb, n_resamples=10_000)
    result["comparison"] = "two_stage_vs_single_stage_retrieval_only"
    result["metric"] = "token_f1_vs_answer"
    result["significant_at_0.05"] = result["p_value"] < 0.05
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--exclude-ids", type=Path, default=DEFAULT_EXCLUDE)
    args = parser.parse_args()

    gold = _load_gold(args.gold)
    exclude = _load_exclude(args.exclude_ids)

    stratified: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}
    for label, rel_path in SYSTEMS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        summary, records = _analyze_system(path, gold, exclude)
        stratified[label] = summary
        all_records[label] = records

    stratified["significance_retrieval_only"] = _retrieval_only_significance(
        all_records["single_stage"],
        all_records["two_stage"],
    )

    latency = {
        label: _latency_breakdown(label, records)
        for label, records in all_records.items()
    }

    OUT_STRATIFIED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_STRATIFIED, "w", encoding="utf-8") as f:
        json.dump(stratified, f, indent=2, ensure_ascii=False)
    with open(OUT_LATENCY, "w", encoding="utf-8") as f:
        json.dump(latency, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUT_STRATIFIED}")
    print(f"Wrote {OUT_LATENCY}")
    for label, s in stratified.items():
        if label == "significance_retrieval_only":
            continue
        print(
            f"  {label}: overall={s['overall_f1']} "
            f"retrieval_only={s['retrieval_only_f1']} (n={s['retrieval_only_n']}) "
            f"clarify={s['clarify_count']}"
        )
    sig = stratified["significance_retrieval_only"]
    print(
        f"  retrieval-only sig: delta={sig['observed_diff']:.4f} "
        f"p={sig['p_value']:.4f}"
    )


if __name__ == "__main__":
    main()
