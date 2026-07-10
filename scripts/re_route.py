"""
re_route.py
===========
Re-labels routing predictions using the trained XGBoost Stage-1 model,
without re-running the full LLM pipeline (~40 minutes).

Fixes over the previous version
--------------------------------
1. os.makedirs crash when output path has no directory component
   (e.g. "predictions_v2.json" → dirname="" → makedirs("") raises FileNotFoundError)
2. Silent data corruption when predictions.json is a dict instead of a list
   (iterating over dict keys gave wrong pred_map, all updates silently dropped)
3. Optional Python < 3.10 compatibility (Optional[...] instead of ... | None)

Usage
-----
  python re_route.py \
      --predictions eval_results/predictions.json \
      --model       router_model/xgb_router.pkl \
      --feat-names  router_model/feature_names.json \
      --benchmark   test_benchmark_v2.json \
      --output      eval_results/predictions_v2.json

What this script does
---------------------
  - Loads the existing predictions.json (Stage-1+Stage-2 routes)
  - Re-predicts the routing label for every query using ONLY the XGBoost
    Stage-1 model (overwriting "predicted_route")
  - Leaves all other fields (retrieved_articles, answer, sources, …) intact
  - Saves the updated list to --output

When to use
-----------
  Use this for the "Stage-1 only" ablation row in the thesis table.
  For the full pipeline row (Stage-1 + Stage-2 LLM verifier), you must
  re-run run_benchmark_eval.py.
"""

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np

# ── Feature extractor ─────────────────────────────────────────────────────────
try:
    from feature_extractor_fixed import VietnameseLegalFeatureExtractor
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_extractor_fixed import VietnameseLegalFeatureExtractor

# ── Label mapping ─────────────────────────────────────────────────────────────
INT2LABEL: Dict[int, str] = {
    0: "dense_retrieval",
    1: "graph_traversal",
    2: "hybrid_reasoning",
}


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS LOADER — handles list and dict formats
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(path: str) -> List[Dict]:
    """
    Load predictions.json and normalise to a list of dicts.

    Handles two formats:
      (a) List:  [{id, predicted_route, ...}, ...]
      (b) Dict:  {query_id: {id, predicted_route, ...}, ...}
              or {predictions: [{id, predicted_route, ...}, ...]}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        # Nested list under a "predictions" key
        if "predictions" in raw and isinstance(raw["predictions"], list):
            return raw["predictions"]
        # Flat dict keyed by query id — extract values
        values = list(raw.values())
        if values and isinstance(values[0], dict):
            return values
        raise ValueError(
            f"predictions.json is a dict but its values are {type(values[0])}, "
            "expected dict. Cannot parse."
        )

    raise ValueError(f"Unexpected predictions format: {type(raw)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Re-route predictions with new XGBoost model")
    parser.add_argument("--predictions", required=True,
                        help="Path to original predictions.json")
    parser.add_argument("--model",       required=True,
                        help="Path to router_model/xgb_router.pkl")
    parser.add_argument("--feat-names",  required=True,
                        help="Path to router_model/feature_names.json")
    parser.add_argument("--benchmark",   required=True,
                        help="Path to test_benchmark_v2.json (for question text)")
    parser.add_argument("--output",      required=True,
                        help="Output path for re-routed predictions.json")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model:        {args.model}")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print(f"Loading feature names: {args.feat_names}")
    with open(args.feat_names, encoding="utf-8") as f:
        feat_names: List[str] = json.load(f)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading benchmark:    {args.benchmark}")
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    print(f"Loading predictions:  {args.predictions}")
    preds = load_predictions(args.predictions)

    # Build id → pred dict (safe for in-place mutation)
    pred_map: Dict[str, Dict] = {p["id"]: p for p in preds if "id" in p}
    print(f"  {len(preds)} predictions loaded, {len(pred_map)} with valid 'id' field")

    # ── Feature extraction + re-routing ───────────────────────────────────────
    extractor = VietnameseLegalFeatureExtractor()
    changed = 0
    skipped = 0

    print("Re-routing queries with XGBoost Stage-1 model...")
    for item in benchmark:
        q_id = item.get("id")
        if not q_id:
            skipped += 1
            continue
        if q_id not in pred_map:
            # Query has no existing prediction — create a minimal entry
            pred_map[q_id] = {"id": q_id}
            preds.append(pred_map[q_id])
            skipped += 1  # count as "new" entry, not an error

        question = item["question"]
        feat_dict = extractor.extract(question)

        # Ensure feature vector follows the exact order from training
        x_vec = [feat_dict.get(name, 0.0) for name in feat_names]
        X = np.array([x_vec], dtype=np.float32)

        pred_idx  = int(model.predict(X)[0])
        new_route = INT2LABEL[pred_idx]
        old_route = pred_map[q_id].get("predicted_route", "<none>")

        if new_route != old_route:
            changed += 1

        pred_map[q_id]["predicted_route"] = new_route

    print(f"  Routing labels changed: {changed} / {len(benchmark)}")
    if skipped:
        print(f"  Queries with no prior prediction (newly added): {skipped}")

    # Distribution of new labels
    from collections import Counter
    new_dist = Counter(p.get("predicted_route", "unknown") for p in preds)
    print("  New routing distribution:")
    for label, count in sorted(new_dist.items()):
        print(f"    {label:<35} {count:>4}  ({count/len(preds)*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    # FIX: only call makedirs if there is actually a directory component
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(preds)} predictions → {args.output}")


if __name__ == "__main__":
    main()
