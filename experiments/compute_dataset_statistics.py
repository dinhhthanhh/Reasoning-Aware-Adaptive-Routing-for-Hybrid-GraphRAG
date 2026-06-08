#!/usr/bin/env python3
"""Compute descriptive statistics of the legal strict QA dataset.

Produces:
  - artifacts/routing_baselines/dataset_statistics.json
  - artifacts/routing_baselines/dataset_statistics.csv

Statistics cover question/answer/gold-context lengths (chars after
Unicode normalization) and hop_count by routing class.  Also reports
question length per split to check for distribution shift.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "qa_pipeline" / "data" / "legal_strict"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "routing_baselines"


def _norm(text: str) -> str:
    """NFC-normalize and strip whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def _desc(series: pd.Series) -> dict:
    """Compute descriptive stats for a numeric series."""
    arr = series.dropna().values.astype(float)
    if len(arr) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
        "min": round(float(np.min(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }


def load_all_splits() -> pd.DataFrame:
    """Load and concatenate train/dev/test with a 'split' column."""
    frames = []
    for split in ("train", "dev", "test"):
        path = DATA_DIR / f"{split}.json"
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_splits()
    total = len(df)
    print(f"Total samples loaded: {total}")

    # --- Lengths (chars after Unicode normalization) ---
    df["question_len"] = df["question"].apply(lambda x: len(_norm(str(x or ""))))
    df["answer_len"] = df["answer"].apply(lambda x: len(_norm(str(x or ""))))
    df["gold_context_len"] = df["gold_context"].apply(lambda x: len(_norm(str(x or ""))))

    # --- Per-field stats (full dataset) ---
    stats: dict = {}
    stats["question_length_chars"] = _desc(df["question_len"])
    stats["answer_length_chars"] = _desc(df["answer_len"])
    stats["gold_context_length_chars"] = _desc(df["gold_context_len"])

    # --- Hop count per routing class ---
    for route in ("dense_retrieval", "graph_traversal", "hybrid_reasoning"):
        subset = df[df["routing_label"] == route]
        stats[f"hop_count_{route}"] = _desc(subset["hop_count"])

    # --- Question length per split (distribution shift check) ---
    stats["question_length_per_split"] = {}
    for split in ("train", "dev", "test"):
        subset = df[df["split"] == split]
        stats["question_length_per_split"][split] = _desc(subset["question_len"])

    # --- Save JSON ---
    json_path = OUTPUT_DIR / "dataset_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    # --- Save CSV (flat) ---
    rows = []
    for key, val in stats.items():
        if key == "question_length_per_split":
            for split_name, split_stats in val.items():
                rows.append({"metric": f"question_length_{split_name}", **split_stats})
        else:
            rows.append({"metric": key, **val})
    csv_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "dataset_statistics.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("DATASET DESCRIPTIVE STATISTICS")
    print("=" * 70)
    for key, val in stats.items():
        if key == "question_length_per_split":
            for split_name, split_stats in val.items():
                print(f"  question_length_{split_name}: "
                      f"n={split_stats['count']}  mean={split_stats['mean']:.2f}  "
                      f"std={split_stats['std']:.2f}  min={split_stats['min']:.0f}  "
                      f"med={split_stats['median']:.0f}  max={split_stats['max']:.0f}")
        else:
            print(f"  {key}: n={val['count']}  mean={val['mean']:.2f}  "
                  f"std={val['std']:.2f}  min={val['min']:.0f}  "
                  f"med={val['median']:.0f}  max={val['max']:.0f}")

    # --- LaTeX output ---
    print("\n" + "=" * 70)
    print("LATEX TABLE ROWS")
    print("=" * 70)

    latex_fields = [
        ("Question length (chars)", "question_length_chars"),
        ("Answer length (chars)", "answer_length_chars"),
        ("Gold context length (chars)", "gold_context_length_chars"),
        ("Hop count: dense class", "hop_count_dense_retrieval"),
        ("Hop count: graph class", "hop_count_graph_traversal"),
        ("Hop count: hybrid class", "hop_count_hybrid_reasoning"),
    ]
    for display, key in latex_fields:
        s = stats[key]
        print(f"{display:35s} & {s['count']} & {s['mean']:.2f} & {s['std']:.2f} \\\\")

    print("=" * 70)


if __name__ == "__main__":
    main()
