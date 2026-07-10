"""Quick stratified summary for phapdien_strict benchmark CSVs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval_results"
OUT = ROOT / "results" / "final" / "phapdien_strict_stratified_f1.json"

SYSTEMS = [
    "pure_vector",
    "pure_graph",
    "single_stage_router",
    "two_stage_hybrid",
]


def analyze(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by_route: dict[str, list[float]] = defaultdict(list)
    f1s = []
    for row in rows:
        f1 = float(row["F1"])
        f1s.append(f1)
        by_route[row["Expected_Route"]].append(f1)
    return {
        "n": len(rows),
        "overall_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "by_expected_route": {
            k: {"n": len(v), "f1": sum(v) / len(v)} for k, v in sorted(by_route.items())
        },
    }


def main() -> None:
    summary = {}
    for sys in SYSTEMS:
        summary[sys] = analyze(EVAL / f"phapdien_strict_{sys}_results.csv")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
