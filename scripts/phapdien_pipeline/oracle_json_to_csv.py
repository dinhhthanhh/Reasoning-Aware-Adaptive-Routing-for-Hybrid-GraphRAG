"""Convert run_oracle_eval.py JSON output to benchmark-style CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data.get("per_sample", [])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ID",
        "System",
        "Query",
        "Ground_Truth",
        "Generated_Answer",
        "Expected_Route",
        "Route",
        "Actual_Route",
        "KG_Source",
        "Sources",
        "Context_Chars",
        "Context_Preview",
        "Steps",
        "Stage2",
        "Stage2_Override",
        "Time_ms",
        "EM",
        "F1",
        "Acc",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ID": f"phapdien_strict_test_{row['idx']:04d}",
                    "System": "oracle",
                    "Query": row.get("question", ""),
                    "Ground_Truth": row.get("ground_truth", ""),
                    "Generated_Answer": row.get("answer", ""),
                    "Expected_Route": row.get("gold_route", ""),
                    "Route": row.get("executed_route", ""),
                    "Actual_Route": row.get("executed_route", ""),
                    "KG_Source": "",
                    "Sources": "",
                    "Context_Chars": 0,
                    "Context_Preview": "",
                    "Steps": 0,
                    "Stage2": 0,
                    "Stage2_Override": 0,
                    "Time_ms": row.get("latency_ms", 0),
                    "EM": row.get("em", 0),
                    "F1": row.get("f1", 0),
                    "Acc": row.get("acc", 0),
                }
            )
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
