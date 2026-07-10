"""Reproducibly clean the legal_strict QA splits and emit an audit report.

What it does
------------
1. Loads the existing strict splits (train/dev/test).
2. Flags records whose ``answer`` is a placeholder / stale / empty value using a
   documented set of patterns (NO answers are fabricated; bad records are only
   removed).
3. Writes cleaned splits to ``qa_pipeline/data/legal_strict_clean/``.
4. Verifies there is no question-text leakage across the cleaned splits.
5. Writes:
     * ``data/audit_reports/qa_quality_audit.md``  (human-readable report)
     * ``results/final/excluded_ids.json``         (all removed ids)
     * ``results/final/excluded_test_ids.json``    (removed test ids only)
   The test-id file feeds ``evaluation.benchmark.rescore_predictions
   --exclude-ids`` so official metrics are computed on the cleaned test set.

Run
---
    python scripts/regenerate_splits.py

Idempotent: re-running reproduces identical outputs (no randomness).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "qa_pipeline" / "data" / "legal_strict"
OUT_DIR = PROJECT_ROOT / "qa_pipeline" / "data" / "legal_strict_clean"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit_reports"
FINAL_DIR = PROJECT_ROOT / "results" / "final"

SPLITS = ("train", "dev", "test")

# Documented placeholder / stale-answer patterns. Each maps reason -> regex.
PLACEHOLDER_PATTERNS: dict[str, re.Pattern[str]] = {
    "dang_cap_nhat": re.compile(r"đang cập nhật", re.IGNORECASE),
    "arrow_ban_prompt": re.compile(r"=>\s*Bạn", re.IGNORECASE),
    "van_ban_dang_cap_nhat": re.compile(r"Văn bản này đang cập nhật", re.IGNORECASE),
}
MIN_ANSWER_CHARS = 10


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def classify(record: dict[str, Any]) -> list[str]:
    """Return the list of failure reasons for a record (empty list = clean)."""
    answer = str(record.get("answer") or "")
    reasons = [
        name for name, pat in PLACEHOLDER_PATTERNS.items() if pat.search(answer)
    ]
    if len(answer.strip()) < MIN_ANSWER_CHARS:
        reasons.append("empty_or_tiny_answer")
    return reasons


def load_split(split: str) -> list[dict[str, Any]]:
    with open(SRC_DIR / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, Any]] = {}
    excluded_ids: list[str] = []
    excluded_test_ids: list[str] = []
    examples: list[tuple[str, str, str]] = []  # (split, id, answer snippet)
    reason_totals: dict[str, int] = {}
    cleaned_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in SPLITS:
        records = load_split(split)
        kept: list[dict[str, Any]] = []
        removed = 0
        route_before: dict[str, int] = {}
        route_after: dict[str, int] = {}

        for rec in records:
            route_before[rec.get("routing_label", "?")] = (
                route_before.get(rec.get("routing_label", "?"), 0) + 1
            )
            reasons = classify(rec)
            if reasons:
                removed += 1
                excluded_ids.append(rec["id"])
                if split == "test":
                    excluded_test_ids.append(rec["id"])
                for r in reasons:
                    reason_totals[r] = reason_totals.get(r, 0) + 1
                if len(examples) < 6:
                    snippet = str(rec.get("answer") or "")[-90:].replace("\n", " ")
                    examples.append((split, rec["id"], snippet))
            else:
                kept.append(rec)
                route_after[rec.get("routing_label", "?")] = (
                    route_after.get(rec.get("routing_label", "?"), 0) + 1
                )

        cleaned_by_split[split] = kept
        with open(OUT_DIR / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)

        summary[split] = {
            "total": len(records),
            "removed": removed,
            "kept": len(kept),
            "route_before": route_before,
            "route_after": route_after,
        }

    # Leakage check on cleaned splits (question-text overlap).
    qsets = {
        s: {_norm_question(r["question"]) for r in recs}
        for s, recs in cleaned_by_split.items()
    }
    leakage = {
        "train_∩_test": sorted(qsets["train"] & qsets["test"]),
        "train_∩_dev": sorted(qsets["train"] & qsets["dev"]),
        "dev_∩_test": sorted(qsets["dev"] & qsets["test"]),
    }
    leakage_counts = {k: len(v) for k, v in leakage.items()}

    with open(FINAL_DIR / "excluded_ids.json", "w", encoding="utf-8") as f:
        json.dump(excluded_ids, f, ensure_ascii=False, indent=2)
    with open(FINAL_DIR / "excluded_test_ids.json", "w", encoding="utf-8") as f:
        json.dump(excluded_test_ids, f, ensure_ascii=False, indent=2)

    _write_audit(summary, reason_totals, leakage_counts, examples, len(excluded_ids))

    print("Cleaning complete.")
    for split in SPLITS:
        s = summary[split]
        print(f"  {split:5s}: {s['total']} -> {s['kept']} (removed {s['removed']})")
    print(f"  total removed: {len(excluded_ids)}")
    print(f"  leakage (cleaned): {leakage_counts}")
    print(f"  cleaned splits -> {OUT_DIR.relative_to(PROJECT_ROOT)}")


def _write_audit(
    summary: dict[str, dict[str, Any]],
    reason_totals: dict[str, int],
    leakage_counts: dict[str, int],
    examples: list[tuple[str, str, str]],
    total_removed: int,
) -> None:
    lines: list[str] = []
    lines.append("# QA Dataset Quality Audit")
    lines.append("")
    lines.append(
        "Generated by `scripts/regenerate_splits.py`. Bad records are **removed**, "
        "never edited or replaced with fabricated answers."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Split | Before | Removed | After |")
    lines.append("|---|---:|---:|---:|")
    for split in SPLITS:
        s = summary[split]
        lines.append(f"| {split} | {s['total']} | {s['removed']} | {s['kept']} |")
    total_before = sum(summary[s]["total"] for s in SPLITS)
    total_after = sum(summary[s]["kept"] for s in SPLITS)
    lines.append(
        f"| **total** | **{total_before}** | **{total_removed}** | **{total_after}** |"
    )
    lines.append("")
    lines.append("## Removal reasons")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|---|---:|")
    for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
        lines.append(f"| `{reason}` | {count} |")
    lines.append("")
    lines.append(
        "A record may match more than one pattern, so reason counts can exceed the "
        "number of removed records."
    )
    lines.append("")
    lines.append("## Route distribution (cleaned)")
    lines.append("")
    lines.append("| Split | dense | graph | hybrid |")
    lines.append("|---|---:|---:|---:|")
    for split in SPLITS:
        ra = summary[split]["route_after"]
        lines.append(
            f"| {split} | {ra.get('dense_retrieval', 0)} | "
            f"{ra.get('graph_traversal', 0)} | {ra.get('hybrid_reasoning', 0)} |"
        )
    lines.append("")
    lines.append("## Leakage check (cleaned splits, by normalised question text)")
    lines.append("")
    lines.append("| Overlap | Count |")
    lines.append("|---|---:|")
    for key, count in leakage_counts.items():
        lines.append(f"| {key} | {count} |")
    lines.append("")
    lines.append("## Example removed records")
    lines.append("")
    for split, rid, snippet in examples:
        lines.append(f"- `{rid}` ({split}) — answer ends: `...{snippet}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- Cleaned splits: `qa_pipeline/data/legal_strict_clean/{train,dev,test}.json`")
    lines.append("- Excluded ids: `results/final/excluded_ids.json`")
    lines.append("- Excluded test ids: `results/final/excluded_test_ids.json`")
    lines.append("")

    with open(AUDIT_DIR / "qa_quality_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
