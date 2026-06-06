"""Build a larger legal QA evaluation set for research reporting.

The default end-to-end benchmark uses only qa_pipeline/data/final/test.json.
For a research-style evaluation, this script creates a deterministic,
stratified subset from the existing verified QA splits while preserving the
original QA fields used by run_benchmark_eval.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = {
    "dense_retrieval": 300,
    "graph_traversal": 150,
    "hybrid_reasoning": 150,
}


def _load_items(input_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for split in ("test", "dev", "train"):
        path = input_dir / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for idx, item in enumerate(data):
            row = dict(item)
            row["source_split"] = split
            row["source_index"] = idx
            items.append(row)
    return items


def _quality_ok(item: dict[str, Any]) -> bool:
    question = " ".join(str(item.get("question") or "").split())
    answer = " ".join(str(item.get("answer") or "").split())
    route = item.get("routing_label")
    if route not in DEFAULT_TARGETS:
        return False
    if len(question) < 12 or len(answer) < 2:
        return False
    low_value = ("không có trong ngữ cảnh", "không có thông tin", "đang cập nhật nội dung")
    text = f"{question} {answer}".lower()
    return not any(pattern in text for pattern in low_value)


def build_eval_set(input_dir: Path, output: Path, seed: int, targets: dict[str, int]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_items = [item for item in _load_items(input_dir) if _quality_ok(item)]

    seen_questions: set[str] = set()
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        question_key = " ".join(str(item.get("question") or "").lower().split())
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        by_route[str(item["routing_label"])].append(item)

    selected: list[dict[str, Any]] = []
    for route, target in targets.items():
        pool = by_route.get(route, [])
        if len(pool) < target:
            raise ValueError(f"Not enough {route} items: requested {target}, available {len(pool)}")
        rng.shuffle(pool)
        selected.extend(pool[:target])

    rng.shuffle(selected)
    for idx, item in enumerate(selected):
        item["id"] = f"legal_research_{idx:04d}"
        item["eval_set"] = "legal_research_600"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 600-sample legal QA research eval set.")
    parser.add_argument("--input-dir", default="qa_pipeline/data/final")
    parser.add_argument("--output", default="qa_pipeline/data/final/legal_research_eval_600.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = build_eval_set(
        input_dir=Path(args.input_dir),
        output=Path(args.output),
        seed=args.seed,
        targets=DEFAULT_TARGETS,
    )
    print(f"Wrote {len(selected)} items to {args.output}")
    print("Route distribution:", dict(Counter(item["routing_label"] for item in selected)))
    print("Source split distribution:", dict(Counter(item["source_split"] for item in selected)))
    print("Question type distribution:", dict(Counter(item.get("question_type", "unknown") for item in selected)))


if __name__ == "__main__":
    main()
