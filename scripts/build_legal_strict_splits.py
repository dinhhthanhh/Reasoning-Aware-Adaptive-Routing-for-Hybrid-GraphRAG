"""Build strict legal QA splits with a large held-out test set.

This avoids train/test leakage for research reporting:
- test is sampled first and never used for router training or tuning;
- dev is sampled from the remaining pool;
- train contains all remaining items.

The script preserves the qa_pipeline JSON schema used by router training and
benchmark evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEST_TARGETS = {
    "dense_retrieval": 300,
    "graph_traversal": 150,
    "hybrid_reasoning": 150,
}

DEV_TARGETS = {
    "dense_retrieval": 50,
    "graph_traversal": 20,
    "hybrid_reasoning": 10,
}


def _load_source_items(input_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
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
    if route not in TEST_TARGETS:
        return False
    if len(question) < 12 or len(answer) < 2:
        return False
    text = f"{question} {answer}".lower()
    low_value = ("không có trong ngữ cảnh", "không có thông tin", "đang cập nhật nội dung")
    return not any(pattern in text for pattern in low_value)


def _deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = " ".join(str(item.get("question") or "").lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _take_stratified(
    by_route: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for route, target in targets.items():
        pool = by_route[route]
        if len(pool) < target:
            raise ValueError(f"Not enough {route}: requested {target}, available {len(pool)}")
        rng.shuffle(pool)
        selected.extend(pool[:target])
        del pool[:target]
    rng.shuffle(selected)
    return selected


def _stamp(items: list[dict[str, Any]], split: str) -> None:
    for idx, item in enumerate(items):
        item["id"] = f"legal_strict_{split}_{idx:04d}"
        item["strict_split"] = split
        item["eval_set"] = "legal_strict"


def build_strict_splits(input_dir: Path, output_dir: Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    items = _deduplicate([item for item in _load_source_items(input_dir) if _quality_ok(item)])

    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_route[str(item["routing_label"])].append(item)

    test = _take_stratified(by_route, TEST_TARGETS, rng)
    dev = _take_stratified(by_route, DEV_TARGETS, rng)

    train: list[dict[str, Any]] = []
    for route_items in by_route.values():
        train.extend(route_items)
    rng.shuffle(train)

    _stamp(train, "train")
    _stamp(dev, "dev")
    _stamp(test, "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_items in (("train", train), ("dev", dev), ("test", test)):
        (output_dir / f"{split}.json").write_text(
            json.dumps(split_items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {"train": train, "dev": dev, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict legal train/dev/test splits.")
    parser.add_argument("--input-dir", default="qa_pipeline/data/final")
    parser.add_argument("--output-dir", default="qa_pipeline/data/legal_strict")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    splits = build_strict_splits(Path(args.input_dir), Path(args.output_dir), args.seed)
    print(f"Wrote strict splits to {args.output_dir}")
    for split, items in splits.items():
        print(split, len(items), dict(Counter(item["routing_label"] for item in items)))


if __name__ == "__main__":
    main()
