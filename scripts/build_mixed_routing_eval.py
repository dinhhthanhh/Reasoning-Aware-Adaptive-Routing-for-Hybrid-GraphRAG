#!/usr/bin/env python3
"""Build a mixed routing eval set from legal QA test + clarify samples."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed routing evaluation set")
    parser.add_argument("--legal-test", default="qa_pipeline/data/final/test.json")
    parser.add_argument("--clarify-eval", default="evaluation/legal_clarify_eval.json")
    parser.add_argument("--output", default="evaluation/legal_mixed_routing_eval.json")
    parser.add_argument("--legal-limit", type=int, default=None)
    parser.add_argument("--clarify-limit", type=int, default=None)
    parser.add_argument(
        "--filter-ambiguous-legal",
        action="store_true",
        help="Drop legal-test negatives that are standalone ambiguous placeholders",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


AMBIGUOUS_LEGAL_PATTERNS = re.compile(
    r"\b(?:"
    r"thông\s+tư\s+này|quyết\s+định\s+này|nghị\s+định\s+này|"
    r"văn\s+bản\s+này|quy\s+định\s+này|điều\s+khoản\s+đó|"
    r"văn\s+bản\s+1|văn\s+bản\s+2|bên\s+a|bên\s+b"
    r")\b",
    re.IGNORECASE,
)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def normalize_legal_item(item: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "id": item.get("id") or item.get("_id") or f"legal_test_{idx:04d}",
        "query": item.get("query") or item.get("question") or "",
        "history": item.get("history", []),
        "expected_route": item.get("expected_route") or item.get("routing_label") or "",
        "expected_complexity": item.get("expected_complexity", ""),
        "ambiguity_type": "",
        "ground_truth_answer": item.get("ground_truth") or item.get("answer"),
        "source": "legal_test",
    }


def is_ambiguous_legal_negative(item: dict[str, Any]) -> bool:
    query = str(item.get("query", ""))
    return bool(AMBIGUOUS_LEGAL_PATTERNS.search(query))


def normalize_clarify_item(item: dict[str, Any], idx: int) -> dict[str, Any]:
    normalized = dict(item)
    normalized["id"] = normalized.get("id") or f"clarify_{idx:04d}"
    normalized["query"] = normalized.get("query") or normalized.get("question") or ""
    normalized["expected_route"] = normalized.get("expected_route") or "clarify"
    normalized["source"] = "clarify_eval"
    return normalized


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    legal_items = [
        normalize_legal_item(item, idx)
        for idx, item in enumerate(load_json_list(Path(args.legal_test)))
    ]
    clarify_items = [
        normalize_clarify_item(item, idx)
        for idx, item in enumerate(load_json_list(Path(args.clarify_eval)))
    ]

    if args.legal_limit is not None:
        legal_items = legal_items[: args.legal_limit]
    if args.filter_ambiguous_legal:
        legal_items = [
            item for item in legal_items
            if not is_ambiguous_legal_negative(item)
        ]
    if args.clarify_limit is not None:
        clarify_items = clarify_items[: args.clarify_limit]

    mixed = legal_items + clarify_items
    rng.shuffle(mixed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mixed, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "legal_items": len(legal_items),
                "clarify_items": len(clarify_items),
                "total": len(mixed),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
