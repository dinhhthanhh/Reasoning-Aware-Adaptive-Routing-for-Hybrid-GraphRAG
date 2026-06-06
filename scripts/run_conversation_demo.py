#!/usr/bin/env python3
"""Small multi-turn demo for conversation-aware routing.

The demo keeps a fixed session id so later questions can use conversation
history for coreference resolution, ambiguity detection, Stage 2 verification,
and answer generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.hybrid_pipeline import HybridPipeline


DEFAULT_TURNS = [
    "Công ty sử dụng lao động chưa thành niên thì cần lưu ý gì?",
    "Công ty đó có bị xử phạt không?",
    "Nếu không đáp ứng được thì phải làm sao?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a conversation-aware routing demo")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--session-id", default="demo_conversation")
    parser.add_argument("--turn", action="append", help="Add one user turn. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    turns = args.turn or DEFAULT_TURNS
    pipeline = HybridPipeline(args.config)

    for idx, query in enumerate(turns, 1):
        response = pipeline.query(query, session_id=args.session_id, verbose=False)
        print(f"\nTurn {idx}")
        print(f"User: {query}")
        if response.resolved_query and response.resolved_query != query:
            print(f"Resolved: {response.resolved_query}")
        print(
            "Route: "
            f"{response.route_used} | stage1={response.stage1_route} | "
            f"stage2={response.stage2_invoked} | kg={response.kg_source or 'n/a'} | "
            f"latency={response.latency_ms:.0f}ms"
        )
        print(f"Answer: {response.answer[:500]}")


if __name__ == "__main__":
    main()
