"""Script to run the hybrid QA pipeline.

Supports interactive mode, batch mode, and evaluation mode.

Usage:
    python scripts/run_pipeline.py                    # Interactive mode
    python scripts/run_pipeline.py --input queries.txt # Batch mode
    python scripts/run_pipeline.py --evaluate          # Run evaluation
    python scripts/run_pipeline.py --verbose           # Verbose output
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import yaml
from loguru import logger

from pipeline.hybrid_pipeline import HybridPipeline


def main() -> None:
    """Run the hybrid QA pipeline."""
    parser = argparse.ArgumentParser(
        description="Vietnamese Legal QA with Reasoning-Aware Adaptive Routing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input file for batch mode (one query per line)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed routing information",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on test queries",
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    )

    # Setup logging
    logger.add(
        "logs/pipeline_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG" if args.verbose else "INFO",
    )

    if args.evaluate:
        _run_evaluation(config_path, args.verbose)
    elif args.input:
        _run_batch(config_path, args.input, args.verbose)
    else:
        _run_interactive(config_path, args.verbose)


def _run_interactive(config_path: str, verbose: bool) -> None:
    """Run pipeline in interactive mode.

    Args:
        config_path: Path to config.yaml.
        verbose: Show routing details.
    """
    print("=" * 60)
    print("Vietnamese Legal QA System")
    print("Reasoning-Aware Adaptive Routing for Hybrid GraphRAG")
    print("=" * 60)
    print("Nhập câu hỏi pháp luật bằng tiếng Việt.")
    print("Gõ 'quit' hoặc 'exit' để thoát.")
    print("Gõ 'new' để bắt đầu phiên hội thoại mới.")
    print("=" * 60)

    pipeline = HybridPipeline(config_path)
    session_id = str(uuid.uuid4())[:8]

    while True:
        try:
            query = input("\n🔍 Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Tạm biệt!")
            break
        if query.lower() == "new":
            session_id = str(uuid.uuid4())[:8]
            pipeline.conversation_manager.clear_session(session_id)
            print("📝 Phiên hội thoại mới đã bắt đầu.")
            continue

        response = pipeline.query(
            query=query,
            session_id=session_id,
            verbose=verbose,
        )

        print(f"\n📋 Trả lời [{response.route_used}]:")
        print(f"   {response.answer}")

        if verbose:
            print(f"\n🔧 Chi tiết routing:")
            print(f"   Route: {response.route_used}")
            print(f"   Confidence: {response.confidence:.3f}")
            print(f"   Stage 2 invoked: {response.stage2_invoked}")
            print(f"   Reasoning: {response.router_reasoning}")
            print(f"   Sources: {response.sources}")
            print(f"   Latency: {response.latency_ms:.0f}ms")
            print(f"   Ambiguous: {response.is_ambiguous}")


def _run_batch(config_path: str, input_path: str, verbose: bool) -> None:
    """Run pipeline in batch mode.

    Args:
        config_path: Path to config.yaml.
        input_path: Path to input file with one query per line.
        verbose: Show routing details.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    queries = [line.strip() for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Processing {len(queries)} queries from {input_path}")

    pipeline = HybridPipeline(config_path)
    session_id = str(uuid.uuid4())[:8]

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Q: {query}")

        response = pipeline.query(
            query=query,
            session_id=session_id,
            verbose=verbose,
        )

        print(f"  A [{response.route_used}]: {response.answer[:200]}")
        if verbose:
            print(f"  Confidence: {response.confidence:.3f} | Latency: {response.latency_ms:.0f}ms")

    print(f"\nBatch processing complete: {len(queries)} queries processed")


def _run_evaluation(config_path: str, verbose: bool) -> None:
    """Run full evaluation.

    Args:
        config_path: Path to config.yaml.
        verbose: Show per-query details.
    """
    from evaluation.evaluate import run_evaluation
    run_evaluation(config_path=config_path, verbose=verbose)


if __name__ == "__main__":
    main()
