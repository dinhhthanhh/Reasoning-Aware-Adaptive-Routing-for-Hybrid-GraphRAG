"""Script to train the router model.

Usage:
    python scripts/train_router.py [--config PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from loguru import logger

from router.train_router import generate_weak_labels, train_and_evaluate


def main() -> None:
    """Train the router XGBoost classifier."""
    parser = argparse.ArgumentParser(
        description="Train the two-stage router XGBoost classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=200,
        help="Maximum training samples per class (default: 200)",
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger.add(
        "logs/train_router_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )

    processed_dir = Path(config["data"]["processed_dir"])

    # Generate training data
    logger.info("Generating weak labels from processed documents...")
    df = generate_weak_labels(
        processed_docs_dir=str(processed_dir),
        config=config,
        max_queries_per_class=args.max_per_class,
    )

    print(f"\nGenerated {len(df)} training samples")
    print(f"  vector:  {len(df[df['label'] == 'vector'])}")
    print(f"  graph:   {len(df[df['label'] == 'graph'])}")
    print(f"  clarify: {len(df[df['label'] == 'clarify'])}")

    # Train
    logger.info("Training router model...")
    report = train_and_evaluate(df, config)

    print(f"\nModel saved to: {config['router']['stage1']['model_path']}")


if __name__ == "__main__":
    main()
