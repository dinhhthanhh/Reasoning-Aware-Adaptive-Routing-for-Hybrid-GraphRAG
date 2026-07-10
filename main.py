#!/usr/bin/env python3
"""
Vietnamese Legal Documents Crawler
===================================
Crawls from 3 sources:
  1. HuggingFace dataset: th1nhng0/vietnamese-legal-documents
       configs: relationships, metadata, content
  2. Pháp Điển: https://phapdien.moj.gov.vn/Pages/chi-tiet-bo-phap-dien.aspx

(VBPL was attempted but the crawler failed and has been archived; see
archive/legacy_vbpl/README.md.)

Usage:
    python main.py --source all --output data/
    python main.py --source huggingface --hf-configs relationships metadata content
    python main.py --source phapdien --max-chu-de 3

Requirements:
    pip install requests beautifulsoup4 lxml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from crawlers.huggingface_crawler import crawl_huggingface, ALL_CONFIGS
from crawlers.phapdien_crawler import crawl_phapdien

# NOTE: The VBPL crawler has been archived (it produced empty output; its HTML
# selectors were fragile). See archive/legacy_vbpl/README.md. VBPL is no longer
# a supported data source.


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("crawler.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese Legal Documents Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["all", "huggingface", "phapdien"],
        default="all",
        help="Which source to crawl (default: all)",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between requests in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # HuggingFace options
    hf_group = parser.add_argument_group("HuggingFace options")
    hf_group.add_argument(
        "--hf-configs",
        nargs="+",
        choices=ALL_CONFIGS,
        default=ALL_CONFIGS,
        metavar="CONFIG",
        help=(
            f"Which HF configs to crawl (default: all). "
            f"Choices: {', '.join(ALL_CONFIGS)}"
        ),
    )
    hf_group.add_argument(
        "--hf-max-rows",
        type=int,
        default=None,
        help="Max rows per split per config for HuggingFace (default: all)",
    )

    # Pháp Điển options
    pd_group = parser.add_argument_group("Pháp Điển options")
    pd_group.add_argument(
        "--max-chu-de",
        type=int,
        default=None,
        help="Max Chủ đề to crawl from Pháp Điển (default: all)",
    )

    return parser.parse_args()


def save_summary(output_dir: Path, results: dict) -> None:
    """Save a combined summary of all crawl results."""
    summary = {
        "crawled_at": datetime.now().isoformat(),
        "sources": {},
    }
    for source, data in results.items():
        if isinstance(data, list):
            count = len(data)
            sample = data[:3]
        elif isinstance(data, dict):
            count = data.get("count", 0)
            sample = data.get("sample", [])
        else:
            count = 0
            sample = []
            
        summary["sources"][source] = {
            "count": count,
            "sample": sample,
        }

    summary_file = output_dir / "crawl_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n{'='*60}")
    print("CRAWL SUMMARY")
    print(f"{'='*60}")
    for source, info in summary["sources"].items():
        print(f"  {source:20s}: {info['count']:,} records")
    print(f"  Summary saved → {summary_file}")
    print(f"{'='*60}\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── HuggingFace ──────────────────────────────────────────────────────────
    if args.source in ("all", "huggingface"):
        logger.info("=" * 60)
        logger.info("Starting HuggingFace crawler...")
        logger.info("=" * 60)
        try:
            hf_results = crawl_huggingface(
                output_dir=output_dir / "huggingface",
                configs=args.hf_configs,
                max_rows=args.hf_max_rows,
                delay=args.delay,
            )
            # Add each config separately to results for a detailed summary
            for config, res in hf_results.items():
                results[f"huggingface:{config}"] = res
        except Exception as e:
            logger.error(f"HuggingFace crawler failed: {e}", exc_info=True)
            results["huggingface"] = {"count": 0, "sample": []}

    # ── Pháp Điển ────────────────────────────────────────────────────────────
    if args.source in ("all", "phapdien"):
        logger.info("=" * 60)
        logger.info("Starting Pháp Điển crawler...")
        logger.info("=" * 60)
        try:
            pd_records = crawl_phapdien(
                output_dir=output_dir / "phapdien",
                max_chu_de=args.max_chu_de,
                delay=args.delay,
            )
            results["phapdien"] = pd_records
        except Exception as e:
            logger.error(f"Pháp Điển crawler failed: {e}", exc_info=True)
            results["phapdien"] = []

    save_summary(output_dir, results)


if __name__ == "__main__":
    main()
