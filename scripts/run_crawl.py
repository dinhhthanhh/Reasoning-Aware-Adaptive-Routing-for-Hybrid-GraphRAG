"""Script to crawl Vietnamese legal documents.

Usage:
    python scripts/run_crawl.py [--max-docs N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from crawlers.legal_crawler import LegalCrawler


def main() -> None:
    """Run the legal document crawler."""
    parser = argparse.ArgumentParser(
        description="Crawl Vietnamese legal documents from public sources."
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=500,
        help="Maximum number of documents to crawl (default: 500)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["web", "hf", "github", "all"],
        default="all",
        help="Data source to crawl (web, hf, github, or all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    # ... (rest of config loading logic)
    config_path = args.config if args.config else Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["crawler"]["max_docs"] = args.max_docs

    total = 0
    
    if args.source in ["web", "all"]:
        from crawlers.legal_crawler import LegalCrawler
        logger.info("Starting web crawler...")
        crawler = LegalCrawler(config)
        total += crawler.crawl()

    if args.source in ["hf", "all"]:
        from crawlers.hf_crawler import HFCrawler
        logger.info("Starting Hugging Face crawler...")
        hf_crawler = HFCrawler(config)
        total += hf_crawler.crawl("namphan1999/data-luat")

    if args.source in ["github", "all"]:
        from crawlers.github_crawler import GitHubCrawler
        logger.info("Starting GitHub crawler...")
        gh_crawler = GitHubCrawler(config)
        total += gh_crawler.crawl("https://github.com/mlalab/VNLegalText.git")

    logger.info("Crawling finished | total_docs={}", total)
    print(f"\nCrawling complete: {total} documents saved to {config['data']['raw_dir']}")


if __name__ == "__main__":
    main()
