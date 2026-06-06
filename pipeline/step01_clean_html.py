"""STEP 1 — Clean HTML to plain Vietnamese text.

Input:  Crawl/raw_data_tvpl.json
Output: data/processed/cleaned_docs.json

Each output record:
  {
    "title": "...",
    "url": "...",
    "doc_number": "...",
    "text": "<cleaned plain text>"
  }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Ensure root is on path so we can import project modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

from loguru import logger


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace from a raw HTML string.

    Args:
        raw_html: Raw HTML content from the crawler.

    Returns:
        Clean plain-text string.
    """
    if not raw_html:
        return ""

    if _BS4_AVAILABLE:
        soup = BeautifulSoup(raw_html, "lxml")
        # Remove script/style elements
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        # Fallback: basic regex stripping
        text = re.sub(r"<[^>]+>", " ", raw_html)

    # Normalize whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def process_file(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Clean all documents from input JSON and write to output.

    Args:
        input_path: Path to raw_data_tvpl.json.
        output_path: Path to write cleaned_docs.json.

    Returns:
        Number of documents processed.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from {}", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    logger.info("Processing {} documents...", len(docs))
    cleaned: list[dict] = []

    for i, doc in enumerate(docs):
        raw_html = doc.get("raw_html", "")
        text = clean_html(raw_html)

        if len(text.strip()) < 50:
            logger.debug("Skipping short document #{}: {}", i, doc.get("title", "")[:60])
            continue

        cleaned.append(
            {
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "doc_number": doc.get("metadata", {}).get("doc_number", ""),
                "text": text,
            }
        )

        if (i + 1) % 50 == 0:
            logger.info("  Processed {}/{}", i + 1, len(docs))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    logger.info(
        "Step 1 done | input={} | output={} | saved={}",
        len(docs),
        output_path,
        len(cleaned),
    )
    return len(cleaned)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Clean HTML → plain text")
    parser.add_argument(
        "--input",
        default=str(ROOT / "Crawl" / "raw_data_tvpl.json"),
        help="Path to input raw JSON file",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "cleaned_docs.json"),
        help="Path to output cleaned JSON file",
    )
    args = parser.parse_args()

    count = process_file(args.input, args.output)
    print(f"\n✅ Step 1 complete: {count} documents cleaned → {args.output}")
