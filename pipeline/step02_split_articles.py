"""STEP 2 — Split cleaned documents into individual legal articles.

Input:  data/processed/cleaned_docs.json
Output: data/processed/articles.json

Each output record:
  {
    "law": "Quyết định 456/QĐ-TTg",
    "doc_number": "456/QĐ-TTg",
    "url": "...",
    "article_id": "Điều 1",
    "article_key": "456/QĐ-TTg::Điều 1",   # unique key for QA checkpoint
    "content": "<full article text including heading>"
  }

Splitting strategy:
  1. Find genuine "Điều X" article headings at line-start (not mid-sentence refs).
  2. Slice text between consecutive anchors.
  3. Deduplicate: if same (law, article_id) appears multiple times (common in
     consolidated/"văn bản hợp nhất" texts with amendments), keep the entry
     with the LONGEST content.
  4. Post-filter: remove boilerplate signature blocks and articles too short
     to generate useful QA pairs.
  5. Clean footnote markers [2], [12] etc. from content.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# A genuine article heading: "Điều N" that appears at the START of a new line
# (possibly indented). This avoids matching mid-sentence references like
# "quy định tại Điều 6 khoản 2 Thông tư này".
# We allow an optional title after the number (e.g. "Điều 3. Giải thích từ ngữ").
ARTICLE_HEADING = re.compile(
    r"(?:^|\n)[ \t]*(Điều\s+\d+[\.\:]?(?:[ \t]+[^\n]{0,150})?)",
    re.UNICODE,
)

# Footnote / amendment reference markers embedded in text: [2], [12] etc.
FOOTNOTE_MARKER = re.compile(r"\s*\[\d+\]\s*")

# Boilerplate header/footer phrases indicating non-substantive content.
# An article containing these near the TOP is likely a signature/distribution block.
BOILERPLATE_PHRASES = [
    "Nơi nhận:",
    "KT. BỘ TRƯỞNG",
    "KT. THỦ TƯỚNG",
    "KT. CHỦ TỊCH",
    "TM. ỦY BAN NHÂN DÂN",
    "PHÓ THỦ TƯỚNG",
    "Phụ lục số",
    "MẪU THẺ",
    "MẪU GIẤY XÁC NHẬN",
    "GIẤY XÁC NHẬN",
    "Lưu: VT,",
    "VPCP: BTCN",
    "XÁC THỰC VĂN BẢN HỢP NHẤT",
    "Mặt trước (hình",
    "Mặt sau (hình",
]

MIN_CONTENT_LEN = 80   # Minimum chars for a useful article


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_content(text: str) -> str:
    """Remove footnote markers and normalise excessive blank lines."""
    text = FOOTNOTE_MARKER.sub(" ", text)
    # Collapse 3+ consecutive blank lines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_boilerplate(content: str) -> bool:
    """Return True if the article is just a signature / distribution block."""
    head = content[:600]
    for phrase in BOILERPLATE_PHRASES:
        if phrase in head:
            return True
    return False


def _extract_article_id(heading_text: str) -> str:
    """Normalize 'Điều 12. Tiêu chuẩn...' → 'Điều 12'."""
    m = re.match(r"(Điều\s+\d+)", heading_text.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else heading_text.strip()[:20]


# ---------------------------------------------------------------------------
# Core splitting
# ---------------------------------------------------------------------------


def split_articles(text: str, title: str, doc_number: str, url: str) -> list[dict]:
    """Split a document's text into articles.

    Args:
        text: Full cleaned text of the document.
        title: Document title (used as ``law`` field).
        doc_number: Document number, e.g. ``"456/QĐ-TTg"``.
        url: Source URL.

    Returns:
        List of raw article dicts (before dedup / quality filter).
    """
    matches = list(ARTICLE_HEADING.finditer(text))

    if not matches:
        # No genuine article headings — treat entire doc as one entry.
        content = _clean_content(text[:4000])
        if len(content) >= MIN_CONTENT_LEN and not _is_boilerplate(content):
            return [
                {
                    "law": title,
                    "doc_number": doc_number,
                    "url": url,
                    "article_id": "Toàn văn",
                    "article_key": f"{doc_number}::Toàn văn",
                    "content": content,
                }
            ]
        return []

    articles: list[dict] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        raw_heading = match.group(1)
        article_id = _extract_article_id(raw_heading)
        content = _clean_content(text[start:end])

        if len(content) < MIN_CONTENT_LEN:
            continue
        if _is_boilerplate(content):
            logger.debug("Boilerplate skipped: {} | {}", article_id, doc_number)
            continue

        articles.append(
            {
                "law": title,
                "doc_number": doc_number,
                "url": url,
                "article_id": article_id,
                "article_key": f"{doc_number}::{article_id}",
                "content": content[:4000],
            }
        )

    return articles


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """Merge duplicate (law, article_id) pairs, keeping the longest content.

    Consolidated texts (*văn bản hợp nhất*) contain multiple amendment
    versions of the same article under the same article_id.  We keep only
    the version with the most content (richest information).

    Args:
        articles: Raw list that may contain duplicates.

    Returns:
        Deduplicated list in first-occurrence order.
    """
    seen: dict[str, int] = {}   # article_key → index in result
    result: list[dict] = []
    replaced = 0

    for article in articles:
        key = article["article_key"]
        if key not in seen:
            seen[key] = len(result)
            result.append(article)
        else:
            idx = seen[key]
            if len(article["content"]) > len(result[idx]["content"]):
                result[idx] = article
                replaced += 1

    duplicates_removed = len(articles) - len(result)
    logger.info(
        "Dedup: {} raw → {} unique (removed {}, content-upgraded {})",
        len(articles),
        len(result),
        duplicates_removed,
        replaced,
    )
    return result


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    min_content_len: int = MIN_CONTENT_LEN,
) -> int:
    """Split all cleaned docs into articles, dedup, filter, and save.

    Args:
        input_path: Path to ``cleaned_docs.json``.
        output_path: Path where ``articles.json`` will be written.
        min_content_len: Minimum content length to include an article.

    Returns:
        Number of articles in the final output.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned docs from {}", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    logger.info("Splitting {} documents...", len(docs))
    raw_articles: list[dict] = []

    for i, doc in enumerate(docs):
        arts = split_articles(
            text=doc["text"],
            title=doc.get("title", ""),
            doc_number=doc.get("doc_number", ""),
            url=doc.get("url", ""),
        )
        arts = [a for a in arts if len(a["content"]) >= min_content_len]
        raw_articles.extend(arts)

        logger.info(
            "  [{}/{}] {} → {} articles (running total: {})",
            i + 1,
            len(docs),
            doc.get("doc_number", "?"),
            len(arts),
            len(raw_articles),
        )

    # Dedup across the entire corpus
    all_articles = deduplicate_articles(raw_articles)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    logger.info(
        "Step 2 done | docs={} | articles={} | saved to {}",
        len(docs),
        len(all_articles),
        output_path,
    )
    return len(all_articles)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: Split documents into legal articles")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "cleaned_docs.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "articles.json"),
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=MIN_CONTENT_LEN,
        help=f"Minimum content length to include (default: {MIN_CONTENT_LEN})",
    )
    args = parser.parse_args()

    count = process_file(args.input, args.output, args.min_len)
    print(f"\n✅ Step 2 complete: {count} clean articles → {args.output}")
