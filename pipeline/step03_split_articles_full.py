"""STEP 3 (Full) — Split documents into individual legal articles.
Optimized for large 518k datasets via streaming (JSONL).

Input:  vbpl_crawler/output/full_dataset.jsonl
Output: data/processed/articles_full.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "full_dataset.jsonl"

# --- Constants from pipeline/step2_split_articles.py ---
ARTICLE_HEADING = re.compile(
    r"(?:^|\n)[ \t]*(Điều\s+\d+[\.\:]?(?:[ \t]+[^\n]{0,150})?)",
    re.UNICODE,
)
FOOTNOTE_MARKER = re.compile(r"\s*\[\d+\]\s*")
MIN_CONTENT_LEN = 80
BOILERPLATE_PHRASES = [
    "Nơi nhận:", "KT. BỘ TRƯỞNG", "KT. THỦ TƯỚNG", "KT. CHỦ TỊCH",
    "TM. ỦY BAN NHÂN DÂN", "PHÓ THỦ TƯỚNG", "Phụ lục số", "Lưu: VT,"
]

def _clean_content(text: str) -> str:
    text = FOOTNOTE_MARKER.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _is_boilerplate(content: str) -> bool:
    head = content[:600]
    for phrase in BOILERPLATE_PHRASES:
        if phrase in head: return True
    return False

def _extract_article_id(heading_text: str) -> str:
    m = re.match(r"(Điều\s+\d+)", heading_text.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else heading_text.strip()[:20]

def split_doc_to_articles(doc: dict) -> list[dict]:
    text = doc.get("content", "")
    title = doc.get("title", "")
    doc_num = doc.get("document_number", doc.get("id", "Unknown"))
    url = doc.get("url", "")
    
    matches = list(ARTICLE_HEADING.finditer(text))
    if not matches:
        content = _clean_content(text[:4000])
        if len(content) >= MIN_CONTENT_LEN and not _is_boilerplate(content):
            return [{
                "law": title, "doc_number": doc_num, "url": url,
                "article_id": "Toàn văn", "article_key": f"{doc_num}::Toàn văn",
                "content": content
            }]
        return []

    articles = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_heading = match.group(1)
        article_id = _extract_article_id(raw_heading)
        content = _clean_content(text[start:end])
        
        if len(content) < MIN_CONTENT_LEN or _is_boilerplate(content): continue
        
        articles.append({
            "law": title, "doc_number": doc_num, "url": url,
            "article_id": article_id, "article_key": f"{doc_num}::{article_id}",
            "content": content[:4000]
        })
    return articles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT_PATH))
    args = parser.parse_args()
    
    in_path = Path(args.input_path)
    out_path = Path("data/processed/articles_full.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We won't do global deduplication here to avoid memory issues,
    # but we can do per-doc dedup if needed.
    
    print(f"Splitting documents from {in_path} into articles...")
    total_articles = 0
    
    # Count lines first for tqdm
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        total_docs = sum(1 for _ in f)

    with open(in_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_docs, desc="Processing docs"):
            try:
                doc = json.loads(line)
                arts = split_doc_to_articles(doc)
                for art in arts:
                    fout.write(json.dumps(art, ensure_ascii=False) + "\n")
                total_articles += len(arts)
            except Exception as e:
                logger.error(f"Error processing doc: {e}")

    print(f"\n✅ Done! Extracted {total_articles} articles to {out_path}.")

if __name__ == "__main__":
    main()
