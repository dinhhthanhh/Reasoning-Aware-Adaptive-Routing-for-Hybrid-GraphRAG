import json
import re
from pathlib import Path
import logging
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PATTERNS = [
    (r'Số:\s*([0-9]+/[A-ZĐ\-]+)', "Số: Format"),
    (r'(\d+/VBHN-[A-Z]+)', "VBHN Format"),
    (r'(\d+/\d{4}/NĐ-CP)', "NĐ-CP"),
    (r'(\d+/\d{4}/TT-[A-ZĐ]+)', "Thông tư"),
    (r'(\d+/\d{4}/QH\d+)', "Luật QH"),
    (r'(\d+/\d{4}/QĐ-[A-ZĐ]+)', "Quyết định"),
    (r'(\d+/\d{4}/[A-ZĐ]+-[A-Z0-9Đ]+)', "General year/code"),
    (r'(\d+/[A-ZĐ]+-[A-Z0-9Đ]+)', "Short Format"),
]

def extract_law_number(text):
    text_subset = text[:2000]
    for pat_str, _ in PATTERNS:
        match = re.search(pat_str, text_subset, re.IGNORECASE)
        if match:
            # Fix Cyrillic 'С'
            return match.group(1).replace('\u0421', 'C').replace('\u0441', 'c').upper()
    return None

def extract_article_number(text):
    """Extracts just the number part (e.g. '8', '1a') from the start of the chunk."""
    match = re.search(r'^\s*(?:\*\*|###\s*|#+\s*)?Điều\s+(\d+[A-Za-zĐđ]?)\s*[\.:]?', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def split_by_dieu(content):
    """Splits content into multiple chunks by Điều boundary."""
    # Split using a positive lookahead to keep the 'Điều ...' at the start of each chunk
    pattern = re.compile(r'(?=^\s*(?:\*\*|###\s*|#+\s*)?Điều\s+\d+[A-Za-zĐđ]?\s*[\.:]?)', re.MULTILINE | re.IGNORECASE)
    splits = pattern.split(content)
    # Filter out empty splits
    return [s.strip() for s in splits if s.strip()]

def main():
    root = Path(__file__).resolve().parent.parent
    hf_in = root / "data" / "processed" / "hf_processed.jsonl"
    hf_out = root / "data" / "processed" / "hf_rechunked.jsonl"
    
    if not hf_in.exists():
        logging.error(f"Input file not found: {hf_in}")
        return
        
    hf_out.parent.mkdir(parents=True, exist_ok=True)
    
    total_docs = 0
    total_chunks = 0
    canonical_chunks = 0
    fallback_chunks = 0
    
    with open(hf_in, "r", encoding="utf-8") as fin, open(hf_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Rechunking HF"):
            if not line.strip(): continue
            total_docs += 1
            try:
                doc = json.loads(line)
                content = doc.get("content_markdown", "") or doc.get("text", "")
                doc_id = str(doc.get("doc_id", f"unknown_{total_docs}"))
                title = doc.get("title", "")
                
                text_to_search = title + "\n" + content
                law_number = extract_law_number(text_to_search)
                
                chunks = split_by_dieu(content)
                if not chunks:
                    chunks = [content]
                    
                for idx, chunk_text in enumerate(chunks):
                    if len(chunk_text) < 10: continue
                    total_chunks += 1
                    
                    art_num_only = extract_article_number(chunk_text)
                    
                    canonical_id = None
                    if law_number and art_num_only:
                        canonical_id = f"{law_number}::{art_num_only}"
                        canonical_chunks += 1
                    else:
                        canonical_id = f"hf_{doc_id}_dieu_{idx}"
                        fallback_chunks += 1
                        
                    out_record = {
                        "canonical_id": canonical_id,
                        "law_number": law_number,
                        "article_number": f"Điều {art_num_only}" if art_num_only else "",
                        "content": chunk_text,
                        "source": "hf",
                        "original_doc_id": doc_id,
                        "chunk_index": idx,
                        "has_canonical_id": bool(law_number and art_num_only)
                    }
                    
                    # Pass through other metadata if useful
                    if "title" in doc: out_record["title"] = doc["title"]
                    if "url" in doc: out_record["url"] = doc["url"]
                        
                    fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                logging.error(f"Error processing document {total_docs}: {e}")
                
    logging.info(f"--- RECHUNK SUMMARY ---")
    logging.info(f"Total input documents: {total_docs}")
    logging.info(f"Total output chunks: {total_chunks}")
    if total_chunks > 0:
        logging.info(f"Chunks with canonical_id: {canonical_chunks} ({(canonical_chunks/total_chunks)*100:.1f}%)")
        logging.info(f"Chunks with fallback_id: {fallback_chunks} ({(fallback_chunks/total_chunks)*100:.1f}%)")
        logging.info(f"Average chunks per doc: {total_chunks/total_docs:.1f}")
        
    if total_chunks < 50000:
        logging.warning("STOP B TRIGGERED: < 50K chunks generated. Splitting failed.")
    elif (canonical_chunks/total_chunks)*100 < 40:
        logging.warning("STOP WARNING: < 40% chunks have canonical ID. Law extraction failing too often.")

if __name__ == "__main__":
    main()
