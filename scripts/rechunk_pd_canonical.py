import json
import re
from pathlib import Path
import logging
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase0.legal_audit_utils import extract_doc_codes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def extract_article_number(text):
    """Extracts just the number part (e.g. '8', '1a') from the text."""
    if not text: return None
    match = re.search(r'Điều\s+(\d+[A-Za-zĐđ]?)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def main():
    root = Path(__file__).resolve().parent.parent
    pd_in = root / "data" / "processed" / "phapdien_processed.jsonl"
    pd_out = root / "data" / "processed" / "pd_rechunked.jsonl"
    
    if not pd_in.exists():
        logging.error(f"Input file not found: {pd_in}")
        return
        
    pd_out.parent.mkdir(parents=True, exist_ok=True)
    
    total_docs = 0
    canonical_chunks = 0
    fallback_chunks = 0
    
    with open(pd_in, "r", encoding="utf-8") as fin, open(pd_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Rechunking PD"):
            if not line.strip(): continue
            total_docs += 1
            try:
                doc = json.loads(line)
                content = doc.get("content_markdown", "")
                doc_id = str(doc.get("doc_id", f"pd_unknown_{total_docs}"))
                source = doc.get("source", "")
                
                raw_meta = doc.get("raw_metadata", {})
                ghi_chu = raw_meta.get("ghi_chu", "")
                
                # Extract law number
                law_number_list = extract_doc_codes(source)
                if not law_number_list:
                    law_number_list = extract_doc_codes(ghi_chu)
                law_number = law_number_list[0] if law_number_list else ""
                
                # Extract article number
                art_num_only = extract_article_number(ghi_chu)
                if not art_num_only:
                    art_num_only = extract_article_number(content[:1000]) # Fallback to beginning of content
                    
                canonical_id = None
                if law_number and art_num_only:
                    # Fix Cyrillic 'С' in law_number
                    law_number = law_number.replace('\u0421', 'C').replace('\u0441', 'c').upper()
                    canonical_id = f"{law_number}::{art_num_only}"
                    canonical_chunks += 1
                else:
                    canonical_id = doc_id
                    fallback_chunks += 1
                    
                out_record = {
                    "canonical_id": canonical_id,
                    "law_number": law_number,
                    "article_number": f"Điều {art_num_only}" if art_num_only else "",
                    "content": content,
                    "source": "phapdien",
                    "original_doc_id": doc_id,
                    "chunk_index": 0,
                    "has_canonical_id": bool(law_number and art_num_only)
                }
                
                # Pass through metadata
                for k in ["title", "url", "type", "theme", "topic"]:
                    if k in doc: out_record[k] = doc[k]
                    
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                
            except Exception as e:
                logging.error(f"Error processing document {total_docs}: {e}")
                
    logging.info(f"--- RECHUNK SUMMARY ---")
    logging.info(f"Total output chunks: {total_docs}")
    if total_docs > 0:
        logging.info(f"Chunks with canonical_id: {canonical_chunks} ({(canonical_chunks/total_docs)*100:.1f}%)")
        logging.info(f"Chunks with fallback_id: {fallback_chunks} ({(fallback_chunks/total_docs)*100:.1f}%)")

if __name__ == "__main__":
    main()
