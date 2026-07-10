import json
import logging
import re
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase0.legal_audit_utils import extract_doc_codes, normalize_article_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Regex to split content by "Điều <number>."
# Matches optional markdown headers (###), "Điều", space, number (with optional letters like 1a, 1đ), optional punctuation.
ARTICLE_SPLIT_RE = re.compile(r"(?im)^(?:\s*#+\s*)?Điều\s+(\d+[a-zđ]?)(?:[\.:]\s*|\s*\n\s*)")

def rechunk_document(record: dict) -> list[dict]:
    doc_id = record.get("doc_id")
    if not doc_id:
        return []
        
    title = str(record.get("title", ""))
    metadata = str(record.get("raw_metadata", ""))
    content = str(record.get("content_markdown", "")).strip()
    
    if not content:
        return []

    # Extract law number
    blob = f"{title} {metadata} {content[:1000]}"
    codes = extract_doc_codes(blob)
    law_number = codes[0] if codes else str(doc_id)
    
    chunks = []
    
    # Split content
    parts = ARTICLE_SPLIT_RE.split(content)
    
    base_record = {
        "title": title,
        "type": record.get("type", "Unknown"),
        "source": record.get("source", "Unknown"),
        "authority": record.get("authority", "Unknown"),
        "raw_metadata": record.get("raw_metadata", {})
    }
    
    # If no "Điều" found, parts length is 1 (the whole text)
    if len(parts) == 1:
        chunk_rec = base_record.copy()
        chunk_rec["doc_id"] = f"{law_number}::TOAN_VAN"
        chunk_rec["content_markdown"] = parts[0].strip()
        chunks.append(chunk_rec)
        return chunks
        
    # parts[0] is preamble (text before the first "Điều")
    preamble = parts[0].strip()
    if len(preamble) > 50:
        chunk_rec = base_record.copy()
        chunk_rec["doc_id"] = f"{law_number}::PREAMBLE"
        chunk_rec["content_markdown"] = preamble
        chunks.append(chunk_rec)
        
    # parts[1] is article 1 ID, parts[2] is article 1 text, etc.
    for i in range(1, len(parts), 2):
        art_id_raw = parts[i]
        art_text = parts[i+1] if i+1 < len(parts) else ""
        
        art_num = normalize_article_id(art_id_raw)
        if not art_num:
            art_num = art_id_raw.strip().lower()
            
        full_text = f"Điều {art_id_raw}. {art_text.strip()}"
        
        chunk_rec = base_record.copy()
        chunk_rec["doc_id"] = f"{law_number}::{art_num}"
        chunk_rec["content_markdown"] = full_text
        chunks.append(chunk_rec)
        
    return chunks

def process_file(input_path: Path, output_path: Path):
    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return
        
    logger.info(f"Rechunking {input_path.name} -> {output_path.name}")
    
    total_docs = 0
    total_chunks = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            try:
                record = json.loads(line)
                total_docs += 1
                
                chunks = rechunk_document(record)
                for c in chunks:
                    f_out.write(json.dumps(c, ensure_ascii=False) + "\n")
                    total_chunks += 1
                    
                if total_docs % 5000 == 0:
                    logger.info(f"Processed {total_docs} docs -> {total_chunks} chunks")
                    
            except Exception as e:
                logger.error(f"Error processing record: {e}")
                
    logger.info(f"Done. {total_docs} docs -> {total_chunks} chunks")

def main():
    data_dir = Path("data/processed")
    hf_input = data_dir / "hf_processed.jsonl"
    hf_output = data_dir / "hf_rechunked.jsonl"
    
    process_file(hf_input, hf_output)
    
    core_input = data_dir / "core_laws_processed.jsonl"
    core_output = data_dir / "core_laws_rechunked.jsonl"
    
    process_file(core_input, core_output)

if __name__ == "__main__":
    main()
