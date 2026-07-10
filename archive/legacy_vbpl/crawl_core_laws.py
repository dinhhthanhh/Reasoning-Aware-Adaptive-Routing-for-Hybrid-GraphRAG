import sys
import json
import uuid
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawlers.vbpl_crawler import crawl_vbpl

logging.basicConfig(level=logging.INFO)

CORE_LAWS = [
    "Luật Hôn nhân và gia đình",
    "Luật Đất đai",
    "Bộ luật Dân sự",
    "Bộ luật Hình sự",
    "Luật Doanh nghiệp",
    "Luật Thương mại"
]

def main():
    output_dir = Path("data/raw/core_laws")
    processed_file = Path("data/processed/core_laws_processed.jsonl")
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    
    for law in CORE_LAWS:
        logging.info(f"Searching and downloading: {law}")
        docs = crawl_vbpl(output_dir, doc_type="luat", max_pages=1, fetch_content=True, keyword=law)
        
        # Take the best match
        if docs:
            # Try to find the exact match by title or just use the first one
            best_doc = docs[0]
            for doc in docs:
                if law.lower() in doc.get("title", "").lower():
                    best_doc = doc
                    break
                    
            logging.info(f"Found: {best_doc.get('title')}")
            
            content = best_doc.get("content", "")
            paragraphs = content.split('\n')
            
            current_chunk = ""
            for p in paragraphs:
                p = p.strip()
                if not p: continue
                if len(current_chunk) + len(p) > 1200 and current_chunk:
                    all_chunks.append({
                        "doc_id": str(uuid.uuid4()),
                        "content_markdown": current_chunk,
                        "title": best_doc.get("title", law),
                        "type": best_doc.get("loai_vb", "Luật"),
                        "source": best_doc.get("url", "VBPL"),
                        "authority": best_doc.get("co_quan_ban_hanh", "Quốc hội")
                    })
                    current_chunk = p
                else:
                    current_chunk += "\n" + p if current_chunk else p
                    
            if current_chunk:
                all_chunks.append({
                    "doc_id": str(uuid.uuid4()),
                    "content_markdown": current_chunk,
                    "title": best_doc.get("title", law),
                    "type": best_doc.get("loai_vb", "Luật"),
                    "source": best_doc.get("url", "VBPL"),
                    "authority": best_doc.get("co_quan_ban_hanh", "Quốc hội")
                })
        else:
            logging.warning(f"Could not find {law}")
            
    with open(processed_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    logging.info(f"Successfully wrote {len(all_chunks)} chunks to {processed_file}")

if __name__ == "__main__":
    main()
