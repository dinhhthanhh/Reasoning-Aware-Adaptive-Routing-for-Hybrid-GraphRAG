import json
from pathlib import Path
from datasets import load_dataset
from loguru import logger
import hashlib
import re

def generate_doc_id(text, title):
    source = f"{text}|{title}"
    hash_val = hashlib.md5(source.encode()).hexdigest()[:12]
    prefix = re.sub(r"[^a-zA-Z0-9_]", "_", title[:30]).strip("_").lower()
    return f"{prefix}_{hash_val}" if prefix else hash_val

def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading dataset from Hugging Face: namphan1999/data-luat")
    dataset = load_dataset("namphan1999/data-luat", split="train")
    
    count = 0
    for i, item in enumerate(dataset):
        # We'll use 'terms' as the primary content and 'question' as part of the title
        content = item.get("terms", "").strip()
        answer = item.get("answer", "").strip()
        question = item.get("question", "").strip()
        
        if not content:
            continue
            
        full_text = f"Nội dung luật: {content}\n\nCâu hỏi liên quan: {question}\nTrả lời: {answer}"
        title = question[:100] if question else f"Legal Document {i}"
        
        doc_id = generate_doc_id(content, title)
        
        doc = {
            "doc_id": doc_id,
            "title": title,
            "source_url": "hf://namphan1999/data-luat",
            "law_name": "Tài liệu luật tổng hợp",
            "article_number": "",
            "chapter": "",
            "content": full_text,
            "effective_date": "",
            "crawled_at": "2026-03-10T00:00:00Z",
            "word_count": len(full_text.split())
        }
        
        with open(output_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            
        count += 1
        if count >= 100: # Let's start with 100 documents for testing
            break
            
    logger.info(f"Processed {count} documents from Hugging Face dataset")

if __name__ == "__main__":
    main()
