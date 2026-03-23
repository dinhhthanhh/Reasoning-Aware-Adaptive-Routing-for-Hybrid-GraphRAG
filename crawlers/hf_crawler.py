"""Crawler for legal datasets from Hugging Face Hub."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger

from crawlers.legal_crawler import CrawledDocument


class HFCrawler:
    """Crawler for Hugging Face legal datasets.
    
    Specifically tailored for namphan1999/data-luat but generalizable.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize HF crawler.
        
        Args:
            config: Full config dict.
        """
        self.output_dir = Path(config["data"]["raw_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_docs = config["crawler"].get("max_docs", 500)

    def crawl(self, dataset_name: str = "namphan1999/data-luat") -> int:
        """Download and process documents from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset on HF Hub.
            
        Returns:
            Number of documents processed.
        """
        logger.info("Loading dataset from HF: {}", dataset_name)
        
        try:
            # Load dataset (usually has 'train' split)
            ds = load_dataset(dataset_name, split="train")
            
            count = 0
            for i, row in enumerate(ds):
                if count >= self.max_docs:
                    break
                
                # Extract fields based on namphan1999/data-luat structure
                question = row.get("question", "")
                answer = row.get("answer", "")
                terms = row.get("terms", "")
                
                if not answer:
                    continue
                
                # Combine question and answer for content
                content = f"Hỏi: {question}\n\nTrả lời: {answer}"
                
                # Create doc_id
                doc_id = f"hf_{i}_{dataset_name.replace('/', '_')}"
                
                doc = CrawledDocument(
                    doc_id=doc_id,
                    title=f"Q&A: {question[:100]}...",
                    source_url=f"https://huggingface.co/datasets/{dataset_name}",
                    law_name=terms or "Pháp luật Việt Nam",
                    content=content,
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    word_count=len(content.split()),
                )
                
                self._save_document(doc)
                count += 1
                
                if count % 50 == 0:
                    logger.info("HF Progress: {} docs processed", count)
                    
            return count
            
        except Exception as exc:
            logger.error("Failed to crawl HF dataset {}: {}", dataset_name, exc)
            return 0

    def _save_document(self, doc: CrawledDocument) -> None:
        """Save a document as JSON."""
        filename = f"{doc.doc_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
