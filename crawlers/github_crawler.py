"""Crawler for legal datasets from GitHub (e.g., VNLegalText)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from crawlers.legal_crawler import CrawledDocument


class GitHubCrawler:
    """Crawler for GitHub-based legal datasets."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize GitHub crawler.
        
        Args:
            config: Full config dict.
        """
        self.output_dir = Path(config["data"]["raw_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_docs = config["crawler"].get("max_docs", 500)
        self.temp_dir = Path("temp_github_crawl")

    def _rmtree(self, path: Path) -> None:
        """Robustly remove a directory, handling read-only files on Windows."""
        def on_error(func, path, exc_info):
            # Change permissions and retry
            os.chmod(path, stat.S_IWRITE)
            func(path)

        if path.exists():
            shutil.rmtree(path, onerror=on_error)

    def crawl(self, repo_url: str) -> int:
        """Clone repository and process files.
        
        Args:
            repo_url: GitHub repository URL.
            
        Returns:
            Number of documents processed.
        """
        logger.info("Cloning repo from GitHub: {}", repo_url)
        
        self._rmtree(self.temp_dir)
            
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(self.temp_dir)],
                check=True,
                capture_output=True
            )
            
            # Special handling for VNLegalText if identified
            if "VNLegalText" in repo_url:
                return self._process_vnlegaltext()
            
            # Generic processing for other subdirectories/files if needed
            return self._process_generic()
            
        except Exception as exc:
            logger.error("Failed to crawl GitHub repo {}: {}", repo_url, exc)
            return 0
        finally:
            self._rmtree(self.temp_dir)

    def _process_vnlegaltext(self) -> int:
        """Specific processing for VNLegalText repository."""
        zip_path = self.temp_dir / "data" / "xml_data.zip"
        if not zip_path.exists():
            logger.error("VNLegalText xml_data.zip not found")
            return 0
            
        extract_dir = self.temp_dir / "extracted_xml"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Unzip using Python's shutil (more portable than system command)
        shutil.unpack_archive(str(zip_path), str(extract_dir), "zip")
        
        dataset_dir = extract_dir / "dataset"
        if not dataset_dir.exists():
            # Sometimes unzipping creates an extra nesting
            nested = list(extract_dir.glob("**/dataset"))
            if nested:
                dataset_dir = nested[0]
            else:
                logger.error("VNLegalText dataset directory not found after unzip")
                return 0
        
        count = 0
        for xml_file in dataset_dir.glob("*.xml"):
            if count >= self.max_docs:
                break
                
            try:
                content = xml_file.read_text(encoding="utf-8")
                if not content:
                    continue
                
                # The files seem to have a title on first line and body on rest
                lines = content.strip().split("\n")
                title = lines[0] if lines else "Bản văn pháp luật"
                body = "\n".join(lines[1:]) if len(lines) > 1 else content
                
                doc_id = f"github_vnlegal_{xml_file.stem}"
                
                doc = CrawledDocument(
                    doc_id=doc_id,
                    title=title[:200],
                    source_url="https://github.com/mlalab/VNLegalText",
                    law_name=title,
                    content=body,
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    word_count=len(body.split()),
                )
                
                self._save_document(doc)
                count += 1
                
                if count % 100 == 0:
                    logger.info("GitHub Progress: {} docs processed", count)
                    
            except Exception as exc:
                logger.warning("Failed to process file {}: {}", xml_file.name, exc)
                
        return count

    def _process_generic(self) -> int:
        """Generic processing for text files in a repo."""
        count = 0
        for ext in ["*.txt", "*.md", "*.json"]:
            for file in self.temp_dir.glob(f"**/{ext}"):
                if count >= self.max_docs:
                    break
                # Basic processing: read file, treat as document
                # ... (omping for brevity as user specified these two)
        return count

    def _save_document(self, doc: CrawledDocument) -> None:
        """Save a document as JSON."""
        filename = f"{doc.doc_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
