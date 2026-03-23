"""Vietnamese legal document crawler.

Crawls public legal documents from thuvienphapluat.vn and luatvietnam.vn.
Implements polite crawling with delays, retry with exponential backoff,
Unicode normalization, and structured JSON output.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class CrawledDocument:
    """A single crawled legal document.

    Attributes:
        doc_id: Unique document identifier.
        title: Document title.
        source_url: Original URL.
        law_name: Name of the law/regulation.
        article_number: Article number if applicable.
        chapter: Chapter designation.
        content: Full text content.
        effective_date: When the law takes effect.
        crawled_at: ISO timestamp of crawl time.
        word_count: Number of words in content.
    """

    doc_id: str = ""
    title: str = ""
    source_url: str = ""
    law_name: str = ""
    article_number: str = ""
    chapter: str = ""
    content: str = ""
    effective_date: str = ""
    crawled_at: str = ""
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_url": self.source_url,
            "law_name": self.law_name,
            "article_number": self.article_number,
            "chapter": self.chapter,
            "content": self.content,
            "effective_date": self.effective_date,
            "crawled_at": self.crawled_at,
            "word_count": self.word_count,
        }


class LegalCrawler:
    """Crawler for Vietnamese legal document websites.

    Supports thuvienphapluat.vn and luatvietnam.vn. Implements:
    - Polite crawling with configurable delay
    - Retry with exponential backoff
    - Unicode NFC normalization
    - URL deduplication
    - Structured JSON output
    """

    HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the legal crawler.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        crawler_config = config["crawler"]
        self.sources: list[dict[str, str]] = crawler_config["sources"]
        self.delay: float = crawler_config.get("delay_seconds", 2.0)
        self.timeout: int = crawler_config.get("timeout_seconds", 30)
        self.max_retries: int = crawler_config.get("max_retries", 3)
        self.max_docs: int = crawler_config.get("max_docs", 500)

        self.output_dir = Path(config["data"]["raw_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._visited_urls: set[str] = set()
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

        logger.info(
            "LegalCrawler initialized | sources={} | max_docs={} | delay={}s",
            len(self.sources),
            self.max_docs,
            self.delay,
        )

    def crawl(self) -> int:
        """Crawl all configured sources.

        Returns:
            Total number of documents successfully crawled.
        """
        total = 0

        for source in self.sources:
            name = source["name"]
            base_url = source["base_url"]
            category = source.get("category", "")

            logger.info("Starting crawl: {} ({})", name, base_url)

            remaining = self.max_docs - total
            if remaining <= 0:
                break

            if "thuvienphapluat" in name:
                count = self._crawl_thuvienphapluat(base_url, category, remaining)
            elif "luatvietnam" in name:
                count = self._crawl_luatvietnam(base_url, category, remaining)
            else:
                logger.warning("Unknown source: {}", name)
                count = 0

            total += count
            logger.info("Finished {} | docs_crawled={}", name, count)

        logger.info("Crawling complete | total_docs={}", total)
        return total

    def _crawl_thuvienphapluat(
        self,
        base_url: str,
        category: str,
        max_docs: int,
    ) -> int:
        """Crawl documents from thuvienphapluat.vn.

        Args:
            base_url: Base URL of the site.
            category: Category to crawl.
            max_docs: Maximum documents to crawl from this source.

        Returns:
            Number of documents crawled.
        """
        count = 0
        page = 1

        while count < max_docs:
            # Build listing page URL
            list_url = f"{base_url}/page/{page}/van-ban-phap-luat.aspx"

            html = self._fetch(list_url)
            if not html:
                break

            soup = BeautifulSoup(html, "lxml")

            # Find document links
            links = self._extract_links_thuvienphapluat(soup, base_url)

            if not links:
                logger.info("No more links found on page {}", page)
                break

            for link in links:
                if count >= max_docs:
                    break

                if link in self._visited_urls:
                    continue

                doc = self._parse_thuvienphapluat(link)
                if doc and doc.content:
                    self._save_document(doc)
                    count += 1
                    self._visited_urls.add(link)

                    if count % 10 == 0:
                        logger.info("Progress: {}/{} docs crawled", count, max_docs)

                time.sleep(self.delay)

            page += 1
            time.sleep(self.delay)

        return count

    def _crawl_luatvietnam(
        self,
        base_url: str,
        category: str,
        max_docs: int,
    ) -> int:
        """Crawl documents from luatvietnam.vn.

        Args:
            base_url: Base URL of the site.
            category: Category to crawl.
            max_docs: Maximum documents to crawl from this source.

        Returns:
            Number of documents crawled.
        """
        count = 0
        page = 1

        while count < max_docs:
            list_url = f"{base_url}/{category}/trang-{page}.html"

            html = self._fetch(list_url)
            if not html:
                break

            soup = BeautifulSoup(html, "lxml")

            links = self._extract_links_luatvietnam(soup, base_url)

            if not links:
                logger.info("No more links found on page {}", page)
                break

            for link in links:
                if count >= max_docs:
                    break

                if link in self._visited_urls:
                    continue

                doc = self._parse_luatvietnam(link)
                if doc and doc.content:
                    self._save_document(doc)
                    count += 1
                    self._visited_urls.add(link)

                    if count % 10 == 0:
                        logger.info("Progress: {}/{} docs crawled", count, max_docs)

                time.sleep(self.delay)

            page += 1
            time.sleep(self.delay)

        return count

    def _extract_links_thuvienphapluat(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> list[str]:
        """Extract document links from thuvienphapluat.vn listing page.

        Args:
            soup: Parsed HTML.
            base_url: Base URL for resolving relative links.

        Returns:
            List of absolute URLs.
        """
        links: list[str] = []
        # Look for document links in listing
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Filter for legal document pages
            if any(pattern in href for pattern in ["/van-ban/", "/vb/", "/noi-dung/"]):
                full_url = urljoin(base_url, href)
                if full_url not in self._visited_urls:
                    links.append(full_url)

        return list(set(links))[:50]  # Limit per page

    def _extract_links_luatvietnam(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> list[str]:
        """Extract document links from luatvietnam.vn listing page.

        Args:
            soup: Parsed HTML.
            base_url: Base URL for resolving relative links.

        Returns:
            List of absolute URLs.
        """
        links: list[str] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if any(pattern in href for pattern in ["/van-ban/", "/tin-phap-luat/"]):
                full_url = urljoin(base_url, href)
                if full_url not in self._visited_urls:
                    links.append(full_url)

        return list(set(links))[:50]

    def _parse_thuvienphapluat(self, url: str) -> CrawledDocument | None:
        """Parse a single document from thuvienphapluat.vn.

        Args:
            url: Document URL.

        Returns:
            CrawledDocument or None if parsing failed.
        """
        html = self._fetch(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")

        title = ""
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            title = self._normalize(title_tag.get_text(strip=True))

        # Extract main content
        content = ""
        content_div = (
            soup.find("div", class_="content1") or
            soup.find("div", class_="noidung") or
            soup.find("div", {"id": "toanvancontent"}) or
            soup.find("article")
        )
        if content_div:
            content = self._normalize(content_div.get_text(separator="\n", strip=True))
        else:
            # Fallback: get all paragraph text
            paragraphs = soup.find_all("p")
            content = self._normalize(
                "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            )

        if not content or len(content) < 50:
            return None

        # Extract metadata
        law_name = self._extract_law_name(title, content)
        article_number = self._extract_article_number(title)
        chapter = self._extract_chapter(content)
        effective_date = self._extract_date(content)

        doc_id = self._generate_doc_id(url, title)

        return CrawledDocument(
            doc_id=doc_id,
            title=title,
            source_url=url,
            law_name=law_name,
            article_number=article_number,
            chapter=chapter,
            content=content,
            effective_date=effective_date,
            crawled_at=datetime.now(timezone.utc).isoformat(),
            word_count=len(content.split()),
        )

    def _parse_luatvietnam(self, url: str) -> CrawledDocument | None:
        """Parse a single document from luatvietnam.vn.

        Args:
            url: Document URL.

        Returns:
            CrawledDocument or None if parsing failed.
        """
        html = self._fetch(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")

        title = ""
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            title = self._normalize(title_tag.get_text(strip=True))

        content = ""
        content_div = (
            soup.find("div", class_="article-body") or
            soup.find("div", class_="content-detail") or
            soup.find("div", class_="box-content") or
            soup.find("article")
        )
        if content_div:
            content = self._normalize(content_div.get_text(separator="\n", strip=True))
        else:
            paragraphs = soup.find_all("p")
            content = self._normalize(
                "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            )

        if not content or len(content) < 50:
            return None

        law_name = self._extract_law_name(title, content)
        article_number = self._extract_article_number(title)
        chapter = self._extract_chapter(content)
        effective_date = self._extract_date(content)
        doc_id = self._generate_doc_id(url, title)

        return CrawledDocument(
            doc_id=doc_id,
            title=title,
            source_url=url,
            law_name=law_name,
            article_number=article_number,
            chapter=chapter,
            content=content,
            effective_date=effective_date,
            crawled_at=datetime.now(timezone.utc).isoformat(),
            word_count=len(content.split()),
        )

    def _fetch(self, url: str) -> str | None:
        """Fetch a URL with retry and exponential backoff.

        Args:
            url: URL to fetch.

        Returns:
            HTML content string or None on failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, timeout=self.timeout)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or "utf-8"
                return response.text
            except requests.RequestException as exc:
                logger.warning(
                    "Fetch attempt {}/{} failed for {}: {}",
                    attempt,
                    self.max_retries,
                    url[:80],
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("Failed to fetch after {} attempts: {}", self.max_retries, url[:80])
        return None

    def _save_document(self, doc: CrawledDocument) -> None:
        """Save a crawled document as JSON.

        Args:
            doc: Document to save.
        """
        filename = f"{doc.doc_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize Vietnamese Unicode text to NFC form.

        Args:
            text: Input text.

        Returns:
            NFC-normalized text with cleaned whitespace.
        """
        text = unicodedata.normalize("NFC", text)
        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _generate_doc_id(url: str, title: str) -> str:
        """Generate a unique document ID from URL and title.

        Args:
            url: Document URL.
            title: Document title.

        Returns:
            Short hash-based ID.
        """
        source = f"{url}|{title}"
        hash_val = hashlib.md5(source.encode()).hexdigest()[:12]
        # Create readable prefix from title
        prefix = re.sub(r"[^a-zA-Z0-9_]", "_", title[:30]).strip("_").lower()
        return f"{prefix}_{hash_val}" if prefix else hash_val

    @staticmethod
    def _extract_law_name(title: str, content: str) -> str:
        """Extract law name from title or content.

        Args:
            title: Document title.
            content: Document content.

        Returns:
            Law name string.
        """
        patterns = [
            re.compile(r"(Bộ\s+luật\s+[^\n,.]{3,50})", re.IGNORECASE),
            re.compile(r"(Luật\s+[^\n,.]{3,50})", re.IGNORECASE),
            re.compile(r"(Nghị\s+định\s+số?\s*\d+[^\n,.]{0,40})", re.IGNORECASE),
            re.compile(r"(Thông\s+tư\s+số?\s*\d+[^\n,.]{0,40})", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(title)
            if match:
                return match.group(1).strip()
        for pattern in patterns:
            match = pattern.search(content[:500])
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_article_number(title: str) -> str:
        """Extract article number from title.

        Args:
            title: Document title.

        Returns:
            Article number string.
        """
        match = re.search(r"Điều\s+(\d+[a-zđ]?)", title, re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_chapter(content: str) -> str:
        """Extract chapter designation from content.

        Args:
            content: Document content.

        Returns:
            Chapter designation string.
        """
        match = re.search(r"(Chương\s+[IVXLCDM]+|Chương\s+\d+)", content, re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_date(content: str) -> str:
        """Extract effective date from content.

        Args:
            content: Document content.

        Returns:
            Date string or empty.
        """
        patterns = [
            re.compile(r"có\s+hiệu\s+lực.*?(\d{1,2}[/-]\d{1,2}[/-]\d{4})", re.IGNORECASE),
            re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(content)
            if match:
                return match.group(0).strip()[:30]
        return ""
