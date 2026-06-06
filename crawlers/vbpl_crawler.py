"""
Crawler for VBPL (Văn bản pháp luật): https://vbpl.vn/pages/portal.aspx
Uses the public search API to fetch legal documents.
"""

import requests
import json
import time
import logging
import re
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlencode

logger = logging.getLogger(__name__)

BASE_URL = "https://vbpl.vn"
SEARCH_URL = "https://vbpl.vn/TW/Pages/vbpq-timkiem.aspx"
DOCUMENT_API = "https://vbpl.vn/TW/Pages/vbpq-van-ban-goc.aspx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Referer": BASE_URL,
}

# Document types (loại văn bản)
DOC_TYPES = {
    "all": "",
    "luat": "1",          # Luật
    "nghi_dinh": "2",     # Nghị định
    "thong_tu": "3",      # Thông tư
    "quyet_dinh": "4",    # Quyết định
    "nghi_quyet": "5",    # Nghị quyết
    "phap_lenh": "6",     # Pháp lệnh
    "chi_thi": "7",       # Chỉ thị
}


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def parse_search_results(html: str) -> list[dict]:
    """Parse search result page to extract document links and metadata."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # VBPL lists documents in table rows or list items
    rows = soup.select("table.styled-table tr, .document-list li, .vbpq-item, ul.list li")
    if not rows:
        # Try alternative selectors
        rows = soup.select("tr[class*='vb'], .vb-item, article.doc-item")

    for row in rows:
        try:
            link_el = row.select_one("a[href*='ItemID'], a[href*='vbpq']")
            if not link_el:
                continue

            href = link_el.get("href", "")
            title = link_el.get_text(strip=True)

            # Extract metadata columns
            cols = row.find_all("td")
            doc = {
                "title": title,
                "url": urljoin(BASE_URL, href),
                "so_hieu": cols[0].get_text(strip=True) if len(cols) > 0 else "",
                "loai_vb": cols[1].get_text(strip=True) if len(cols) > 1 else "",
                "co_quan_ban_hanh": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                "ngay_ban_hanh": cols[3].get_text(strip=True) if len(cols) > 3 else "",
                "tinh_trang": cols[4].get_text(strip=True) if len(cols) > 4 else "",
            }
            results.append(doc)
        except Exception as e:
            logger.debug(f"Row parse error: {e}")
            continue

    return results


def get_total_pages(html: str) -> int:
    """Extract total number of search result pages."""
    soup = BeautifulSoup(html, "html.parser")
    pager = soup.select_one(".pager, .pagination, [class*='pager']")
    if not pager:
        return 1
    page_links = pager.select("a")
    page_nums = []
    for a in page_links:
        text = a.get_text(strip=True)
        if text.isdigit():
            page_nums.append(int(text))
    return max(page_nums) if page_nums else 1


def fetch_document_content(session: requests.Session, url: str) -> dict:
    """Fetch full content of a single legal document."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=90)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract main content
            content_el = (
                soup.select_one(".content-detail, #toanvan, .vbpq-content, article")
                or soup.select_one("div[class*='content']")
            )

            # Extract metadata
            meta = {}
            for row in soup.select(".vb-info tr, .info-table tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).rstrip(":")
                    val = cells[1].get_text(strip=True)
                    meta[key] = val

            return {
                "url": url,
                "content": content_el.get_text(separator="\n", strip=True) if content_el else "",
                "metadata": meta,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch document {url} after {max_retries} attempts: {e}")
                return {"url": url, "content": "", "error": str(e)}
            logger.warning(f"Retry {attempt + 1} for {url}...")
            time.sleep(2 * (attempt + 1))


def crawl_vbpl(
    output_dir: Path,
    doc_type: str = "all",
    max_pages: Optional[int] = None,
    fetch_content: bool = False,
    delay: float = 1.5,
) -> list[dict]:
    """
    Crawl legal documents from VBPL.

    Args:
        output_dir: Directory to save results.
        doc_type: One of DOC_TYPES keys (default 'all').
        max_pages: Max search result pages to crawl.
        fetch_content: Whether to fetch full document text.
        delay: Seconds between requests.

    Returns:
        List of document records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session = get_session()
    all_docs = []
    page = 1

    while True:
        logger.info(f"VBPL: fetching search page {page}...")
        params = {
            "Page": page,
            "DanhMucID": DOC_TYPES.get(doc_type, ""),
        }
        
        # Retry logic for search page
        html = None
        for attempt in range(3):
            try:
                resp = session.get(SEARCH_URL, params=params, timeout=90)
                resp.raise_for_status()
                html = resp.text
                break
            except Exception as e:
                logger.warning(f"Search page {page} attempt {attempt+1} failed: {e}")
                time.sleep(5)
        
        if not html:
            logger.error(f"Failed to fetch search page {page} after 3 attempts. Stopping.")
            break

        docs = parse_search_results(html)
        if not docs:
            logger.info(f"No documents found on page {page}. Stopping.")
            break

        total_pages = get_total_pages(html)
        logger.info(f"  Found {len(docs)} docs on page {page}/{total_pages}")

        if fetch_content:
            for doc in docs:
                detail = fetch_document_content(session, doc["url"])
                doc.update(detail)
                time.sleep(delay)

        all_docs.extend(docs)

        # Save per-page checkpoint
        ckpt = output_dir / f"vbpl_page_{page:04d}.json"
        ckpt.write_text(json.dumps(docs, ensure_ascii=False, indent=2))

        if max_pages and page >= max_pages:
            break
        if page >= total_pages:
            break

        page += 1
        time.sleep(delay)

    out_file = output_dir / "vbpl_all.json"
    out_file.write_text(json.dumps(all_docs, ensure_ascii=False, indent=2))
    logger.info(f"VBPL total: {len(all_docs)} documents → {out_file}")
    return all_docs
