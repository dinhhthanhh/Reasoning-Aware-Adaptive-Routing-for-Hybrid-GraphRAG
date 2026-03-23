"""
vanban.chinhphu.vn Crawler — GraphRAG Dataset Builder
=======================================================
Source: https://vanban.chinhphu.vn/he-thong-van-ban
URL văn bản: https://vanban.chinhphu.vn/?pageid=27160&docid=XXX&classid=1

Cài đặt:
    pip install requests beautifulsoup4 lxml pdfplumber tqdm
"""

import re
import json
import time
import random
import logging
import argparse
from io import BytesIO
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# ─────────────────────────────────────────────────────────────────
# CONFIG — verified từ vanban.chinhphu.vn
# ─────────────────────────────────────────────────────────────────

BASE            = "https://chinhphu.vn"
URL_LIST        = f"{BASE}/he-thong-van-ban"
URL_DOC         = f"{BASE}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://chinhphu.vn",
}

# typegroupid trên vanban.chinhphu.vn (từ URL params quan sát)
# classid=1 = văn bản quy phạm pháp luật
DOC_TYPES = {
    "luat":       {"typegroupid": "1", "classid": "1", "label": "Luật / Bộ luật"},
    "nghi_dinh":  {"typegroupid": "2", "classid": "1", "label": "Nghị định"},
    "quyet_dinh": {"typegroupid": "3", "classid": "1", "label": "Quyết định"},
    "thong_tu":   {"typegroupid": "4", "classid": "1", "label": "Thông tư"},
    "chi_thi":    {"typegroupid": "5", "classid": "1", "label": "Chỉ thị"},
    "nghi_quyet": {"typegroupid": "6", "classid": "1", "label": "Nghị quyết"},
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "crawl.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ─────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────

@dataclass
class Document:
    node_id: str
    doc_id: str
    so_hieu: str
    loai_van_ban: str
    trich_yeu: str
    co_quan_ban_hanh: str
    ngay_ban_hanh: str
    ngay_hieu_luc: str
    tinh_trang: str
    linh_vuc: str
    nguoi_ky: str
    url: str
    url_pdf: str
    noi_dung: str
    chuong_muc: list
    crawled_at: str

    def to_node(self) -> dict:
        return {
            "id": self.node_id,
            "type": "VanBan",
            "properties": {
                "doc_id":           self.doc_id,
                "so_hieu":          self.so_hieu,
                "loai_van_ban":     self.loai_van_ban,
                "trich_yeu":        self.trich_yeu,
                "co_quan_ban_hanh": self.co_quan_ban_hanh,
                "ngay_ban_hanh":    self.ngay_ban_hanh,
                "ngay_hieu_luc":    self.ngay_hieu_luc,
                "tinh_trang":       self.tinh_trang,
                "linh_vuc":         self.linh_vuc,
                "nguoi_ky":         self.nguoi_ky,
                "url":              self.url,
                "url_pdf":          self.url_pdf,
                "noi_dung":         self.noi_dung,
                "chuong_muc":       self.chuong_muc,
                "crawled_at":       self.crawled_at,
            },
        }


@dataclass
class Article:
    node_id: str
    van_ban_id: str
    so_dieu: str
    ten_dieu: str
    noi_dung: str
    chuong: str
    muc: str

    def to_node(self) -> dict:
        return {
            "id": self.node_id,
            "type": "Dieu",
            "properties": {
                "so_dieu":    self.so_dieu,
                "ten_dieu":   self.ten_dieu,
                "noi_dung":   self.noi_dung,
                "chuong":     self.chuong,
                "muc":        self.muc,
                "van_ban_id": self.van_ban_id,
            },
        }


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: str
    mo_ta: str = ""

    def to_edge(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type":   self.relation_type,
            "properties": {"mo_ta": self.mo_ta},
        }


# ─────────────────────────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────────────────────────

def safe_get(url: str, params: dict = None, retries: int = 3, timeout: int = 20) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r
        except requests.RequestException as e:
            wait = 2 ** attempt + random.uniform(0.5, 1.5)
            if attempt < retries - 1:
                log.warning(f"  Retry {attempt+1}: {e}")
                time.sleep(wait)
    log.error(f"  FAILED: {url}")
    return None


def polite_sleep(lo=1.0, hi=2.5):
    time.sleep(random.uniform(lo, hi))


# ─────────────────────────────────────────────────────────────────
# STEP 1: INDEX CRAWL — lấy danh sách docid
# ─────────────────────────────────────────────────────────────────

def get_viewstate(typegroupid: str, classid: str) -> dict:
    """
    Lấy __VIEWSTATE và các hidden fields từ page 1.
    chinhphu.vn dùng ASP.NET WebForms __doPostBack để phân trang.
    """
    params = {"classid": classid, "typegroupid": typegroupid, "mode": "1"}
    r = safe_get(URL_LIST, params=params)
    if not r:
        return {}
    soup = BeautifulSoup(r.text, "lxml")
    fields = {}
    for inp in soup.find_all("input", type="hidden"):
        name = inp.get("name", "")
        val  = inp.get("value", "")
        if name:
            fields[name] = val
    # Tìm tên GridView control (dùng cho __doPostBack)
    # Dạng: __doPostBack('ctrl_191017_163$grvDocument','Page$2')
    script_text = soup.get_text()
    m = re.search(r"__doPostBack\('([^']+\$grvDocument)'", soup.decode())
    if m:
        fields["__GRID_CONTROL"] = m.group(1)
    else:
        # fallback: tìm trong href
        for a in soup.find_all("a", href=re.compile(r"__doPostBack")):
            href = a.get("href", "")
            m2 = re.search(r"__doPostBack\('([^']+)'", href)
            if m2:
                fields["__GRID_CONTROL"] = m2.group(1)
                break
    log.info(f"    ViewState: {len(fields)} hidden fields, grid={fields.get('__GRID_CONTROL','?')}")
    return fields


def parse_index_html(soup: BeautifulSoup) -> tuple:
    """Parse danh sách văn bản và thông tin phân trang từ HTML."""
    results = []
    seen = set()

    # Lấy văn bản từ table rows
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # Tìm link văn bản (có docid hoặc dẫn đến trang chi tiết)
        link = None
        for td in cells:
            a = td.find("a", href=re.compile(r"docid=\d+|pageid=\d+", re.I))
            if a:
                link = a
                break

        if not link:
            continue

        href = link.get("href", "")
        m = re.search(r"docid=(\d+)", href, re.I)
        if not m:
            # Thử lấy từ onclick hoặc data attribute
            continue
        doc_id = m.group(1)
        if doc_id in seen:
            continue
        seen.add(doc_id)

        # Lấy số hiệu (thường ở cột đầu)
        so_hieu = cells[0].get_text(strip=True) if cells else ""
        # Lấy ngày (cột 2 nếu có)
        ngay = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        # Tên văn bản (text của link hoặc cột 3)
        ten = link.get_text(strip=True)
        if not ten and len(cells) > 2:
            ten = cells[2].get_text(strip=True)

        # PDF đính kèm trực tiếp từ danh sách
        pdf_url = ""
        for td in cells:
            pdf_a = td.find("a", href=re.compile(r"\.pdf", re.I))
            if pdf_a:
                pdf_href = pdf_a.get("href", "")
                pdf_url = pdf_href if pdf_href.startswith("http") else BASE + pdf_href
                break

        results.append({
            "doc_id":    doc_id,
            "so_hieu":   so_hieu,
            "ngay":      ngay,
            "trich_yeu": ten,
            "url":       BASE + href if href.startswith("/") else href,
            "url_pdf":   pdf_url,
        })

    # Tổng số trang từ "X - Y | Z" display
    total_pages = 1
    page_info = soup.select_one(".document-page-info, [id*='document_page']")
    if page_info:
        # "201 - 250 | 500" → total=500, per_page=50 → 10 pages
        m = re.search(r"(\d+)\s*-\s*(\d+)\s*[|]\s*(\d+)", page_info.get_text())
        if m:
            per_page = int(m.group(2)) - int(m.group(1)) + 1
            total = int(m.group(3))
            total_pages = (total + per_page - 1) // per_page
            log.info(f"    Tổng: {total} văn bản → {total_pages} trang")

    return results, total_pages


def get_index_page(typegroupid: str, classid: str, page: int = 1,
                   viewstate_cache: dict = None) -> tuple:
    """
    Crawl trang danh sách văn bản.
    Page 1: GET request bình thường.
    Page 2+: POST with __doPostBack (ASP.NET WebForms).
    """
    params = {"classid": classid, "typegroupid": typegroupid, "mode": "1"}

    if page == 1:
        r = safe_get(URL_LIST, params=params)
        if not r:
            return [], 1
        soup = BeautifulSoup(r.text, "lxml")

        # Cache viewstate cho các trang tiếp
        if viewstate_cache is not None:
            for inp in soup.find_all("input", type="hidden"):
                name = inp.get("name", "")
                if name:
                    viewstate_cache[name] = inp.get("value", "")
            # Tìm tên grid control
            for a in soup.find_all("a", href=re.compile(r"__doPostBack")):
                m = re.search(r"__doPostBack\('([^']+)'", a.get("href", ""))
                if m and "grv" in m.group(1).lower():
                    viewstate_cache["__GRID_CONTROL"] = m.group(1)
                    break

        results, total_pages = parse_index_html(soup)
        log.info(f"    page=1: {len(results)} văn bản | total_pages={total_pages}")
        return results, total_pages

    else:
        # POST with __doPostBack
        if not viewstate_cache:
            log.warning("    Không có viewstate cache, bỏ qua page > 1")
            return [], 1

        grid_ctrl = viewstate_cache.get("__GRID_CONTROL", "")
        if not grid_ctrl:
            log.warning("    Không tìm được grid control name")
            return [], 1

        post_data = dict(viewstate_cache)
        post_data.pop("__GRID_CONTROL", None)
        post_data["__EVENTTARGET"]   = grid_ctrl
        post_data["__EVENTARGUMENT"] = f"Page${page}"

        url = f"{URL_LIST}?classid={classid}&typegroupid={typegroupid}&mode=1"
        try:
            r = SESSION.post(url, data=post_data, timeout=20)
            r.raise_for_status()
            r.encoding = "utf-8"
        except requests.RequestException as e:
            log.error(f"    POST failed page {page}: {e}")
            return [], 1

        soup = BeautifulSoup(r.text, "lxml")

        # Cập nhật viewstate mới từ response
        for inp in soup.find_all("input", type="hidden"):
            name = inp.get("name", "")
            if name and name != "__GRID_CONTROL":
                viewstate_cache[name] = inp.get("value", "")

        results, total_pages = parse_index_html(soup)
        log.info(f"    page={page}: {len(results)} văn bản")
        return results, total_pages


def crawl_index_all(typegroupid: str, classid: str, max_pages: int = 999) -> list:
    all_items = []
    viewstate_cache = {}

    # Page 1 (GET)
    log.info(f"  → Index page 1...")
    items, total_pages = get_index_page(typegroupid, classid, 1, viewstate_cache)
    if not items:
        log.info("  → Dừng: không có kết quả ở page 1")
        return []
    all_items.extend(items)
    polite_sleep(1.0, 2.0)

    actual_max = min(total_pages, max_pages)
    log.info(f"  → Sẽ crawl {actual_max} trang")

    # Page 2+ (POST)
    for page in range(2, actual_max + 1):
        log.info(f"  → Index page {page}/{actual_max}...")
        items, _ = get_index_page(typegroupid, classid, page, viewstate_cache)
        if not items:
            log.info(f"  → Dừng tại page {page}")
            break
        all_items.extend(items)
        polite_sleep(1.0, 2.0)

    return all_items


# ─────────────────────────────────────────────────────────────────
# STEP 2: DETAIL CRAWL — nội dung từng văn bản
# ─────────────────────────────────────────────────────────────────

def parse_date(raw: str) -> str:
    if not raw:
        return ""
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", raw)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return raw.strip()


def classify_relation(text: str) -> str:
    t = text.lower()
    if "sửa đổi" in t and "bổ sung" in t:
        return "SUA_DOI_BO_SUNG"
    if "sửa đổi" in t:
        return "SUA_DOI"
    if "bổ sung" in t:
        return "BO_SUNG"
    if "hủy bỏ" in t or "bãi bỏ" in t:
        return "HUY_BO"
    if "thay thế" in t:
        return "THAY_THE"
    if "hướng dẫn" in t:
        return "HUONG_DAN"
    if "căn cứ" in t:
        return "CAN_CU"
    return "LIEN_QUAN"


def crawl_doc(doc_id: str, doc_url: str) -> dict:
    """
    Crawl trang văn bản chi tiết.
    URL: https://vanban.chinhphu.vn/?pageid=27160&docid=XXX
    """
    if not doc_url or "docid" not in doc_url:
        doc_url = f"{BASE}/?pageid=27160&docid={doc_id}"

    r = safe_get(doc_url)
    if not r:
        return {}

    soup = BeautifulSoup(r.text, "lxml")
    data = {"url": doc_url, "doc_id": doc_id}

    # ── Metadata từ bảng thuộc tính ──
    # vanban.chinhphu.vn thường dùng table hoặc dl để hiển thị thuộc tính
    for row in soup.select("table.info tr, table.metadata tr, .doc-info tr, table tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True).lower()
        value = cells[1].get_text(separator=" ", strip=True)

        if "số hiệu" in label or "số ký hiệu" in label:
            data["so_hieu"] = value
        elif "loại văn bản" in label:
            data["loai_van_ban"] = value
        elif "cơ quan ban hành" in label or "nơi ban hành" in label:
            data["co_quan_ban_hanh"] = value
        elif "ngày ban hành" in label or "ngày ký" in label:
            data["ngay_ban_hanh"] = parse_date(value)
        elif "ngày có hiệu lực" in label or "ngày hiệu lực" in label:
            data["ngay_hieu_luc"] = parse_date(value)
        elif "tình trạng" in label or "hiệu lực" in label:
            data["tinh_trang"] = value
        elif "lĩnh vực" in label:
            data["linh_vuc"] = value
        elif "người ký" in label or "ký bởi" in label:
            data["nguoi_ky"] = value
        elif "trích yếu" in label or "tên văn bản" in label:
            data["trich_yeu"] = value

    # Fallback metadata từ thẻ meta/title
    if not data.get("trich_yeu"):
        og_title = soup.find("meta", property="og:title")
        if og_title:
            data["trich_yeu"] = og_title.get("content", "").strip()
    if not data.get("trich_yeu"):
        h1 = soup.select_one("h1, .doc-title, .title")
        if h1:
            data["trich_yeu"] = h1.get_text(strip=True)

    # ── Nội dung toàn văn ──
    noi_dung = ""
    for sel in [
        ".doc-content",
        "#doc-content",
        ".van-ban-content",
        ".content-doc",
        "#ContentPlaceHolder1_divContent",
        ".article-content",
        "article",
        ".entry-content",
    ]:
        tag = soup.select_one(sel)
        if tag and len(tag.get_text(strip=True)) > 200:
            noi_dung = tag.get_text(separator="\n", strip=True)
            break

    if not noi_dung:
        # Fallback: body trừ nav/header/footer
        for unwanted in soup.select("nav, header, footer, script, style, .sidebar, .menu, .pager, .breadcrumb"):
            unwanted.decompose()
        body = soup.find("body")
        if body:
            noi_dung = body.get_text(separator="\n", strip=True)

    data["noi_dung"] = noi_dung

    # ── Link PDF ──
    for a in soup.find_all("a", href=re.compile(r"\.pdf", re.I)):
        href = a.get("href", "")
        data["url_pdf"] = href if href.startswith("http") else BASE + href
        break
    if "url_pdf" not in data:
        data["url_pdf"] = ""

    # ── Relations (văn bản liên quan) ──
    relations = []
    for a in soup.find_all("a", href=re.compile(r"docid=\d+", re.I)):
        href = a.get("href", "")
        m = re.search(r"docid=(\d+)", href, re.I)
        if m and m.group(1) != doc_id:
            text = a.get_text(strip=True)
            relations.append({
                "target_doc_id": m.group(1),
                "relation_type": classify_relation(text),
                "mo_ta": text[:100],
            })
    data["relations"] = relations

    return data


def extract_pdf_text(pdf_url: str) -> str:
    if not HAS_PDFPLUMBER or not pdf_url:
        return ""
    try:
        r = SESSION.get(pdf_url, timeout=30)
        r.raise_for_status()
        with pdfplumber.open(BytesIO(r.content)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        log.warning(f"  PDF failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────
# STEP 3: PARSE ĐIỀU KHOẢN
# ─────────────────────────────────────────────────────────────────

def parse_articles(doc_id: str, full_text: str) -> tuple:
    articles = []
    structure = []
    if not full_text:
        return articles, structure

    re_chuong = re.compile(r"^chương\s+[\wIVXivx]+", re.I)
    re_muc    = re.compile(r"^mục\s+[\wIVXivx\d]+", re.I)
    re_dieu   = re.compile(r"^điều\s+(\d+)[.:\s]\s*(.*)", re.I)

    current_chuong = ""
    current_muc = ""
    current_dieu = None
    current_lines = []

    def flush():
        if not current_dieu:
            return
        nd = "\n".join(current_lines).strip()
        if nd:
            articles.append(Article(
                node_id=f"article_{doc_id}_dieu_{current_dieu['so']}",
                van_ban_id=f"doc_{doc_id}",
                so_dieu=current_dieu["so"],
                ten_dieu=current_dieu["ten"],
                noi_dung=nd,
                chuong=current_chuong,
                muc=current_muc,
            ))

    for line in full_text.split("\n"):
        line = line.strip()
        if not line:
            if current_dieu:
                current_lines.append("")
            continue
        if re_chuong.match(line):
            flush(); current_dieu, current_lines = None, []
            current_chuong, current_muc = line, ""
            structure.append({"type": "CHUONG", "label": line, "dieu": []})
            continue
        if re_muc.match(line):
            flush(); current_dieu, current_lines = None, []
            current_muc = line
            if structure:
                structure[-1].setdefault("muc", []).append({"label": line})
            continue
        m = re_dieu.match(line)
        if m:
            flush()
            current_dieu = {"so": m.group(1), "ten": m.group(2).strip()}
            current_lines = []
            if structure:
                structure[-1].setdefault("dieu", []).append(f"Điều {m.group(1)}")
            continue
        if current_dieu:
            current_lines.append(line)

    flush()
    return articles, structure


# ─────────────────────────────────────────────────────────────────
# STEP 4: RELATIONS TỪ TEXT
# ─────────────────────────────────────────────────────────────────

REL_PATTERNS = [
    (r"sửa đổi[,\s]+bổ sung[^.]{0,80}?([\d]+[/\-][\d]{4}[/\-][\w\-]+)", "SUA_DOI_BO_SUNG"),
    (r"bãi bỏ[^.]{0,80}?([\d]+[/\-][\d]{4}[/\-][\w\-]+)",               "HUY_BO"),
    (r"thay thế[^.]{0,80}?([\d]+[/\-][\d]{4}[/\-][\w\-]+)",              "THAY_THE"),
    (r"hướng dẫn[^.]{0,80}?([\d]+[/\-][\d]{4}[/\-][\w\-]+)",            "HUONG_DAN"),
    (r"căn cứ[^.]{0,80}?([\d]+[/\-][\d]{4}[/\-][\w\-]+)",               "CAN_CU"),
]


def extract_text_relations(source_id: str, text: str) -> list:
    rels = []
    t = text[:8000].lower()
    for pattern, rel_type in REL_PATTERNS:
        for m in re.finditer(pattern, t):
            rels.append(Relation(
                source_id=source_id,
                target_id=f"ref_sohieu_{m.group(1).upper()}",
                relation_type=rel_type,
                mo_ta=m.group(1).upper(),
            ))
    return rels


# ─────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.done: set = set()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.done = set(json.load(f))
            log.info(f"Resume: {len(self.done)} items đã crawl")

    def mark(self, doc_id: str):
        self.done.add(doc_id)
        if len(self.done) % 50 == 0:
            self.save()

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(list(self.done), f)

    def is_done(self, doc_id: str) -> bool:
        return doc_id in self.done


# ─────────────────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────────────────

def verify_dataset(nodes, edges, articles) -> dict:
    stats = {
        "total_documents":  len(nodes),
        "total_articles":   len(articles),
        "total_edges":      len(edges),
        "by_loai":          {},
        "by_tinh_trang":    {},
        "missing_content":  0,
        "missing_so_hieu":  0,
        "has_pdf":          0,
        "edge_types":       {},
    }
    for n in nodes:
        p = n.get("properties", {})
        loai = p.get("loai_van_ban", "Unknown")
        stats["by_loai"][loai] = stats["by_loai"].get(loai, 0) + 1
        tt = p.get("tinh_trang", "Unknown")
        stats["by_tinh_trang"][tt] = stats["by_tinh_trang"].get(tt, 0) + 1
        if not p.get("noi_dung", "").strip():
            stats["missing_content"] += 1
        if not p.get("so_hieu", "").strip():
            stats["missing_so_hieu"] += 1
        if p.get("url_pdf", ""):
            stats["has_pdf"] += 1
    for e in edges:
        et = e.get("type", "UNKNOWN")
        stats["edge_types"][et] = stats["edge_types"].get(et, 0) + 1
    return stats


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_pipeline(doc_types=None, max_pages=999, output_dir=OUTPUT_DIR):
    if doc_types is None:
        doc_types = list(DOC_TYPES.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    ckpt = Checkpoint(output_dir / "checkpoint.json")

    all_nodes = []
    all_articles = []
    all_relations = []
    so_hieu_map = {}

    # Resume
    nodes_path = output_dir / "nodes.json"
    if nodes_path.exists():
        with open(nodes_path, encoding="utf-8") as f:
            all_nodes = json.load(f)
        for n in all_nodes:
            sh = n["properties"].get("so_hieu", "")
            if sh:
                so_hieu_map[sh] = n["id"]
        log.info(f"Resume: loaded {len(all_nodes)} nodes")

    # ── PHASE 1: Thu thập danh sách ──────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1: Thu thập danh sách văn bản")
    log.info(f"  Source: {URL_LIST}")

    all_index = []
    for key in doc_types:
        if key not in DOC_TYPES:
            log.warning(f"  Không hợp lệ: {key}")
            continue
        info = DOC_TYPES[key]
        log.info(f"  Loại: {info['label']} (typegroupid={info['typegroupid']})")
        items = crawl_index_all(info["typegroupid"], info["classid"], max_pages=max_pages)
        log.info(f"  → {len(items)} văn bản")
        for item in items:
            item["doc_type_key"] = key
        all_index.extend(items)

    # Dedup
    seen = {}
    deduped = []
    for item in all_index:
        if item["doc_id"] not in seen:
            seen[item["doc_id"]] = True
            deduped.append(item)
    log.info(f"Tổng: {len(all_index)} | Unique: {len(deduped)}")

    if not deduped:
        log.error("❌ Không lấy được danh sách văn bản!")
        log.error("   Kiểm tra kết nối mạng hoặc mở trình duyệt vào:")
        log.error(f"   {URL_LIST}?classid=1&typegroupid=1&mode=1")
        return {}

    # ── PHASE 2: Crawl chi tiết ───────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2: Crawl nội dung chi tiết")

    for item in tqdm(deduped, desc="Crawling"):
        doc_id = item["doc_id"]
        if ckpt.is_done(doc_id):
            continue

        detail = crawl_doc(doc_id, item.get("url", ""))
        polite_sleep(0.8, 1.5)

        noi_dung = detail.get("noi_dung", "")
        pdf_url  = detail.get("url_pdf", "")

        # PDF fallback
        if not noi_dung.strip() and pdf_url:
            log.info(f"  [{doc_id}] Không có HTML → thử PDF")
            noi_dung = extract_pdf_text(pdf_url)
            polite_sleep(1.0, 2.0)

        articles, structure = parse_articles(doc_id, noi_dung)
        all_articles.extend(articles)

        doc = Document(
            node_id=f"doc_{doc_id}",
            doc_id=doc_id,
            so_hieu=detail.get("so_hieu", item.get("so_hieu", "")),
            loai_van_ban=detail.get("loai_van_ban", DOC_TYPES.get(item.get("doc_type_key",""), {}).get("label", "")),
            trich_yeu=detail.get("trich_yeu", item.get("trich_yeu", "")),
            co_quan_ban_hanh=detail.get("co_quan_ban_hanh", ""),
            ngay_ban_hanh=detail.get("ngay_ban_hanh", ""),
            ngay_hieu_luc=detail.get("ngay_hieu_luc", ""),
            tinh_trang=detail.get("tinh_trang", ""),
            linh_vuc=detail.get("linh_vuc", ""),
            nguoi_ky=detail.get("nguoi_ky", ""),
            url=detail.get("url", ""),
            url_pdf=pdf_url,
            noi_dung=noi_dung,
            chuong_muc=structure,
            crawled_at=datetime.now().isoformat(),
        )

        all_nodes.append(doc.to_node())
        if doc.so_hieu:
            so_hieu_map[doc.so_hieu] = doc.node_id

        # Edges từ trang chi tiết
        for rel in detail.get("relations", []):
            all_relations.append(Relation(
                source_id=doc.node_id,
                target_id=f"doc_{rel['target_doc_id']}",
                relation_type=rel["relation_type"],
                mo_ta=rel["mo_ta"],
            ))

        # Edges từ nội dung text
        all_relations.extend(extract_text_relations(doc.node_id, noi_dung[:6000]))
        ckpt.mark(doc_id)

        # Incremental save
        if len(all_nodes) % 100 == 0:
            with open(output_dir / "nodes.json", "w", encoding="utf-8") as f:
                json.dump(all_nodes, f, ensure_ascii=False)
            log.info(f"  [Auto-save] {len(all_nodes)} nodes")

    # ── PHASE 3: Resolve relations ────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 3: Resolve relations")

    resolved = []
    for rel in all_relations:
        if rel.target_id.startswith("ref_sohieu_"):
            sh = rel.target_id.replace("ref_sohieu_", "")
            rid = so_hieu_map.get(sh)
            if rid:
                rel.target_id = rid
                resolved.append(rel)
        else:
            resolved.append(rel)

    for a in all_articles:
        resolved.append(Relation(
            source_id=a.node_id,
            target_id=a.van_ban_id,
            relation_type="THUOC_VAN_BAN",
        ))

    all_edges = [r.to_edge() for r in resolved]

    # ── PHASE 4: Save & Verify ─────────────────────────────────────
    ckpt.save()
    article_nodes = [a.to_node() for a in all_articles]

    graph = {
        "meta": {
            "created_at":  datetime.now().isoformat(),
            "source":      "vanban.chinhphu.vn",
            "description": "Văn bản pháp luật Việt Nam - GraphRAG Dataset",
            "total_nodes": len(all_nodes) + len(article_nodes),
            "total_edges": len(all_edges),
        },
        "nodes": all_nodes + article_nodes,
        "edges": all_edges,
    }

    with open(output_dir / "nodes.json", "w", encoding="utf-8") as f:
        json.dump(all_nodes, f, ensure_ascii=False, indent=2)
    with open(output_dir / "article_nodes.json", "w", encoding="utf-8") as f:
        json.dump(article_nodes, f, ensure_ascii=False, indent=2)
    with open(output_dir / "edges.json", "w", encoding="utf-8") as f:
        json.dump(all_edges, f, ensure_ascii=False, indent=2)
    with open(output_dir / "graphrag_dataset.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    stats = verify_dataset(all_nodes, all_edges, all_articles)
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info("VERIFICATION REPORT")
    log.info(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vanban.chinhphu.vn Crawler")
    parser.add_argument(
        "--types", nargs="+",
        choices=list(DOC_TYPES.keys()),
        default=list(DOC_TYPES.keys()),
        help="Loại văn bản cần crawl",
    )
    parser.add_argument(
        "--max-pages", type=int, default=999,
        help="Số trang tối đa mỗi loại (dùng 1-3 để test)",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Thư mục output",
    )
    args = parser.parse_args()

    log.info("🚀 vanban.chinhphu.vn Crawler bắt đầu")
    log.info(f"   Source: {URL_LIST}")
    log.info(f"   Loại: {args.types}")
    log.info(f"   Max pages: {args.max_pages}")

    stats = run_pipeline(
        doc_types=args.types,
        max_pages=args.max_pages,
        output_dir=Path(args.output),
    )

    if stats:
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH!")
        print(f"   Văn bản:    {stats.get('total_documents', 0):,}")
        print(f"   Điều khoản: {stats.get('total_articles', 0):,}")
        print(f"   Edges:      {stats.get('total_edges', 0):,}")
        print(f"   → {args.output}/graphrag_dataset.json")
