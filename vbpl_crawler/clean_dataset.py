"""
clean_dataset.py — GraphRAG Dataset Cleaner
=============================================
Thực hiện 4 bước clean:
  1. Fix tình trạng hiệu lực rỗng (suy luận từ nội dung)
  2. Chuẩn hóa loại văn bản
  3. Loại bỏ văn bản trùng lặp / không liên quan
  4. Crawl thêm Quyết định bị thiếu

Chạy:
    python clean_dataset.py
    python clean_dataset.py --skip-crawl   # bỏ qua bước crawl thêm
"""

import re
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

INPUT_FILE  = Path("output/graphrag_dataset.json")
OUTPUT_FILE = Path("output/graphrag_dataset_clean.json")
LOG_FILE    = Path("output/clean.log")

BASE     = "https://chinhphu.vn"
URL_LIST = f"{BASE}/he-thong-van-ban"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": BASE,
}

# typegroupid trên vanban.chinhphu.vn
# typegroupid=3 is Quyết định
DOC_TYPES = {
    "quyet_dinh": {"typegroupid": "3", "classid": "1", "label": "Quyết định"},
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# STEP 1: FIX TÌNH TRẠNG HIỆU LỰC
# ─────────────────────────────────────────────────────────────────

# Từ khóa trong nội dung để suy luận tình trạng
STILL_VALID_PATTERNS = [
    r"còn hiệu lực",
    r"đang có hiệu lực",
    r"có hiệu lực thi hành",
    r"có hiệu lực từ ngày",
    r"có hiệu lực kể từ",
]

EXPIRED_PATTERNS = [
    r"hết hiệu lực",
    r"không còn hiệu lực",
    r"bị bãi bỏ",
    r"bị hủy bỏ",
    r"được thay thế",
    r"thay thế bởi",
]

# Văn bản trước 1975 hoặc quá cũ → đánh dấu "Lịch sử"
HISTORICAL_LOAI = {"Sắc lệnh", "Sắc luật", "Sắt luật", "Lệnh"}


def infer_tinh_trang(node: dict) -> str:
    """Suy luận tình trạng hiệu lực từ nội dung và metadata."""
    props = node.get("properties", {})
    noi_dung = (props.get("noi_dung", "") or "").lower()
    trich_yeu = (props.get("trich_yeu", "") or "").lower()
    ngay_ban_hanh = props.get("ngay_ban_hanh", "") or ""
    loai = props.get("loai_van_ban", "") or ""

    # Văn bản lịch sử (trước 1975)
    year_match = re.search(r"(\d{4})", ngay_ban_hanh)
    if year_match and int(year_match.group(1)) < 1975:
        return "Lịch sử"

    # Loại văn bản đã lỗi thời
    if loai in HISTORICAL_LOAI:
        return "Hết hiệu lực"

    # Tìm trong nội dung
    for pat in EXPIRED_PATTERNS:
        if re.search(pat, noi_dung[:2000]) or re.search(pat, trich_yeu):
            return "Hết hiệu lực"

    for pat in STILL_VALID_PATTERNS:
        if re.search(pat, noi_dung[:2000]):
            return "Còn hiệu lực"

    # Mặc định: văn bản sau 2000 → giả định còn hiệu lực
    if year_match and int(year_match.group(1)) >= 2000:
        return "Còn hiệu lực (chưa xác nhận)"

    return "Chưa xác định"


def fix_tinh_trang(nodes: list) -> tuple:
    """Fix tình trạng hiệu lực for all nodes."""
    fixed = 0
    dist = defaultdict(int)

    for node in nodes:
        if node.get("type") != "VanBan":
            continue
        props = node["properties"]
        tt = props.get("tinh_trang", "").strip()

        # Skip if already has a valid status
        if tt and tt not in ["", "Tỷ lệ phí bảo hiểm chưa được hưởng"]:
            dist[tt] += 1
            continue

        new_tt = infer_tinh_trang(node)
        props["tinh_trang"] = new_tt
        dist[new_tt] += 1
        fixed += 1

    log.info(f"[Step 1] Fixed {fixed} tình trạng rỗng")
    log.info(f"  Phân bổ: {dict(dist)}")
    return nodes, fixed


# ─────────────────────────────────────────────────────────────────
# STEP 2: CHUẨN HÓA LOẠI VĂN BẢN
# ─────────────────────────────────────────────────────────────────

# Map original type → normalized type
LOAI_MAP = {
    # OCR/Typo fixes
    "Sắt luật":               "Sắc luật",
    "Sắc lệnh":               "Sắc lệnh",
    "Sắc luật":               "Sắc luật",

    # Merge similar groups
    "Thông tư liên tịch":     "Thông tư liên tịch",
    "Thông tư liên bộ":       "Thông tư liên tịch",

    # Case normalization
    "hiến pháp":              "Hiến pháp",
    "luật":                   "Luật",
    "nghị định":              "Nghị định",
    "thông tư":               "Thông tư",
    "quyết định":             "Quyết định",
    "nghị quyết":             "Nghị quyết",
    "pháp lệnh":              "Pháp lệnh",
    "chỉ thị":                "Chỉ thị",
}

# Valid types for the Legal domain
VALID_LOAI = {
    "Hiến pháp", "Luật", "Bộ luật", "Pháp lệnh",
    "Nghị định", "Quyết định", "Thông tư", "Thông tư liên tịch",
    "Nghị quyết", "Chỉ thị", "Lệnh", "Sắc lệnh", "Sắc luật",
}


def normalize_loai(loai: str) -> str:
    if not loai:
        return "Không xác định"
    loai = loai.strip()
    # Exact match
    if loai in LOAI_MAP:
        return LOAI_MAP[loai]
    # Case-insensitive
    loai_lower = loai.lower()
    if loai_lower in LOAI_MAP:
        return LOAI_MAP[loai_lower]
    return loai


def fix_loai_van_ban(nodes: list) -> tuple:
    fixed = 0
    dist = defaultdict(int)

    for node in nodes:
        if node.get("type") != "VanBan":
            continue
        props = node["properties"]
        old_loai = props.get("loai_van_ban", "")
        new_loai = normalize_loai(old_loai)
        if new_loai != old_loai:
            props["loai_van_ban"] = new_loai
            fixed += 1
        dist[new_loai] += 1

    log.info(f"[Step 2] Chuẩn hóa {fixed} loại văn bản")
    for loai, count in sorted(dist.items(), key=lambda x: -x[1]):
        log.info(f"  {loai:<35} {count:>6,}")
    return nodes, fixed


# ─────────────────────────────────────────────────────────────────
# STEP 3: LOẠI BỎ TRÙNG LẶP / KHÔNG LIÊN QUAN
# ─────────────────────────────────────────────────────────────────

# Keywords in trich_yeu to discard irrelevant documents
IRRELEVANT_KEYWORDS = [
    "kế hoạch tuyển dụng",
    "thông báo tuyển sinh",
    "lịch công tác",
    "thông báo họp",
    "kết quả xổ số",
    "giá xăng dầu",
]


def remove_duplicates_and_irrelevant(nodes: list, edges: list) -> tuple:
    """
    - Remove duplicate VanBan nodes by so_hieu (keep newest)
    - Remove irrelevant documents
    - Update edges accordingly
    """
    vanban_nodes = [n for n in nodes if n.get("type") == "VanBan"]
    other_nodes  = [n for n in nodes if n.get("type") != "VanBan"]

    # Dedup by so_hieu (keep newest crawled_at)
    so_hieu_seen = {}
    dupes = 0
    for node in vanban_nodes:
        sh = node["properties"].get("so_hieu", "").strip()
        if not sh:
            continue
        if sh in so_hieu_seen:
            existing = so_hieu_seen[sh]
            if node["properties"].get("crawled_at","") > existing["properties"].get("crawled_at",""):
                so_hieu_seen[sh] = node
            dupes += 1
        else:
            so_hieu_seen[sh] = node

    # Keep all nodes without so_hieu (can't safely dedup)
    no_sohieu = [n for n in vanban_nodes if not n["properties"].get("so_hieu","").strip()]
    deduped = list(so_hieu_seen.values()) + no_sohieu

    # Filter irrelevant
    irrelevant = 0
    kept = []
    for node in deduped:
        trich_yeu = (node["properties"].get("trich_yeu","") or "").lower()
        if any(kw in trich_yeu for kw in IRRELEVANT_KEYWORDS):
            irrelevant += 1
            continue
        kept.append(node)

    valid_ids = {n["id"] for n in kept} | {n["id"] for n in other_nodes}

    # Filter edges
    clean_edges = [
        e for e in edges
        if e["source"] in valid_ids and e["target"] in valid_ids
    ]

    log.info(f"[Step 3] Xóa {dupes} bản trùng số hiệu")
    log.info(f"[Step 3] Xóa {irrelevant} văn bản không liên quan")
    log.info(f"[Step 3] Còn lại: {len(kept):,} VanBan nodes, {len(clean_edges):,} edges")

    return kept + other_nodes, clean_edges, dupes + irrelevant


# ─────────────────────────────────────────────────────────────────
# STEP 4: CRAWL THÊM QUYẾT ĐỊNH
# ─────────────────────────────────────────────────────────────────

def safe_get(url, params=None, timeout=20):
    for attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt + random.uniform(0.5, 1))
    return None


def parse_index_html_simple(soup) -> list:
    """Extract docid and metadata from list page."""
    results = []
    seen = set()
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        link = None
        for td in cells:
            a = td.find("a", href=re.compile(r"docid=\d+", re.I))
            if a:
                link = a
                break
        if not link:
            continue
        href = link.get("href", "")
        m = re.search(r"docid=(\d+)", href, re.I)
        if not m:
            continue
        doc_id = m.group(1)
        if doc_id in seen:
            continue
        seen.add(doc_id)

        so_hieu = cells[0].get_text(strip=True) if cells else ""
        ngay    = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        ten     = link.get_text(strip=True)
        pdf_url = ""
        for td in cells:
            pdf_a = td.find("a", href=re.compile(r"\.pdf", re.I))
            if pdf_a:
                h = pdf_a.get("href","")
                pdf_url = h if h.startswith("http") else BASE + h
                break

        results.append({
            "doc_id": doc_id, "so_hieu": so_hieu,
            "ngay": ngay, "trich_yeu": ten,
            "url": BASE + href if href.startswith("/") else href,
            "url_pdf": pdf_url,
        })
    return results


def crawl_quyet_dinh(existing_ids: set, max_pages: int = 999) -> list:
    """Crawl missing Quyết định documents (typegroupid=3)."""
    log.info("[Step 4] Crawl thêm Quyết định...")
    all_items = []
    viewstate = {}

    # Page 1 GET
    params = {"classid": "1", "typegroupid": "3", "mode": "1"}
    r = safe_get(URL_LIST, params=params)
    if not r:
        log.error("  Không kết nối được chinhphu.vn")
        return []

    soup = BeautifulSoup(r.text, "lxml")

    # Get viewstate
    for inp in soup.find_all("input", type="hidden"):
        name = inp.get("name","")
        if name:
            viewstate[name] = inp.get("value","")
    for a in soup.find_all("a", href=re.compile(r"__doPostBack")):
        m = re.search(r"__doPostBack\('([^']+)'", a.get("href",""))
        if m and "grv" in m.group(1).lower():
            viewstate["__GRID_CONTROL"] = m.group(1)
            break

    # Total pages
    page_info = soup.select_one("[id*='document_page']")
    total_pages = 1
    if page_info:
        m = re.search(r"(\d+)\s*-\s*(\d+)\s*[|]\s*(\d+)", page_info.get_text())
        if m:
            per_page = int(m.group(2)) - int(m.group(1)) + 1
            total = int(m.group(3))
            total_pages = min((total + per_page - 1) // per_page, max_pages)
            log.info(f"  Quyết định: {total} văn bản, {total_pages} trang")

    items_p1 = parse_index_html_simple(soup)
    new_items = [x for x in items_p1 if x["doc_id"] not in existing_ids]
    all_items.extend(new_items)
    log.info(f"  Page 1: {len(items_p1)} items, {len(new_items)} mới")

    time.sleep(random.uniform(1, 2))

    # Page 2+
    grid_ctrl = viewstate.get("__GRID_CONTROL", "")
    if grid_ctrl and total_pages > 1:
        for page in tqdm(range(2, total_pages + 1), desc="  Quyết định pages"):
            post_data = {k: v for k, v in viewstate.items() if k != "__GRID_CONTROL"}
            post_data["__EVENTTARGET"]   = grid_ctrl
            post_data["__EVENTARGUMENT"] = f"Page${page}"

            url = f"{URL_LIST}?classid=1&typegroupid=3&mode=1"
            try:
                r = SESSION.post(url, data=post_data, timeout=20)
                r.raise_for_status()
                r.encoding = "utf-8"
            except Exception as e:
                log.warning(f"  POST page {page} failed: {e}")
                continue

            soup = BeautifulSoup(r.text, "lxml")
            for inp in soup.find_all("input", type="hidden"):
                name = inp.get("name","")
                if name and name != "__GRID_CONTROL":
                    viewstate[name] = inp.get("value","")

            items = parse_index_html_simple(soup)
            new = [x for x in items if x["doc_id"] not in existing_ids]
            all_items.extend(new)
            time.sleep(random.uniform(1, 2))

    log.info(f"  → {len(all_items)} Quyết định mới cần crawl chi tiết")
    return all_items


def crawl_doc_simple(doc_id: str, doc_url: str) -> dict:
    """Crawl full document content."""
    if not doc_url or "docid" not in doc_url:
        doc_url = f"{BASE}/?pageid=27160&docid={doc_id}"
    r = safe_get(doc_url)
    if not r:
        return {}

    soup = BeautifulSoup(r.text, "lxml")
    data = {"url": doc_url, "noi_dung": "", "url_pdf": ""}

    for sel in [".doc-content","#doc-content",".van-ban-content",
                "#ContentPlaceHolder1_divContent","article",".entry-content"]:
        tag = soup.select_one(sel)
        if tag and len(tag.get_text(strip=True)) > 200:
            data["noi_dung"] = tag.get_text(separator="\n", strip=True)
            break

    if not data["noi_dung"]:
        for bad in soup.select("nav,header,footer,script,style,.menu,.pager"):
            bad.decompose()
        body = soup.find("body")
        if body:
            data["noi_dung"] = body.get_text(separator="\n", strip=True)

    for a in soup.find_all("a", href=re.compile(r"\.pdf", re.I)):
        h = a.get("href","")
        data["url_pdf"] = h if h.startswith("http") else BASE + h
        break

    for row in soup.select("table tr"):
        cells = row.find_all(["td","th"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True).lower()
        value = cells[1].get_text(separator=" ", strip=True)
        if "số hiệu" in label:
            data["so_hieu"] = value
        elif "loại văn bản" in label:
            data["loai_van_ban"] = value
        elif "cơ quan ban hành" in label:
            data["co_quan_ban_hanh"] = value
        elif "ngày ban hành" in label or "ngày ký" in label:
            data["ngay_ban_hanh"] = value
        elif "người ký" in label:
            data["nguoi_ky"] = value
        elif "lĩnh vực" in label:
            data["linh_vuc"] = value

    return data


def build_node_from_item(item: dict, detail: dict) -> dict:
    doc_id = item["doc_id"]
    return {
        "id": f"doc_{doc_id}",
        "type": "VanBan",
        "properties": {
            "doc_id":           doc_id,
            "so_hieu":          detail.get("so_hieu", item.get("so_hieu","")),
            "loai_van_ban":     normalize_loai(detail.get("loai_van_ban","Quyết định")),
            "trich_yeu":        item.get("trich_yeu",""),
            "co_quan_ban_hanh": detail.get("co_quan_ban_hanh",""),
            "ngay_ban_hanh":    item.get("ngay",""),
            "ngay_hieu_luc":    "",
            "tinh_trang":       "Còn hiệu lực (chưa xác nhận)",
            "linh_vuc":         detail.get("linh_vuc",""),
            "nguoi_ky":         detail.get("nguoi_ky",""),
            "url":              item.get("url",""),
            "url_pdf":          detail.get("url_pdf", item.get("url_pdf","")),
            "noi_dung":         detail.get("noi_dung",""),
            "chuong_muc":       [],
            "crawled_at":       datetime.now().isoformat(),
        }
    }


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main(skip_crawl: bool = False):
    log.info("=" * 60)
    log.info("GraphRAG Dataset Cleaner")
    log.info(f"Input:  {INPUT_FILE}")
    log.info(f"Output: {OUTPUT_FILE}")

    # Load dataset
    if not INPUT_FILE.exists():
        log.error(f"Input file not found: {INPUT_FILE}")
        return

    log.info("Loading dataset...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    edges = graph["edges"]
    log.info(f"Loaded: {len(nodes):,} nodes, {len(edges):,} edges")

    vanban_before = sum(1 for n in nodes if n.get("type") == "VanBan")
    log.info(f"VanBan nodes: {vanban_before:,}")

    # ── Step 1: Fix tình trạng ────────────────────────────────────
    log.info("\n── STEP 1: Fix tình trạng hiệu lực ──")
    nodes, fixed1 = fix_tinh_trang(nodes)

    # ── Step 2: Chuẩn hóa loại văn bản ───────────────────────────
    log.info("\n── STEP 2: Chuẩn hóa loại văn bản ──")
    nodes, fixed2 = fix_loai_van_ban(nodes)

    # ── Step 3: Loại bỏ trùng lặp ────────────────────────────────
    log.info("\n── STEP 3: Loại bỏ trùng lặp / không liên quan ──")
    nodes, edges, removed3 = remove_duplicates_and_irrelevant(nodes, edges)

    # ── Step 4: Crawl thêm Quyết định ────────────────────────────
    new_nodes_added = 0
    if not skip_crawl:
        log.info("\n── STEP 4: Crawl thêm Quyết định ──")
        existing_ids = {
            n["properties"].get("doc_id","")
            for n in nodes if n.get("type") == "VanBan"
        }

        qd_items = crawl_quyet_dinh(existing_ids)

        if qd_items:
            log.info(f"  Crawl chi tiết {len(qd_items)} Quyết định...")
            for item in tqdm(qd_items, desc="  Chi tiết"):
                detail = crawl_doc_simple(item["doc_id"], item.get("url",""))
                node = build_node_from_item(item, detail)
                nodes.append(node)
                new_nodes_added += 1
                time.sleep(random.uniform(0.8, 1.5))
            log.info(f"  → Thêm {new_nodes_added} Quyết định")
    else:
        log.info("\n── STEP 4: Bỏ qua (--skip-crawl) ──")

    # ── Save ──────────────────────────────────────────────────────
    vanban_after = sum(1 for n in nodes if n.get("type") == "VanBan")
    dieu_after   = sum(1 for n in nodes if n.get("type") == "Dieu")

    graph_clean = {
        "meta": {
            "created_at":   datetime.now().isoformat(),
            "source":       "chinhphu.vn",
            "description":  "Văn bản pháp luật Việt Nam - GraphRAG Dataset (Cleaned)",
            "version":      "2.0-clean",
            "total_nodes":  len(nodes),
            "total_edges":  len(edges),
            "clean_log": {
                "fixed_tinh_trang":    fixed1,
                "fixed_loai_van_ban":  fixed2,
                "removed_dupes":       removed3,
                "added_quyet_dinh":    new_nodes_added,
            },
        },
        "nodes": nodes,
        "edges": edges,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_clean, f, ensure_ascii=False, indent=2)

    # Stats
    log.info("\n" + "=" * 60)
    log.info("CLEAN HOÀN THÀNH")
    log.info(f"  VanBan:  {vanban_before:>8,} → {vanban_after:>8,}")
    log.info(f"  Dieu:    {dieu_after:>8,}")
    log.info(f"  Edges:   {len(edges):>8,}")
    log.info(f"  Output:  {OUTPUT_FILE}")

    # Final distribution
    from collections import Counter
    loai_dist = Counter(
        n["properties"].get("loai_van_ban","?")
        for n in nodes if n.get("type") == "VanBan"
    )
    log.info("\n  Loại văn bản sau clean:")
    for loai, count in loai_dist.most_common():
        log.info(f"    {loai:<35} {count:>6,}")

    tt_dist = Counter(
        n["properties"].get("tinh_trang","?")
        for n in nodes if n.get("type") == "VanBan"
    )
    log.info("\n  Tình trạng hiệu lực sau clean:")
    for tt, count in tt_dist.most_common():
        log.info(f"    {tt:<40} {count:>6,}")

    return graph_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Dataset Cleaner")
    parser.add_argument(
        "--skip-crawl", action="store_true",
        help="Bỏ qua bước crawl thêm Quyết định (chỉ clean data cũ)"
    )
    parser.add_argument(
        "--input", type=str, default="output/graphrag_dataset.json",
        help="File input (default: output/graphrag_dataset.json)"
    )
    parser.add_argument(
        "--output", type=str, default="output/graphrag_dataset_clean.json",
        help="File output (default: output/graphrag_dataset_clean.json)"
    )
    args = parser.parse_args()

    INPUT_FILE  = Path(args.input)
    OUTPUT_FILE = Path(args.output)

    main(skip_crawl=args.skip_crawl)
