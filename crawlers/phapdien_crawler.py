"""
Crawler for Pháp Điển (Bộ pháp điển): https://phapdien.moj.gov.vn
Currently updated to process from offline directory (BoPhapDienDienTu) instead of HTTP scraping.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def _extract_js_var(js_text: str, var_name: str) -> str:
    """Extract JSON string from var name = [...];"""
    pattern = rf"var\s+{re.escape(var_name)}\s*=\s*"
    m = re.search(pattern, js_text)
    if not m:
        raise ValueError(f"Không tìm thấy biến: {var_name}")
    start = m.end()
    first_char = js_text[start]
    open_b, close_b = ("[", "]") if first_char == "[" else ("{", "}")
    depth, in_string, escape, i = 0, False, False, start
    quote_char = None
    while i < len(js_text):
        ch = js_text[i]
        if escape: 
            escape = False
        elif ch == "\\" and in_string: 
            escape = True
        elif ch in ('"', "'"):
            if in_string:
                if ch == quote_char:
                    in_string = False
            else:
                in_string = True
                quote_char = ch
        elif not in_string:
            if ch == open_b: 
                depth += 1
            elif ch == close_b:
                depth -= 1
                if depth == 0: 
                    return js_text[start:i+1]
        i += 1
    return "[]"

def _parse_js_array(raw: str) -> list:
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(cleaned)

def parse_html_demuc(html_path: Path) -> list[dict]:
    if not html_path.exists():
        logger.warning(f"File not found: {html_path}")
        return []
    try:
        soup = BeautifulSoup(html_path.read_bytes(), "html.parser", from_encoding="utf-8")
        content_div = soup.find("div", class_="_content") or soup
        dieu_list = []
        
        current_dieu = None
        current_parts = []
        TARGET_CLASSES = ("pNoiDung", "pDan", "pItem")
        
        for tag in content_div.find_all(True):
            cls = " ".join(tag.get("class", []))
            text = tag.get_text(" ", strip=True)
            if "pDieu" in cls:
                if current_dieu:
                    dieu_list.append({
                        "so_dieu": current_dieu["so_dieu"],
                        "tieu_de": current_dieu["tieu_de"],
                        "ghi_chu": current_dieu.get("ghi_chu", ""),
                        "noi_dung": "\n".join(current_parts)
                    })
                # Extract so dieu
                m = re.search(r"Điều\s+([\d\.]+a?)", text, re.IGNORECASE)
                so_dieu = m.group(1) if m else ""
                current_dieu = {"so_dieu": so_dieu, "tieu_de": text, "ghi_chu": ""}
                current_parts = []
            elif "pGhiChu" in cls and current_dieu:
                # Bắt thẻ ghi chú chứa tên luật
                if text:
                    current_dieu["ghi_chu"] = text.strip("()")
            elif any(c in cls for c in TARGET_CLASSES) and current_dieu:
                if text:
                    current_parts.append(text)
                    
        if current_dieu:
            dieu_list.append({
                "so_dieu": current_dieu["so_dieu"],
                "tieu_de": current_dieu["tieu_de"],
                "ghi_chu": current_dieu.get("ghi_chu", ""),
                "noi_dung": "\n".join(current_parts)
            })
            
        return dieu_list
    except Exception as e:
        logger.error(f"Error parsing {html_path}: {e}")
        return []

def crawl_phapdien(
    output_dir: Path, 
    max_chu_de: Optional[int] = None, 
    delay: float = 0.0, 
    offline_dir: Path = Path("BoPhapDienDienTu")
) -> list[dict]:

    js_path = offline_dir / "jsonData.js"
    demuc_dir = offline_dir / "demuc"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Parsing local Phap Dien from {offline_dir}...")
    
    if not js_path.exists():
        logger.error(f"{js_path} does not exist!")
        return []

    with open(js_path, encoding="utf-8") as f:
        js_text = f.read()
    
    chude_list = _parse_js_array(_extract_js_var(js_text, "jdChuDe"))
    demuc_list = _parse_js_array(_extract_js_var(js_text, "jdDeMuc"))
    
    demuc_by_chude = {}
    for dm in demuc_list:
        cd = dm.get("ChuDe", "")
        demuc_by_chude.setdefault(cd, []).append(dm)
        
    results = []
    if max_chu_de:
        chude_list = chude_list[:max_chu_de]
        
    for i, cd in enumerate(chude_list, 1):
        logger.info(f"[{i}/{len(chude_list)}] Processing Chủ đề: {cd.get('Text', '')}")
        cd_result = {
            "id": cd.get("Value", ""),
            "ten_chu_de": cd.get("Text", ""),
            "de_muc_list": []
        }
        demuc_count = 0
        dieu_count = 0
        for dm in demuc_by_chude.get(cd.get("Value", ""), []):
            dm_id = dm.get("Value", "")
            html_path = demuc_dir / f"{dm_id}.html"
            dieu_list = parse_html_demuc(html_path)
            
            cd_result["de_muc_list"].append({
                "id": dm_id,
                "ten_de_muc": dm.get("Text", ""),
                "dieu_list": dieu_list
            })
            demuc_count += 1
            dieu_count += len(dieu_list)
            
        logger.info(f"  -> Extracted {demuc_count} Đề mục, {dieu_count} Điều.")
        results.append(cd_result)
        
        ckpt = output_dir / f"chu_de_{i:03d}.json"
        ckpt.write_text(json.dumps(cd_result, ensure_ascii=False, indent=2))
        
        if delay > 0:
            time.sleep(delay)
            
    out_file = output_dir / "phapdien_all.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"Pháp Điển total: {len(results)} Chủ đề → {out_file}")
    
    return results
