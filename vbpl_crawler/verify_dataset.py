"""
verify_dataset.py
=================
Kiểm tra chất lượng dataset sau khi crawl.
Chạy: python verify_dataset.py --input output/graphrag_dataset.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def verify(filepath: str):
    path = Path(filepath)
    if not path.exists():
        print(f"❌ File không tồn tại: {filepath}")
        return

    print(f"📂 Đang verify: {filepath}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta  = data.get("meta", {})
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    van_ban_nodes = [n for n in nodes if n.get("type") == "VanBan"]
    dieu_nodes    = [n for n in nodes if n.get("type") == "Dieu"]

    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Nguồn:           {meta.get('source', 'N/A')}")
    print(f"  Tạo lúc:         {meta.get('created_at', 'N/A')}")
    print(f"  VanBan nodes:    {len(van_ban_nodes):,}")
    print(f"  Dieu nodes:      {len(dieu_nodes):,}")
    print(f"  Edges:           {len(edges):,}")

    # Phân loại văn bản
    print("\n── Phân loại văn bản ──")
    loai_counter = Counter()
    for n in van_ban_nodes:
        loai = n.get("properties", {}).get("loai_van_ban", "Unknown")
        loai_counter[loai] += 1
    for loai, count in sorted(loai_counter.items(), key=lambda x: -x[1]):
        print(f"   {loai:<30} {count:>6,}")

    # Tình trạng hiệu lực
    print("\n── Tình trạng hiệu lực ──")
    tt_counter = Counter()
    for n in van_ban_nodes:
        tt = n.get("properties", {}).get("tinh_trang", "Unknown")
        tt_counter[tt] += 1
    for tt, count in sorted(tt_counter.items(), key=lambda x: -x[1]):
        print(f"   {tt:<30} {count:>6,}")

    # Chất lượng dữ liệu
    print("\n── Chất lượng dữ liệu ──")
    missing_content  = sum(1 for n in van_ban_nodes if not n.get("properties", {}).get("noi_dung", "").strip())
    missing_so_hieu  = sum(1 for n in van_ban_nodes if not n.get("properties", {}).get("so_hieu", "").strip())
    missing_ngay     = sum(1 for n in van_ban_nodes if not n.get("properties", {}).get("ngay_ban_hanh", "").strip())
    has_pdf          = sum(1 for n in van_ban_nodes if n.get("properties", {}).get("url_pdf", "").strip())

    print(f"   Có nội dung:     {len(van_ban_nodes) - missing_content:>6,} / {len(van_ban_nodes):,}  ({100*(len(van_ban_nodes)-missing_content)/max(len(van_ban_nodes),1):.1f}%)")
    print(f"   Có số hiệu:      {len(van_ban_nodes) - missing_so_hieu:>6,} / {len(van_ban_nodes):,}  ({100*(len(van_ban_nodes)-missing_so_hieu)/max(len(van_ban_nodes),1):.1f}%)")
    print(f"   Có ngày ban hành:{len(van_ban_nodes) - missing_ngay:>6,} / {len(van_ban_nodes):,}  ({100*(len(van_ban_nodes)-missing_ngay)/max(len(van_ban_nodes),1):.1f}%)")
    print(f"   Có link PDF:     {has_pdf:>6,} / {len(van_ban_nodes):,}  ({100*has_pdf/max(len(van_ban_nodes),1):.1f}%)")

    # Edge types
    print("\n── Phân loại Edges ──")
    edge_counter = Counter()
    for e in edges:
        edge_counter[e.get("type", "UNKNOWN")] += 1
    for et, count in sorted(edge_counter.items(), key=lambda x: -x[1]):
        print(f"   {et:<30} {count:>6,}")

    # Sample 3 records
    print("\n── 3 VanBan mẫu ──")
    for n in van_ban_nodes[:3]:
        p = n.get("properties", {})
        noi_dung = (p.get("noi_dung", "") or "")[:120].replace("\n", " ")
        print(f"\n   [{p.get('so_hieu','N/A')}] {p.get('trich_yeu','')[:60]}")
        print(f"   Loại: {p.get('loai_van_ban','')} | Hiệu lực: {p.get('tinh_trang','')}")
        print(f"   Nội dung (120 ký tự): {noi_dung}...")

    # Sample 3 edges
    print("\n── 3 Edges mẫu ──")
    for e in edges[:3]:
        print(f"   {e['source']} --[{e['type']}]--> {e['target']}")
        if e.get("properties", {}).get("mo_ta"):
            print(f"   Mo tả: {e['properties']['mo_ta']}")

    print("\n" + "=" * 55)

    # Đánh giá tổng thể
    score = 0
    if len(van_ban_nodes) > 0: score += 20
    if missing_content / max(len(van_ban_nodes), 1) < 0.3: score += 20
    if missing_so_hieu / max(len(van_ban_nodes), 1) < 0.2: score += 20
    if len(dieu_nodes) > 0: score += 20
    if len(edges) > 0: score += 20

    grade = "✅ TỐT" if score >= 80 else "⚠️  CẦN CẢI THIỆN" if score >= 40 else "❌ CẦN KIỂM TRA"
    print(f"  Đánh giá tổng thể: {grade} ({score}/100)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/graphrag_dataset.json")
    args = parser.parse_args()
    verify(args.input)
