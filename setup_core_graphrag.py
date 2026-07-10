import json
import os
import shutil
from pathlib import Path

workspace = Path("data/graphrag_workspace")
input_dir = workspace / "input"
output_dir = workspace / "output"
cache_dir = workspace / "cache"

print("Cleaning up old data...")
for d in [input_dir, output_dir, cache_dir]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

target_files = [
    "data/phapdien/chu_de_009.json", # Dân sự
    "data/phapdien/chu_de_016.json"  # Hình sự
]

total_extracted = 0

for file_path in target_files:
    print(f"Reading {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        topic_name = data.get("ten_chu_de", "unknown")
        for demuc in data.get("de_muc_list", []):
            for dieu in demuc.get("dieu_list", []):
                if "noi_dung" in dieu and len(dieu["noi_dung"]) > 50:
                    doc_path = input_dir / f"doc_{total_extracted}.txt"
                    with open(doc_path, "w", encoding="utf-8") as out_f:
                        out_f.write(f"Chủ đề: {topic_name}\n")
                        out_f.write(f"Tiêu đề: {dieu.get('tieu_de', '')}\n\n")
                        out_f.write(f"{dieu['noi_dung']}\n")
                    total_extracted += 1
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

print(f"Successfully extracted {total_extracted} articles to {input_dir}")
