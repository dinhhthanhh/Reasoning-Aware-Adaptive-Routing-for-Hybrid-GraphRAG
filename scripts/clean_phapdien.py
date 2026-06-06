import json
import logging
from pathlib import Path
import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def flatten_phapdien(input_file: Path, output_file: Path):
    """
    Converts hierarchical Phap Dien JSON to flat Article-level JSONL.
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading hierarchical Phap Dien from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Flattening into articles...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    count_articles = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for cd_idx, chu_de in enumerate(data):
            cd_name = chu_de.get("ten_chu_de", "Unknown")
            
            for dm_idx, de_muc in enumerate(chu_de.get("de_muc_list", [])):
                dm_name = de_muc.get("ten_de_muc", "Unknown")
                
                for d_idx, dieu in enumerate(de_muc.get("dieu_list", [])):
                    so_dieu = dieu.get("so_dieu", "")
                    tieu_de = dieu.get("tieu_de", "")
                    noi_dung = dieu.get("noi_dung", "")
                    
                    # Clean title
                    full_title = f"{so_dieu} {tieu_de}".strip()
                    
                    # Create a deterministic ID
                    pd_id = f"pd_{cd_idx:03d}_{dm_idx:03d}_{d_idx:04d}"
                    
                    record = {
                        "doc_id": pd_id,
                        "title": full_title,
                        "type": "Điều (Pháp điển)",
                        "theme": cd_name,
                        "topic": dm_name,
                        "content_markdown": f"### {full_title}\n\n{noi_dung}",
                        "source": "Pháp Điển",
                        "raw_metadata": {
                            "chu_de": cd_name,
                            "de_muc": dm_name,
                            "so_dieu": so_dieu
                        }
                    }
                    
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count_articles += 1

    logger.info(f"Phap Dien flattening complete. Total articles: {count_articles:,}")

if __name__ == "__main__":
    IN_FILE = Path("data/phapdien/phapdien_all.json")
    OUT_FILE = Path("data/processed/phapdien_processed.jsonl")
    flatten_phapdien(IN_FILE, OUT_FILE)
