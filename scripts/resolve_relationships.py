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

def resolve_relationships(data_dir: Path, output_file: Path):
    """
    Validates relationships against existing processed documents.
    """
    processed_hf = data_dir / "processed" / "hf_processed.jsonl"
    all_rels_path = data_dir / "huggingface" / "relationships" / "relationships_all.jsonl"
    
    if not processed_hf.exists() or not all_rels_path.exists():
        logger.error("Required files for relationship resolution not found.")
        return

    # 1. Load available IDs from processed documents
    logger.info("Loading valid document IDs...")
    valid_ids = set()
    with open(processed_hf, "r", encoding="utf-8") as f:
        for line in f:
            try:
                valid_ids.add(json.loads(line).get("doc_id"))
            except:
                continue
    logger.info(f"Total valid IDs: {len(valid_ids):,}")

    # 2. Filter relationships
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count_valid = 0
    count_invalid = 0
    
    logger.info(f"Filtering relationships -> {output_file}")
    with open(all_rels_path, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in tqdm.tqdm(f_in, desc="Validating links"):
            try:
                rel = json.loads(line)
                # Convert to string to match our document IDs
                src = str(rel.get("doc_id"))
                tgt = str(rel.get("other_doc_id"))
                
                if src in valid_ids and tgt in valid_ids:
                    # Keep formatted consistently
                    clean_rel = {
                        "source": src,
                        "target": tgt,
                        "type": rel.get("relationship", "Unknown")
                    }
                    f_out.write(json.dumps(clean_rel, ensure_ascii=False) + "\n")
                    count_valid += 1
                else:
                    count_invalid += 1
            except:
                continue

    logger.info("Relationship resolution complete.")
    logger.info(f"  Valid links: {count_valid:,}")
    logger.info(f"  Invalid (orphan) links: {count_invalid:,}")
    logger.info(f"  Total processed: {count_valid + count_invalid:,}")

if __name__ == "__main__":
    DATA_DIR = Path("data")
    OUT_FILE = DATA_DIR / "processed" / "relationships_final.jsonl"
    resolve_relationships(DATA_DIR, OUT_FILE)
