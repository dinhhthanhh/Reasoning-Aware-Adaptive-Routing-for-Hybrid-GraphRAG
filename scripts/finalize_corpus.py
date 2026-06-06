import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def finalize_corpus(data_dir: Path, output_file: Path):
    """
    Combines processed HF and Phap Dien documents into a single corpus.
    Writes a summary of the finalized dataset.
    """
    hf_path = data_dir / "processed" / "hf_processed.jsonl"
    pd_path = data_dir / "processed" / "phapdien_processed.jsonl"
    
    if not hf_path.exists() or not pd_path.exists():
        logger.error("Processed files missing. Ensure previous steps are complete.")
        return

    logger.info(f"Merging all processed documents into {output_file}...")
    
    counts = {"hf": 0, "phapdien": 0}
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        # 1. HF documents
        with open(hf_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                f_out.write(line)
                counts["hf"] += 1
        
        # 2. Phap Dien documents
        with open(pd_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                f_out.write(line)
                counts["phapdien"] += 1

    # Generate global summary
    summary = {
        "finalized_at": datetime.now().isoformat(),
        "total_documents": counts["hf"] + counts["phapdien"],
        "breakdown": counts,
        "files": {
            "corpus": str(output_file),
            "relationships": str(data_dir / "processed" / "relationships_final.jsonl")
        }
    }
    
    summary_path = data_dir / "processed" / "corpus_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Corpus finalization complete.")
    logger.info(f"  Total Documents: {summary['total_documents']:,}")
    logger.info(f"  HF: {counts['hf']:,}")
    logger.info(f"  Phap Dien: {counts['phapdien']:,}")
    logger.info(f"  Summary saved -> {summary_path}")

if __name__ == "__main__":
    DATA_DIR = Path("data")
    FINAL_CORPUS = DATA_DIR / "processed" / "final_corpus.jsonl"
    finalize_corpus(DATA_DIR, FINAL_CORPUS)
