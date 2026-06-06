"""Download full validation datasets for English benchmarks.

Downloads HotpotQA, TriviaQA, and Natural Questions (as proxies for single/multi-hop).
Saves queries and ground truth answers for evaluation, along with full context
for vector database indexing.
"""

import json
import logging
from pathlib import Path
import yaml
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_hotpot(processed_dir: Path):
    output_file = processed_dir / "hotpot_full.jsonl"
    eval_file = processed_dir / "hotpot_eval.jsonl"
    
    logger.info("Downloading HotpotQA (validation split)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    with open(output_file, "w", encoding="utf-8") as f_out, \
         open(eval_file, "w", encoding="utf-8") as f_eval:
        for item in dataset:
            # 1. Save Evaluation Query & Ground Truth
            f_eval.write(json.dumps({
                "id": f"hotpot_{item['id']}",
                "query": item["question"],
                "ground_truth": item["answer"],
                "level": item["level"]
            }, ensure_ascii=False) + "\n")
            
            # 2. Save Context for VectorDB & Graph
            titles = item.get("context", {}).get("title", [])
            sentences = item.get("context", {}).get("sentences", [])
            
            for i, (title, sents) in enumerate(zip(titles, sentences)):
                content = " ".join(sents)
                doc_id = f"hotpot_{item['id']}_{i}"
                doc = {
                    "doc_id": doc_id,
                    "title": title,
                    "type": "Wikipedia",
                    "source": "HotpotQA",
                    "content": content,
                    "content_markdown": content,
                    "issue_date": "2024-01-01"
                }
                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Saved HotpotQA context to {output_file} and eval queries to {eval_file}")

def main():
    config_path = Path("configs/config_en.yaml")
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config.get("data", {}).get("processed_dir", "data/en_benchmark/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the datasets
    process_hotpot(processed_dir)
    # Additional datasets (TriviaQA, etc.) can be added here following the same pattern
    logger.info("Benchmark datasets downloaded and prepared for evaluation.")

if __name__ == "__main__":
    main()
