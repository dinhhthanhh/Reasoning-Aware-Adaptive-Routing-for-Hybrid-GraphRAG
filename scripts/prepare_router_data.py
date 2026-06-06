"""Prepare training data for English Router Model from HotpotQA.

Extracts a subset of the HotpotQA evaluation dataset and uses heuristics
to map queries into the 3 routing labels to ensure a balanced dataset
for the XGBoost model.
"""

import json
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def prepare_router_data(input_file: Path, output_file: Path, max_samples: int = 2000):
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Reading from {input_file}...")
    
    samples = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
                
            item = json.loads(line.strip())
            query = item["query"]
            words = query.split()
            word_count = len(words)
            
            # Heuristic labeling to ensure all classes are present for XGBoost
            query_lower = query.lower()
            if word_count <= 10 and "both" not in query_lower and "and" not in query_lower:
                routing_label = "dense_retrieval"
                difficulty = 0.2
                hop_count = 1
            elif word_count > 16 or "both" in query_lower or "first" in query_lower:
                routing_label = "hybrid_reasoning"
                difficulty = 0.9
                hop_count = 3
            else:
                routing_label = "graph_traversal"
                difficulty = 0.6
                hop_count = 2
                
            samples.append({
                "question": query,
                "routing_label": routing_label,
                "hop_count": hop_count,
                "is_cross_doc": (hop_count > 1),
                "difficulty": difficulty
            })
            
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        
    # Print distribution
    dist = {}
    for s in samples:
        dist[s["routing_label"]] = dist.get(s["routing_label"], 0) + 1
        
    logger.info(f"Successfully generated {len(samples)} training samples.")
    logger.info(f"Distribution: {dist}")
    logger.info(f"Saved to {output_file}")

if __name__ == "__main__":
    input_path = Path("data/en_benchmark/processed/hotpot_eval.jsonl")
    output_path = Path("data/en_benchmark/router_training/train.json")
    prepare_router_data(input_path, output_path, max_samples=2000)
