import json
import glob
from pathlib import Path
import sys

# Thêm đường dẫn project
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics.id_normalizer import compute_hit_at_k, compute_mrr
from scripts.run_comparison_eval import get_gold_articles

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval_results")
    ap.add_argument("--test-path", default="qa_pipeline/data/phapdien_strict_backup/test.json")
    args = ap.parse_args()
    
    eval_dir = Path(args.eval_dir)
    json_files = glob.glob(str(eval_dir / "per_sample_*.json"))
    
    if not json_files:
        print(f"No per_sample_*.json files found in {eval_dir}")
        return
        
    with open(args.test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    for jf in json_files:
        config_name = Path(jf).stem.replace("per_sample_", "")
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not data:
            continue
            
        hit1, hit3, hit5, mrr = 0, 0, 0, 0
        n = len(data)
        for sample in data:
            idx = sample["idx"]
            gold_articles = get_gold_articles(test_data[idx])
            sources = sample.get("sources", [])
            
            hit1 += compute_hit_at_k(sources, gold_articles, k=1, mode="article")
            hit3 += compute_hit_at_k(sources, gold_articles, k=3, mode="article")
            hit5 += compute_hit_at_k(sources, gold_articles, k=5, mode="article")
            mrr += compute_mrr(sources, gold_articles, mode="article")
            
        print(f"[{config_name}] Hit@1={hit1/n:.3f} | Hit@3={hit3/n:.3f} | Hit@5={hit5/n:.3f} | MRR={mrr/n:.3f}")

if __name__ == "__main__":
    main()
