import argparse
import json
import logging
from pathlib import Path
from statistics import mean
import sys
import time

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics.id_normalizer import compute_hit_at_k, compute_mrr
import chromadb
from vector_store.safe_embedding import SafeEmbeddingFunction
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_retrieval(collection, queries, gold_articles, emb_fn, top_k=10):
    start = time.perf_counter()
    embeddings = emb_fn(queries)
    emb_time = time.perf_counter() - start
    
    start = time.perf_counter()
    results = collection.query(
        query_embeddings=embeddings,
        n_results=top_k
    )
    query_time = time.perf_counter() - start
    
    r5 = r10 = mrr = 0
    total = len(queries)
    
    for i in range(total):
        retrieved_ids = results["ids"][i]
        gold = gold_articles[i]
        
        if compute_hit_at_k(retrieved_ids, gold, 5, mode="strict"):
            r5 += 1
        if compute_hit_at_k(retrieved_ids, gold, 10, mode="strict"):
            r10 += 1
            
        mrr += compute_mrr(retrieved_ids, gold, mode="strict")
        
    return {
        "recall_at_5": r5 / total,
        "recall_at_10": r10 / total,
        "mrr": mrr / total,
        "emb_time_sec": emb_time,
        "query_time_sec": query_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="qa_pipeline/data/legal_strict/test_covered.json")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    test_data = json.load(open(args.test_file, encoding="utf-8"))
    queries = [item.get("question", item.get("query", "")) for item in test_data]
    gold_arts = [item.get("relevant_articles", []) for item in test_data]
    
    # Init embedding function
    emb_config = config.get("embedding", {})
    model_name = emb_config.get("model_name", "microsoft/Harrier-OSS-v1-0.6B")
    device = emb_config.get("device", "cuda")
    emb_fn = SafeEmbeddingFunction(model_name=model_name, device=device, max_seq_length=512)
    
    # Init Chroma
    chroma_path = Path("data/vector_store/chroma_full")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    collections_to_test = ["legal_docs", "chroma_full_v2"]
    
    print(f"=== Evaluating on {len(test_data)} covered queries ===")
    
    for coll_name in collections_to_test:
        try:
            collection = client.get_collection(coll_name)
            logger.info(f"Evaluating collection: {coll_name} (docs: {collection.count()})")
            
            # Batch queries to avoid OOM
            batch_size = 100
            total_r5 = 0
            total_r10 = 0
            total_mrr = 0
            
            for i in tqdm(range(0, len(queries), batch_size)):
                batch_q = queries[i:i+batch_size]
                batch_g = gold_arts[i:i+batch_size]
                
                res = evaluate_retrieval(collection, batch_q, batch_g, emb_fn, top_k=10)
                n_batch = len(batch_q)
                total_r5 += res["recall_at_5"] * n_batch
                total_r10 += res["recall_at_10"] * n_batch
                total_mrr += res["mrr"] * n_batch
                
            n = len(queries)
            print(f"\nResults for {coll_name}:")
            print(f"  Recall@5 : {total_r5/n:.4f}")
            print(f"  Recall@10: {total_r10/n:.4f}")
            print(f"  MRR      : {total_mrr/n:.4f}")
            print("-" * 40)
            
        except ValueError:
            logger.warning(f"Collection {coll_name} not found. Skipping.")
        except Exception as e:
            logger.error(f"Error evaluating {coll_name}: {e}")

if __name__ == "__main__":
    main()
