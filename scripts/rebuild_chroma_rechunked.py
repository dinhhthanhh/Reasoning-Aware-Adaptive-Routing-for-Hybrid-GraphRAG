import json
import logging
import sys
import time
import traceback
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import tqdm
import unicodedata
import chromadb
from vector_store.safe_embedding import SafeEmbeddingFunction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def get_embedding_function(config):
    import torch
    emb_config = config.get("embedding", {})
    model_name = emb_config.get("model_name", "microsoft/Harrier-OSS-v1-0.6B")
    max_length = emb_config.get("max_length", 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading embedding model: {model_name} on {device}")
    return SafeEmbeddingFunction(model_name=model_name, device=device, max_seq_length=max_length)

def flush_batch(collection, emb_fn, documents, metadatas, ids):
    try:
        embeddings = emb_fn(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        logger.error(f"Batch index failed: {e}")
        raise

def main():
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    chroma_path = root / config["chroma"]["path"]
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    # We will rebuild into chroma_full_v2_rechunked
    collection_name = "chroma_full_v2_rechunked"
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection {collection_name}")
    except:
        pass
        
    emb_fn = get_embedding_function(config)
    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    batch_size = config.get("embedding", {}).get("batch_size", 96)
    chunk_lengths = []
    seen_ids = set()
    total_indexed = 0
    duplicates_resolved = 0
    started_at = time.perf_counter()
    
    for file_name in ["hf_rechunked.jsonl", "pd_rechunked.jsonl"]:
        file_path = root / "data" / "processed" / file_name
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Indexing {file_name}")
        documents, metadatas, ids = [], [], []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc=f"Indexing {file_name}"):
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    
                    # Core metadata
                    canonical_id = item.get("canonical_id")
                    if not canonical_id: continue
                    
                    content = item.get("content", "").strip()
                    if not content or len(content) < 50: continue  # Skip micro-chunks
                    
                    text_to_index = unicodedata.normalize("NFC", content[:8000])
                    text_to_index = " ".join(text_to_index.split())
                    
                    chunk_lengths.append(len(text_to_index))
                    
                    # Deduplicate IDs: if same canonical_id appears again, append suffix
                    final_id = canonical_id
                    if final_id in seen_ids:
                        counter = 2
                        while f"{canonical_id}_{counter}" in seen_ids:
                            counter += 1
                        final_id = f"{canonical_id}_{counter}"
                        duplicates_resolved += 1
                    seen_ids.add(final_id)
                    
                    documents.append(text_to_index)
                    ids.append(final_id)
                    metadatas.append({
                        "law_number": str(item.get("law_number", "")),
                        "article_number": str(item.get("article_number", "")),
                        "canonical_id": str(canonical_id),  # Original, not deduped
                        "source": str(item.get("source", "")),
                        "has_canonical_id": str(item.get("has_canonical_id", False))
                    })
                    
                    if len(documents) >= batch_size:
                        flush_batch(collection, emb_fn, documents, metadatas, ids)
                        total_indexed += len(documents)
                        documents, metadatas, ids = [], [], []
                        if total_indexed % 1000 < batch_size:
                            logger.info(f"Progress: {total_indexed} indexed | {duplicates_resolved} dups resolved | {time.perf_counter()-started_at:.0f}s")
                        
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    
            if documents:
                flush_batch(collection, emb_fn, documents, metadatas, ids)
                total_indexed += len(documents)
    
    logger.info(f"Duplicates resolved: {duplicates_resolved}")
                
    logger.info(f"Rebuild complete! Total vectors: {collection.count()}")
    
    if chunk_lengths:
        arr = np.array(chunk_lengths)
        logger.info(f"--- Chunk Length Stats ---")
        logger.info(f"Mean: {np.mean(arr):.1f}")
        logger.info(f"p50:  {np.percentile(arr, 50):.1f}")
        logger.info(f"p90:  {np.percentile(arr, 90):.1f}")
        logger.info(f"p99:  {np.percentile(arr, 99):.1f}")
        logger.info(f"Max:  {np.max(arr)}")

if __name__ == "__main__":
    main()
