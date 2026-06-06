import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import tqdm
import unicodedata
import chromadb
from chromadb.utils import embedding_functions
from vector_store.safe_embedding import SafeEmbeddingFunction

# Setup logging
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
    
    # Target GPU for RTX 4060 Ti
    device = emb_config.get("device", "cuda")
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Loading embedding model: {model_name} on {device} (max_length={max_length})")
    return SafeEmbeddingFunction(
        model_name=model_name,
        device=device,
        max_seq_length=max_length
    )

def index_jsonl(file_path: Path, collection, batch_size=500):
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return

    documents = []
    metadatas = []
    ids = []
    source_prefix = file_path.stem
    
    # We use streaming to avoid OOM on large 6GB files
    with open(file_path, "r", encoding="utf-8") as f:
        # User requested to avoid double open for counting. 
        # We'll use tqdm without total or with manual estimation if needed.
        for line in tqdm.tqdm(f, desc=f"Indexing {file_path.name}", unit="line"):
            try:
                line = line.strip()
                if not line: continue
                
                item = json.loads(line)
                doc_id = item.get("doc_id")
                content = item.get("content_markdown", "").strip()
                
                if doc_id is None:
                    continue
                
                if not content:
                    continue
                
                # Chunking: take first 8000 chars and normalize
                text_to_index = unicodedata.normalize("NFC", content[:8000])
                text_to_index = " ".join(text_to_index.split())  # remove control chars and extra whitespace
                
                documents.append(text_to_index)
                # Prefixed ID to avoid collisions
                ids.append(f"{source_prefix}_{doc_id}")
                
                metadatas.append({
                    "title": str(item.get("title", "")),
                    "type": str(item.get("type", "")),
                    "source": str(item.get("source", "Unknown")),
                    "authority": str(item.get("authority", "") or "Unknown")
                })
                
                if len(documents) >= batch_size:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents, metadatas, ids = [], [], []
            except json.JSONDecodeError as e:
                logger.warning(f"Malformed JSON in {file_path.name}: {e}")
            except Exception as e:
                logger.error(f"Error indexing line: {e}")
        
        # Final batch
        if documents:
            try:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            except Exception as e:
                logger.error(f"Error adding final batch: {e}")

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    try:
        config_path = Path(args.config)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        is_en = "config_en" in args.config
        data_dir = config.get("data", {}).get("processed_dir", "data/processed")

        chroma_path = Path(config["chroma"]["path"])
        # Fix: ensure chroma_path itself (as a dir) is created
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Opening ChromaDB at {chroma_path}...")
        client = chromadb.PersistentClient(path=str(chroma_path))
        
        collection_name = config["chroma"]["collection_name"]
        batch_size = config["embedding"].get("batch_size", 500)
        
        try:
            client.delete_collection(collection_name)
            logger.warning(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass # Collection probably doesn't exist
        
        emb_fn = get_embedding_function(config)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )

        if is_en:
            index_jsonl(Path(data_dir) / "hotpot_full.jsonl", collection, batch_size=batch_size)
        else:
            index_jsonl(Path(data_dir) / "hf_processed.jsonl", collection, batch_size=batch_size)
            index_jsonl(Path(data_dir) / "phapdien_processed.jsonl", collection, batch_size=batch_size)

        logger.info(f"Indexing complete. Total docs: {collection.count():,}")
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()
