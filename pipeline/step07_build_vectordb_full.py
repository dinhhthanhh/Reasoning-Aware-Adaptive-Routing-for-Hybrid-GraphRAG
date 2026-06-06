import json
import os
import sys
from pathlib import Path
import yaml
from loguru import logger
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

# Add current directory to path
sys.path.append(os.getcwd())

def main():
    # 1. Load config
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.error("Config file not found at configs/config.yaml")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Initialize ChromaDB
    chroma_path = Path("data/vector_store/chroma_full")
    chroma_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing ChromaDB at {chroma_path}...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Use Vietnamese SBERT for embeddings
    model_name = "keepitreal/vietnamese-sbert"
    
    logger.info(f"Using embedding model: {model_name} (GPU/CUDA mode)")
    # Force CUDA device
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device="cuda"
    )
    
    collection = client.get_or_create_collection(
        name="legal_articles_full",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Check current count for resumption
    current_count = collection.count()
    logger.info(f"Current collection count: {current_count}. Resuming from here...")

    # 3. Stream Articles JSONL
    dataset_path = Path("data/processed/articles_full.jsonl")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Streaming and indexing articles from {dataset_path}...")
    
    batch_size = 256
    batch_docs = []
    batch_metadatas = []
    batch_ids = []
    
    with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(tqdm(f, desc="Indexing articles", total=2100569)):
            # Skip already indexed
            if i < current_count:
                continue
                
            try:
                item = json.loads(line)
                text = item.get("content", "")
                if not text:
                    continue
                
                doc_id = f"art_{i}"
                
                batch_docs.append(text)
                batch_ids.append(doc_id)
                batch_metadatas.append({
                    "law": item.get("law", ""),
                    "title": item.get("title", ""),
                    "index": i
                })
                
                if len(batch_docs) >= batch_size:
                    collection.add(
                        documents=batch_docs,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    batch_docs = []
                    batch_metadatas = []
                    batch_ids = []
                
            except Exception as e:
                logger.error(f"Error at line {i}: {e}")
                continue
        
        # Final batch
        if batch_docs:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

    logger.info(f"Indexing complete! Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main()
