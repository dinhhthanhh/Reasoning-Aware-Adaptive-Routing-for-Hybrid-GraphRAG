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
    chroma_path = Path("data/vector_store/chroma")
    chroma_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing ChromaDB at {chroma_path}...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Use Vietnamese SBERT for embeddings (as in config)
    emb_config = config.get("embedding", {})
    model_name = emb_config.get("model_name", "keepitreal/vietnamese-sbert")
    
    logger.info(f"Using embedding model: {model_name}")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    collection = client.get_or_create_collection(
        name="legal_docs",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Load Cleaned Dataset
    dataset_path = Path("vbpl_crawler/output/graphrag_dataset_clean.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    
    # 4. Filter and Prepare Documents (VanBan and Dieu)
    documents = []
    metadatas = []
    ids = []
    
    logger.info("Preparing documents for indexing...")
    for node in nodes:
        props = node.get("properties", {})
        
        # We index both VanBan (summary) and Dieu (content)
        if node["type"] == "VanBan":
            text = f"{props.get('trich_yeu', '')}\n{props.get('noi_dung', '')[:1000]}" # Limit snippet for VanBan
            doc_id = props.get("doc_id", node["id"])
            
            documents.append(text)
            ids.append(f"doc_{doc_id}")
            metadatas.append({
                "type": "VanBan",
                "doc_id": doc_id,
                "so_hieu": props.get("so_hieu", ""),
                "trich_yeu": props.get("trich_yeu", "")
            })
            
        elif node["type"] == "Dieu":
            text = props.get("content", "")
            if not text: continue
            
            doc_id = props.get("doc_id", "")
            article_id = node["id"]
            
            documents.append(text)
            ids.append(article_id)
            metadatas.append({
                "type": "Dieu",
                "doc_id": doc_id,
                "article_id": article_id,
                "title": props.get("title", "")
            })

    # 5. Batch Indexing
    batch_size = 100
    logger.info(f"Indexing {len(documents)} documents in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(documents), batch_size)):
        end = i + batch_size
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )

    logger.info(f"Indexing complete! Total documents in collection: {collection.count()}")

    # 6. Test Query
    test_query = "quyền sở hữu đất đai"
    logger.info(f"Testing search with query: '{test_query}'")
    results = collection.query(
        query_texts=[test_query],
        n_results=3
    )
    
    for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        score = 1 - dist # Cosine similarity
        logger.info(f"  {i+1}. [{score:.2f}] {meta.get('doc_id')} - {meta.get('title') or meta.get('trich_yeu')}")

if __name__ == "__main__":
    main()
