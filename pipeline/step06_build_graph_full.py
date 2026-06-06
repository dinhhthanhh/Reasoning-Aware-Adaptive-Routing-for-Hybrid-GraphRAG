import json
import os
import sys
from pathlib import Path
import yaml
from loguru import logger
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.getcwd())

from graph.neo4j_client import Neo4jClient

def sanitize_properties(props):
    """Sanitize properties for Neo4j."""
    sanitized = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized

def main():
    # 1. Load config
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.error("Config file not found at configs/config.yaml")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Initialize Neo4j Client
    logger.info("Connecting to Neo4j...")
    client = Neo4jClient(config["neo4j"])
    
    if not client.health_check():
        logger.error("Could not connect to Neo4j.")
        return

    # 3. Clean existing nodes (Optional, but good for full rebuild)
    # logger.warning("Cleaning existing nodes in Neo4j...")
    # client.execute_query("MATCH (n) DETACH DELETE n")

    # 4. Stream Articles JSONL
    dataset_path = Path("data/processed/articles_full.jsonl")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Streaming and indexing nodes into Neo4j from {dataset_path}...")
    
    batch_size = 1000
    nodes_batch = []
    
    with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(tqdm(f, desc="Ingesting nodes")):
            try:
                item = json.loads(line)
                law_title = item.get("law", "Unknown Law")
                article_title = item.get("title", "Unknown Article")
                content = item.get("content", "")
                
                # Node for Article (Dieu)
                nodes_batch.append({
                    "name": f"{law_title} - {article_title}",
                    "type": "Dieu",
                    "properties": sanitize_properties({
                        "law": law_title,
                        "title": article_title,
                        "content": content[:5000], # Limit content size in graph
                        "full_id": f"art_{i}"
                    })
                })
                
                if len(nodes_batch) >= batch_size:
                    client.batch_insert_nodes(nodes_batch)
                    nodes_batch = []
                    
            except Exception as e:
                logger.error(f"Error at line {i}: {e}")
                continue
                
        # Final batch
        if nodes_batch:
            client.batch_insert_nodes(nodes_batch)

    logger.info("Creating indexes...")
    client.create_indexes()
    
    stats = client.get_stats()
    logger.info(f"Build complete! Stats: {stats}")
    client.close()

if __name__ == "__main__":
    main()
