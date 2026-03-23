import json
import os
import sys
from pathlib import Path
import yaml
from loguru import logger
from tqdm import tqdm

# Add current directory to path so we can import internal modules
sys.path.append(os.getcwd())

from graph.neo4j_client import Neo4jClient

def sanitize_properties(props):
    """Sanitize properties for Neo4j (no nested dicts/mixed lists)."""
    sanitized = {}
    for k, v in props.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            sanitized[k] = v
        elif isinstance(v, list):
            if all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
                sanitized[k] = v
            else:
                sanitized[k] = json.dumps(v, ensure_ascii=False)
        else:
            sanitized[k] = json.dumps(v, ensure_ascii=False)
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
        logger.error("Could not connect to Neo4j. Please check if Neo4j is running and credentials are correct.")
        return

    # 3. Load Cleaned Dataset
    dataset_path = Path("vbpl_crawler/output/graphrag_dataset_clean.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    logger.info(f"Loaded {len(nodes)} nodes and {len(edges)} edges.")

    # 4. Insert Nodes
    logger.info("Inserting nodes into Neo4j...")
    formatted_nodes = []
    for node in nodes:
        name = node["properties"].get("trich_yeu") or node["properties"].get("doc_id") or node["id"]
        formatted_nodes.append({
            "name": name,
            "type": node["type"],
            "properties": sanitize_properties(node["properties"])
        })
    
    client.batch_insert_nodes(formatted_nodes)

    # 5. Insert Relations
    logger.info("Inserting relations into Neo4j...")
    id_to_name = {node["id"]: (node["properties"].get("trich_yeu") or node["properties"].get("doc_id") or node["id"]) for node in nodes}
    
    formatted_edges = []
    for edge in edges:
        source_name = id_to_name.get(edge["source"])
        target_name = id_to_name.get(edge["target"])
        
        if source_name and target_name:
            formatted_edges.append({
                "source": source_name,
                "target": target_name,
                "relation_type": edge["type"],
                "properties": sanitize_properties(edge.get("properties", {}))
            })

    client.batch_insert_relations(formatted_edges)

    # 6. Finalize
    logger.info("Creating indexes...")
    client.create_indexes()
    
    stats = client.get_stats()
    logger.info(f"Build complete! Stats: {stats}")
    
    client.close()

if __name__ == "__main__":
    main()
