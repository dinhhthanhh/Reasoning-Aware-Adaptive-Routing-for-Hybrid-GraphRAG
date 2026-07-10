"""Embed Semantic Nodes in Neo4j.

Extracts semantic nodes (LEGAL_CONCEPT, ACTOR, RIGHT, etc.),
generates embeddings using SentenceTransformer, and writes them
back to Neo4j to support Semantic Vector Routing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from graph.neo4j_client import Neo4jClient

CONFIG = ROOT / "configs/config.yaml"
TARGET_DATABASE = "phapdien"

def embed_nodes(client: Neo4jClient, model: SentenceTransformer, batch_size: int = 256):
    """Fetch nodes without embeddings, embed them, and update Neo4j."""
    
    labels = ["LEGAL_CONCEPT", "ACTOR", "RIGHT", "OBLIGATION", "CONDITION", "PROCEDURE", "PENALTY", "AUTHORITY"]
    
    total_embedded = 0
    with client._get_driver().session(database=client.database) as session:
        for label in labels:
            logger.info(f"Processing label: {label}")
            
            # Create vector index if not exists
            index_query = f"""
            CREATE VECTOR INDEX semantic_{label.lower()}_emb IF NOT EXISTS
            FOR (n:{label}) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }}}}
            """
            try:
                session.run(index_query)
            except Exception as e:
                logger.debug(f"Index creation for {label} skipped: {e}")

            # Fetch batch
            while True:
                fetch_query = f"""
                MATCH (n:{label})
                WHERE n.embedding IS NULL AND n.name IS NOT NULL
                RETURN elementId(n) AS id, n.name AS text
                LIMIT $limit
                """
                records = session.run(fetch_query, limit=batch_size).data()
                
                if not records:
                    break
                    
                ids = [r["id"] for r in records]
                texts = [r["text"] for r in records]
                
                # Encode
                logger.info(f"Encoding {len(texts)} {label} nodes...")
                embeddings = model.encode(texts, convert_to_numpy=True).tolist()
                
                # Update
                update_query = f"""
                UNWIND $batch AS row
                MATCH (n:{label})
                WHERE elementId(n) = row.id
                SET n.embedding = row.embedding
                """
                batch_data = [{"id": ids[i], "embedding": embeddings[i]} for i in range(len(ids))]
                session.run(update_query, batch=batch_data)
                
                total_embedded += len(ids)
                logger.info(f"Embedded {total_embedded} nodes total.")
                
    logger.info("Semantic Node Embedding complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Database
    neo_cfg = dict(cfg["neo4j"])
    neo_cfg["database"] = TARGET_DATABASE
    client = Neo4jClient(neo_cfg)
    
    if not client.verify_connection():
        logger.error(f"Cannot connect to Neo4j database '{TARGET_DATABASE}'.")
        sys.exit(1)

    # Embedding Model
    model_name = cfg.get("embedding", {}).get("model_name", "BAAI/bge-m3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    try:
        embed_nodes(client, model, batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
