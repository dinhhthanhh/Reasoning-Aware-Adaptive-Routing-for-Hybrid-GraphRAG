import os
import sys
from pathlib import Path
import yaml
from loguru import logger

# Add current directory to path
sys.path.append(os.getcwd())

from graph.neo4j_client import Neo4jClient

def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Connecting to Neo4j to clear database...")
    client = Neo4jClient(config["neo4j"])
    
    if not client.health_check():
        logger.error("Could not connect to Neo4j.")
        return

    with client._get_session() as session:
        logger.info("Deleting all nodes and relationships...")
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared.")

    client.close()

if __name__ == "__main__":
    main()
