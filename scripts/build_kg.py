"""Script to build the knowledge graph and FAISS index.

Usage:
    python scripts/build_kg.py [--config PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from loguru import logger

from graph.build_kg import build_knowledge_graph
from graph.neo4j_client import Neo4jClient
from ner.vi_ner import ViNER
from vector_store.vector_retriever import VectorRetriever


def main() -> None:
    """Build knowledge graph and FAISS index from crawled documents."""
    parser = argparse.ArgumentParser(
        description="Build knowledge graph and FAISS index from legal documents."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--skip-kg",
        action="store_true",
        help="Skip knowledge graph building (Neo4j)",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip FAISS index building",
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger.add(
        "logs/build_kg_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    kg_dir = Path(config["data"]["kg_dir"])

    # Ensure dirs exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    kg_dir.mkdir(parents=True, exist_ok=True)

    # Copy raw docs to processed (as-is for now)
    if not any(processed_dir.glob("*.json")):
        import shutil
        for f in raw_dir.glob("*.json"):
            shutil.copy2(f, processed_dir / f.name)
        logger.info("Copied {} raw docs to processed dir", len(list(raw_dir.glob("*.json"))))

    # Build Knowledge Graph
    if not args.skip_kg:
        logger.info("Building Knowledge Graph...")
        neo4j_client = Neo4jClient(config["neo4j"])

        if not neo4j_client.health_check():
            logger.error(
                "Neo4j is not reachable at {}. Please start Neo4j first.",
                config["neo4j"]["uri"],
            )
            print("ERROR: Neo4j is not running. Please start Neo4j Community Edition.")
        else:
            ner_model = ViNER(config["ner"])
            report = build_knowledge_graph(
                processed_dir=str(processed_dir),
                neo4j_client=neo4j_client,
                ner_model=ner_model,
                kg_export_dir=str(kg_dir),
            )
            print(f"\nKG Build Report:")
            print(f"  Documents processed: {report.documents_processed}")
            print(f"  Entities extracted:  {report.entities_extracted}")
            print(f"  Triples extracted:   {report.triples_extracted}")
            print(f"  Nodes inserted:      {report.nodes_inserted}")
            print(f"  Relations inserted:  {report.relations_inserted}")
            if report.errors:
                print(f"  Errors: {len(report.errors)}")

            # Deduplicate
            dedup_count = neo4j_client.deduplicate()
            print(f"  Deduplicated nodes:  {dedup_count}")

            stats = neo4j_client.get_stats()
            print(f"\nGraph Stats: {stats['node_count']} nodes, {stats['relation_count']} relations")
            neo4j_client.close()

    # Build FAISS Index
    if not args.skip_faiss:
        logger.info("Building FAISS index...")
        retriever = VectorRetriever(config)
        chunk_count = retriever.index_documents(str(processed_dir))
        print(f"\nFAISS Index: {chunk_count} chunks indexed")

    print("\nBuild complete!")


if __name__ == "__main__":
    main()
