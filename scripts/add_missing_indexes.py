"""Add missing Neo4j indexes to the live graph (no rebuild required).

The architectural audit (see audit/performance_audit.md) confirmed three indexes
are missing from the build scripts. Their absence causes full node scans on every
article lookup and multi-hop expansion. This script creates them in-place against
the live Neo4j instance — it changes no data and requires no rebuild.

Usage:
    python scripts/add_missing_indexes.py [--config configs/config.yaml]

Safe to run repeatedly: all statements use IF NOT EXISTS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.neo4j_client import Neo4jClient  # noqa: E402
from loguru import logger  # noqa: E402


# (index_name, cypher) — every statement is idempotent.
MISSING_INDEXES: list[tuple[str, str]] = [
    (
        "article_id_index",
        "CREATE INDEX article_id_index IF NOT EXISTS "
        "FOR (a:LegalArticle) ON (a.article_id)",
    ),
    (
        "article_doc_id_index",
        "CREATE INDEX article_doc_id_index IF NOT EXISTS "
        "FOR (a:LegalArticle) ON (a.doc_id)",
    ),
    (
        "chunk_id_index",
        "CREATE INDEX chunk_id_index IF NOT EXISTS "
        "FOR (c:VectorChunk) ON (c.chunk_id)",
    ),
]


def add_indexes(client: Neo4jClient) -> None:
    with client._get_driver().session(database=client.database) as session:
        for name, cypher in MISSING_INDEXES:
            try:
                session.run(cypher)
                logger.info("Ensured index: {}", name)
            except Exception as exc:  # pragma: no cover - depends on live DB
                logger.error("Failed to create index {}: {}", name, exc)


# The three target (label, property) schemas we want covered by *some* index.
TARGET_SCHEMAS: list[tuple[str, str]] = [
    ("LegalArticle", "article_id"),
    ("LegalArticle", "doc_id"),
    ("VectorChunk", "chunk_id"),
]


def verify_indexes(client: Neo4jClient) -> None:
    """Confirm each target (label, property) schema is covered by an index.

    Coverage may come from a named index OR from a uniqueness-constraint-backed
    range index. We verify by schema, not by name, because constraint-backed
    indexes use auto-generated names (e.g. constraint_81851638).
    """
    rows = client.query("SHOW INDEXES")
    logger.info("Total indexes in database: {}", len(rows))

    covered: dict[tuple[str, str], str] = {}
    for row in rows:
        labels = row.get("labelsOrTypes") or []
        props = row.get("properties") or []
        name = str(row.get("name"))
        for label in labels:
            for prop in props:
                covered.setdefault((label, prop), name)

    for label, prop in TARGET_SCHEMAS:
        backing = covered.get((label, prop))
        status = f"COVERED by '{backing}'" if backing else "NOT COVERED"
        logger.info("  ({}.{}) -> {}", label, prop, status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: {}", config_path)
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = Neo4jClient(config["neo4j"])
    if not client.verify_connection():
        logger.error("Could not connect to Neo4j. Indexes not created.")
        client.close()
        return

    try:
        add_indexes(client)
        verify_indexes(client)
        logger.info("Index creation complete. No data was modified.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
