"""Check Neo4j graph quality for GraphRAG experiments.

This script is intentionally read-only. It reports graph size, label and
relationship distributions, basic connectivity, and sample multi-hop context
for a few queries.

Usage:
    python scripts/check_neo4j_graph_quality.py --config configs/config.yaml
    python scripts/check_neo4j_graph_quality.py --config configs/config_hotpot.yaml --query "Were Scott Derrickson and Ed Wood of the same nationality?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.neo4j_client import Neo4jClient


DEFAULT_QUERIES = {
    "vi": [
        "So sánh quy định về điều kiện kết hôn và ly hôn trong pháp luật Việt Nam.",
        "Cơ quan nào có thẩm quyền xử phạt trong trường hợp vi phạm lao động?",
    ],
    "en": [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What government position was held by the woman who portrayed Corliss Archer in Kiss and Tell?",
    ],
}


def _run_scalar(client: Neo4jClient, cypher: str, params: dict[str, Any] | None = None) -> Any:
    rows = client.query(cypher, params or {})
    if not rows:
        return None
    return next(iter(rows[0].values()))


def inspect_graph(client: Neo4jClient) -> dict[str, Any]:
    """Collect read-only graph quality statistics."""
    stats: dict[str, Any] = {}

    stats["node_count"] = _run_scalar(client, "MATCH (n) RETURN count(n) AS count") or 0
    stats["relationship_count"] = _run_scalar(client, "MATCH ()-[r]->() RETURN count(r) AS count") or 0

    stats["labels"] = client.query(
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
        LIMIT 20
        """
    )

    stats["relationship_types"] = client.query(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
        LIMIT 20
        """
    )

    stats["avg_degree"] = _run_scalar(
        client,
        """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        RETURN avg(degree) AS avg_degree
        """,
    ) or 0.0

    stats["isolated_nodes"] = _run_scalar(
        client,
        """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) AS count
        """,
    ) or 0

    stats["sample_nodes"] = client.query(
        """
        MATCH (n)
        RETURN labels(n) AS labels,
               coalesce(n.name, n.title, n.doc_id, "Unknown") AS name,
               coalesce(n.type, "") AS type
        LIMIT 10
        """
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Neo4j graph quality check")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--query", action="append", default=None, help="Sample query to test multi-hop context")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = Neo4jClient(config["neo4j"])
    try:
        if not client.verify_connection():
            raise RuntimeError("Cannot connect to Neo4j with the selected config")

        stats = inspect_graph(client)
        language = config.get("language", "vi")
        queries = args.query or DEFAULT_QUERIES.get(language, DEFAULT_QUERIES["vi"])

        stats["sample_contexts"] = []
        for query in queries:
            context = client.get_multi_hop_context(query, top_k=3)
            stats["sample_contexts"].append({
                "query": query,
                "context_preview": context[:2000],
                "context_length": len(context),
            })

        print(json.dumps(stats, ensure_ascii=False, indent=2))

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Graph quality report saved to {}", out)
    finally:
        client.close()


if __name__ == "__main__":
    main()
