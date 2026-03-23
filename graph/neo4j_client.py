"""Neo4j graph database client.

Provides connection management, batch node/relation insertion,
index creation, neighbor querying, and deduplication for the
Vietnamese legal knowledge graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from neo4j import GraphDatabase, Driver, Session


class Neo4jClient:
    """Neo4j driver wrapper for knowledge graph operations.

    Handles connection lifecycle, batch CRUD, indexing,
    and graph traversal queries.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Neo4j client.

        Args:
            config: Neo4j config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config["neo4j"]

        self.uri: str = config["uri"]
        self.user: str = config["user"]
        self.password: str = config["password"]
        self.database: str = config.get("database", "neo4j")
        self.batch_size: int = config.get("batch_size", 100)

        self._driver: Driver | None = None
        logger.info("Neo4jClient initialized | uri={} | database={}", self.uri, self.database)

    def _get_driver(self) -> Driver:
        """Get or create the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Neo4j driver created")
        return self._driver

    def _get_session(self) -> Session:
        """Create a new session."""
        return self._get_driver().session(database=self.database)

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    def health_check(self) -> bool:
        """Check Neo4j connectivity.

        Returns:
            True if the database is reachable.
        """
        try:
            with self._get_session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j health check passed")
            return True
        except Exception as exc:
            logger.error("Neo4j health check failed: {}", exc)
            return False

    def batch_insert_nodes(self, nodes: list[dict[str, Any]]) -> int:
        """Insert nodes in batches using MERGE to avoid duplicates.

        Each node dict should have:
            - name (str): Node name/label text
            - type (str): Node type (e.g., "LegalArticle", "Organization")
            - properties (dict): Additional properties

        Args:
            nodes: List of node dictionaries.

        Returns:
            Total number of nodes inserted/merged.
        """
        total = 0
        with self._get_session() as session:
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i : i + self.batch_size]
                query = """
                UNWIND $nodes AS node
                MERGE (n:Entity {name: node.name})
                SET n.type = node.type,
                    n += node.properties
                RETURN count(n) AS cnt
                """
                result = session.run(query, nodes=batch)
                record = result.single()
                count = record["cnt"] if record else 0
                total += count

        logger.info("Batch inserted/merged {} nodes", total)
        return total

    def batch_insert_relations(self, relations: list[dict[str, Any]]) -> int:
        """Insert relations in batches using MERGE.

        Each relation dict should have:
            - source (str): Source node name
            - target (str): Target node name
            - relation_type (str): Relation type (e.g., "THUỘC", "THAM_CHIẾU")
            - properties (dict): Additional properties

        Args:
            relations: List of relation dictionaries.

        Returns:
            Total number of relations inserted/merged.
        """
        total = 0
        with self._get_session() as session:
            for i in range(0, len(relations), self.batch_size):
                batch = relations[i : i + self.batch_size]
                query = """
                UNWIND $rels AS rel
                MATCH (a:Entity {name: rel.source})
                MATCH (b:Entity {name: rel.target})
                CALL apoc.merge.relationship(a, rel.relation_type, {}, rel.properties, b) YIELD rel AS r
                RETURN count(r) AS cnt
                """
                # Fallback if APOC not available: use dynamic relationship type via workaround
                try:
                    result = session.run(query, rels=batch)
                    record = result.single()
                    count = record["cnt"] if record else 0
                except Exception:
                    # Fallback without APOC: create with fixed relation type per batch
                    count = self._insert_relations_no_apoc(session, batch)
                total += count

        logger.info("Batch inserted/merged {} relations", total)
        return total

    def _insert_relations_no_apoc(self, session: Session, batch: list[dict[str, Any]]) -> int:
        """Insert relations without APOC plugin by grouping by relation type.

        Args:
            session: Active Neo4j session.
            batch: List of relation dicts.

        Returns:
            Number of relations created.
        """
        from collections import defaultdict

        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rel in batch:
            by_type[rel["relation_type"]].append(rel)

        total = 0
        for rel_type, rels in by_type.items():
            # Sanitize relation type for Cypher
            safe_type = rel_type.replace(" ", "_").replace("-", "_")
            query = f"""
            UNWIND $rels AS rel
            MATCH (a:Entity {{name: rel.source}})
            MATCH (b:Entity {{name: rel.target}})
            MERGE (a)-[r:{safe_type}]->(b)
            SET r += rel.properties
            RETURN count(r) AS cnt
            """
            result = session.run(query, rels=rels)
            record = result.single()
            total += record["cnt"] if record else 0

        return total

    def create_indexes(self) -> None:
        """Create indexes for efficient graph queries."""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
        ]
        with self._get_session() as session:
            for idx_query in indexes:
                try:
                    session.run(idx_query)
                except Exception as exc:
                    logger.warning("Index creation warning: {}", exc)

        logger.info("Neo4j indexes created")

    def query_neighbors(self, entity: str, depth: int = 2) -> list[dict[str, Any]]:
        """Query neighbors of an entity up to a given depth.

        Args:
            entity: Entity name to start traversal from.
            depth: Maximum traversal depth (hops).

        Returns:
            List of dicts with 'source', 'relation', 'target', 'depth' keys.
        """
        query = """
        MATCH path = (n:Entity {name: $entity})-[r*1..$depth]-(m:Entity)
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS src, type(rel) AS relType, endNode(rel) AS tgt, length(path) AS d
        RETURN DISTINCT src.name AS source, relType AS relation, tgt.name AS target, d AS depth
        ORDER BY d
        LIMIT 100
        """
        results: list[dict[str, Any]] = []
        with self._get_session() as session:
            records = session.run(query, entity=entity, depth=depth)
            for record in records:
                results.append({
                    "source": record["source"],
                    "relation": record["relation"],
                    "target": record["target"],
                    "depth": record["depth"],
                })

        logger.debug("Found {} neighbor triples for '{}'", len(results), entity)
        return results

    def query_by_type(self, entity_type: str, limit: int = 50) -> list[dict[str, Any]]:
        """Query entities by type.

        Args:
            entity_type: Type of entities to retrieve.
            limit: Maximum number of results.

        Returns:
            List of entity property dicts.
        """
        query = """
        MATCH (n:Entity {type: $type})
        RETURN n
        LIMIT $limit
        """
        results: list[dict[str, Any]] = []
        with self._get_session() as session:
            records = session.run(query, type=entity_type, limit=limit)
            for record in records:
                node = record["n"]
                results.append(dict(node))

        return results

    def deduplicate(self) -> int:
        """Merge duplicate entities with the same name.

        Returns:
            Number of duplicate nodes removed.
        """
        query = """
        MATCH (n:Entity)
        WITH n.name AS name, collect(n) AS nodes
        WHERE size(nodes) > 1
        CALL {
            WITH nodes
            WITH head(nodes) AS keep, tail(nodes) AS duplicates
            UNWIND duplicates AS dup
            OPTIONAL MATCH (dup)-[r]-()
            DELETE r, dup
            RETURN count(dup) AS removed
        }
        RETURN sum(removed) AS total_removed
        """
        with self._get_session() as session:
            try:
                result = session.run(query)
                record = result.single()
                count = record["total_removed"] if record else 0
            except Exception as exc:
                logger.warning("Deduplication query failed (may need Neo4j 5.x): {}", exc)
                count = 0

        logger.info("Deduplicated {} nodes", count)
        return count

    def get_stats(self) -> dict[str, int]:
        """Get basic graph statistics.

        Returns:
            Dict with 'node_count' and 'relation_count'.
        """
        with self._get_session() as session:
            node_count_result = session.run("MATCH (n) RETURN count(n) AS cnt")
            node_count = node_count_result.single()["cnt"]

            rel_count_result = session.run("MATCH ()-[r]-() RETURN count(r) AS cnt")
            rel_count = rel_count_result.single()["cnt"]

        return {"node_count": node_count, "relation_count": rel_count}
