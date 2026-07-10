"""Build Neo4j graph from Pháp Điển canonical chunks only.

Target database (default ``phapdien``) holds **only** Bộ Pháp điển data:
  (:LegalDoc:PD {doc_id, source:'phapdien'})-[:HAS_ARTICLE]->(:LegalArticle:PD)

Optional ``--build-semantics`` adds Layer-2 semantic nodes/edges (LEGAL_CONCEPT,
MENTIONS, AMENDS, …) extracted from article text.

Use ``--wipe`` before a full rebuild of database ``phapdien``.
Use ``--replace-pd-only`` only for incremental structural updates (keeps semantic
nodes that are not labeled :PD).

Usage:
    python scripts/phapdien_pipeline/build_neo4j.py --wipe
    python scripts/phapdien_pipeline/build_neo4j.py --wipe --build-semantics
    python scripts/phapdien_pipeline/build_neo4j.py --replace-pd-only
    python scripts/phapdien_pipeline/build_neo4j.py --database phapdien
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from graph.neo4j_client import Neo4jClient
from llm.openai_client import OpenAIClient
import re

PD_RECHUNKED = ROOT / "data/processed/pd_rechunked.jsonl"
CONFIG = ROOT / "configs/build_kg_no_ner.yaml"
DEFAULT_DATABASE = "phapdien"


def ensure_database(client: Neo4jClient, database: str) -> None:
    """Create a separate Neo4j database when multi-db is supported."""
    if database in ("neo4j", "system"):
        return
    try:
        with client._get_driver().session(database="system") as session:
            session.run(f"CREATE DATABASE `{database}` IF NOT EXISTS")
        logger.info("Using Neo4j database '{}'", database)
    except Exception as exc:
        logger.warning(
            "Could not create database '{}' (Community Edition = single DB only): {}",
            database,
            exc,
        )


def replace_pd_only(client: Neo4jClient) -> None:
    """Remove only :PD structural nodes before re-ingest."""
    logger.warning("Removing existing :PD structural nodes...")
    deleted = 0
    while True:
        rows = client.query(
            "MATCH (n:PD) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c"
        )
        batch = rows[0]["c"] if rows else 0
        if batch == 0:
            break
        deleted += batch
    logger.info("Removed {} :PD nodes", deleted)


def wipe_graph(client: Neo4jClient) -> None:
    """Delete ALL nodes — only use when explicitly requested."""
    logger.warning("Wiping entire Neo4j database (batched)...")
    deleted = 0
    while True:
        rows = client.query(
            "MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c"
        )
        batch = rows[0]["c"] if rows else 0
        if batch == 0:
            break
        deleted += batch
        if deleted % 20000 == 0:
            logger.info("Deleted {} nodes so far...", deleted)
    logger.info("Wipe complete — removed {} nodes total", deleted)


def create_schema(client: Neo4jClient) -> None:
    statements = [
        "CREATE CONSTRAINT legaldoc_doc_id IF NOT EXISTS FOR (d:LegalDoc) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT legalarticle_id IF NOT EXISTS FOR (a:LegalArticle) REQUIRE a.article_id IS UNIQUE",
        "CREATE INDEX legalarticle_law_id IF NOT EXISTS FOR (a:LegalArticle) ON (a.law_id)",
        "CREATE INDEX legaldoc_source IF NOT EXISTS FOR (d:LegalDoc) ON (d.source)",
        "CREATE INDEX legalarticle_source IF NOT EXISTS FOR (a:LegalArticle) ON (a.source)",
    ]
    for stmt in statements:
        try:
            client.query(stmt)
        except Exception as exc:
            logger.debug("Schema stmt skipped: {} | {}", stmt[:60], exc)


def _ingest_batch_tx(tx, batch: list[dict]) -> None:
    cypher = """
    UNWIND $batch AS row
    MERGE (d:LegalDoc {doc_id: row.law_number})
      ON CREATE SET d.source = 'phapdien', d.type = 'Pháp Điển'
      ON MATCH SET d.source = coalesce(d.source, 'phapdien')
    SET d:PD
    MERGE (a:LegalArticle {article_id: row.canonical_id})
      ON CREATE SET
        a.law_id = row.law_number,
        a.article_number = row.article_number,
        a.title = row.title,
        a.content = row.content,
        a.source = 'phapdien'
      ON MATCH SET
        a.law_id = row.law_number,
        a.article_number = row.article_number,
        a.title = coalesce(row.title, a.title),
        a.content = row.content,
        a.source = 'phapdien'
    SET a:PD
    MERGE (d)-[:HAS_ARTICLE]->(a)
    """
    tx.run(cypher, batch=batch)


def ingest_batch(client: Neo4jClient, batch: list[dict]) -> None:
    with client._get_driver().session(database=client.database) as session:
        session.execute_write(_ingest_batch_tx, batch)


def build_fulltext(client: Neo4jClient) -> None:
    try:
        client.query("DROP INDEX legalArticleFulltext IF EXISTS")
    except Exception:
        pass
    client.query(
        """
        CREATE FULLTEXT INDEX legalArticleFulltext IF NOT EXISTS
        FOR (a:LegalArticle)
        ON EACH [a.title, a.content, a.article_id, a.law_id, a.article_number]
        """
    )


def extract_semantics(llm_client: OpenAIClient, content: str) -> list[dict]:
    """Extract semantic triples from legal text using LLM."""
    prompt = """
    Bạn là một chuyên gia về Biểu diễn Tri thức Pháp luật (Legal Knowledge Graph Architect).
    Hãy trích xuất các thực thể và mối quan hệ ngữ nghĩa từ đoạn văn bản pháp luật sau.
    
    Các loại thực thể (Node Labels) cho phép:
    LEGAL_CONCEPT, ACTOR, AUTHORITY, OBLIGATION, RIGHT, PROCEDURE, CONDITION, PENALTY
    
    Các loại quan hệ (Relations) cho phép:
    REGULATES, GOVERNS, APPLIES_TO, OBLIGATES, PERMITS, PROHIBITS, REQUIRES, HAS_CONDITION, 
    HAS_EXCEPTION, ELIGIBLE_FOR, NOT_ELIGIBLE_FOR, RESPONSIBLE_FOR, AUTHORIZES, SUPERVISES, 
    ENFORCES, ISSUED_BY, GUIDED_BY, IMPLEMENTS, REFERS_TO, AMENDS, REPLACES, ABOLISHES, 
    PART_OF_PROCEDURE, NEXT_STEP, RELATED_CONCEPT
    
    Định dạng JSON đầu ra (chỉ trả về JSON, không giải thích):
    [
      {"subject_label": "ACTOR", "subject_name": "hộ kinh doanh", "relation": "APPLIES_TO", "object_label": "LEGAL_CONCEPT", "object_name": "miễn giảm tiền thuê đất"},
      {"subject_label": "LegalArticle", "subject_name": "ARTICLE_NODE", "relation": "REGULATES", "object_label": "LEGAL_CONCEPT", "object_name": "miễn giảm tiền thuê đất"}
    ]
    (Quy ước: Nếu quan hệ là từ chính Điều luật này tới một thực thể, dùng subject_label: "LegalArticle" và subject_name: "ARTICLE_NODE").
    
    Văn bản:
    """ + content

    try:
        response = llm_client.generate(prompt=prompt, system_prompt="You return valid JSON arrays only.", temperature=0.1)
        # Find JSON array in the response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except Exception as e:
        logger.warning(f"Failed to extract semantics: {e}")
        return []

def _ingest_semantics_tx(tx, batch: list[dict]):
    """Ingest semantic triples into Neo4j."""
    for row in batch:
        sub_label = row.get("subject_label")
        sub_name = row.get("subject_name")
        rel = row.get("relation")
        obj_label = row.get("object_label")
        obj_name = row.get("object_name")
        article_id = row.get("article_id")
        
        # Validations to prevent injection or bad schemas
        valid_labels = {"LEGAL_CONCEPT", "ACTOR", "AUTHORITY", "OBLIGATION", "RIGHT", "PROCEDURE", "CONDITION", "PENALTY", "LegalArticle"}
        valid_rels = {"REGULATES", "GOVERNS", "APPLIES_TO", "OBLIGATES", "PERMITS", "PROHIBITS", "REQUIRES", "HAS_CONDITION", "HAS_EXCEPTION", "ELIGIBLE_FOR", "NOT_ELIGIBLE_FOR", "RESPONSIBLE_FOR", "AUTHORIZES", "SUPERVISES", "ENFORCES", "ISSUED_BY", "GUIDED_BY", "IMPLEMENTS", "REFERS_TO", "AMENDS", "REPLACES", "ABOLISHES", "PART_OF_PROCEDURE", "NEXT_STEP", "RELATED_CONCEPT"}
        
        if sub_label not in valid_labels or obj_label not in valid_labels or rel not in valid_rels:
            continue
            
        if sub_label == "LegalArticle" and sub_name == "ARTICLE_NODE":
            # Link from the Article to the object
            cypher = f"""
            MATCH (a:LegalArticle:PD {{article_id: $article_id}})
            MERGE (o:{obj_label} {{name: $obj_name}})
            MERGE (a)-[:{rel}]->(o)
            """
            tx.run(cypher, article_id=article_id, obj_name=obj_name)
        else:
            # Link between two semantic nodes
            cypher = f"""
            MERGE (s:{sub_label} {{name: $sub_name}})
            MERGE (o:{obj_label} {{name: $obj_name}})
            MERGE (s)-[:{rel}]->(o)
            """
            tx.run(cypher, sub_name=sub_name, obj_name=obj_name)

def ingest_semantics_batch(client: Neo4jClient, batch: list[dict]):
    with client._get_driver().session(database=client.database) as session:
        session.execute_write(_ingest_semantics_tx, batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PD_RECHUNKED)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--replace-pd-only",
        action="store_true",
        help="Delete only :PD nodes before ingest (default: merge in-place)",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete ENTIRE database (destructive — not recommended)",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help="Target Neo4j database (default: phapdien)",
    )
    parser.add_argument(
        "--build-semantics",
        action="store_true",
        help="Use LLM to extract and build Layer 2 Semantic Knowledge Graph",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    neo_cfg = dict(cfg["neo4j"])
    neo_cfg["database"] = args.database

    client = Neo4jClient(neo_cfg)

    try:
        ensure_database(client, args.database)

        if args.wipe:
            wipe_graph(client)
        elif args.replace_pd_only:
            replace_pd_only(client)

        create_schema(client)
        
        # Initialize LLM if semantics extraction is enabled
        llm_client = OpenAIClient(cfg.get("openai", cfg.get("ollama", {}))) if args.build_semantics else None

        batch: list[dict] = []
        total = 0
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("has_canonical_id"):
                    continue
                content = (r.get("content") or "").strip()
                if len(content) < 20:
                    continue
                batch.append(
                    {
                        "canonical_id": r["canonical_id"],
                        "law_number": r["law_number"],
                        "article_number": r.get("article_number", ""),
                        "title": r.get("title", ""),
                        "content": content[:8000],
                    }
                )
                if len(batch) >= args.batch_size:
                    ingest_batch(client, batch)
                    total += len(batch)
                    
                    if args.build_semantics and llm_client:
                        semantic_batch = []
                        for item in batch:
                            triples = extract_semantics(llm_client, item["content"])
                            for t in triples:
                                t["article_id"] = item["canonical_id"]
                                semantic_batch.append(t)
                        if semantic_batch:
                            ingest_semantics_batch(client, semantic_batch)
                    
                    batch = []
                    if total % 5000 == 0:
                        logger.info("Ingested {} articles...", total)
        if batch:
            ingest_batch(client, batch)
            total += len(batch)
            if args.build_semantics and llm_client:
                semantic_batch = []
                for item in batch:
                    triples = extract_semantics(llm_client, item["content"])
                    for t in triples:
                        t["article_id"] = item["canonical_id"]
                        semantic_batch.append(t)
                if semantic_batch:
                    ingest_semantics_batch(client, semantic_batch)

        build_fulltext(client)

        stats = {
            "database": client.database,
            "pd_legal_doc": client.query(
                "MATCH (d:LegalDoc:PD) RETURN count(d) AS c"
            )[0]["c"],
            "pd_legal_article": client.query(
                "MATCH (a:LegalArticle:PD) RETURN count(a) AS c"
            )[0]["c"],
            "pd_has_article": client.query(
                "MATCH (:PD)-[r:HAS_ARTICLE]->(:PD) RETURN count(r) AS c"
            )[0]["c"],
            "total_nodes": client.query("MATCH (n) RETURN count(n) AS c")[0]["c"],
            "co_occurred": client.query(
                "MATCH ()-[r:CO_OCCURRED]->() RETURN count(r) AS c"
            )[0]["c"],
        }
        logger.info("Neo4j Pháp Điển ingest complete: {}", stats)
    finally:
        client.close()


if __name__ == "__main__":
    main()
