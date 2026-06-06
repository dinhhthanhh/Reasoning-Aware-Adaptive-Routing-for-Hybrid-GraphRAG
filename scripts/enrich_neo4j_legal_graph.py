"""Incrementally enrich the Vietnamese legal Neo4j graph.

This script does not clear existing Neo4j data. It adds article-level
evidence from SQLiteKG and optional vector evidence links from Chroma.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from neo4j import GraphDatabase
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DOC_REF_RE = re.compile(
    r"\b\d{1,4}/(?:\d{4}/)?[A-ZĐ]{1,12}(?:-[A-ZĐ]{1,12}){0,4}\b",
    flags=re.UNICODE,
)
ARTICLE_NO_RE = re.compile(r"Điều\s+(\d+[a-zA-Z]?)", flags=re.IGNORECASE | re.UNICODE)

RELATION_TYPE_MAP = {
    "intra_doc_reference": "INTRA_DOC_REFERENCE",
    "cross_doc_reference": "CROSS_REFERENCES",
    "references": "REFERENCES_ARTICLE",
}

CONCEPT_KEYWORDS = {
    "kết hôn": ["kết hôn", "đăng ký kết hôn", "hôn nhân"],
    "ly hôn": ["ly hôn", "chấm dứt hôn nhân"],
    "lao động": ["lao động", "người lao động", "người sử dụng lao động"],
    "xử phạt hành chính": ["xử phạt", "vi phạm hành chính", "thẩm quyền xử phạt"],
    "hàng không": ["hàng không", "chuyến bay", "tàu bay", "phép bay"],
    "vốn ODA": ["oda", "vốn vay ưu đãi", "vốn vay nước ngoài"],
    "quốc tịch": ["quốc tịch", "nhập quốc tịch", "thôi quốc tịch"],
    "đất đai": ["đất đai", "quyền sử dụng đất", "thu hồi đất"],
    "thuế": ["thuế", "hóa đơn", "nghĩa vụ thuế"],
    "môi trường": ["môi trường", "chất thải", "ô nhiễm"],
    "doanh nghiệp": ["doanh nghiệp", "kinh doanh", "đầu tư"],
    "người tiêu dùng": ["người tiêu dùng", "bảo vệ quyền lợi người tiêu dùng"],
}


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _truncate(text: str | None, max_len: int) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text[:max_len]


def _article_sort_key(article_id: str) -> tuple[int, str]:
    match = ARTICLE_NO_RE.search(article_id)
    if not match:
        return (10**9, article_id)
    raw = match.group(1)
    num = int(re.match(r"\d+", raw).group(0)) if re.match(r"\d+", raw) else 10**9
    return (num, article_id)


def _infer_doc_relation(text: str, start: int, end: int) -> str:
    window = text[max(0, start - 80) : min(len(text), end + 80)].lower()
    if "căn cứ" in window:
        return "LEGAL_BASIS"
    if "hướng dẫn" in window or "quy định chi tiết" in window:
        return "GUIDES"
    if "sửa đổi" in window or "bổ sung" in window:
        return "AMENDS"
    if "bãi bỏ" in window or "hết hiệu lực" in window:
        return "REPEALS"
    if "chuyển tiếp" in window:
        return "TRANSITIONAL_TO"
    if "thi hành" in window:
        return "IMPLEMENTS"
    return "MENTIONS_DOC"


def _extract_doc_refs(article_id: str, content: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in DOC_REF_RE.finditer(content or ""):
        doc_ref = match.group(0)
        rel_type = _infer_doc_relation(content, match.start(), match.end())
        key = (doc_ref, rel_type)
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            "article_id": article_id,
            "doc_id": doc_ref,
            "rel_type": rel_type,
        })
    return refs


def _extract_concepts(article_id: str, title: str, content: str) -> list[dict[str, str]]:
    text = f"{title}\n{content}".lower()
    rows: list[dict[str, str]] = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            rows.append({"article_id": article_id, "concept": concept})
    return rows


def _vector_doc_id(chunk_id: str) -> str:
    if chunk_id.startswith("hf_processed_"):
        return chunk_id.removeprefix("hf_processed_")
    if chunk_id.startswith("phapdien_processed_"):
        return chunk_id.removeprefix("phapdien_processed_")
    return chunk_id


class LegalGraphEnricher:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        neo4j_cfg = config["neo4j"]
        self.driver = GraphDatabase.driver(
            neo4j_cfg["uri"],
            auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
        )
        self.database = neo4j_cfg.get("database", "neo4j")
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def create_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:LegalDoc) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:LegalArticle) REQUIRE a.article_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:VectorChunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:LegalConcept) REQUIRE c.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (a:LegalArticle) ON (a.law_id)",
            "CREATE INDEX IF NOT EXISTS FOR (a:LegalArticle) ON (a.title)",
            "CREATE INDEX IF NOT EXISTS FOR (c:VectorChunk) ON (c.doc_id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:LegalDoc) ON (d.title)",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

            fulltext_statements = [
                (
                    "CREATE FULLTEXT INDEX legal_article_fulltext IF NOT EXISTS "
                    "FOR (a:LegalArticle) ON EACH [a.title, a.content]"
                ),
                (
                    "CREATE FULLTEXT INDEX vector_chunk_fulltext IF NOT EXISTS "
                    "FOR (c:VectorChunk) ON EACH [c.title, c.content_preview]"
                ),
            ]
            for statement in fulltext_statements:
                try:
                    session.run(statement)
                except Exception as exc:
                    logger.warning("Fulltext index creation skipped: {}", exc)

    def ingest_sqlite_articles(self, sqlite_path: Path, batch_size: int, limit: int | None) -> dict[str, int]:
        con = sqlite3.connect(sqlite_path)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT article_id, law_id, title, content FROM nodes")]
        if limit:
            rows = rows[:limit]

        article_rows = []
        concept_rows = []
        doc_ref_rows = []
        by_law: dict[str, list[str]] = defaultdict(list)

        for row in rows:
            article_id = str(row.get("article_id") or "")
            law_id = str(row.get("law_id") or "")
            title = str(row.get("title") or article_id)
            content = str(row.get("content") or "")
            if not article_id:
                continue

            article_rows.append({
                "article_id": article_id,
                "law_id": law_id,
                "title": title,
                "content": _truncate(content, 6000),
                "content_preview": _truncate(content, 800),
            })
            if law_id:
                by_law[law_id].append(article_id)
            concept_rows.extend(_extract_concepts(article_id, title, content))
            doc_ref_rows.extend(_extract_doc_refs(article_id, content))

        edge_rows = [dict(r) for r in con.execute(
            "SELECT src_article, dst_article, relation_type, question_ctx FROM edges"
        )]
        if limit:
            edge_rows = edge_rows[:limit]
        con.close()

        next_rows: list[dict[str, str]] = []
        for law_id, article_ids in by_law.items():
            ordered = sorted(article_ids, key=_article_sort_key)
            for src, dst in zip(ordered, ordered[1:]):
                next_rows.append({"src": src, "dst": dst, "law_id": law_id})

        with self.driver.session(database=self.database) as session:
            for batch in tqdm(_chunks(article_rows, batch_size), desc="LegalArticle"):
                session.execute_write(self._merge_articles_tx, batch)

            for batch in tqdm(_chunks(edge_rows, batch_size), desc="SQLite edges"):
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in batch:
                    rel_type = RELATION_TYPE_MAP.get(
                        str(row.get("relation_type") or "references"),
                        "REFERENCES_ARTICLE",
                    )
                    grouped[rel_type].append(row)
                for rel_type, rel_rows in grouped.items():
                    session.execute_write(self._merge_article_edges_tx, rel_rows, rel_type)

            for batch in tqdm(_chunks(next_rows, batch_size), desc="NEXT_ARTICLE"):
                session.execute_write(self._merge_next_article_tx, batch)

            for batch in tqdm(_chunks(doc_ref_rows, batch_size), desc="Doc references"):
                grouped = defaultdict(list)
                for row in batch:
                    grouped[row["rel_type"]].append(row)
                for rel_type, rel_rows in grouped.items():
                    session.execute_write(self._merge_doc_ref_tx, rel_rows, rel_type)

            for batch in tqdm(_chunks(concept_rows, batch_size), desc="Concept links"):
                session.execute_write(self._merge_concepts_tx, batch)

        return {
            "articles": len(article_rows),
            "sqlite_edges": len(edge_rows),
            "next_article_edges": len(next_rows),
            "doc_reference_edges": len(doc_ref_rows),
            "concept_edges": len(concept_rows),
        }

    @staticmethod
    def _merge_articles_tx(tx, rows: list[dict[str, Any]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (a:LegalArticle {article_id: row.article_id})
            SET a.law_id = row.law_id,
                a.title = row.title,
                a.content = row.content,
                a.content_preview = row.content_preview,
                a.source = "SQLiteLegalKG"
            WITH a, row
            MERGE (d:LegalDoc {doc_id: row.law_id})
            ON CREATE SET d.title = row.law_id,
                          d.type = "Legal document",
                          d.authority = "",
                          d.source = "SQLiteLegalKG",
                          d.issue_date = ""
            MERGE (d)-[:HAS_ARTICLE]->(a)
            """,
            rows=rows,
        )

    @staticmethod
    def _merge_article_edges_tx(tx, rows: list[dict[str, Any]], rel_type: str) -> None:
        tx.run(
            f"""
            UNWIND $rows AS row
            MATCH (src:LegalArticle {{article_id: row.src_article}})
            MATCH (dst:LegalArticle {{article_id: row.dst_article}})
            MERGE (src)-[r:{rel_type}]->(dst)
            SET r.source = "SQLiteLegalKG",
                r.question_ctx = coalesce(row.question_ctx, ""),
                r.original_type = coalesce(row.relation_type, "")
            """,
            rows=rows,
        )

    @staticmethod
    def _merge_next_article_tx(tx, rows: list[dict[str, str]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (src:LegalArticle {article_id: row.src})
            MATCH (dst:LegalArticle {article_id: row.dst})
            MERGE (src)-[r:NEXT_ARTICLE]->(dst)
            SET r.law_id = row.law_id
            """,
            rows=rows,
        )

    @staticmethod
    def _merge_doc_ref_tx(tx, rows: list[dict[str, str]], rel_type: str) -> None:
        tx.run(
            f"""
            UNWIND $rows AS row
            MATCH (a:LegalArticle {{article_id: row.article_id}})
            MERGE (d:LegalDoc {{doc_id: row.doc_id}})
            ON CREATE SET d.title = row.doc_id,
                          d.type = "Mentioned legal document",
                          d.authority = "",
                          d.source = "MentionedReference",
                          d.issue_date = ""
            MERGE (a)-[r:{rel_type}]->(d)
            SET r.source = "regex_content"
            """,
            rows=rows,
        )

    @staticmethod
    def _merge_concepts_tx(tx, rows: list[dict[str, str]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a:LegalArticle {article_id: row.article_id})
            MERGE (c:LegalConcept {name: row.concept})
            MERGE (a)-[:REGULATES_CONCEPT]->(c)
            """,
            rows=rows,
        )

    def ingest_chroma_vectors(
        self,
        chroma_path: Path,
        collection_name: str,
        batch_size: int,
        limit: int | None,
        content_chars: int,
    ) -> int:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(collection_name)
        total = collection.count()
        max_items = min(total, limit) if limit else total
        offset = 0
        written = 0

        with self.driver.session(database=self.database) as session:
            while offset < max_items:
                size = min(batch_size, max_items - offset)
                res = collection.get(
                    limit=size,
                    offset=offset,
                    include=["metadatas", "documents"],
                )
                rows = []
                ids = res.get("ids", [])
                metadatas = res.get("metadatas", [])
                documents = res.get("documents", [])
                for i, chunk_id in enumerate(ids):
                    metadata = metadatas[i] or {}
                    document = documents[i] or ""
                    doc_id = _vector_doc_id(str(chunk_id))
                    rows.append({
                        "chunk_id": str(chunk_id),
                        "doc_id": doc_id,
                        "title": str(metadata.get("title") or ""),
                        "source": str(metadata.get("source") or ""),
                        "type": str(metadata.get("type") or ""),
                        "authority": str(metadata.get("authority") or ""),
                        "content_preview": _truncate(document, content_chars),
                    })

                if rows:
                    session.execute_write(self._merge_vector_chunks_tx, rows)
                    written += len(rows)
                offset += size
                tqdm.write(f"Vector chunks linked: {written}/{max_items}")

        return written

    def materialize_article_chunks(self, batch_size: int, content_chars: int) -> int:
        """Create evidence chunks directly from LegalArticle nodes.

        The vector corpus may not contain chunks keyed by law_id for the small
        SQLite article graph. Materializing article evidence keeps graph
        traversal article-centric while still exposing full answer text through
        the same HAS_VECTOR_CHUNK evidence edge used by other graph nodes.
        """
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                MATCH (a:LegalArticle)
                RETURN a.article_id AS article_id,
                       a.law_id AS law_id,
                       a.title AS title,
                       coalesce(a.content, a.content_preview, "") AS content
                ORDER BY a.law_id, a.article_id
                """
            ).data()

            prepared = []
            for row in rows:
                article_id = str(row.get("article_id") or "")
                if not article_id:
                    continue
                content = _truncate(str(row.get("content") or ""), content_chars)
                prepared.append({
                    "article_id": article_id,
                    "chunk_id": f"article::{article_id}",
                    "doc_id": str(row.get("law_id") or ""),
                    "title": str(row.get("title") or article_id),
                    "content_preview": content,
                })

            for batch in tqdm(_chunks(prepared, batch_size), desc="Article evidence chunks"):
                session.execute_write(self._materialize_article_chunks_tx, batch)

        return len(prepared)

    @staticmethod
    def _materialize_article_chunks_tx(tx, rows: list[dict[str, str]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a:LegalArticle {article_id: row.article_id})
            MERGE (d:LegalDoc {doc_id: row.doc_id})
            ON CREATE SET d.title = row.doc_id,
                          d.type = "Legal document",
                          d.authority = "",
                          d.source = "SQLiteLegalKG",
                          d.issue_date = ""
            MERGE (c:VectorChunk {chunk_id: row.chunk_id})
            SET c.doc_id = row.doc_id,
                c.title = row.title,
                c.source = "SQLiteLegalKG",
                c.type = "LegalArticleEvidence",
                c.authority = "",
                c.content_preview = row.content_preview
            MERGE (a)-[:HAS_VECTOR_CHUNK]->(c)
            MERGE (d)-[:HAS_VECTOR_CHUNK]->(c)
            """,
            rows=rows,
        )

    @staticmethod
    def _merge_vector_chunks_tx(tx, rows: list[dict[str, str]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (d:LegalDoc {doc_id: row.doc_id})
            ON CREATE SET d.title = coalesce(row.title, row.doc_id),
                          d.type = row.type,
                          d.authority = row.authority,
                          d.source = row.source,
                          d.issue_date = ""
            SET d.vector_id = row.chunk_id,
                d.content_preview = coalesce(d.content_preview, row.content_preview)
            MERGE (c:VectorChunk {chunk_id: row.chunk_id})
            SET c.doc_id = row.doc_id,
                c.title = row.title,
                c.source = row.source,
                c.type = row.type,
                c.authority = row.authority,
                c.content_preview = row.content_preview
            MERGE (d)-[:HAS_VECTOR_CHUNK]->(c)
            """,
            rows=rows,
        )

    def stats(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            labels = session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(*) AS count
                ORDER BY count DESC
                """
            ).data()
            rels = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY count DESC
                LIMIT 30
                """
            ).data()
        return {"labels": labels, "relationships": rels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich legal Neo4j graph with evidence nodes.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--limit-vector-chunks", type=int, default=0)
    parser.add_argument("--vector-content-chars", type=int, default=1200)
    parser.add_argument("--materialize-article-chunks", action="store_true")
    parser.add_argument("--skip-sqlite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else (
        Path(config["data"]["kg_dir"]) / config["data"].get("kg_filename", "legal_kg.db")
    )
    chroma_cfg = config.get("chroma", {})
    chroma_path = Path(args.chroma_path or chroma_cfg.get("path", "data/vector_store/chroma_full"))
    collection = args.collection or chroma_cfg.get("collection_name", "legal_docs")

    if args.dry_run:
        logger.info("Dry run only.")
        logger.info("SQLite path: {}", sqlite_path)
        logger.info("Chroma path: {} | collection={}", chroma_path, collection)
        if sqlite_path.exists():
            con = sqlite3.connect(sqlite_path)
            node_count = con.execute("SELECT count(*) FROM nodes").fetchone()[0]
            edge_count = con.execute("SELECT count(*) FROM edges").fetchone()[0]
            con.close()
            logger.info("SQLite nodes={} edges={}", node_count, edge_count)
        return

    enricher = LegalGraphEnricher(config)
    try:
        enricher.create_schema()
        if not args.skip_sqlite:
            result = enricher.ingest_sqlite_articles(
                sqlite_path=sqlite_path,
                batch_size=args.batch_size,
                limit=args.limit_articles,
            )
            logger.info("SQLite enrichment: {}", result)

        if args.limit_vector_chunks != 0:
            vector_limit = args.limit_vector_chunks if args.limit_vector_chunks > 0 else None
            written = enricher.ingest_chroma_vectors(
                chroma_path=chroma_path,
                collection_name=collection,
                batch_size=args.batch_size,
                limit=vector_limit,
                content_chars=args.vector_content_chars,
            )
            logger.info("Vector chunks linked: {}", written)

        if args.materialize_article_chunks:
            written = enricher.materialize_article_chunks(
                batch_size=args.batch_size,
                content_chars=args.vector_content_chars,
            )
            logger.info("Article evidence chunks materialized: {}", written)

        logger.info("Neo4j enrichment stats: {}", enricher.stats())
    finally:
        enricher.close()


if __name__ == "__main__":
    main()
