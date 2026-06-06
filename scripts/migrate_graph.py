"""Week 1 graph schema migration for Vietnamese legal GraphRAG.

The migration is intentionally narrow and idempotent:
1. Inspect the current graph state.
2. Promote Phap Dien article-like LegalDoc nodes to LegalArticle.
3. Create parent LegalDoc groups and HAS_ARTICLE relations.
4. Link VectorChunk nodes back to LegalDoc where keys match.
5. Rebuild fulltext indexes used by graph retrieval.
6. Write a Markdown report and JSON summary with before/after numbers.

Run from the repository root:
    python scripts/migrate_graph.py
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


REPORT_MD = Path("reports/week1_graph_migration_report.md")
REPORT_JSON = Path("reports/week1_graph_migration_summary.json")


def _scalar(client: Neo4jClient, cypher: str, key: str = "count") -> Any:
    rows = client.query(cypher)
    if not rows:
        return 0
    return rows[0].get(key, next(iter(rows[0].values())))


def _compact_value(value: Any, max_len: int = 500) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len] + "..."
    if isinstance(value, dict):
        return {k: _compact_value(v, max_len=max_len) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact_value(v, max_len=max_len) for v in value]
    return value


def _print_json(title: str, value: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(_compact_value(value), ensure_ascii=False, indent=2))


def _pd_predicate(alias: str = "n") -> str:
    return (
        "("
        f"{alias}.type = 'Điều (Pháp điển)' "
        f"OR {alias}.source = 'Pháp Điển' "
        f"OR 'PD' IN labels({alias})"
        ") "
        f"AND coalesce({alias}.type, '') <> 'Nhóm Pháp Điển'"
    )


def inspect_graph(client: Neo4jClient) -> dict[str, Any]:
    """Collect before/after statistics and representative samples."""
    stats: dict[str, Any] = {}

    stats["label_counts"] = client.query(
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC, label
        """
    )

    stats["legal_doc_type_counts"] = client.query(
        """
        MATCH (n:LegalDoc)
        RETURN coalesce(n.type, "<missing>") AS type, count(*) AS count
        ORDER BY count DESC, type
        LIMIT 30
        """
    )

    stats["legal_article_count"] = _scalar(
        client,
        "MATCH (n:LegalArticle) RETURN count(n) AS count",
    )
    stats["legal_article_with_content"] = _scalar(
        client,
        """
        MATCH (n:LegalArticle)
        WHERE coalesce(n.content, n.content_preview, "") <> ""
        RETURN count(n) AS count
        """,
    )
    stats["has_article_count"] = _scalar(
        client,
        "MATCH (:LegalDoc)-[:HAS_ARTICLE]->(:LegalArticle) RETURN count(*) AS count",
    )
    stats["vector_chunk_count"] = _scalar(
        client,
        "MATCH (n:VectorChunk) RETURN count(n) AS count",
    )
    stats["vector_chunk_linked_count"] = _scalar(
        client,
        """
        MATCH (c:VectorChunk)-[:BELONGS_TO]->(:LegalDoc)
        RETURN count(DISTINCT c) AS count
        """,
    )
    stats["vector_chunk_belongs_to_count"] = _scalar(
        client,
        "MATCH (:VectorChunk)-[:BELONGS_TO]->(:LegalDoc) RETURN count(*) AS count",
    )
    stats["doc_article_coverage"] = client.query(
        """
        MATCH (d:LegalDoc)
        OPTIONAL MATCH (d)-[r:HAS_ARTICLE]->(:LegalArticle)
        WITH d, count(r) AS article_count
        RETURN
          sum(CASE WHEN article_count = 0 THEN 1 ELSE 0 END) AS no_articles,
          sum(CASE WHEN article_count > 0 THEN 1 ELSE 0 END) AS has_articles,
          sum(article_count) AS total_articles
        """
    )[0]

    stats["pd_legal_doc_samples"] = client.query(
        f"""
        MATCH (n:LegalDoc)
        WHERE {_pd_predicate("n")}
        RETURN labels(n) AS labels, properties(n) AS properties
        LIMIT 3
        """
    )
    stats["hf_legal_doc_samples"] = client.query(
        """
        MATCH (n:LegalDoc)
        WHERE 'HF' IN labels(n)
           OR toLower(coalesce(n.source, "")) CONTAINS "hugging"
           OR coalesce(n.vector_id, "") STARTS WITH "hf_processed"
        RETURN labels(n) AS labels, properties(n) AS properties
        LIMIT 3
        """
    )
    stats["vector_chunk_samples"] = client.query(
        """
        MATCH (c:VectorChunk)
        RETURN labels(c) AS labels, properties(c) AS properties
        LIMIT 3
        """
    )
    stats["fulltext_indexes"] = get_fulltext_indexes(client)

    return stats


def promote_pd_articles(client: Neo4jClient) -> int:
    """Add LegalArticle label/properties to Phap Dien article nodes."""
    rows = client.query(
        f"""
        MATCH (n:LegalDoc)
        WHERE {_pd_predicate("n")}
        CALL {{
          WITH n
          WITH n,
               coalesce(n.doc_id, n.id, n.article_id, elementId(n)) AS raw_id,
               split(coalesce(n.doc_id, n.id, n.article_id, elementId(n)), "_") AS parts
          WITH n, raw_id,
               CASE
                 WHEN n.law_id IS NOT NULL AND n.law_id <> "" THEN n.law_id
                 WHEN raw_id STARTS WITH "pd_" AND size(parts) >= 3
                   THEN parts[0] + "_" + parts[1] + "_" + parts[2]
                 WHEN size(parts) >= 1 THEN parts[0]
                 ELSE raw_id
               END AS inferred_law_id
          SET n:LegalArticle
          SET n.article_id = CASE
                WHEN n.article_id IS NULL OR n.article_id = ""
                THEN coalesce(n.doc_id, n.id, elementId(n))
                ELSE n.article_id
              END,
              n.content = CASE
                WHEN n.content IS NULL OR n.content = ""
                THEN coalesce(n.content_preview, n.text, n.content)
                ELSE n.content
              END,
              n.law_id = CASE
                WHEN n.law_id IS NULL OR n.law_id = ""
                THEN inferred_law_id
                ELSE n.law_id
              END
          RETURN count(n) AS promoted
        }} IN TRANSACTIONS OF 5000 ROWS
        RETURN sum(promoted) AS promoted
        """
    )
    return int(rows[0]["promoted"] or 0) if rows else 0


def create_pd_parent_relations(client: Neo4jClient) -> int:
    """Create parent Phap Dien LegalDoc groups and HAS_ARTICLE relations."""
    rows = client.query(
        f"""
        MATCH (a:LegalArticle)
        WHERE ({_pd_predicate("a")})
          AND a.law_id IS NOT NULL
          AND a.law_id <> ""
        WITH a, a.law_id AS law_id
        CALL {{
          WITH a, law_id
          MERGE (d:LegalDoc {{doc_id: law_id}})
          ON CREATE SET d.title = "Pháp Điển - " + law_id,
                        d.type = "Nhóm Pháp Điển",
                        d.source = "Pháp Điển Group"
          SET d.title = CASE
                WHEN d.title IS NULL OR d.title = "" THEN "Pháp Điển - " + law_id
                ELSE d.title
              END,
              d.type = CASE
                WHEN d.type IS NULL OR d.type = "" THEN "Nhóm Pháp Điển"
                ELSE d.type
              END,
              d.source = CASE
                WHEN d.type = "Nhóm Pháp Điển" THEN "Pháp Điển Group"
                WHEN d.source IS NULL OR d.source = "" THEN "Pháp Điển Group"
                ELSE d.source
              END
          WITH d, a
          WHERE elementId(d) <> elementId(a)
          MERGE (d)-[:HAS_ARTICLE]->(a)
          RETURN count(*) AS linked
        }} IN TRANSACTIONS OF 5000 ROWS
        RETURN sum(linked) AS linked
        """
    )
    return int(rows[0]["linked"] or 0) if rows else 0


def cleanup_parent_article_labels(client: Neo4jClient) -> int:
    """Remove accidental LegalArticle labels from synthetic Phap Dien parent docs."""
    rows = client.query(
        """
        MATCH (n:LegalDoc:LegalArticle)
        WHERE n.type = "Nhóm Pháp Điển"
        REMOVE n:LegalArticle
        SET n.source = "Pháp Điển Group"
        RETURN count(n) AS cleaned
        """
    )
    return int(rows[0]["cleaned"] or 0) if rows else 0


def _run_link_strategy(client: Neo4jClient, name: str, cypher: str, warnings: list[str]) -> int:
    try:
        rows = client.query(cypher)
        linked = int(rows[0]["linked"]) if rows else 0
        logger.info("Vector link strategy '{}' matched {} pairs", name, linked)
        return linked
    except Exception as exc:
        warning = f"Vector link strategy '{name}' failed: {exc}"
        logger.warning(warning)
        warnings.append(warning)
        return 0


def link_vector_chunks(client: Neo4jClient, warnings: list[str]) -> dict[str, int]:
    """Create BELONGS_TO edges from VectorChunk to LegalDoc where keys match."""
    strategy_counts: dict[str, int] = {}

    strategy_counts["doc_id_to_doc_id"] = _run_link_strategy(
        client,
        "doc_id_to_doc_id",
        """
        MATCH (c:VectorChunk)
        WHERE c.doc_id IS NOT NULL AND c.doc_id <> ""
        CALL {
          WITH c
          MATCH (d:LegalDoc {doc_id: c.doc_id})
          MERGE (c)-[:BELONGS_TO]->(d)
          RETURN count(*) AS linked
        } IN TRANSACTIONS OF 5000 ROWS
        RETURN sum(linked) AS linked
        """,
        warnings,
    )

    strategy_counts["source_id_to_doc_id"] = _run_link_strategy(
        client,
        "source_id_to_doc_id",
        """
        MATCH (c:VectorChunk)
        WHERE c.source_id IS NOT NULL AND c.source_id <> ""
        CALL {
          WITH c
          MATCH (d:LegalDoc {doc_id: c.source_id})
          MERGE (c)-[:BELONGS_TO]->(d)
          RETURN count(*) AS linked
        } IN TRANSACTIONS OF 5000 ROWS
        RETURN sum(linked) AS linked
        """,
        warnings,
    )

    strategy_counts["chunk_id_to_doc_vector_id"] = _run_link_strategy(
        client,
        "chunk_id_to_doc_vector_id",
        """
        MATCH (d:LegalDoc)
        WHERE d.vector_id IS NOT NULL AND d.vector_id <> ""
        CALL {
          WITH d
          MATCH (c:VectorChunk {chunk_id: d.vector_id})
          MERGE (c)-[:BELONGS_TO]->(d)
          RETURN count(*) AS linked
        } IN TRANSACTIONS OF 5000 ROWS
        RETURN sum(linked) AS linked
        """,
        warnings,
    )

    vector_chunk_with_vector_id = _scalar(
        client,
        """
        MATCH (c:VectorChunk)
        WHERE c.vector_id IS NOT NULL AND c.vector_id <> ""
        RETURN count(c) AS count
        """,
    )
    if vector_chunk_with_vector_id:
        warnings.append(
            "VectorChunk.vector_id exists but c.vector_id = d.vector_id was not executed "
            "because LegalDoc.vector_id has no dedicated index in the current schema."
        )
    else:
        logger.info("Skipping vector_id_to_vector_id: VectorChunk.vector_id is absent")

    return strategy_counts


def rebuild_fulltext_indexes(client: Neo4jClient, errors: list[str], warnings: list[str]) -> None:
    """Drop and recreate fulltext indexes used by graph retrieval."""
    drop_statements = [
        "DROP INDEX legal_article_fulltext IF EXISTS",
        "DROP INDEX vector_chunk_fulltext IF EXISTS",
        "DROP INDEX legal_doc_fulltext IF EXISTS",
    ]
    create_statements = [
        (
            "legal_article_fulltext",
            """
            CREATE FULLTEXT INDEX legal_article_fulltext IF NOT EXISTS
            FOR (n:LegalArticle)
            ON EACH [n.title, n.content, n.content_preview, n.article_id, n.law_id, n.doc_id]
            """,
        ),
        (
            "vector_chunk_fulltext",
            """
            CREATE FULLTEXT INDEX vector_chunk_fulltext IF NOT EXISTS
            FOR (n:VectorChunk)
            ON EACH [n.title, n.content, n.content_preview, n.doc_id, n.chunk_id, n.source_id]
            """,
        ),
        (
            "legal_doc_fulltext",
            """
            CREATE FULLTEXT INDEX legal_doc_fulltext IF NOT EXISTS
            FOR (n:LegalDoc)
            ON EACH [n.title, n.doc_id, n.type, n.authority, n.source]
            """,
        ),
    ]

    for statement in drop_statements:
        try:
            client.query(statement)
            logger.info("Executed: {}", statement)
        except Exception as exc:
            warning = f"Index drop skipped/failed for '{statement}': {exc}"
            logger.warning(warning)
            warnings.append(warning)

    for name, statement in create_statements:
        try:
            client.query(statement)
            logger.info("Created fulltext index: {}", name)
        except Exception as exc:
            message = f"Index creation failed for {name}: {exc}"
            logger.error(message)
            errors.append(message)

    try:
        client.query("CALL db.awaitIndexes(300)")
    except Exception as exc:
        warning = f"db.awaitIndexes failed or timed out: {exc}"
        logger.warning(warning)
        warnings.append(warning)


def get_fulltext_indexes(client: Neo4jClient) -> list[dict[str, Any]]:
    try:
        return client.query(
            """
            SHOW INDEXES
            YIELD name, type, state
            WHERE type = "FULLTEXT"
            RETURN name, state
            ORDER BY name
            """
        )
    except Exception:
        return client.query(
            """
            SHOW INDEXES
            YIELD name, type, state
            WHERE type = 'FULLTEXT'
            RETURN name, state
            ORDER BY name
            """
        )


def verify_after(client: Neo4jClient, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    verification: dict[str, Any] = {}
    verification["legal_article_count"] = _scalar(
        client,
        "MATCH (n:LegalArticle) RETURN count(n) AS count",
    )
    verification["legal_article_with_content"] = _scalar(
        client,
        """
        MATCH (n:LegalArticle)
        WHERE coalesce(n.content, n.content_preview, "") <> ""
        RETURN count(n) AS count
        """,
    )
    verification["has_article_count"] = _scalar(
        client,
        "MATCH (:LegalDoc)-[:HAS_ARTICLE]->(:LegalArticle) RETURN count(*) AS count",
    )
    verification["vector_chunk_linked_count"] = _scalar(
        client,
        """
        MATCH (c:VectorChunk)-[:BELONGS_TO]->(:LegalDoc)
        RETURN count(DISTINCT c) AS count
        """,
    )
    verification["fulltext_indexes"] = get_fulltext_indexes(client)

    try:
        verification["sample_fulltext_article"] = client.query(
            """
            CALL db.index.fulltext.queryNodes(
              "legal_article_fulltext",
              $term,
              {limit: 3}
            )
            YIELD node, score
            RETURN score,
                   labels(node) AS labels,
                   node.article_id AS article_id,
                   node.law_id AS law_id,
                   node.title AS title,
                   coalesce(node.content, node.content_preview, "") AS content
            LIMIT 3
            """,
            {"term": '"quản lý chất thải"'},
        )
    except Exception as exc:
        message = f"Sample legal_article_fulltext search failed: {exc}"
        logger.warning(message)
        warnings.append(message)
        verification["sample_fulltext_article"] = []

    verification["sample_doc_to_article"] = client.query(
        """
        MATCH (d:LegalDoc)-[:HAS_ARTICLE]->(a:LegalArticle)
        RETURN d.doc_id AS parent_doc_id,
               d.title AS parent_title,
               a.article_id AS article_id,
               a.title AS article_title
        LIMIT 5
        """
    )
    verification["sample_vector_to_doc"] = client.query(
        """
        MATCH (c:VectorChunk)-[:BELONGS_TO]->(d:LegalDoc)
        RETURN c.chunk_id AS chunk_id,
               c.doc_id AS chunk_doc_id,
               d.doc_id AS doc_id,
               d.title AS doc_title
        LIMIT 5
        """
    )

    offline_indexes = [
        row for row in verification["fulltext_indexes"]
        if row.get("name") in {
            "legal_article_fulltext",
            "legal_doc_fulltext",
            "vector_chunk_fulltext",
        }
        and row.get("state") != "ONLINE"
    ]
    if offline_indexes:
        errors.append(f"Fulltext indexes not ONLINE: {offline_indexes}")

    if verification["legal_article_count"] <= 272:
        errors.append("LegalArticle count did not increase beyond the initial 272")
    if verification["legal_article_with_content"] <= 50000:
        errors.append("LegalArticle with content count is not above 50k")
    if verification["has_article_count"] <= 272:
        errors.append("HAS_ARTICLE count did not increase clearly")

    return verification


def run_idempotence_check(client: Neo4jClient, warnings: list[str]) -> dict[str, Any]:
    """Check for duplicate relationships without another heavy write pass."""
    has_article_duplicates = _scalar(
        client,
        """
        MATCH (d:LegalDoc)-[r:HAS_ARTICLE]->(a:LegalArticle)
        WITH elementId(d) AS source_id, elementId(a) AS target_id, count(r) AS rel_count
        WHERE rel_count > 1
        RETURN coalesce(sum(rel_count - 1), 0) AS count
        """,
    )
    belongs_to_duplicates = _scalar(
        client,
        """
        MATCH (c:VectorChunk)-[r:BELONGS_TO]->(d:LegalDoc)
        WITH elementId(c) AS source_id, elementId(d) AS target_id, count(r) AS rel_count
        WHERE rel_count > 1
        RETURN coalesce(sum(rel_count - 1), 0) AS count
        """,
    )
    return {
        "has_article_duplicate_relationships": has_article_duplicates,
        "belongs_to_duplicate_relationships": belongs_to_duplicates,
        "pass": has_article_duplicates == 0 and belongs_to_duplicates == 0,
    }


def write_reports(
    before: dict[str, Any],
    after: dict[str, Any],
    verification: dict[str, Any],
    migration: dict[str, Any],
    idempotence: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    status = "PASS" if not errors and idempotence.get("pass") else "FAIL"
    next_action = "READY_FOR_MINI_EVAL" if status == "PASS" else "NEED_FIX"

    before_coverage = before["doc_article_coverage"]
    fulltext_index_states = verification.get("fulltext_indexes", [])

    summary = {
        "status": status,
        "before": {
            "legal_article_count": before["legal_article_count"],
            "legal_doc_without_articles": before_coverage.get("no_articles", 0),
            "legal_doc_with_articles": before_coverage.get("has_articles", 0),
            "vector_chunk_count": before["vector_chunk_count"],
        },
        "after": {
            "legal_article_count": after["legal_article_count"],
            "legal_article_with_content": after["legal_article_with_content"],
            "has_article_count": after["has_article_count"],
            "vector_chunk_linked_count": after["vector_chunk_linked_count"],
        },
        "indexes": fulltext_index_states,
        "errors": errors,
        "warnings": warnings,
        "idempotence_check": idempotence,
        "next_action": next_action,
    }
    REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _table_row(metric: str, before_value: Any, after_value: Any) -> str:
        return f"| {metric} | {before_value} | {after_value} |"

    md_lines = [
        "# Week 1 Graph Migration Report",
        "",
        "## 1. Objective",
        "",
        (
            "Mục tiêu tuần 1 là kiểm tra hiện trạng graph, phát hiện lỗi schema "
            "và chuẩn hóa dữ liệu để phục vụ retrieval tốt hơn."
        ),
        "",
        "## 2. Initial Graph Status",
        "",
        f"- LegalArticle count: {before['legal_article_count']}",
        f"- LegalArticle with content: {before['legal_article_with_content']}",
        f"- LegalDoc without articles: {before_coverage.get('no_articles', 0)}",
        f"- LegalDoc with articles: {before_coverage.get('has_articles', 0)}",
        f"- HAS_ARTICLE count: {before['has_article_count']}",
        f"- VectorChunk count: {before['vector_chunk_count']}",
        f"- VectorChunk linked count: {before['vector_chunk_linked_count']}",
        "",
        "Top LegalDoc types before migration:",
        "",
        "| Type | Count |",
        "|---|---:|",
    ]
    md_lines.extend(
        f"| {row.get('type')} | {row.get('count')} |"
        for row in before.get("legal_doc_type_counts", [])[:12]
    )
    md_lines.extend([
        "",
        "## 3. Problem Diagnosis",
        "",
        (
            "Graph không thiếu dữ liệu. Vấn đề chính là các node Pháp Điển "
            "có `type = \"Điều (Pháp điển)\"` đang tồn tại dưới label `LegalDoc`, "
            "trong khi về mặt retrieval chúng tương đương `LegalArticle`. Vì vậy "
            "fulltext index và traversal theo `LegalArticle` không khai thác được "
            "khoảng 70k điều luật có nội dung."
        ),
        "",
        "## 4. Migration Actions",
        "",
        f"- Promoted Pháp Điển nodes to `LegalArticle`: {migration.get('promoted_pd_articles')}",
        f"- Cleaned parent stubs accidentally labeled as `LegalArticle`: {migration.get('cleaned_parent_article_labels', 0)}",
        f"- Created/merged parent `HAS_ARTICLE` pairs: {migration.get('pd_has_article_pairs')}",
        f"- VectorChunk link strategies: `{json.dumps(migration.get('vector_link_strategies', {}), ensure_ascii=False)}`",
        "- Rebuilt fulltext indexes: `legal_article_fulltext`, `vector_chunk_fulltext`, `legal_doc_fulltext`.",
        "- Verified sample fulltext retrieval and graph traversal.",
        "",
        "## 5. Before / After Comparison",
        "",
        "| Metric | Before | After |",
        "|---|---:|---:|",
        _table_row("LegalArticle count", before["legal_article_count"], after["legal_article_count"]),
        _table_row("LegalArticle with content", before["legal_article_with_content"], after["legal_article_with_content"]),
        _table_row("HAS_ARTICLE count", before["has_article_count"], after["has_article_count"]),
        _table_row("VectorChunk linked count", before["vector_chunk_linked_count"], after["vector_chunk_linked_count"]),
        "",
        "Fulltext indexes after migration:",
        "",
        "| Name | State |",
        "|---|---|",
    ])
    md_lines.extend(
        f"| {row.get('name')} | {row.get('state')} |"
        for row in fulltext_index_states
    )
    md_lines.extend([
        "",
        "## 6. Verification Result",
        "",
        f"- Sample legal_article_fulltext rows: {len(verification.get('sample_fulltext_article', []))}",
        f"- Sample LegalDoc -> LegalArticle rows: {len(verification.get('sample_doc_to_article', []))}",
        f"- Sample VectorChunk -> LegalDoc rows: {len(verification.get('sample_vector_to_doc', []))}",
        f"- Idempotence check pass: {idempotence.get('pass')}",
        "",
        "Sample fulltext result:",
        "",
        "```json",
        json.dumps(_compact_value(verification.get("sample_fulltext_article", []), 350), ensure_ascii=False, indent=2),
        "```",
        "",
        "Sample LegalDoc -> LegalArticle traversal:",
        "",
        "```json",
        json.dumps(_compact_value(verification.get("sample_doc_to_article", []), 350), ensure_ascii=False, indent=2),
        "```",
        "",
        "Sample VectorChunk -> LegalDoc traversal:",
        "",
        "```json",
        json.dumps(_compact_value(verification.get("sample_vector_to_doc", []), 350), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 7. Remaining Risks",
        "",
        "- HF nodes có thể vẫn cần chuẩn hóa sâu hơn theo văn bản/điều/khoản nếu muốn traversal chính xác hơn.",
        "- Một số VectorChunk có thể có khóa liên kết chưa được biểu diễn bằng `doc_id` hoặc `chunk_id`.",
        "- `law_id` của Pháp Điển đang được suy ra theo prefix `pd_xxx_xxx`; có thể cần normalize chi tiết hơn ở bước sau.",
        "- Full evaluation chưa chạy ở bước này.",
        "",
        "Warnings:",
        "",
    ])
    if warnings:
        md_lines.extend(f"- {warning}" for warning in warnings)
    else:
        md_lines.append("- None")
    md_lines.extend([
        "",
        "Errors:",
        "",
    ])
    if errors:
        md_lines.extend(f"- {error}" for error in errors)
    else:
        md_lines.append("- None")
    md_lines.extend([
        "",
        "## 8. Next Step",
        "",
        "- Chạy mini evaluation 50 câu.",
        "- Kiểm tra retrieval hit rate.",
        "- Sau đó mới chạy full evaluation.",
        "",
        f"Decision: `{next_action}`",
        "",
    ])
    REPORT_MD.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1 Neo4j graph schema migration")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = Neo4jClient(config["neo4j"])
    errors: list[str] = []
    warnings: list[str] = []
    migration: dict[str, Any] = {}

    try:
        if not client.verify_connection():
            raise RuntimeError("Cannot connect to Neo4j with the selected config")

        print("\nSTEP 1 - Inspecting graph before migration")
        before = inspect_graph(client)
        _print_json("Node counts by label", before["label_counts"])
        _print_json("LegalDoc counts by type", before["legal_doc_type_counts"][:20])
        _print_json("Doc-Article coverage", before["doc_article_coverage"])
        _print_json("Sample LegalDoc Phap Dien", before["pd_legal_doc_samples"])
        _print_json("Sample LegalDoc HF", before["hf_legal_doc_samples"])
        _print_json("Sample VectorChunk", before["vector_chunk_samples"])

        print("\nSTEP 2 - Promoting Phap Dien nodes to LegalArticle")
        migration["promoted_pd_articles"] = promote_pd_articles(client)
        print(f"Promoted/matched Phap Dien articles: {migration['promoted_pd_articles']}")
        migration["cleaned_parent_article_labels"] = cleanup_parent_article_labels(client)
        print(
            "Cleaned accidental LegalArticle labels from parent stubs: "
            f"{migration['cleaned_parent_article_labels']}"
        )

        print("\nSTEP 3 - Creating parent LegalDoc and HAS_ARTICLE relations")
        migration["pd_has_article_pairs"] = create_pd_parent_relations(client)
        print(f"Created/merged parent HAS_ARTICLE pairs: {migration['pd_has_article_pairs']}")

        print("\nSTEP 4 - Linking VectorChunk nodes to LegalDoc")
        migration["vector_link_strategies"] = link_vector_chunks(client, warnings)
        _print_json("Vector link strategy counts", migration["vector_link_strategies"])

        print("\nSTEP 5 - Rebuilding fulltext indexes")
        rebuild_fulltext_indexes(client, errors, warnings)

        print("\nSTEP 6 - Verifying graph after migration")
        after = inspect_graph(client)
        verification = verify_after(client, errors, warnings)
        idempotence = run_idempotence_check(client, warnings)

        _print_json("After summary", {
            "legal_article_count": after["legal_article_count"],
            "legal_article_with_content": after["legal_article_with_content"],
            "has_article_count": after["has_article_count"],
            "vector_chunk_linked_count": after["vector_chunk_linked_count"],
            "vector_chunk_belongs_to_count": after["vector_chunk_belongs_to_count"],
        })
        _print_json("Fulltext indexes", verification["fulltext_indexes"])
        _print_json("Sample fulltext search", verification["sample_fulltext_article"])
        _print_json("Sample LegalDoc -> LegalArticle", verification["sample_doc_to_article"])
        _print_json("Sample VectorChunk -> LegalDoc", verification["sample_vector_to_doc"])
        _print_json("Idempotence check", idempotence)

        write_reports(before, after, verification, migration, idempotence, errors, warnings)
        status = "PASS" if not errors and idempotence.get("pass") else "FAIL"
        print(f"\nWeek 1 graph migration status: {status}")
        print(f"Report written: {REPORT_MD}")
        print(f"Summary written: {REPORT_JSON}")
    except Exception as exc:
        errors.append(str(exc))
        logger.exception("Week 1 graph migration failed")
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(
            json.dumps({
                "status": "FAIL",
                "before": {},
                "after": {},
                "indexes": [],
                "errors": errors,
                "warnings": warnings,
                "next_action": "NEED_FIX",
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
