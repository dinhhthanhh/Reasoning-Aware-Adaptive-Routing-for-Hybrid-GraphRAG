"""Extract article-level cross-references from LegalArticle.content.

The audit (audit/graph_density_analysis.md) found only 133 CROSS_REFERENCES edges
for 70,347 articles (0.19% coverage). This script performs a regex pass over the
content of existing LegalArticle nodes to extract explicit article citations and
inserts CROSS_REFERENCES edges — without rebuilding the graph.

Design constraints (per promt.md, Task 2):
- Only create an edge when BOTH source and target articles exist in the graph.
- Use MERGE to avoid duplicates.
- Never create edges to articles not already present.
- Log total new edges created.

Two resolution modes are supported for the citation target:
  1. Same-document citation ("Điều 5 của Luật này") -> resolve within source doc_id.
  2. Cross-document citation ("Điều 12 Luật X / Nghị định Y") -> resolve by the
     referenced article number within the cited document, when that document is in
     the graph.

Usage:
    python scripts/extract_cross_references.py [--config configs/config.yaml]
                                               [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.neo4j_client import Neo4jClient  # noqa: E402
from loguru import logger  # noqa: E402


# Same-document reference: "Điều N của Luật này", "Điều N của Pháp lệnh này", bare "Điều N"
SAME_DOC_PATTERNS = [
    re.compile(r"Điều\s+(\d+[a-zđ]?)\s+của\s+(?:Luật|Pháp\s+lệnh|Nghị\s+định|Thông\s+tư)\s+này", re.IGNORECASE),
    re.compile(r"(?:căn\s+cứ|theo\s+quy\s+định\s+tại|quy\s+định\s+tại)\s+Điều\s+(\d+[a-zđ]?)(?!\d)", re.IGNORECASE),
]

# Cross-document reference: "Điều N (của) Luật/Nghị định/Thông tư <id-or-name>"
CROSS_DOC_PATTERN = re.compile(
    r"Điều\s+(\d+[a-zđ]?)\s+(?:của\s+)?"
    r"(Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Pháp\s+lệnh)\s+"
    r"([^\s,;.]+(?:\s+[^\s,;.]+){0,4})",
    re.IGNORECASE,
)

DOC_NUM_PATTERN = re.compile(r"\d{1,4}/\d{4}/[A-ZĐ\-]+", re.IGNORECASE)


def fetch_articles(client: Neo4jClient, limit: int | None) -> list[dict]:
    cypher = """
    MATCH (a:LegalArticle)
    WITH a, coalesce(a.content, a.content_preview, "") AS body
    WHERE body <> ""
    RETURN a.article_id AS article_id,
           a.doc_id     AS doc_id,
           a.law_id     AS law_id,
           body         AS content,
           a.title      AS title
    """
    if limit:
        cypher += f"\nLIMIT {int(limit)}"
    return client.query(cypher)


def extract_same_doc_targets(content: str) -> set[str]:
    """Return set of cited article numbers that point at the same document."""
    found: set[str] = set()
    for pat in SAME_DOC_PATTERNS:
        for m in pat.finditer(content):
            found.add(m.group(1).lower())
    return found


def extract_cross_doc_targets(content: str) -> list[tuple[str, str]]:
    """Return list of (article_number, cited_doc_token) cross-document citations."""
    out: list[tuple[str, str]] = []
    for m in CROSS_DOC_PATTERN.finditer(content):
        art_num = m.group(1).lower()
        tail = m.group(3)
        doc_num = DOC_NUM_PATTERN.search(tail)
        token = doc_num.group(0) if doc_num else tail.strip()
        if token:
            out.append((art_num, token))
    return out


def build_edge_batch(client: Neo4jClient, articles: list[dict]) -> list[dict]:
    """Resolve citations to concrete (source_article_id, target_article_id) pairs.

    Resolution is done with MATCH queries so we only emit edges where the target
    node actually exists in the graph.
    """
    edges: list[dict] = []
    # Index existing articles by (doc_id, normalized article number) for same-doc lookup.
    index = client.query(
        """
        MATCH (a:LegalArticle)
        RETURN a.article_id AS article_id, a.doc_id AS doc_id,
               toLower(a.article_id) AS aid_lower, a.title AS title
        """
    )
    by_doc_num: dict[tuple[str, str], str] = {}
    num_re = re.compile(r"điều\s+(\d+[a-zđ]?)", re.IGNORECASE)
    for row in index:
        doc_id = str(row.get("doc_id") or "")
        aid = str(row.get("article_id") or "")
        hay = f"{row.get('aid_lower') or ''} {str(row.get('title') or '').lower()}"
        m = num_re.search(hay)
        if doc_id and aid and m:
            by_doc_num.setdefault((doc_id, m.group(1).lower()), aid)

    seen: set[tuple[str, str]] = set()
    for art in articles:
        src_id = str(art.get("article_id") or "")
        src_doc = str(art.get("doc_id") or art.get("law_id") or "")
        content = str(art.get("content") or "")
        if not src_id or not src_doc or not content:
            continue

        for num in extract_same_doc_targets(content):
            tgt = by_doc_num.get((src_doc, num))
            if tgt and tgt != src_id and (src_id, tgt) not in seen:
                seen.add((src_id, tgt))
                edges.append({"source": src_id, "target": tgt, "kind": "same_doc"})

    logger.info("Resolved {} candidate CROSS_REFERENCES edges", len(edges))
    return edges


def insert_edges(client: Neo4jClient, edges: list[dict], dry_run: bool) -> int:
    if dry_run:
        logger.info("[dry-run] would insert {} edges", len(edges))
        return 0
    inserted = 0
    batch_size = 500
    with client._get_driver().session(database=client.database) as session:
        for i in range(0, len(edges), batch_size):
            chunk = edges[i : i + batch_size]
            result = session.run(
                """
                UNWIND $rows AS row
                MATCH (s:LegalArticle {article_id: row.source})
                MATCH (t:LegalArticle {article_id: row.target})
                MERGE (s)-[r:CROSS_REFERENCES]->(t)
                SET r.extracted = true, r.method = row.kind
                RETURN count(r) AS cnt
                """,
                rows=chunk,
            )
            rec = result.single()
            inserted += rec["cnt"] if rec else 0
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: {}", config_path)
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = Neo4jClient(config["neo4j"])
    if not client.verify_connection():
        logger.error("Could not connect to Neo4j.")
        client.close()
        return

    try:
        articles = fetch_articles(client, args.limit)
        logger.info("Fetched {} LegalArticle nodes with content", len(articles))
        edges = build_edge_batch(client, articles)
        inserted = insert_edges(client, edges, args.dry_run)
        logger.info("CROSS_REFERENCES edges inserted/merged: {}", inserted)
    finally:
        client.close()


if __name__ == "__main__":
    main()
