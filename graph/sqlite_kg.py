"""SQLite-based Knowledge Graph for multi-hop legal reasoning.

Provides a lightweight, Neo4j-free alternative for graph traversal.
Built from the QA dataset's evidence and relevant_articles fields,
and from the articles_full.jsonl for cross-reference edges.

Schema:
  - nodes: (article_id, law_id, doc_number, content, title)
  - edges: (src_article, dst_article, relation_type, weight)
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from loguru import logger
import tqdm

from pipeline.i18n import get_template

_DB_DEFAULT = Path("data/kg/legal_kg.db")


@contextmanager
def _conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


ARTICLE_REF_PATTERN = re.compile(
    r"(?:Điều\s+(\d+[a-zA-Z]?)).*?(?:của\s+)?"
    r"(Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định)\s+([^\s,;.]+)",
    re.IGNORECASE,
)

DOC_NUM_PATTERN = re.compile(
    r"\b(\d{1,4}/\d{4}/[A-ZĐÁÀẢÃẠ\-]+)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_kg_from_hotpot(
    hotpot_path: str | Path = "data/en_benchmark/raw/hotpot_train_v1.1.json",
    db_path: str | Path = _DB_DEFAULT,
) -> int:
    """Build SQLite KG from HotpotQA dataset.
    
    Supports:
    1. Original JSON: List of objects with 'context' and 'supporting_facts'.
    2. Processed JSONL: Flat lines of docs with 'doc_id' as 'hotpot_<qid>_<idx>'.
    
    Args:
        hotpot_path: Path to dataset file.
        db_path: SQLite database output path.
        
    Returns:
        Number of edges added.
    """
    db_path = Path(db_path)
    if db_path.parent.exists() and not db_path.parent.is_dir():
        logger.warning("Found file at {}, removing to create directory", db_path.parent)
        db_path.parent.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    hotpot_path = Path(hotpot_path)
    if not hotpot_path.exists():
        logger.error("HotpotQA dataset not found: {}", hotpot_path)
        return 0

    # Detect format: JSON or JSONL
    is_jsonl = hotpot_path.suffix.lower() == ".jsonl"
    
    samples = []
    if is_jsonl:
        with open(hotpot_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        logger.info("Loaded {} documents from JSONL corpus", len(samples))
    else:
        with open(hotpot_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        logger.info("Loaded {} samples from JSON dataset", len(samples))

    with _conn(db_path) as con:
        _create_schema(con)
        edges_added = 0

        # Case 1: Original JSON (with supporting_facts)
        if not is_jsonl and isinstance(samples, list) and samples and "supporting_facts" in samples[0]:
            for item in tqdm.tqdm(samples, desc="Building Hotpot KG (JSON)"):
                context_dict = {title: " ".join(sents) 
                               for title, sents in item.get("context", [])}
                supporting = item.get("supporting_facts", [])
                supporting_titles = list({sf[0] for sf in supporting})
                
                for title in supporting_titles:
                    content = context_dict.get(title, "")
                    con.execute(
                        "INSERT OR IGNORE INTO nodes (article_id, law_id, content, title) VALUES (?, ?, ?, ?)",
                        (title, "", content[:3000], title),
                    )

                if len(supporting_titles) >= 2:
                    for i in range(len(supporting_titles)):
                        for j in range(i + 1, len(supporting_titles)):
                            con.execute(
                                "INSERT OR IGNORE INTO edges (src_article, dst_article, relation_type, question_ctx) VALUES (?, ?, ?, ?)",
                                (supporting_titles[i], supporting_titles[j], "co_supporting", item.get("question", "")[:200]),
                            )
                            edges_added += 1

        # Case 2: Flattened JSONL Corpus (group by qid)
        else:
            # Group by qid: hotpot_<qid>_<idx>
            groups = {}
            for doc in samples:
                doc_id = doc.get("doc_id", "")
                if doc_id.startswith("hotpot_"):
                    # Extract qid: everything between first and last underscore
                    parts = doc_id.split("_")
                    if len(parts) >= 3:
                        qid = parts[1]
                        if qid not in groups: groups[qid] = []
                        groups[qid].append(doc)
                
                # Still insert node individually
                title = doc.get("title", "")
                content = doc.get("content", "")
                con.execute(
                    "INSERT OR IGNORE INTO nodes (article_id, law_id, content, title) VALUES (?, ?, ?, ?)",
                    (title, "", content[:3000], title),
                )

            # Create edges within groups (cliques)
            for qid, docs in tqdm.tqdm(groups.items(), desc="Creating edges from qid groups"):
                titles = [d.get("title", "") for d in docs if d.get("title")]
                if len(titles) >= 2:
                    for i in range(len(titles)):
                        for j in range(i + 1, len(titles)):
                            con.execute(
                                "INSERT OR IGNORE INTO edges (src_article, dst_article, relation_type, question_ctx) VALUES (?, ?, ?, ?)",
                                (titles[i], titles[j], "qid_group", f"Group {qid}"),
                            )
                            edges_added += 1
        
        node_count = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    
    logger.info("HotpotQA KG built | nodes={} | edges={} | db={}", node_count, edge_count, db_path)
    return edges_added


def build_kg_from_qa(
    qa_train_path: str | Path = "qa_pipeline/data/final/train.json",
    db_path: str | Path = _DB_DEFAULT,
) -> int:
    """Build SQLite KG from the QA training dataset.

    Extracts nodes from relevant_articles and edges from co-occurrence
    within the same QA pair (multi-hop samples).

    Args:
        qa_train_path: Path to QA train split JSON.
        db_path: SQLite database output path.

    Returns:
        Total number of edges created.
    """
    db_path = Path(db_path)
    if db_path.parent.exists() and not db_path.parent.is_dir():
        logger.warning("Found file at {}, removing to create directory", db_path.parent)
        db_path.parent.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    qa_path = Path(qa_train_path)
    if not qa_path.exists():
        logger.error("QA dataset not found: {}", qa_path)
        return 0

    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Building SQLite KG from {} QA samples...", len(data))

    with _conn(db_path) as con:
        _create_schema(con)

        nodes_added = 0
        edges_added = 0

        for item in data:
            relevant = item.get("relevant_articles", [])
            evidence = item.get("evidence", "")
            question = item.get("question", "")
            hop = item.get("hop_count", 1)
            is_cross = item.get("is_cross_doc", False)

            # Add nodes
            for art in relevant:
                art_id = art.get("article_id", "")
                law_id = art.get("law_id", "")
                if art_id and law_id:
                    con.execute(
                        """INSERT OR IGNORE INTO nodes
                           (article_id, law_id, content, title)
                           VALUES (?, ?, ?, ?)""",
                        (
                            f"{art_id}_{law_id}",
                            law_id,
                            evidence[:2000] if isinstance(evidence, str) else "",
                            f"{art_id} — {law_id}",
                        ),
                    )
                    nodes_added += 1

            # Add edges for multi-hop: connect pairs of relevant_articles
            if hop >= 2 and len(relevant) >= 2:
                relation = "cross_doc_reference" if is_cross else "intra_doc_reference"
                for i in range(len(relevant)):
                    for j in range(i + 1, len(relevant)):
                        src_id = f"{relevant[i].get('article_id')}_{relevant[i].get('law_id')}"
                        dst_id = f"{relevant[j].get('article_id')}_{relevant[j].get('law_id')}"
                        if src_id and dst_id:
                            con.execute(
                                """INSERT OR IGNORE INTO edges
                                   (src_article, dst_article, relation_type, question_ctx)
                                   VALUES (?, ?, ?, ?)""",
                                (src_id, dst_id, relation, question[:200]),
                            )
                            edges_added += 1

        # Count results
        node_count = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    logger.info(
        "SQLite KG built | nodes={} | edges={} | db={}",
        node_count, edge_count, db_path,
    )
    return edge_count


def build_kg_from_articles(
    articles_path: str | Path = "data/processed/articles.json",
    db_path: str | Path = _DB_DEFAULT,
    max_articles: int = 10000,
) -> int:
    """Extend SQLite KG from processed articles JSON.

    Extracts cross-references between articles mentioned in the text.

    Args:
        articles_path: Path to articles.json.
        db_path: SQLite database path (will extend existing).
        max_articles: Maximum articles to process.

    Returns:
        Number of new edges added.
    """
    db_path = Path(db_path)
    if db_path.parent.exists() and not db_path.parent.is_dir():
        logger.warning("Found file at {}, removing to create directory", db_path.parent)
        db_path.parent.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    articles_path = Path(articles_path)
    if not articles_path.exists():
        logger.warning("Articles file not found: {}", articles_path)
        return 0

    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    logger.info("Extending KG from {} articles...", min(len(articles), max_articles))

    edges_added = 0
    with _conn(db_path) as con:
        _create_schema(con)

        for art in articles[:max_articles]:
            art_id = art.get("article_id", art.get("id", ""))
            law_id = art.get("law_id", art.get("doc_number", ""))
            content = art.get("content", "")
            title = art.get("title", "")

            if not art_id:
                continue

            node_key = f"{art_id}_{law_id}" if law_id else art_id
            con.execute(
                """INSERT OR REPLACE INTO nodes
                   (article_id, law_id, content, title)
                   VALUES (?, ?, ?, ?)""",
                (node_key, law_id, content[:3000], title),
            )

            # Extract cross-references from content
            doc_refs = DOC_NUM_PATTERN.findall(content)
            for ref_num in doc_refs[:10]:
                dst_id = f"*_{ref_num}"
                con.execute(
                    """INSERT OR IGNORE INTO edges
                       (src_article, dst_article, relation_type, question_ctx)
                       VALUES (?, ?, ?, ?)""",
                    (node_key, dst_id, "cites", title[:100]),
                )
                edges_added += 1

    logger.info("Extended KG | new edges~{} | db={}", edges_added, db_path)
    return edges_added


# ─────────────────────────────────────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────────────────────────────────────

class SQLiteKG:
    """Query interface for the SQLite Knowledge Graph.

    Supports neighbor lookup, multi-hop traversal, and
    text search across article content.
    """

    def __init__(self, db_path: str | Path = _DB_DEFAULT, language: str = "vi") -> None:
        self.db_path = Path(db_path)
        self.language = language
        if not self.db_path.exists():
            if self.language == "en":
                logger.error(
                    "English SQLite KG not found at {}. Please build it manually using: "
                    "python graph/sqlite_kg.py --hotpot data/en_benchmark/raw/hotpot_train_v1.1.json "
                    "--db {}", 
                    self.db_path, self.db_path
                )
            else:
                logger.warning("SQLite KG not found at {}, building from QA data...", self.db_path)
                build_kg_from_qa(db_path=self.db_path)
                build_kg_from_articles(db_path=self.db_path)

    def is_available(self) -> bool:
        """Check if the KG has data."""
        if not self.db_path.exists():
            return False
        with _conn(self.db_path) as con:
            cnt = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        return cnt > 0

    def search_nodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Full-text search over article content and titles.

        Args:
            query: Search text.
            top_k: Maximum results to return.

        Returns:
            List of node dicts with article_id, law_id, title, content.
        """
        with _conn(self.db_path) as con:
            # Simple LIKE search (fast enough for small KG)
            raw_words = query.lower().split()
            
            # Clean punctuation from words first
            cleaned_words = [re.sub(r"[^\w]", "", w) for w in raw_words]
            cleaned_words = [w for w in cleaned_words if w] # Remove empty
            
            # Filter stopwords for English to improve LIKE quality
            if self.language == "en":
                stopwords = {"what", "is", "the", "a", "an", "of", "and", "or", "in", "on", "with", "for", "to", "was", "were", "first", "last"}
                words = [w for w in cleaned_words if w not in stopwords][:5]
            else:
                words = raw_words[:5]

            if not words:
                words = cleaned_words[:3] # Fallback to cleaned words if all filtered

            conditions = " OR ".join(
                f"(content LIKE ? OR title LIKE ?)" for _ in words
            )
            params = []
            for w in words:
                params.extend([f"%{w}%", f"%{w}%"])

            rows = con.execute(
                f"SELECT article_id, law_id, title, content FROM nodes "
                f"WHERE {conditions} LIMIT ?",
                params + [top_k],
            ).fetchall()

        return [dict(r) for r in rows]

    def get_neighbors(
        self,
        article_id: str,
        depth: int = 2,
        max_nodes: int = 20,
    ) -> list[dict[str, Any]]:
        """BFS traversal from an article node.

        Args:
            article_id: Starting article node key.
            depth: Maximum hop depth.
            max_nodes: Maximum nodes to return.

        Returns:
            List of edge dicts: {src, dst, relation, content}.
        """
        results: list[dict[str, Any]] = []
        visited: set[str] = {article_id}
        frontier: list[tuple[str, int]] = [(article_id, 0)]

        with _conn(self.db_path) as con:
            while frontier and len(results) < max_nodes:
                current, hop = frontier.pop(0)
                if hop >= depth:
                    continue

                rows = con.execute(
                    """SELECT e.src_article, e.dst_article, e.relation_type,
                              n.content, n.title
                       FROM edges e
                       LEFT JOIN nodes n ON n.article_id = e.dst_article
                       WHERE e.src_article = ? OR e.dst_article = ?
                       LIMIT 10""",
                    (current, current),
                ).fetchall()

                for row in rows:
                    neighbor = row["dst_article"] if row["src_article"] == current else row["src_article"]
                    results.append({
                        "src": row["src_article"],
                        "dst": row["dst_article"],
                        "relation": row["relation_type"],
                        "content": (row["content"] or "")[:500],
                        "title": row["title"] or "",
                    })
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append((neighbor, hop + 1))

        return results[:max_nodes]

    def multi_hop_context(self, query: str, top_k: int = 3) -> str:
        """Get formatted multi-hop context string for a query.

        Searches for relevant nodes, then retrieves their neighbors.

        Args:
            query: Legal query text.
            top_k: Starting nodes to explore.

        Returns:
            Formatted context string for LLM consumption.
        """
        start_nodes = self.search_nodes(query, top_k=top_k)
        if not start_nodes:
            return get_template(self.language, "no_graph_found")

        context_parts: list[str] = []
        seen_ids: set[str] = set()

        for node in start_nodes:
            nid = node["article_id"]
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            title = node.get("title", nid)
            content = (node.get("content") or "")[:600]
            context_parts.append(f"[{title}]\n{content}")

            # Get neighbors
            neighbors = self.get_neighbors(nid, depth=2, max_nodes=5)
            for nb in neighbors[:3]:
                nb_content = nb.get("content", "")[:300]
                nb_title = nb.get("title", nb.get("dst", ""))
                relation = nb.get("relation", "related" if self.language == "en" else "liên quan")
                if nb_content and nb_title not in seen_ids:
                    seen_ids.add(nb_title)
                    context_parts.append(
                        f"  → [{nb_title}] ({relation}):\n  {nb_content}"
                    )

        if not context_parts:
            return get_template(self.language, "no_graph_found")
            
        return "\n\n".join(context_parts)


def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            article_id  TEXT PRIMARY KEY,
            law_id      TEXT,
            content     TEXT,
            title       TEXT
        );
        CREATE TABLE IF NOT EXISTS edges (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            src_article     TEXT NOT NULL,
            dst_article     TEXT NOT NULL,
            relation_type   TEXT DEFAULT 'references',
            question_ctx    TEXT,
            UNIQUE(src_article, dst_article, relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_article);
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_article);
        CREATE INDEX IF NOT EXISTS idx_nodes_law ON nodes(law_id);
    """)


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Build SQLite Knowledge Graph")
    parser.add_argument("--qa", help="Path to Vietnamese Legal QA train JSON")
    parser.add_argument("--hotpot", help="Path to HotpotQA train JSON")
    parser.add_argument("--articles", help="Path to processed articles JSON")
    parser.add_argument("--db", default=str(_DB_DEFAULT), help="Output SQLite DB path")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    
    if args.hotpot:
        build_kg_from_hotpot(args.hotpot, db_path)
    elif args.qa:
        build_kg_from_qa(args.qa, db_path)
        if args.articles:
            build_kg_from_articles(args.articles, db_path)
    else:
        print("Please specify either --qa or --hotpot to build the KG.")
        sys.exit(1)
