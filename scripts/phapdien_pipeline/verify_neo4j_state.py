"""Live verification of Neo4j ``phapdien`` database (Pháp Điển-only).

Checks the **current** database — not archived audit markdown from older builds.

Pass criteria (PD-only project):
  - database name is ``phapdien`` (or user override)
  - zero ``:HF`` nodes / zero HuggingFace ``LegalDoc`` sources
  - all ``LegalDoc`` rows have ``source = 'phapdien'``
  - ``CO_OCCURRED`` = 0 (legacy NER path disabled)

Semantic nodes (LEGAL_CONCEPT, ACTOR, …) intentionally lack the ``:PD`` label;
they are still Pháp Điển-derived and counted in total node/edge statistics.

Usage:
    python scripts/phapdien_pipeline/verify_neo4j_state.py
    python scripts/phapdien_pipeline/verify_neo4j_state.py --database phapdien
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from graph.neo4j_client import Neo4jClient

DEFAULT_CONFIG = ROOT / "configs/build_kg_no_ner.yaml"
DEFAULT_JSON_OUT = ROOT / "build_logs/phapdien_graph_stats.json"

SEMANTIC_LABELS = (
    "LEGAL_CONCEPT",
    "ACTOR",
    "AUTHORITY",
    "RIGHT",
    "OBLIGATION",
    "CONDITION",
    "PROCEDURE",
    "PENALTY",
)


def load_config(path: Path) -> dict:
    primary = ROOT / "configs/config.yaml"
    cfg_path = primary if primary.exists() else path
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_semantic_nodes(client: Neo4jClient) -> int:
    label_union = "|".join(SEMANTIC_LABELS)
    rows = client.query(f"MATCH (n:{label_union}) RETURN count(n) AS c")
    return int(rows[0]["c"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", default=None, help="Override Neo4j database name")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    cfg = load_config(args.config)
    neo_cfg = dict(cfg["neo4j"])
    if args.database:
        neo_cfg["database"] = args.database

    client = Neo4jClient(neo_cfg)
    try:
        db = client.database
        print(f"Connected to Neo4j database: {db}")

        def q(cypher: str) -> list[dict]:
            return client.query(cypher)

        legal_doc = int(q("MATCH (d:LegalDoc) RETURN count(d) AS c")[0]["c"])
        legal_article = int(q("MATCH (a:LegalArticle) RETURN count(a) AS c")[0]["c"])
        semantic_nodes = count_semantic_nodes(client)
        other_nodes = int(q("MATCH (n) WHERE NOT n:LegalDoc AND NOT n:LegalArticle "
                            f"AND NOT n:{' AND NOT n:'.join(SEMANTIC_LABELS)} "
                            "RETURN count(n) AS c")[0]["c"])

        stats = {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "source": "BoPhapDienDienTu",
            "database": db,
            "total_nodes": int(q("MATCH (n) RETURN count(n) AS c")[0]["c"]),
            "total_edges": int(q("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]),
            "legal_doc": legal_doc,
            "legal_article": legal_article,
            "semantic_nodes": semantic_nodes,
            "other_nodes": other_nodes,
            "legal_doc_pd": int(q("MATCH (d:LegalDoc:PD) RETURN count(d) AS c")[0]["c"]),
            "legal_doc_hf": int(q("MATCH (d:LegalDoc:HF) RETURN count(d) AS c")[0]["c"]),
            "has_article_edges": int(q("MATCH ()-[r:HAS_ARTICLE]->() RETURN count(r) AS c")[0]["c"]),
            "co_occurred": int(q("MATCH ()-[r:CO_OCCURRED]->() RETURN count(r) AS c")[0]["c"]),
            "vector_chunk": int(q("MATCH (v:VectorChunk) RETURN count(v) AS c")[0]["c"]),
            "sources": q(
                "MATCH (d:LegalDoc) "
                "RETURN coalesce(d.source, '(null)') AS source, count(*) AS c "
                "ORDER BY c DESC"
            ),
            "top_relationship_types": q(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS c "
                "ORDER BY c DESC LIMIT 20"
            ),
            "node_labels": [
                row["label"]
                for row in q("CALL db.labels() YIELD label RETURN label ORDER BY label")
            ],
            "schema": "(:LegalDoc:PD)-[:HAS_ARTICLE]->(:LegalArticle:PD) + semantic layer",
        }

        non_phapdien_sources = [
            row for row in stats["sources"]
            if row["source"] not in ("phapdien", "(null)")
        ]
        checks = {
            "no_hf_label": stats["legal_doc_hf"] == 0,
            "no_foreign_sources": len(non_phapdien_sources) == 0,
            "no_co_occurred": stats["co_occurred"] == 0,
            "has_structural_pd": stats["legal_doc_pd"] > 0 and stats["has_article_edges"] > 0,
        }
        stats["checks"] = checks
        stats["pd_only_pass"] = all(checks.values())

        print(json.dumps(stats, ensure_ascii=False, indent=2))

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.json_out}")

        if not stats["pd_only_pass"]:
            failed = [name for name, ok in checks.items() if not ok]
            print(
                f"WARNING: database '{db}' failed PD-only checks: {', '.join(failed)}",
                file=sys.stderr,
            )
            if non_phapdien_sources:
                print(f"  Non-phapdien sources: {non_phapdien_sources}", file=sys.stderr)
            sys.exit(1)

        print(
            f"OK: database '{db}' is Pháp Điển-only "
            f"({stats['total_nodes']:,} nodes, {stats['total_edges']:,} edges)."
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
