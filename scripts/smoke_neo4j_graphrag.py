"""Read-only smoke test for the GraphRAGAdapter Neo4j path.

This validates that the same adapter/client path used by the benchmark can
resolve Neo4j settings from config or environment and run a read-only query.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

from rag.graph_rag_adapter import GraphRAGAdapter


def main() -> int:
    load_dotenv()
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    adapter = GraphRAGAdapter(config)
    client = getattr(adapter, "_neo4j_client", None)
    if client is None:
        print("neo4j_adapter_smoke: failed (client not initialized)")
        return 1

    try:
        if not client.verify_connection():
            print("neo4j_adapter_smoke: failed (connectivity check failed)")
            return 1
        rows = client.query("RETURN 1 AS ok")
        if rows != [{"ok": 1}]:
            print("neo4j_adapter_smoke: failed (unexpected read query result)")
            return 1
        print("neo4j_adapter_smoke: passed")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
