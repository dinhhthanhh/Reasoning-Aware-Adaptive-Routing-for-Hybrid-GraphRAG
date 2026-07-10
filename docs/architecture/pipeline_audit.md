# Graph Ingestion Pipeline Audit — Which Pipeline Is Canonical

## Problem

Two scripts can write nodes representing legal content into Neo4j, using
**incompatible schemas**:

| Script | Node label created | Key | Source data |
|---|---|---|---|
| `scripts/build_kg.py` | `:LegalDoc` (+ `:HF`/`:PD`/`:CoreLaws`), `:Entity` | `doc_id` (unique) / `name` | `*_processed.jsonl` + `relationships_final.jsonl` |
| `pipeline/step06_build_graph_full.py` | `:Entity` (via `Neo4jClient.batch_insert_nodes`) | `name` | `articles_full.jsonl` |

`step06` runs (in `graph/neo4j_client.py::_insert_nodes_tx`):

```cypher
MERGE (n:Entity {name: node.name})
SET n.type = node.type, n += node.properties
```

So an article ingested by `step06` becomes `:Entity {name: "<law> - <title>", type: "Dieu"}`,
**not** `:LegalArticle`. The query layer (`get_multi_hop_context_by_chunks`,
`_find_start_nodes`, `_expand_paths`) matches and label-boosts `:LegalArticle`
nodes (boost 8.0). `:Entity` article nodes score 0 and are effectively invisible
to graph retrieval.

## Verification queries (run against the live instance)

```cypher
// How many proper article nodes exist?
MATCH (n:LegalArticle) RETURN count(n) AS legal_article_count;

// How many article-like Entity nodes from step06 exist?
MATCH (n:Entity) WHERE n.type = 'Dieu' RETURN count(n) AS dieu_entity_count;

// Sanity: confirm LegalArticle nodes carry article_id / doc_id
MATCH (n:LegalArticle) RETURN n.article_id, n.doc_id LIMIT 5;
```

## Finding (from live graph snapshot)

`results_final_unified/e2e_benchmark/post_migration_graph_quality.json` reports:

- `LegalArticle`: **70,347** nodes
- No `Dieu`-typed `:Entity` article nodes appear in the live label/relationship
  breakdown.

**Conclusion: the live 70,347 `LegalArticle` nodes were produced by the
`build_kg.py` / migration path, NOT by `step06_build_graph_full.py`.** The
`step06` path is not represented in the production graph.

## Canonical pipeline

> **`scripts/build_kg.py` is the canonical article/document ingestion pipeline.**
> `pipeline/step06_build_graph_full.py` is **deprecated for article ingestion**
> and must not be re-run. A deprecation banner has been added to the top of
> `step06_build_graph_full.py`.

## Action taken

1. Deprecation comment added to `pipeline/step06_build_graph_full.py` (header).
2. This audit recorded as the source of truth for pipeline provenance.
3. No data was modified; no rebuild was performed.

## Residual risk

If `step06` is ever re-run by mistake, it will inject `:Entity {type:"Dieu"}`
nodes that duplicate article content under an unqueryable label. The deprecation
banner is the mitigation. A future rebuild should delete `step06` or convert it
to emit `:LegalArticle` nodes via `build_kg.py`'s ingestion path.
