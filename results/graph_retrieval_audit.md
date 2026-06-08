# Graph Retrieval Audit

## Scope

This audit checks the current graph retrieval design and post-migration graph quality. It does not rebuild Neo4j, migrate graph data, retrain the router, or change retrieval code.

## Current Implementation

- Graph route entry point: `rag/graph_rag_adapter.py`.
- Neo4j client: `graph/neo4j_client.py`.
- Pipeline fallback policy: `pipeline/hybrid_pipeline.py`.
- Neo4j retrieval tries `Neo4jClient.get_multi_hop_context(query, top_k=...)` first, then falls back to SQLite KG if Neo4j returns no useful context.
- Start-node retrieval uses full-text indexes over `LegalArticle` and `VectorChunk`, plus metadata search over `LegalDoc`.
- Path expansion retrieves 1-2 hop paths from candidate start nodes and prioritizes `LegalArticle`/`LegalDoc` endpoints over generic nodes.
- Hybrid retrieval merges vector context and graph context, then falls back to vector-only answer generation if hybrid synthesis is uninformative.

## Post-migration Graph Status

From `eval_results/post_migration_graph_quality.json`:

- Nodes: `419,251`
- Relationships: `1,239,542`
- `LegalArticle`: `70,347`
- `VectorChunk`: `199,530`
- `HAS_ARTICLE`: `70,347`
- `BELONGS_TO`: `199,530`
- Average degree: `5.91`
- Isolated nodes: `3,102`

The graph migration fixed the major schema problem: Pháp điển article nodes are now visible as `LegalArticle` and connected through `HAS_ARTICLE`. Sample graph contexts contain real legal text, so graph retrieval is no longer blocked by empty article content.

## Strengths

- Article-level graph retrieval is now viable after the `LegalArticle` migration.
- `BELONGS_TO` links connect vector chunks back to legal documents, which supports hybrid evidence serialization.
- Full-text indexes over article/chunk/doc layers give the graph route a practical lexical entry point.
- Pipeline fallback reduces user-facing failures when graph retrieval returns no useful answer.

## Main Risks

- Relation distribution is imbalanced. `REFERENCES` has `769,000` edges, while curated legal-effect edges are sparse: `AMENDS=27`, `REPEALS=19`, `GUIDES=25`, `IMPLEMENTS=3`.
- Because generic references dominate, graph traversal can retrieve broad related context instead of legally decisive amendment/repeal/effective-date evidence.
- Candidate start-node selection is still mostly lexical/full-text based. It does not yet expose a calibrated entity-linking confidence score.
- `_expand_paths` excludes `VectorChunk` endpoints during graph path expansion, so graph-only context may miss useful chunk text unless the start node or hybrid route already provides it.
- There is no dedicated graph-retrieval-only diagnostic set measuring whether retrieved paths contain the gold article/document before answer generation.

## Recommended Next Steps

1. Add relation-specific retrieval templates for amendment, repeal, effective date, issuing authority, article membership, and transitional clauses.
2. Add graph retrieval diagnostics before LLM generation: start-node hit rate, gold article hit rate, gold document hit rate, path relation type distribution, and context length.
3. Add a small manually checked graph-reasoning set focused on `AMENDS`, `REPEALS`, `HAS_ARTICLE`, `REFERENCES`, and effective-date questions.
4. Enrich sparse legal-effect relations from document metadata and legal text patterns.
5. Log graph context source and relation chain for each graph/hybrid answer so route errors can be separated from retrieval errors.

## Current Conclusion

The graph is now good enough for thesis experiments and demo after migration, but graph retrieval quality is still limited by relation sparsity and lexical start-node matching. The next improvement should be graph retrieval diagnostics and relation-specific traversal, not another full graph rebuild.
