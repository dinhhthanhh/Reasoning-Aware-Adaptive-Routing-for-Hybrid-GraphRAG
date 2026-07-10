# Graph Contribution — Honest Scope Statement

This statement defines exactly what the graph component contributes, based on the
architectural audit (`audit/`). Use it in the defense presentation and paper
revision. It is deliberately conservative.

## What the graph provides (confirmed by audit)

1. **Structured article-level storage** for 70,347 Pháp Điển articles with a
   parent-document hierarchy (`LegalDoc → LegalArticle` via `HAS_ARTICLE`).

2. **Sequential article adjacency** via `NEXT_ARTICLE` (239 edges), enabling
   neighbouring-article retrieval for Pháp Điển content.

3. **Cross-article reference traversal** via `CROSS_REFERENCES` (133 original
   edges, extended by `scripts/extract_cross_references.py` using regex extraction
   of explicit citations from article text — real extracted signal, not invented
   data).

4. **Vector-to-graph bridging:** `VectorChunk` nodes (≈199K) connect dense
   retrieval hits back to structured article and document nodes
   (`HAS_VECTOR_CHUNK` / `BELONGS_TO`), enabling hybrid context assembly.

5. **Structured fulltext indexing** via Neo4j's `legal_article_fulltext` index
   over article titles and content, with weighted keyword ranking and
   domain-contradiction penalties in `graph/neo4j_client.py`.

## What the graph does NOT provide (by current audit)

1. Multi-hop legal-effect reasoning — `AMENDS`/`GUIDES`/`REPEALS` coverage
   <0.015% at document level.
2. Sub-article hierarchy — `Khoản` (clause) and `Điểm` (sub-clause) are not
   represented as nodes.
3. Article decomposition for the HuggingFace corpus — 149K documents, 0 article
   nodes.
4. A normalized entity layer — entity extraction is disabled in the production
   build (`LegalConcept` has 10 nodes).
5. Temporal validity / legal status filtering — no reliable `effective_date` or
   `status` property.
6. Colloquial-to-formal term bridging — e.g., "sổ đỏ" → "Giấy chứng nhận quyền
   sử dụng đất".

## Honest framing for thesis defense

The graph component of the Hybrid GraphRAG system functions as:

> A structured retrieval index that provides article-level granularity for Pháp
> Điển content, cross-article adjacency exploration, and vector-to-graph context
> bridging — as a practical first step toward full legal knowledge graph
> reasoning.

The **primary contribution** of the thesis is the **two-stage adaptive router**
(XGBoost Stage 1 + Qwen3-32B-AWQ LLM verifier Stage 2), which selects between
dense retrieval, graph retrieval, and clarification based on query complexity,
reasoning demand, and latency constraints. The graph serves as the retrieval
backend for complex queries; the router determines when graph retrieval is
warranted.

This framing is accurate, defensible, and still novel.
