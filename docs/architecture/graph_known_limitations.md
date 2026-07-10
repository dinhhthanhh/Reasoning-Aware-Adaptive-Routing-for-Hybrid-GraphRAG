# Graph — Known Limitations

This document records confirmed limitations of the production Neo4j graph,
established by the architectural audit (`audit/`). It is intended to be cited in
the thesis defense and paper revision so that limitations are stated honestly
rather than discovered by an examiner.

---

## Known Limitation 1: Untyped REFERENCES edges

769,000 `REFERENCES` relationships exist at document level. All have
`type = "Unknown"` due to missing type information in the source relationship
file (`relationships_final.jsonl`); the ingester defaults to `"Unknown"` when no
type is present (`scripts/build_kg.py`).

**Impact:** These edges cannot support semantic cross-document reasoning
(e.g., "which law amends X?"). They are used only for broad adjacency exploration
in path expansion. Typed legal-effect coverage is negligible:
`AMENDS` (27), `GUIDES` (25), `REPEALS` (19), `IMPLEMENTS` (3) — all <0.015% of
the 219K document nodes.

**Mitigation:** The system falls back to VectorRAG for cross-document relational
queries. The adaptive router routes such queries to GraphRAG only when Pháp Điển
article-level structure is available; otherwise dense retrieval handles them.

---

## Known Limitation 2: No HuggingFace article decomposition

149,051 HuggingFace `LegalDoc` nodes (≈68% of all document nodes) have **zero**
child `LegalArticle` nodes. Article-level structure exists only for the Pháp Điển
corpus (70,347 articles).

**Impact:** Article-level Hit@k metrics are only meaningful for the Pháp Điển
subset. For HF-sourced gold articles, graph retrieval returns the parent document
and relies on the LLM to locate the relevant clause.

**Mitigation:** Honest scoping — graph advantage is claimed only for the Pháp Điển
subset (see `docs/defense_materials/graph_contribution_statement.md`). HF article
decomposition is scoped as future work.

---

## Known Limitation 3: Sparse article-level cross-references

Before extraction: 133 `CROSS_REFERENCES` edges for 70,347 articles (0.19%).
After `scripts/extract_cross_references.py`: increased via regex extraction of
explicit "Điều N của Luật này" / "căn cứ Điều N" citations, restricted to pairs
where both source and target articles exist in the graph.

**Impact:** Multi-hop article-to-article traversal is limited. The extracted
edges are real signal from article text, not invented data.

**Mitigation:** Documented as a partial improvement; full citation graph
construction is future work.

---

## Known Limitation 4: Entity layer disabled in production

NER-based `Entity` extraction (`ner/vi_ner.py`, invoked by `build_kg.py`) is not
enabled in the production build. Two reasons:

1. `CO_OCCURRED` edge generation is O(N²) in entities per document, making
   full-corpus build time impractical.
2. Without entity canonicalization, surface forms fragment synonyms
   ("GPLX" vs "Giấy phép lái xe" vs "bằng lái") into separate nodes, reducing
   rather than improving retrieval quality.

**Impact:** No normalized entity/concept bridge layer exists. `LegalConcept` has
only 10 nodes.

**Mitigation:** Entity layer documented as future work with the recommended fix:
add a Vietnamese legal-term canonicalization dictionary before `MERGE`, and
replace `CO_OCCURRED` with semantically typed relationships.

---

## Known Limitation 5: No temporal validity / legal status

No node carries reliable `effective_date`, `expiry_date`, or `status`
(active/repealed/superseded). The 19 `REPEALS` edges cover <0.01% of documents.

**Impact:** The system may return superseded articles as authoritative. Cannot
filter traversal to in-effect law.

**Mitigation:** Documented as future work; out of scope for the current
graduation-thesis implementation window.

---

## Known Limitation 6: Pháp Điển IDs not resolved to canonical legal IDs

Pháp Điển article identifiers (e.g., `pd_007_003_0044`, `Điều 8.4.LQ.8`) are
structural codes, not canonical Vietnamese law references ("Điều 8, Luật Hôn nhân
và gia đình 2014"). No mapping table exists.

**Impact:** Retrieval evaluation using canonical gold IDs requires the
`evaluation/metrics/id_normalizer.py` layer; PD structural codes still cannot be
fully resolved to canonical form.

**Mitigation:** Documented as future work (build a PD-code → canonical-law map).

---

## Summary

The graph functions as a **structured retrieval index** for Pháp Điển
content, not a reasoning-capable knowledge graph over the full corpus. All
limitations above are scoped as known limitations / future work in the thesis,
and the primary contribution is reframed onto the two-stage adaptive router
(see `docs/defense_materials/graph_contribution_statement.md`).
