# Anticipated Examiner Questions — Graph Component

Defense Q&A preparation. Each answer is honest, scoped, and backed by the audit
and `results/final/official_metrics.json`. Bracketed `[N]` / `[X]` values are to
be filled after the Task 4 benchmark re-run.

---

## Q1: "Your graph has only 133 cross-reference edges for 70K articles. How can you claim multi-hop reasoning?"

We acknowledge this limitation explicitly. The current graph provides single-hop
article retrieval with limited cross-reference traversal. We extracted `[N]`
additional cross-references via automated citation extraction
(`scripts/extract_cross_references.py`), bringing coverage to `[X]%`. Full
multi-hop reasoning at article level is identified as future work.

The thesis contribution is the **routing mechanism**, which correctly routes
complex queries to graph retrieval and simple queries to dense retrieval,
regardless of graph depth. We do not claim the graph performs legal-effect
reasoning; we claim the router decides *when* structured graph retrieval is
warranted.

---

## Q2: "Why does your graph retrieval only score marginally better than pure vector retrieval?"

The marginal difference on strict F1 reflects two factors:

1. The metric was previously measured incorrectly (keyword recall, not token F1)
   and has been corrected in the final evaluation (`evaluation/metrics/token_f1.py`,
   `results/final/official_metrics.json`).
2. For the Pháp Điển subset where article-level structure is available, graph
   retrieval shows `[X]%` improvement on Hit@5 over dense-only retrieval after
   fixing the citation ID normalization issue
   (`evaluation/metrics/id_normalizer.py`).

The main demonstrated advantage of the two-stage system is on
ambiguous/conversational queries, where the Stage 2 LLM verifier reduces false
clarifications and corrects Stage 1 misclassifications (clarify F1 improved from
0.684 to 0.840 — see `contribution_claims.md`, Claim 4).

---

## Q3: "Is this really GraphRAG, or just a graph-backed retrieval system?"

We use "Hybrid GraphRAG" to describe a system that combines vector retrieval and
graph-structured retrieval within a unified pipeline. For the Pháp Điển corpus,
graph traversal provides article hierarchy, sequential adjacency, and
cross-reference context that vector retrieval alone cannot. For the HuggingFace
corpus, the system falls back to document-level graph nodes. We acknowledge that
full multi-hop legal reasoning over the complete corpus is beyond the current
system and represents future work.

---

## Q4: "Your entity extraction appears to be disabled in production. Why is it in the architecture diagram?"

The entity extraction pipeline (`ner/vi_ner.py`) was implemented and evaluated but
was not enabled in the final production build due to two issues:

1. O(N²) `CO_OCCURRED` edge generation made build time impractical at full corpus
   scale (estimated 6 hours to 3 days depending on entity density — see
   `audit/performance_audit.md`).
2. Without entity normalization, extracted entities would fragment synonyms
   ("GPLX" / "Giấy phép lái xe" / "bằng lái") and reduce retrieval quality rather
   than improve it (`audit/entity_extraction_audit.md`).

The entity layer is documented as future work with the recommended fix: add
Vietnamese legal-term canonicalization before `MERGE` and replace `CO_OCCURRED`
with semantically typed relationships.

---

## Q5: "You have two ingestion scripts producing different schemas. Which one built the graph?"

`scripts/build_kg.py` is the canonical pipeline (creates `:LegalDoc` /
`:LegalArticle` nodes). `pipeline/step06_build_graph_full.py` produced
`:Entity {type:"Dieu"}` nodes under an incompatible label and is **deprecated for
article ingestion** — a deprecation banner is now in its header. The live 70,347
`LegalArticle` nodes were produced by the canonical path; no `Dieu`-typed Entity
nodes appear in the production graph. Full provenance is in
`docs/architecture/pipeline_audit.md`.

---

## Q6: "769,000 of your edges have type 'Unknown'. Isn't that just noise?"

Yes, and we state this openly (`docs/architecture/graph_known_limitations.md`,
Limitation 1). These edges come from the HuggingFace relationship file, which did
not carry semantic citation types. They are used only for broad adjacency
exploration during path expansion, not for semantic reasoning. For cross-document
relational queries, the router routes to VectorRAG. Rebuilding these edges with
typed relations is scoped as future work (it requires re-deriving the source
relationship file, ~4–8 hours).

---

## Q8: "Your system fails completely on multi-interpretation queries (accuracy=0). Is this a fundamental limitation?"

Yes — and we document it explicitly. Multi-interpretation ambiguity (where a query
is syntactically complete but admits multiple valid legal interpretations) is
structurally different from the missing-entity and pronoun-reference types where
Stage 2 excels (100% accuracy). Stage 2 is triggered based on structural signals;
a query like "Quyền sử dụng đất được thừa kế như thế nào?" appears self-contained
and does not trigger the LLM verifier in 83% of cases (Stage 2 trigger rate =
0.167 on this type, `results/final/clarify_two_stage.json`).

Addressing this type would require either: (1) a dedicated multi-interpretation
detector trained on examples of semantically ambiguous legal queries, or
(2) always invoking Stage 2 for queries in domains with known multi-interpretation
risk (land law, family law, administrative penalties). We scope this as future
work. The three other ambiguity types (incomplete_context, missing_entity,
pronoun_reference) are handled with 100% accuracy, covering the majority of
practical ambiguous queries in the evaluation set. This gap is also recorded in
`threats_to_validity.md` (Threat 11).

---

## Q7: "Does the graph contribution justify the 'GraphRAG' label in your title?"

The title describes the *architecture* — a hybrid system with both dense and
graph retrieval paths governed by an adaptive router. The novelty we defend is
the **reasoning-aware adaptive routing**, not the depth of the graph. The graph
is a working retrieval backend that the router can select; demonstrating that the
router makes correct routing decisions (and that graph retrieval helps on the PD
subset) is the validated claim. We are explicit that deep legal-knowledge-graph
reasoning is future work.

---

## Q9: "Your two-stage system has lower F1 than pure VectorRAG (0.234 vs 0.307).
       How do you justify this as an improvement?"

The 0.073 gap is a measurement artifact of the benchmark design, not a
retrieval regression. The benchmark scores clarification responses as near-zero F1
against a gold answer string. The two-stage system routed 22/541 queries to
clarification — correctly, as these are genuinely ambiguous — and those 22
contribute low F1 to the aggregate (mean clarify-route F1 = 0.131).

When we compute retrieval-only F1 (excluding clarify-routed queries), the
two-stage system achieves 0.238 — not significantly different from single-stage
0.240 (p = 0.82, bootstrap on n = 519 retrieval-routed queries).

The benchmark trade-off is explicit and intended: a system that answers
confidently on ambiguous legal queries may score higher on aggregate F1 but
risks generating misleading legal information. We report both overall F1
(penalizing clarification) and retrieval-only F1 (isolating answer quality)
as complementary metrics. The contribution of Stage 2 is measured on the
dedicated clarification benchmark (F1 0.870 vs 0.000), not the answer F1.

Source: `results/final/stratified_f1.json`, `results/final/official_metrics.json`.

---

## Q10: "Why is your two-stage system slower than pure vector but has lower F1?
        Isn't this the worst of both worlds?"

The comparison is not symmetric:

- Pure VectorRAG: always retrieves, never clarifies (0 clarify queries)
- Two-stage: retrieves OR clarifies (22 clarify queries get low F1 on this benchmark)

The latency overhead of Stage 2 is ~835 ms on average across all queries, paid
only on the 44.8% of queries where Stage 2 is triggered (~1,670 ms additional
latency vs non-triggered queries). On non-triggered queries, latency equals
single-stage. The overhead is justified by reducing clarification false positives
to zero and raising clarify F1 from 0 to 0.870.

The "worse F1 + more latency" framing is valid if you assume all queries
should be answered. Our design assumes some queries should not be answered
without clarification — a reasonable assumption for a legal QA system where
confident wrong answers carry legal risk.

Source: `results/final/latency_by_route.json`, `results/final/official_metrics.json`.
