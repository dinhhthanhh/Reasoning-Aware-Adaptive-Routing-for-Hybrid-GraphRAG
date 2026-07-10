# Contribution Claims (Evidence-Backed)

Each claim below states what was implemented, what was measured, and what **cannot**
be claimed. All numbers trace to files under `results/final/` or
`router_model/training_report.json`.

---

## Claim 1: Adaptive routing beats fixed baselines

**Implemented:** Two-stage router (Stage 1 XGBoost + selective Stage 2 LLM verifier)
routes queries to `dense_retrieval`, `graph_traversal`, `hybrid_reasoning`, or
`clarify`.

**Measured (routing, 5-fold CV on strict train set):**

| Metric | Value | Source |
|---|---:|---|
| CV accuracy | **0.8517 ± 0.0249** | `router_model/training_report.json` |
| CV macro-F1 | 0.8399 ± 0.0262 | same |

**Measured (offline re-score, cleaned test n=541):**

| System | Routing acc. | Token F1 |
|---|---:|---:|
| Router (honest run) | 0.8503 | 0.3408 |
| Pure Graph | 0.2773 | 0.3390 |
| Pure Vector (retrieval only) | — | 0.0000* |

\*Pure vector file has no generated answers.

**What this shows:** The router assigns routes far more accurately than always-vector
(0.50 structural) or always-graph (0.25 structural) baselines.

**Cannot claim:** End-to-end routing accuracy of 96.67% — that is the rejected
data-leakage figure (`router_run_b_LEAKED` in official_metrics.json).

---

## Claim 2: Reasoning-first, clarify-last design

**Implemented:** `router/two_stage_router.py` + `router/history_resolver.py`

- History resolution runs **before** clarify decisions.
- Resolved history **blocks** unnecessary clarification
  (`resolved_history_blocks_auto_clarify`).
- Ambiguity override is **denied by default** for relation-heavy queries.

**Measured (conversation stress test, diagnostic):**

| Metric | Value | Source |
|---|---:|---|
| Clarify F1 (post Phase 3) | 0.862 | `results_phase3/conversation_ambiguity_summary.json` |
| History-resolution accuracy | 0.850 | same |

**Cannot claim:** Conversation stress test replaces the strict 600-query benchmark.

---

## Claim 3: Two-stage adds robustness on ambiguous queries

**Implemented:** Stage 2 LLM verifier invoked selectively (~43% trigger rate on
canonical snapshot).

**Measured (canonical snapshot, legacy keyword-recall F1):**

| System | F1 | Routing acc. | Latency |
|---|---:|---:|---:|
| Single-stage | 0.4231 | 0.9350 | 2,209 ms |
| Two-stage | 0.4235 | 0.9283 | 3,913 ms |

Delta F1 = **0.0004** — not meaningful on strict benchmark.

**Significance (real token F1, router vs pure graph, n=541):**

| Comparison | Δ F1 | p-value | Significant? |
|---|---:|---:|---|
| router vs pure_graph | +0.0018 | 0.793 | **No** |

**Honest framing:** Two-stage does **not** significantly beat single-stage on
strict end-to-end answer F1. Its value is on **ambiguous/conversational** cases
where Stage 2 catches Stage 1 misclassifications and prevents false clarifications
when history resolves referents. The +1.7 s latency cost is justified only for
those cases, not for all queries.

---

## Claim 4: Clarification behaviour

**Measured (clarify benchmark, n=234, diagnostic):**

| Metric | Pre-Phase-3 | Post-Phase-3 |
|---|---:|---:|
| Clarify F1 | 0.684 | **0.840** |

Source: `results_phase3/clarify_eval_summary.json`

**Cannot claim:** Stage 1 was trained on 3 retrieval labels (no `clarify` label).
Stage 1 clarify F1 = 0 is expected, not a bug.

---

## Claim 5: Graph provides structured retrieval for Pháp Điển content

**Implemented:** Neo4j graph (419K nodes, 1.24M relationships) with
`LegalDoc → LegalArticle` hierarchy, `NEXT_ARTICLE` adjacency, `CROSS_REFERENCES`,
and `VectorChunk` bridging. See `audit/graph_schema.md`.

**Honest scope (per `audit/` and `docs/architecture/graph_known_limitations.md`):**

- The graph is a **structured retrieval index** for Pháp Điển content, not a
  reasoning-capable knowledge graph over the full corpus.
- Article-level structure exists only for the 70,347 Pháp Điển articles; the
  149K HuggingFace documents are document-level only.

**Reframed claims (replace prior overstatements):**

- ✅ "The adaptive router correctly routes multi-hop queries to the graph
  retrieval path, where article-level structure for Pháp Điển content provides
  richer context than document-level retrieval."
- ✅ "The two-stage routing mechanism preserves GraphRAG's structured retrieval
  benefits for Pháp Điển content while avoiding unnecessary graph traversal for
  simple lookup queries."

**Cannot claim:**

- ❌ "The graph enables multi-hop legal reasoning" — `AMENDS`/`GUIDES`/`REPEALS`
  coverage <0.015%; not supported.
- ❌ "GraphRAG outperforms VectorRAG on complex queries" — not demonstrated with
  the corrected token F1 metric; the PD-subset comparison (Task 4) is the only
  honest venue for any graph-advantage claim.

---

## What we explicitly do NOT claim

1. Two-stage beats single-stage on strict answer F1 (Δ ≈ 0).
2. Hit@5 > 5% without fixing the Pháp Điển ↔ VBPL ID scheme mismatch.
3. VBPL as a data source (crawler failed; archived).
4. 96.67% routing accuracy (data leakage).
5. The old "F1 = 0.4235" as standard token F1 (it was keyword recall).
