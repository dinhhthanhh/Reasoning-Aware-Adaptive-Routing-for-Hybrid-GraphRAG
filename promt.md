# CORPUS REBUILD DECISION PROMPT
# Strategic pivot: evaluate Pháp Điển-only corpus as the primary fix.
# Hypothesis: 61.8% unresolvable IDs + 93% multi-Điều chunks = corpus structure
#             problem, not retrieval ranking problem.
# BM25 is deprioritized until corpus structure is confirmed clean.
#
# DO NOT modify routing system.
# DO NOT run entity extraction or CO_OCCURRED.
# DO NOT modify finalized LaTeX thesis or paper files.

---

## CONTEXT — WHY THIS PROMPT SUPERSEDES THE PREVIOUS ONE

The previous prompt proposed BM25 + PD canonical mapping as the primary fix.
This prompt revises that based on new analysis:

The 61.8% unresolvable ID rate is NOT primarily a mapping bug.
It is a corpus architecture problem:
- Gold test answers reference canonical law IDs: `52/2014/QH13::Điều_8`
- Corpus documents are indexed with heterogeneous IDs:
    `pd_007_003_0044` (Pháp Điển structural)
    `hf_processed_4260` (HuggingFace position)
    `Document_4266` (unknown origin)
- Chunking cuts across Điều boundaries (93% multi-Điều chunks confirmed)
- BM25 cannot fix retrieval when the indexed units are wrong-granularity fragments

The correct fix order is:
  1. Audit corpus coverage (does Pháp Điển cover the test set?)
  2. If yes → rebuild corpus from Pháp Điển only, with correct structure
  3. Then benchmark
  4. Then decide if BM25 is still needed

---

## PHASE 0 — EXTENDED AUDIT (complete all tasks, no code changes)

Phase 0 from the previous prompt remains valid. Run it fully.
This phase adds ONE critical new task: corpus coverage audit.

---
0.0A – Pháp Điển Source Verification

Verify:

- phapdien_processed.jsonl exists
- full article text exists
- law_number exists
- article_number exists

Output:

docs/retrieval/phapdien_source_audit.md

If any field missing:
    STOP

### 0.0 — Corpus Coverage Audit (NEW — run this first)

This is the gate condition for everything that follows.

**Task:** For the 600 test questions, determine what % of gold articles
exist in the Pháp Điển corpus (the 70,075 LegalArticle nodes in Neo4j).

```python
# scripts/audit_corpus_coverage.py
"""
For each test question in data/datasets/legal_strict_test_cleaned.jsonl:
1. Extract gold article reference: law_number + article_number
   e.g. {"law_id": "52/2014/QH13", "article_id": "Điều 8"}
2. Attempt to find this article in:
   (a) Pháp Điển corpus (Neo4j LegalArticle nodes)
   (b) HuggingFace corpus (Neo4j LegalDoc nodes, HF label)
   (c) Not found in either

Output per question:
{
  "test_id": "legal_strict_test_0001",
  "gold_law": "52/2014/QH13",
  "gold_article": "Điều 8",
  "in_phapdien": true | false,
  "in_hf": true | false,
  "in_neither": true | false,
  "law_type": "Luật" | "Nghị định" | "Thông tư" | "Quyết định" | "VBHN" | "other"
}

Aggregate:
{
  "total": 600,
  "in_phapdien_only": N,
  "in_hf_only": N,
  "in_both": N,
  "in_neither": N,
  "phapdien_coverage_pct": X.X,
  "by_law_type": {
    "Luật": {"total": N, "in_phapdien": N, "pct": X.X},
    "Nghị định": {"total": N, "in_phapdien": N, "pct": X.X},
    "Thông tư": {"total": N, "in_phapdien": N, "pct": X.X},
    "Quyết định": {"total": N, "in_phapdien": N, "pct": X.X},
    "VBHN": {"total": N, "in_phapdien": N, "pct": X.X}
  }
}
"""
```

Generate: `docs/retrieval/corpus_coverage_audit.md`
Generate: `results/final/corpus_coverage_audit.json`

---

### 0.1 — Retrieval ID Audit (from previous prompt, still needed)

Investigate the ID mismatch failure modes.
Generate: `docs/retrieval/retrieval_id_audit.md`

---

### 0.2 — Chunk Audit (from previous prompt, still needed)

Measure chunk boundary quality. Specifically:
- What % of chunks contain multiple Điều?
- For PD articles (typically short, 1-5 lines): are they kept whole or split?
- For HF documents (longer): where do chunk boundaries fall?

Generate: `docs/retrieval/chunk_audit.md`

---

### 0.3 — Oracle Recall Analysis (from previous prompt, still needed)

Run after 0.0 and 0.1 to understand actual retrieval recall.
Generate: `docs/retrieval/retrieval_recall_report.md`
Generate: `results/final/retrieval_recall_analysis.json`

---

### 0.4 — Decision Gate

Based on 0.0 through 0.3, complete this decision matrix:

Generate: `docs/retrieval/rebuild_decision.md`

```markdown
# Corpus Rebuild Decision

## Coverage Result
Pháp Điển coverage of test set: ___%
- Luật: ___% | Nghị định: ___% | Thông tư: ___% | QĐ: ___% | VBHN: ___%

## Decision

[ ] IF coverage >= 70%:
    → PROCEED with Pháp Điển-first rebuild (Phase 1A)
    → HuggingFace corpus: ARCHIVE (move to archive/, remove from active pipeline)
    → Rationale: 70%+ coverage means we can rebuild a clean, canonical corpus
      and accept ~30% coverage gap (documented as limitation)

[ ] IF coverage 50-69%:
    → PARTIAL rebuild: keep Pháp Điển as primary, supplement with targeted HF docs
    → Only keep HF documents whose law_number appears in test set gold answers
    → Rebuild chunking for both sources: 1 Điều = 1 chunk
    → Assign canonical IDs to all documents from both sources

[ ] IF coverage < 50%:
    → Do NOT rebuild corpus

→ Rechunk existing corpus:
   - PD: 1 Điều = 1 chunk
   - HF: semantic chunk

→ Fix evaluator ID normalization

→ Keep heterogeneous corpus

→ Re-benchmark

## Chunk Strategy (all paths)
Regardless of coverage, the chunk strategy must change:
  Current: variable-length chunks, 93% cross Điều boundaries
  Target:  1 LegalArticle = 1 chunk (for PD content)
           1 semantic unit per chunk (for any non-PD content)
```
---
### 0.5 — Pháp Điển Source Audit (MANDATORY)

Before any rebuild, inspect the actual Pháp Điển source.

Report:

- total article count
- total unique laws
- law types:
    - Luật
    - Nghị định
    - Thông tư
    - Quyết định
    - VBHN
- percentage containing:
    - law_number
    - article_number
    - full article text
- percentage missing canonical identifiers

Generate:

docs/retrieval/phapdien_source_audit.md

STOP if:
- law_number missing > 20%
- full article text unavailable
---

## PHASE 1A — PHÁP ĐIỂN REBUILD (if coverage >= 70%)

**This is the primary path if coverage audit supports it.**
Estimated effort: 6-10 hours.
Expected impact: Eliminate ID mismatch, correct chunk structure, clean evaluation.

### 1A.1 — Design canonical ID format

All documents must use a single canonical ID format from day 1:

```
Format:  {law_number}::{article_number}
Example: 52/2014/QH13::Điều_8
         100/2019/NĐ-CP::Điều_23
         15/2012/QH13::Điều_86
```

This format must be used consistently in:
- Neo4j node properties (doc_id, article_id)
- ChromaDB chunk metadata
- Evaluation scripts
- Gold test set references (verify gold IDs match this format)

### 1A.2 — Rebuild the Pháp Điển processing pipeline

Audit `data/raw/phapdien_processed.jsonl` for available fields.
Each Pháp Điển article should have:
- Source law name and number (e.g. "52/2014/QH13")
- Article number (e.g. "Điều 8")
- Full article content (NOT truncated preview)
- Chủ đề (topic/domain)
- Đề mục (subject heading)

Create: `pipelines/preprocessing/build_phapdien_canonical.py`

```python
"""
Rebuild Pháp Điển corpus with canonical IDs and clean structure.

Input:  data/raw/phapdien_processed.jsonl
Output: data/processed/phapdien_canonical.jsonl

Each output record:
{
  "canonical_id": "52/2014/QH13::Điều_8",
  "law_number": "52/2014/QH13",
  "law_name": "Luật Hôn nhân và Gia đình",
  "article_number": "Điều 8",
  "article_title": "Điều kiện kết hôn",
  "content": "[full article text, not truncated]",
  "chu_de": "Hôn nhân - Gia đình",
  "de_muc": "Kết hôn",
  "source": "phapdien"
}

Requirements:
- canonical_id must be unique across all records
- content must be complete article text (if current JSONL has truncated
  content_preview, investigate where full text is stored)
- If law_number cannot be extracted: log and SKIP the record (do not invent IDs)
- Report: total records, skipped records, coverage by law type
"""
```
---
### 1A.2b — Coverage Dry Run

Create a temporary index using 5,000 randomly sampled
Pháp Điển articles.

Run:

- 50-question benchmark
- Recall@5
- Hit@5 strict
- Token F1

Compare against current baseline.

If:
    Recall improvement < 20%
    AND
    Hit@5 improvement < 10%

STOP.

Do not continue full rebuild.
---

### 1A.3 — Rebuild ChromaDB index with 1 article = 1 chunk

```python
# pipelines/indexing/index_phapdien_canonical.py
"""
Index phapdien_canonical.jsonl into ChromaDB.

Chunking strategy: 1 LegalArticle = 1 chunk.
- If article content < 512 tokens: store as single chunk
- If article content > 512 tokens: split by Khoản boundaries
  (NOT by token count — split at "Khoản 1.", "Khoản 2.", etc.)
- Each chunk gets metadata: canonical_id, law_number, article_number, chunk_index

chunk_id format: {canonical_id}::chunk_{index}
Example: 52/2014/QH13::Điều_8::chunk_0
         52/2014/QH13::Điều_8::chunk_1 (if split by Khoản)
"""
```

### 1A.4 — Rebuild Neo4j with canonical IDs

DO NOT delete existing graph.

Build a parallel graph:

database:
    legal_graphrag_phapdien_only

Keep the original graph intact.

Only switch after benchmark comparison.

```python
# scripts/build_phapdien_only_graph.py
"""
Build Neo4j graph from phapdien_canonical.jsonl only.

Node structure:
  (:LegalDoc {
    doc_id: "52/2014/QH13",         ← canonical law number
    law_name: "Luật Hôn nhân và Gia đình",
    law_type: "Luật",               ← Luật/NĐ/TT/QĐ
    source: "phapdien"
  })

  (:LegalArticle {
    article_id: "52/2014/QH13::Điều_8",  ← canonical article ID
    law_id: "52/2014/QH13",
    article_number: "Điều 8",
    article_title: "Điều kiện kết hôn",
    content: "[full text]",
    chu_de: "Hôn nhân - Gia đình"
  })

  (:VectorChunk {
    chunk_id: "52/2014/QH13::Điều_8::chunk_0",
    article_id: "52/2014/QH13::Điều_8",
    content: "[chunk text]"
  })

Relationships:
  (LegalDoc)-[:HAS_ARTICLE]->(LegalArticle)
  (LegalArticle)-[:HAS_CHUNK]->(VectorChunk)
  (VectorChunk)-[:BELONGS_TO_ARTICLE]->(LegalArticle)

NO CO_OCCURRED.
NO Entity nodes.
NO HuggingFace nodes.
"""
```

Guard: after rebuild, verify:
```cypher
MATCH ()-[r:CO_OCCURRED]->() RETURN count(r);  // must be 0
MATCH (n:LegalArticle) RETURN count(n);         // must be ~70,000
MATCH (n:LegalDoc) RETURN count(n);             // must be ~number of unique laws
MATCH (n:VectorChunk) RETURN count(n);          // must be ~70,000 to ~140,000
```

### 1A.5 — Verify gold test set compatibility

After rebuild, run the compatibility check:
```python
# scripts/verify_test_set_compatibility.py
"""
For each test question:
1. Extract gold canonical_id from supporting_facts
2. Look up canonical_id in new Neo4j graph
3. Report: % found, % not found
Target: >= 00% found (matching corpus_coverage_audit result)
"""
```

If < 80% of test set gold articles are found:
- Add targeted supplementary documents (only the specific laws in the test set
  that Pháp Điển is missing)
- Re-run the compatibility check

---

## PHASE 2 — REBUILD BENCHMARK (after Phase 1A)

Run the full-corpus benchmark after Phase 1A is complete.

```bash
# Pure vector baseline with rebuilt corpus
python scripts/run_benchmark_eval.py \
  --dataset data/datasets/legal_strict_test_cleaned.jsonl \
  --systems pure_vector \
  --retriever dense \
  --output_dir results/final/phapdien_only_benchmark/

# Then two_stage if pure_vector looks good
python scripts/run_benchmark_eval.py \
  --dataset data/datasets/legal_strict_test_cleaned.jsonl \
  --systems two_stage_hybrid \
  --retriever dense \
  --output_dir results/final/phapdien_only_benchmark/
```

Expected outcomes after clean corpus rebuild:
- Unresolvable ID rate: < 10% (was 61.8%)
- Hit@5 strict: > 0.10 (was 0.024)
- Pure Vector F1: > 0.35 (was 0.307) — assuming retrieval now finds correct articles

If Pure Vector F1 >= 0.35 after rebuild → STOP. This is sufficient.
Update official_metrics.json and thesis.

---

## STOP CONDITIONS (same as previous prompt)

```
STOP A: Pure Vector F1 >= 0.35 after Phase 1A rebuild
STOP B: Two-stage hybrid F1 (retrieval-only) >= 0.35
STOP C: Oracle gap < 0.10 (actual F1 > 0.45)
STOP D: Phase 0 shows coverage < 50% → switch to partial fix path, not rebuild
STOP E: Time budget exhausted — document limitations, do not sacrifice defense prep
STOP F:

If phapdien_coverage_pct < 50%

OR

coverage of law types appearing in benchmark
(NĐ + VBHN + QĐ)

< 40%

Then:

Abort Pháp Điển-only rebuild.

Reason:
Benchmark distribution is incompatible with corpus.
```

---

## WHAT IS DEFERRED (not in this prompt)

These are explicitly deferred until after Phase 1A + Phase 2:

```
BM25 hybrid retrieval    → defer until corpus structure is correct
                           (BM25 on wrong-granularity chunks = wasted effort)

Entity extraction        → NOT in scope for thesis
Agency nodes             → good for demo, not for F1
Temporal metadata        → good for production, not for F1
LegalConcept taxonomy    → consider only if corpus rebuild is insufficient
NEXT_ARTICLE extension   → low priority
```

---

## DELIVERABLES (in order)

Phase 0:
```
results/final/corpus_coverage_audit.json     ← GATE CONDITION
docs/retrieval/corpus_coverage_audit.md
docs/retrieval/retrieval_id_audit.md
docs/retrieval/chunk_audit.md
docs/retrieval/retrieval_recall_report.md
docs/retrieval/rebuild_decision.md           ← DECISION DOCUMENT
```

Phase 1A (if coverage >= 70%):
```
data/processed/phapdien_canonical.jsonl
pipelines/preprocessing/build_phapdien_canonical.py
pipelines/indexing/index_phapdien_canonical.py
scripts/build_phapdien_only_graph.py
scripts/verify_test_set_compatibility.py
[rebuilt Neo4j graph + ChromaDB index]
```

Phase 2:
```
results/final/phapdien_only_benchmark/       ← new benchmark results
results/final/official_metrics.json          ← updated
docs/chapter5_results_draft.tex              ← updated tables
docs/paper_errata.md                         ← updated
```

---

## ABSOLUTE CONSTRAINTS

1. DO NOT start Phase 1A without completing Phase 0.0 (coverage audit).
2. DO NOT rebuild if coverage < 80% — use the partial fix path instead.
3. DO NOT enable CO_OCCURRED at any point.
4. DO NOT fabricate benchmark numbers.
5. DO NOT modify routing system.
6. DO NOT modify finalized LaTeX thesis or paper files.
7. Archive existing Neo4j graph and ChromaDB before wiping:
   `archive/graph_backups/pre_phapdien_rebuild_{date}/`
8. If corpus rebuild reveals full article text is missing (only content_preview
   exists in phapdien_processed.jsonl), STOP and report — do not proceed with
   truncated content.