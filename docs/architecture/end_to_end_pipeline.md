# End-to-End Pipeline

```
DATA SOURCES (Pháp Điển offline + HuggingFace corpus)
    ↓
DATA CLEANING (remove 84 placeholder answers)
    ↓
DATASET GENERATION (regenerate splits → scripts/regenerate_splits.py)
    ↓
INDEXING (ChromaDB dense index + Neo4j knowledge graph)
    ↓
RETRIEVAL (dense via ChromaDB, graph traversal via Neo4j)
    ↓
ROUTING (Stage 1 XGBoost → optional Stage 2 Qwen3-32B-AWQ)
    ↓
ANSWER GENERATION (LLM with retrieved context)
    ↓
EVALUATION (token F1, BERTScore, Hit@5, MRR, route accuracy, latency)
    ↓
FINAL RESULTS (results/final/official_metrics.json)
```

---

## Stage 1: Data Sources

| Source | Script | Output | Size |
|---|---|---|---|
| HuggingFace `th1nhng0/vietnamese-legal-documents` | `main.py --source huggingface` | `data/huggingface/` | ~1.23M rows |
| Pháp Điển offline bundle | `main.py --source phapdien` | `data/phapdien/phapdien_all.json` | 70,075 articles |
| ~~VBPL portal~~ | **ARCHIVED** — produced empty output | `archive/legacy_vbpl/` | 0 records |

Merged corpus: `data/processed/final_corpus.jsonl` (248,740 documents).

**Dependencies:** `requests`, `beautifulsoup4`, local `BoPhapDienDienTu/` bundle.

---

## Stage 2: Data Cleaning

| Input | Script | Output |
|---|---|---|
| Raw corpus JSONL | `scripts/clean_phapdien.py`, `scripts/clean_and_merge_hf.py` | `data/processed/` |
| QA raw | `qa_pipeline/pipeline/step*.py` | `qa_pipeline/data/final/` |
| Strict splits | `scripts/build_legal_strict_splits.py` | `qa_pipeline/data/legal_strict/` |
| **Placeholder removal** | **`scripts/regenerate_splits.py`** | **`qa_pipeline/data/legal_strict_clean/`** |

Audit report: `data/audit_reports/qa_quality_audit.md`

---

## Stage 3: Indexing

| Index | Script | Storage | Config key |
|---|---|---|---|
| Dense (Chroma) | `scripts/build_vectordb.py` | `data/vector_store/chroma_harrier_oss_0_6b` | `vector_store.collection` |
| Graph (Neo4j) | `scripts/build_kg.py` | Neo4j instance | `neo4j.uri` |
| GraphRAG workspace | `graphrag_wrapper/setup_graphrag.py` | `data/graphrag_workspace/` | optional |

Embedding model: `microsoft/Harrier-OSS-v1-0.6B` (config example).

---

## Stage 4: Retrieval

| Backend | Module | When used |
|---|---|---|
| VectorRAG | `rag/vector_rag.py` + `vector_store/chroma_store.py` | `dense_retrieval` route |
| GraphRAG | `rag/graph_rag_adapter.py` + `graph/neo4j_client.py` | `graph_traversal` route |
| Hybrid | Both, merged in `pipeline/hybrid_pipeline.py` | `hybrid_reasoning` route |

**Known issue:** Retrieved article IDs use Pháp Điển structural codes; gold QA
uses VBPL document numbers. Normalizer at `evaluation/metrics/id_normalizer.py`.

---

## Stage 5: Routing

See `docs/routing_design.md` for full detail.

| Component | File |
|---|---|
| Orchestrator | `router/two_stage_router.py` |
| Stage 1 model | `router/router_model.py` |
| Stage 2 verifier | `router/llm_reasoning_verifier.py` |
| History resolver | `router/history_resolver.py` |
| Ambiguity detector | `router/ambiguity_detector.py` |
| Feature extraction | `router/features.py` |

Training: `scripts/run_router_training.py` → `data/router_training/legal_strict/router_model.pkl`

---

## Stage 6: Answer Generation

| Component | File |
|---|---|
| Pipeline orchestrator | `pipeline/hybrid_pipeline.py` |
| Legal citation prompts | `pipeline/legal_citation_prompts.py` |
| LLM client | `llm/openai_client.py` |
| Conversation manager | `pipeline/conversation_manager.py` |
| Rate limiting / retry | `pipeline/llm_retry_utils.py` |

---

## Stage 7: Evaluation

| Metric | Module | Notes |
|---|---|---|
| Token F1 / EM | `evaluation/metrics/token_f1.py` | Vietnamese word segmentation |
| Retrieval Hit@k / MRR | `evaluation/metrics/id_normalizer.py` | Canonical ID matching |
| BERTScore | `evaluation/metrics/bertscore_eval.py` | `vinai/phobert-base` |
| Routing accuracy | `evaluation/metrics/legacy.py` | Exact route match |
| Significance | `evaluation/significance/bootstrap_test.py` | Paired bootstrap |
| Re-scoring | `evaluation/benchmark/rescore_predictions.py` | Offline from stored preds |

Benchmark scripts:

```bash
python scripts/run_benchmark_eval.py          # full e2e (needs serving stack)
python -m evaluation.benchmark.rescore_predictions --build-official  # offline
python scripts/evaluate_conversation_ambiguity.py  # diagnostic
python scripts/run_clarify_eval.py                   # clarify benchmark
```

---

## Stage 8: API & Frontend (optional)

| Component | Path | Notes |
|---|---|---|
| FastAPI backend | `api/main.py` | `/query` endpoint |
| Next.js frontend | `frontend/` | Does not send conversation history |

Defense demo: `python scripts/demo_conversation_routing.py` (recommended).

---

## Final Results

Single canonical file: **`results/final/official_metrics.json`**

See `results/final/README.md` for provenance and reproduction steps.
