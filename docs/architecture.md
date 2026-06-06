# System Architecture

> Reasoning-Aware Adaptive Routing for Hybrid GraphRAG
> Vietnamese Legal Question Answering System

---

## Pipeline Overview

The system processes user queries through a multi-stage pipeline that adaptively
routes each query to the most appropriate retrieval strategy — balancing latency
against reasoning depth.

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Coreference   │  Resolve pronouns using
              │  Resolution    │  conversation history
              └───────┬────────┘
                      │
          ┌───────────▼────────────┐
          │  Feature Extraction    │  25 handcrafted features:
          │  (router/features.py)  │  NER, multi-hop score,
          │                        │  legal references, etc.
          └───────────┬────────────┘
                      │
          ┌───────────▼────────────┐
          │  Ambiguity Detection   │  Pronoun density,
          │  (router/ambiguity_    │  vague terms,
          │   detector.py)         │  missing entities
          └───────────┬────────────┘
                      │
     ┌────────────────▼─────────────────┐
     │  STAGE 1: XGBoost Classifier     │  Fast statistical routing
     │  (router/router_model.py)        │  with 25 features
     │                                   │
     │  Output: route + confidence       │
     └────────────────┬─────────────────┘
                      │
              ┌───────▼───────┐
              │ Confidence ≥ θ │
              │ & no override  │
              │ rules trigger? │
              └──┬──────────┬──┘
            Yes  │          │  No
                 │    ┌─────▼──────────────────┐
                 │    │ STAGE 2: LLM Reasoning │  Chain-of-thought
                 │    │ Verifier               │  verification via
                 │    │ (router/llm_reasoning_ │  OpenAI-compatible
                 │    │  verifier.py)          │  LLM
                 │    └─────┬──────────────────┘
                 │          │
         ┌───────▼──────────▼────────┐
         │   Final Route Decision    │
         └──┬──────┬──────┬──────┬───┘
            │      │      │      │
    ┌───────▼┐ ┌───▼────┐ ┌▼─────┐ ┌▼───────┐
    │ dense_ │ │ graph_ │ │hybrid│ │clarify │
    │retriev.│ │travers.│ │_reas.│ │        │
    └───┬────┘ └───┬────┘ └──┬───┘ └───┬────┘
        │          │         │         │
        ▼          ▼         ▼         ▼
    VectorRAG   GraphRAG  Hybrid    Generate
    (FAISS/     (Neo4j/   (Vector   clarification
     Chroma)    SQLite)    +Graph)  question
        │          │         │
        └──────────┼─────────┘
                   ▼
         ┌─────────────────┐
         │ LLM Answer      │  OpenAI-compatible API
         │ Generation      │  (vLLM / Ollama)
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Pipeline         │  JSONL logging for
         │ Response + Log   │  thesis analysis
         └─────────────────┘
```

## Module Architecture

### Core Pipeline (`pipeline/`)

| File | Role |
|------|------|
| `hybrid_pipeline.py` | **Main orchestrator** — routes queries through the two-stage router, dispatches to the appropriate RAG pipeline, manages conversation history, and produces structured `PipelineResponse` objects |
| `conversation_manager.py` | Multi-turn conversation tracking with coreference resolution |
| `i18n.py` | Internationalization — prompt templates for Vietnamese and English |
| `step01_clean_html.py` – `step07_build_vectordb_full.py` | Data processing pipeline: HTML cleaning → article splitting → QA generation → verification → KG/vector DB building |

### Two-Stage Router (`router/`)

| File | Role |
|------|------|
| `features.py` | **25 handcrafted features** — query length, entity count/density, legal reference patterns, multi-hop signals, cross-document indicators, comparison markers, procedural cues |
| `query_complexity.py` | Adaptive-RAG complexity estimation — sub-question detection, authority chain analysis, token-level signals |
| `router_model.py` | **Stage 1** — XGBoost classifier wrapper with confidence calibration |
| `llm_reasoning_verifier.py` | **Stage 2** — LLM chain-of-thought verifier for uncertain predictions |
| `two_stage_router.py` | **Router orchestrator** — applies override rules, ambiguity thresholds, and manages Stage 1→2 escalation |
| `ambiguity_detector.py` | Detects ambiguous queries (pronouns, vague terms, missing entities) |
| `scripts/run_router_training.py` | Training logic for the XGBoost router model |

### RAG Backends (`rag/`)

| File | Role |
|------|------|
| `vector_rag.py` | **VectorRAG** — retrieves chunks via FAISS/Chroma, generates answers with LLM. Used for single-hop, direct legal questions |
| `graph_rag_adapter.py` | **GraphRAG** — entity extraction → Neo4j/SQLite graph traversal → LLM synthesis. Used for multi-hop, relational reasoning |

### Knowledge Graph (`graph/`)

| File | Role |
|------|------|
| `neo4j_client.py` | Neo4j driver — multi-hop traversal, entity search, Cypher queries |
| `sqlite_kg.py` | SQLite fallback KG — lightweight graph for environments without Neo4j |
| `build_kg.py` | KG construction from processed legal articles |

### Vector Store (`vector_store/`)

| File | Role |
|------|------|
| `embedder.py` | Sentence-transformer embedding (default: `microsoft/Harrier-OSS-v1-0.6B`) |
| `faiss_store.py` | FAISS index management |
| `chroma_store.py` | ChromaDB collection management |
| `vector_retriever.py` | Unified retrieval interface across FAISS/Chroma |
| `safe_embedding.py` | Token-safe embedding wrapper for ChromaDB |

### LLM Clients (`llm/`)

| File | Role |
|------|------|
| `openai_client.py` | OpenAI-compatible API client (vLLM, LM Studio, OpenAI) with retry logic and thinking-tag stripping |
| `ollama_client.py` | Ollama-native client |

### NER (`ner/`)

| File | Role |
|------|------|
| `vi_ner.py` | Vietnamese NER using `NlpHUST/ner-vietnamese-electra-base` + rule-based legal patterns |
| `en_ner.py` | English NER using `dslim/bert-base-NER` |
| `factory.py` | NER model factory based on config language |

### Evaluation (`evaluation/`)

| File | Role |
|------|------|
| `evaluate.py` | End-to-end evaluation orchestrator |
| `metrics.py` | Token F1, Exact Match, BLEU, legal citation metrics |
| `faithfulness_eval.py` | LLM-based faithfulness evaluation |

### QA Data Pipeline (`qa_pipeline/`)

| Directory | Role |
|-----------|------|
| `pipeline/` | 9-step QA dataset generation: load → parse → enrich → filter → dedup → augment → split → report |
| `data/` | Input/output datasets with train/dev/test splits |
| `reports/` | Pipeline execution reports |

## Route Decision Logic

The router supports four routes:

| Route | Trigger Conditions | Pipeline |
|-------|-------------------|----------|
| `dense_retrieval` | Single-hop, direct legal questions; high Stage 1 confidence | VectorRAG (FAISS/Chroma) |
| `graph_traversal` | Multi-hop, relational questions; entity-dense queries | GraphRAG (Neo4j/SQLite) |
| `hybrid_reasoning` | Cross-document synthesis; comparison questions | Vector + Graph merged context |
| `clarify` | Ambiguous queries with pronouns, vague terms, or missing entities | Clarification question generation |

### Override Rules (configured in `configs/config.yaml`)

- **High-confidence dense skip**: If Stage 1 predicts `dense_retrieval` with confidence ≥ 0.95, skip Stage 2
- **Ambiguity force clarify**: If ambiguity score ≥ 0.8, force `clarify` route
- **Ambiguity force Stage 2**: If ambiguity score ≥ 0.6, escalate to Stage 2
- **Reasoning force Stage 2**: If reasoning complexity ≥ 0.45 and confidence < 0.90, escalate to Stage 2
- **Graph priority**: If graph-relevant features detected and confidence ≥ 0.55, boost `graph_traversal`

## Fallback Chain

```
graph_traversal  ──(empty answer)──→  dense_retrieval
dense_retrieval  ──(empty answer)──→  graph_traversal
hybrid_reasoning ──(empty answer)──→  dense_retrieval (vector only)
```

## Configuration

All system parameters are in `configs/config.yaml`. Key sections:

- `router.stage1` — XGBoost model path, confidence threshold, feature version
- `router.stage2` — LLM verifier enable/disable, max reasoning tokens
- `router.override_rules` — Threshold tuning for Stage 2 escalation
- `router.scoring` — Multi-hop normalization, comparison/cross-doc boosts
- `rag` — Context limits, fallback toggles, top-k settings
- `neo4j` — Graph database connection
- `openai` / `ollama` — LLM endpoint configuration
- `embedding` — Model name, batch size, device
- `ambiguity` — Pronoun list, vague terms, score threshold

## Data Flow

### Offline (Build Phase)

```
Legal documents (crawl/HuggingFace)
    │
    ├──→ pipeline/step01..03 → Cleaned articles
    │
    ├──→ scripts/build_kg.py → Neo4j graph + SQLite KG
    │
    ├──→ scripts/build_vectordb.py → FAISS/Chroma index
    │
    └──→ qa_pipeline/ → Train/dev/test QA dataset
              │
              └──→ scripts/run_router_training.py → XGBoost model (.pkl)
```

### Online (Query Phase)

```
User query → HybridPipeline.query()
    ├── ConversationManager.resolve_coreference()
    ├── TwoStageRouter.route()
    │   ├── FeatureExtractor.extract()
    │   ├── AmbiguityDetector.detect()
    │   ├── RouterModel.predict()          ← Stage 1
    │   └── LLMReasoningVerifier.verify()  ← Stage 2 (conditional)
    ├── VectorRAG / GraphRAGAdapter / Hybrid execution
    ├── OpenAIClient.generate()            ← Answer synthesis
    └── Log to JSONL
```

## Reproducibility

Entry point scripts for the full experimental pipeline:

```bash
# 1. Data collection
python main.py --source all --output data/

# 2. Knowledge graph construction
python scripts/build_kg.py --config configs/config.yaml

# 3. Vector index construction
python scripts/build_vectordb.py --config configs/config.yaml

# 4. Router training
python scripts/run_router_training.py --config configs/config.yaml \
    --train_path qa_pipeline/data/final/train.json \
    --dev_path qa_pipeline/data/final/dev.json \
    --test_path qa_pipeline/data/final/test.json

# 5. End-to-end evaluation
python scripts/run_benchmark_eval.py --config configs/config.yaml \
    --dataset legal --systems all --eval-answer-style

# 6. Routing baseline comparison
python experiments/run_routing_baselines.py --config configs/config.yaml
```
