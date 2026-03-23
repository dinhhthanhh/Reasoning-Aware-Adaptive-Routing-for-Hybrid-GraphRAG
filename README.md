# Reasoning-Aware Adaptive Routing for Hybrid GraphRAG
## Vietnamese Legal Question Answering System

**Author:** Nguyễn Đình Thành  
**Research Area:** NLP × Knowledge Graph × Adaptive Retrieval  
**Thesis:** Graduation thesis — publication-ready (ACL/EMNLP SRW target)

---

## Project Overview

This project introduces a **two-stage reasoning-aware router** for hybrid GraphRAG in Vietnamese legal question answering. The core novelty is a routing mechanism that combines:

- **Stage 1 (Statistical):** XGBoost classifier with 17 handcrafted features for fast initial routing
- **Stage 2 (Reasoning):** LLM-based chain-of-thought verifier that activates only when Stage 1 confidence is low

The router adaptively directs queries to either **Vector RAG** (simple lookups), **Graph RAG** (multi-hop reasoning), or a **Clarification** pathway — achieving both speed and accuracy.

## Architecture

```
                    ┌─────────────┐
                    │  User Query │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Coreference │
                    │  Resolution  │
                    └──────┬──────┘
                           │
               ┌───────────▼───────────┐
               │   STAGE 1: XGBoost    │
               │   Feature Classifier  │
               └───────────┬───────────┘
                           │
                  ┌────────▼────────┐
                  │ Confidence ≥ θ? │
                  └───┬─────────┬───┘
                 Yes  │         │  No
                      │   ┌─────▼──────────┐
                      │   │ STAGE 2: LLM   │
                      │   │ Reasoning       │
                      │   │ Verifier        │
                      │   └─────┬──────────┘
                      │         │
              ┌───────▼─────────▼───────┐
              │    Final Route Decision  │
              └─┬──────────┬──────────┬─┘
                │          │          │
         ┌──────▼──┐ ┌────▼────┐ ┌──▼──────┐
         │ Vector  │ │ Graph   │ │ Clarify │
         │   RAG   │ │   RAG   │ │         │
         └────┬────┘ └────┬────┘ └────┬────┘
              │           │           │
              └─────────┬─┘           │
                   ┌────▼────┐   ┌────▼────┐
                   │  Answer  │   │ Ask User│
                   └──────────┘   └─────────┘
```

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (RTX 3060 or better recommended)
- Neo4j 5.x Community Edition
- Ollama with `llama3:8b` model

### Setup

```bash
# 1. Clone repository
git clone https://github.com/dinhhthanhh/Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG.git
cd Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Neo4j (download from https://neo4j.com/download/)
# Update password in configs/config.yaml

# 5. Start Ollama and pull model
ollama pull llama3:8b
```

## Usage

### Step-by-step pipeline:

```bash
# Step 1: Crawl legal documents
python scripts/run_crawl.py --max-docs 500

# Step 2: Build knowledge graph & FAISS index
python scripts/build_kg.py

# Step 3: Train the router
python scripts/train_router.py

# Step 4: Run QA system (interactive)
python scripts/run_pipeline.py

# Step 4b: Run with verbose routing info
python scripts/run_pipeline.py --verbose

# Step 4c: Batch mode
python scripts/run_pipeline.py --input queries.txt

# Step 5: Run evaluation
python scripts/run_pipeline.py --evaluate
```

## Project Structure

```
├── configs/config.yaml          # Centralized configuration
├── crawlers/legal_crawler.py    # Vietnamese legal document crawler
├── ner/vi_ner.py               # Vietnamese NER (PhoBERT + rules)
├── vector_store/               # FAISS embeddings & retrieval
├── graph/                      # Neo4j KG builder & client
├── rag/                        # Vector RAG + Graph RAG adapters
├── graphrag_wrapper/           # Microsoft GraphRAG integration
├── router/                     # ★ Two-stage router (NOVELTY)
│   ├── features.py             #   17-feature extraction
│   ├── router_model.py         #   Stage 1: XGBoost classifier
│   ├── llm_reasoning_verifier.py #  Stage 2: LLM verifier
│   ├── two_stage_router.py     #   Orchestrator
│   └── ambiguity_detector.py   #   Ambiguity detection
├── pipeline/                   # Main orchestrator + conversation
├── evaluation/                 # Metrics, evaluation, test queries
├── scripts/                    # CLI entry points
└── logs/routing_log.jsonl      # Research logging (JSONL)
```

## Evaluation Results

| Metric | Value |
|--------|-------|
| Routing Accuracy | _TBD_ |
| Ambiguity F1 | _TBD_ |
| Answer F1 | _TBD_ |
| Latency Mean (ms) | _TBD_ |
| Stage 2 Trigger Rate | _TBD_ |
| Stage 2 Override Rate | _TBD_ |

_Results will be populated after running evaluation._

## Key Features

- **Two-stage adaptive routing** — statistical speed + LLM reasoning depth
- **Vietnamese legal NER** — transformer + rule-based legal term extraction
- **Hybrid RAG** — Vector (FAISS) + Graph (Neo4j + Microsoft GraphRAG)
- **Ambiguity detection** — pronoun, vague reference, missing entity checks
- **Conversation management** — multi-turn with coreference resolution
- **Research logging** — JSONL logs for thesis analysis

## Citation

```bibtex
@thesis{nguyen2024reasoning,
    title={Reasoning-Aware Adaptive Routing for Hybrid GraphRAG 
           in Vietnamese Legal Question Answering},
    author={Nguyễn Đình Thành},
    year={2024},
    type={Graduation Thesis}
}
```

## License

This project is for academic research purposes.
