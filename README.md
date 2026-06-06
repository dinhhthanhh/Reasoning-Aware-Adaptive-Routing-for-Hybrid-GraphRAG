# Reasoning-Aware Adaptive Routing for Hybrid GraphRAG
## Vietnamese Legal Question Answering System

**Author:** Nguyễn Đình Thành  
**Research Area:** NLP × Knowledge Graph × Adaptive Retrieval  
**Thesis:** Graduation thesis — publication-ready (ACL/EMNLP SRW target)

---

## Project Overview

This project introduces a **two-stage reasoning-aware router** for hybrid GraphRAG in Vietnamese legal question answering. The core novelty is a routing mechanism that combines:

- **Stage 1 (Statistical):** XGBoost classifier with 25 handcrafted and complexity-aware features for fast initial routing
- **Stage 2 (Reasoning):** LLM-based chain-of-thought verifier that activates only when Stage 1 confidence is low

The router adaptively directs queries to **Vector RAG** (`dense_retrieval`), **Graph RAG** (`graph_traversal`), **Hybrid RAG** (`hybrid_reasoning`), or a **Clarification** pathway (`clarify`) — balancing latency, reasoning depth, and ambiguity handling.

For research reporting, Vietnamese Legal QA is the primary benchmark. ViMQA can be used as a secondary Vietnamese multi-hop generalization benchmark, while HotpotQA should remain a separate English/open-domain benchmark with separate model and index artifacts.

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

For detailed architecture documentation, see [`docs/architecture.md`](docs/architecture.md).

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (RTX 3060 or better recommended)
- Neo4j 5.x Community Edition
- Ollama with `llama3:8b` model, or an OpenAI-compatible inference server (e.g., vLLM)

### Setup

```bash
# 1. Clone repository
git clone https://github.com/dinhhthanhh/Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG.git
cd Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
cp configs/config.yaml.example configs/config.yaml
# Edit .env and configs/config.yaml with your local settings:
#   - Neo4j connection (uri, user, password)
#   - LLM endpoint (OpenAI-compatible base_url or Ollama)
#   - API keys if required

# 5. Start Neo4j (download from https://neo4j.com/download/)
# Ensure Neo4j is running and credentials match configs/config.yaml

# 6. Start Ollama and pull model (if using Ollama)
ollama pull llama3:8b
```

### Alternative: Conda Setup

```bash
conda env create -f environment.yml
conda activate hybrid-graphrag
```

## Usage

### Full Reproducible Pipeline (Vietnamese Legal QA)

```bash
# Step 1: Crawl legal documents
python main.py --source all --output data/

# Step 2: Build knowledge graph & vector index
python scripts/build_kg.py --config configs/config.yaml
python scripts/build_vectordb.py --config configs/config.yaml

# Step 3: Check Neo4j graph quality
python scripts/check_neo4j_graph_quality.py \
    --config configs/config.yaml \
    --output eval_results/legal_graph_quality.json

# Step 4: Generate and prepare QA dataset
python scripts/run_qa_generation.py

# Step 5: Train the Stage 1 router
python scripts/run_router_training.py \
    --config configs/config.yaml \
    --train_path qa_pipeline/data/final/train.json \
    --dev_path qa_pipeline/data/final/dev.json \
    --test_path qa_pipeline/data/final/test.json

# Step 6: Run end-to-end baseline comparison
python scripts/run_benchmark_eval.py \
    --config configs/config.yaml \
    --dataset legal --systems all --eval-answer-style

# Step 7: Run routing baseline comparison
python experiments/run_routing_baselines.py \
    --config configs/config.yaml
```

### Interactive QA

```bash
# Interactive single-query mode
python scripts/run_pipeline.py --config configs/config.yaml --verbose

# Multi-turn conversation demo
python scripts/run_conversation_demo.py --config configs/config.yaml
```

### API Server

```bash
# Start FastAPI backend
python api/main.py
# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Additional Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_clarify_eval.py` | Evaluate ambiguity/clarify detection |
| `scripts/eval_stage2_mini.py` | Mini evaluation of Stage 2 verifier |
| `scripts/ablation_study.py` | Run ablation experiments |
| `scripts/enrich_neo4j_legal_graph.py` | Enrich Neo4j graph with additional legal relationships |
| `scripts/migrate_graph.py` | Migrate graph data between formats |
| `scripts/reporting/run_midterm_eval.py` | Generate midterm evaluation reports |

Secondary benchmark commands (HotpotQA, ViMQA) are documented in [`scripts/README.md`](scripts/README.md).

## Project Structure

```
├── configs/                    # Configuration files
│   ├── config.yaml             # Primary Vietnamese Legal QA config (local, not committed)
│   ├── config.yaml.example     # Sanitized example config for publication
│   ├── config_hotpot.yaml      # Secondary English HotpotQA benchmark
│   └── config_vimqa.yaml       # Secondary Vietnamese ViMQA benchmark
├── crawlers/                   # Legal data crawlers (HuggingFace, VBPL, Pháp Điển)
├── data/                       # (gitignored) Raw/processed corpora, vector stores, KG files
├── docs/                       # Research notes, paper LaTeX, architecture documentation
│   └── architecture.md         # Detailed system architecture documentation
├── evaluation/                 # Metrics and end-to-end evaluation helpers
├── experiments/                # Routing baseline comparisons
├── graph/                      # Neo4j client, SQLite KG, KG builders
├── llm/                        # OpenAI-compatible and Ollama LLM clients
├── ner/                        # Vietnamese/English NER (transformer + rule-based)
├── qa_pipeline/                # Legal QA data generation pipeline (9 steps)
├── rag/                        # VectorRAG and GraphRAG adapters
├── router/                     # ★ Two-stage reasoning-aware router
│   ├── features.py             #   25-feature extraction
│   ├── query_complexity.py     #   Adaptive-RAG complexity signals
│   ├── router_model.py         #   Stage 1: XGBoost classifier
│   ├── llm_reasoning_verifier.py #  Stage 2: LLM verifier
│   ├── two_stage_router.py     #   Router orchestrator
│   └── ambiguity_detector.py   #   Clarification/ambiguity detection
├── pipeline/                   # System pipeline: hybrid orchestrator + data processing steps
├── scripts/                    # Reproducible experiment entry points
├── vector_store/               # FAISS/Chroma retrieval backends
├── api/                        # FastAPI REST backend
├── frontend/                   # Next.js web UI
├── eval_results/               # Evaluation results (CSV, JSON, logs)
├── artifacts/                  # Experiment artifacts and analysis scripts
├── reports/                    # Weekly and midterm reports
├── tests/                      # Unit tests
└── logs/                       # (gitignored) JSONL routing logs
```

### Data Regeneration

The `BoPhapDienDienTu/` directory contains raw crawl data from the Vietnamese legal code portal.
This data can be regenerated by running the crawler:

```bash
python main.py --source phapdien --output data/
```

For reproducibility, the raw crawl data can also be provided separately upon request.

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
- **Hybrid RAG** — Vector (FAISS/Chroma) + Graph (Neo4j/SQLite)
- **Ambiguity detection** — pronoun, vague reference, missing entity checks
- **Conversation management** — multi-turn with coreference resolution
- **Research logging** — JSONL logs for thesis analysis
- **Fallback chains** — automatic fallback between RAG pipelines when answers lack evidence

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
