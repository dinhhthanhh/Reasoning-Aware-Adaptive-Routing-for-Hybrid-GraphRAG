# Reasoning-Aware Adaptive Routing for Hybrid GraphRAG

Vietnamese Legal Question Answering system with adaptive routing across
`dense_retrieval`, `graph_traversal`, `hybrid_reasoning`, and `clarify`.

The project studies a practical question in Legal RAG: not every query should
use the same retrieval path. Direct lookup queries are often best served by
dense retrieval, relation-heavy questions need graph evidence, mixed questions
benefit from hybrid evidence, and underspecified questions should ask for
clarification before retrieval.

## Architecture Summary

The system is built upon a **Two-Stage Routing** mechanism that dynamically selects among **4 distinct routes** (`dense_retrieval`, `graph_traversal`, `hybrid_reasoning`, `clarify`):

- **Stage 1 Router (XGBoost):** A lightweight, low-latency classifier (`n_estimators=200`) trained to predict the optimal route using **27 handcrafted features** (e.g., `is_yes_no_question`, `legal_reference_count`, `ambiguity_score`).
- **Stage 2 Verifier (LLM - Qwen3.5-35B):** Triggered conditionally ($\Gamma = 1$) only when Stage 1 confidence is low (< 0.5), or when high ambiguity/multi-hop reasoning is detected. It overrides Stage 1 if necessary to prevent hallucination.
- **2-Layer Graph Backend (Neo4j):**
  - *Structural Layer:* Models the physical hierarchy of documents (`LegalDoc` $\rightarrow$ `LegalArticle`).
  - *Semantic Layer:* Extracted via LLM, representing abstract legal concepts (`LEGAL_CONCEPT`, `ACTOR`, `PENALTY`) and their logical relations (`AMENDS`, `HAS_CONDITION`, `REGULATES`).
- **Dense Backend (ChromaDB):** Vector store holding raw text chunks (256 words/chunk), embedded using the **zero-shot `microsoft/Harrier-OSS-v1-0.6B`** model (1024-dimensional space).
- **Conversation Management:** Uses deterministic history-referent resolution for ambiguity-aware routing, tracking context across multi-turn sessions.

## Main Results (Final Thesis Benchmark - B.5.2)

The following metrics are the **final official results** submitted in the graduation thesis (evaluated on a strict 600-query benchmark).

### 1. End-to-End Token F1 Score
| System | Token F1 |
|---|---:|
| Vector (Chroma Dense) | **0.663** |
| Graph (Neo4j Traversal) | 0.633 |
| Hybrid (Vector + Graph) | 0.660 |
| Single-stage Router | 0.474 |
| **Two-stage Router (Đề xuất)** | **0.636** |
| Always-on Verifier | 0.564 |

### 2. Routing Performance & Latency (B.5.3 / 3.6 / 5.7)
- **Stage 1 (XGBoost) Routing Accuracy:** **0.995** (5-fold CV: 99.53% ± 0.40)
- **XGBoost Inference Latency:** ~8.9 ms
- **Stage 2 (LLM Verifier) Latency:** ~4,017 ms
- **Total Routing-only Latency:** ~33.7 ms

*Note: The Two-stage Router dynamically balances performance and cost. It successfully prevents hallucination on complex multi-hop and out-of-distribution questions by intelligently triggering the Verifier only when Stage 1 confidence is low (< 0.5) or ambiguity is high.*

## Important Links

- **Official metrics:** [results/final/official_metrics.json](results/final/official_metrics.json)
- Routing design: [docs/routing_design.md](docs/routing_design.md)
- End-to-end pipeline: [docs/architecture/end_to_end_pipeline.md](docs/architecture/end_to_end_pipeline.md)
- Defense materials: [docs/defense_materials/](docs/defense_materials/)
- Paper errata (before journal submit): [docs/paper_errata.md](docs/paper_errata.md)
- QA quality audit: [data/audit_reports/qa_quality_audit.md](data/audit_reports/qa_quality_audit.md)
- Final paper PDF: [docs/AI(PM)_ver 2.3.pdf](<docs/AI(PM)_ver 2.3.pdf>)
- Defense evidence pack: [docs/defense_evidence_pack.md](docs/defense_evidence_pack.md)
- Setup guide: [docs/SETUP.md](docs/SETUP.md)
- Demo guide: [docs/DEMO.md](docs/DEMO.md)
- Reproducibility: [docs/defense_materials/reproducibility_checklist.md](docs/defense_materials/reproducibility_checklist.md)

## Quick Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
copy configs\config.yaml.example configs\config.yaml
```

Edit `.env` and `configs/config.yaml` with local Neo4j, Chroma, and
OpenAI-compatible LLM settings. Do not commit local secrets.

## Smoke Tests and Demos

To quickly test the 4 routing paths (Dense, Graph, Hybrid, Clarify), you can use the interactive single-query demo:

```bash
python run_test_demo.py
```

For multi-turn conversation and ambiguity resolution testing:

```bash
python scripts/run_conversation_demo.py
```

The full 600-query end-to-end benchmark should only be rerun when Neo4j,
Chroma, router checkpoint, and the LLM endpoint are ready.

## Data Sources

**📥 Dataset Download:** [Google Drive Link (Data & Vector Store)](https://drive.google.com/drive/folders/1eF8unFzgKfOnCNVr5_e-gr5sQuotH2hD?usp=sharing)

The project uses a single, highly curated legal source: **Bộ Pháp Điển** (offline bundle `BoPhapDienDienTu/`). No other sources (like HuggingFace or VBPL portal) were used in the final system to ensure strict legal accuracy and structural consistency.

**Corpus Statistics (Deployed in Hybrid System):**
- **Documents (Văn bản gốc):** 4,598
- **Articles (Điều luật):** 68,570
- **Chunks (Đoạn):** 68,835 (chunked by 256 words, 32 overlap via `text.split()`)
- **Graph Nodes (Nút đồ thị):** 804,322
- **Vector Embeddings:** 1024-dimensional using `microsoft/Harrier-OSS-v1-0.6B`

## Data and Artifact Policy

Large/local artifacts are intentionally not committed:

- `data/` vector stores and local model checkpoints
- Neo4j database files or dumps
- `.env` and local configs containing credentials
- logs, caches, LaTeX build artifacts, and frontend build artifacts

Small final result artifacts used for defense are copied into
`docs/final_results_snapshot/`.

## Recommended Defense Demo

Use the CLI routing demo as the primary defense path:

```bash
# Multi-turn demo
python scripts/run_conversation_demo.py

# Single query manual test
python run_test_demo.py
```

The existing frontend is optional. It can display route/stage metadata returned by the backend, but the CLI demo is clearer for defending the routing contribution, as it exposes Stage 1 confidence, Stage 2 triggers, latency, and exact log outputs.

## Citation

```bibtex
@thesis{nguyen2026reasoning,
  title={Reasoning-aware Adaptive Routing for Hybrid GraphRAG},
  author={Nguyen Dinh Thanh},
  year={2026},
  type={Graduation Thesis}
}
```
