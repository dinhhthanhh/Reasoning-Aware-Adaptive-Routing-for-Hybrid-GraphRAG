# Reasoning-Aware Adaptive Routing for Hybrid GraphRAG

Vietnamese Legal Question Answering system with adaptive routing across
`dense_retrieval`, `graph_traversal`, `hybrid_reasoning`, and `clarify`.

The project studies a practical question in Legal RAG: not every query should
use the same retrieval path. Direct lookup queries are often best served by
dense retrieval, relation-heavy questions need graph evidence, mixed questions
benefit from hybrid evidence, and underspecified questions should ask for
clarification before retrieval.

## Architecture Summary

- Stage 1 router: XGBoost classifier over reasoning and ambiguity features.
- Stage 2 verifier: LLM-based verifier triggered only for uncertain,
  ambiguous, or relation-heavy cases.
- Dense backend: Chroma vector store over Vietnamese legal chunks.
- Graph backend: Neo4j legal graph with `LegalDoc`, `LegalArticle`,
  `VectorChunk`, legal concepts, and relation edges.
- Conversation support: deterministic history-referent resolution used for
  ambiguity-aware routing.

## Main Results

**Canonical metrics file:** [`results/final/official_metrics.json`](results/final/official_metrics.json)
(see [`results/final/README.md`](results/final/README.md) for provenance).

> **Metric correction (2026-06):** The previously reported "F1" (~0.42) was
> keyword *recall*, not token F1. Corrected Vietnamese token F1 on the cleaned
> test set (n=541) is ~**0.34**. Routing accuracy: **0.8517 ± 0.0249** (5-fold CV).

| System | Token F1 (corrected) | Routing Acc. | Source |
|---|---:|---:|---|
| Router (honest run) | 0.3408 | 0.8503 | `official_metrics.json` |
| Pure Graph | 0.3390 | 0.2773 | same |
| Stage 1 CV (authoritative routing) | — | **0.8517 ± 0.0249** | `router_model/training_report.json` |

Legacy snapshot (keyword-recall "F1", superseded):
[`docs/final_results_snapshot/legal_strict_full_summary.json`](docs/final_results_snapshot/legal_strict_full_summary.json)

Conversation and clarification results are diagnostic stress tests, not
replacements for the strict end-to-end table.

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

```bash
python -m compileall router graph llm scripts
python -m pytest tests/ -v
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

Routing-only diagnostic examples:

```bash
python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results_demo --limit 10 --use-cache
python scripts/evaluate_strict_routing_only.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results_demo
```

The full 600-query end-to-end benchmark should only be rerun when Neo4j,
Chroma, router checkpoint, and the LLM endpoint are ready.

## Data Sources

Final corpus (248,740 documents):

| Source | Records | Script |
|---|---:|---|
| HuggingFace `th1nhng0/vietnamese-legal-documents` | 178,665 | `main.py --source huggingface` |
| Pháp Điển offline bundle (`BoPhapDienDienTu/`) | 70,075 | `main.py --source phapdien` |

VBPL portal crawler was attempted but **failed** (empty output) and is archived
at `archive/legacy_vbpl/`. It is not a data source for any final experiment.

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
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

The existing frontend is optional. It can display route/stage metadata returned
by the backend, but it does not currently send conversation history or expose
resolved referents, so the CLI demo is clearer for defending the routing
contribution.

## Citation

```bibtex
@thesis{nguyen2026reasoning,
  title={Reasoning-aware Adaptive Routing for Hybrid GraphRAG},
  author={Nguyen Dinh Thanh},
  year={2026},
  type={Graduation Thesis}
}
```
