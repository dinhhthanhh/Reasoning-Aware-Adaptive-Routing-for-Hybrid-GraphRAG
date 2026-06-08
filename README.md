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

Strict 600-query Vietnamese legal QA benchmark:

| System | F1 | Routing Acc. | Avg Latency |
|---|---:|---:|---:|
| Pure Vector | 0.3626 | 0.5000 | 1,270.7 ms |
| Pure Graph | 0.3556 | 0.2500 | 2,283.4 ms |
| Single-stage Router | 0.4231 | 0.9350 | 2,209.2 ms |
| Two-stage Hybrid | 0.4235 | 0.9283 | 3,913.4 ms |

Conversation and clarification results are diagnostic stress tests, not
replacements for the strict end-to-end table. See the evidence pack for the
full mapping from paper numbers to artifact files.

## Important Links

- Final paper PDF: [docs/AI(PM)_ver 2.3.pdf](<docs/AI(PM)_ver 2.3.pdf>)
- Defense evidence pack: [docs/defense_evidence_pack.md](docs/defense_evidence_pack.md)
- Final results snapshot: [docs/final_results_snapshot/MANIFEST.md](docs/final_results_snapshot/MANIFEST.md)
- Setup guide: [docs/SETUP.md](docs/SETUP.md)
- Demo guide: [docs/DEMO.md](docs/DEMO.md)
- Reproducibility notes: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

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
