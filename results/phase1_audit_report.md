# Phase 1 Audit Report

## Scope and Guardrails

Phase 1 was performed as diagnostics and audit only. No router retraining, vector-store rebuild, Neo4j migration/rebuild, retrieval logic change, Stage 2 prompt change, or paper metric update was performed.

## Inventory

| Item | Path |
|---|---|
| Stage 1 checkpoint | `data/router_training/legal_strict/router_model.pkl` |
| Stage 1 train/eval script | `scripts/run_router_training.py` |
| Router training helper | `router/train_router.py` |
| Strict benchmark script | `scripts/run_benchmark_eval.py` |
| Ambiguity eval script | `scripts/run_clarify_eval.py` |
| Strict train/dev/test | `qa_pipeline/data/legal_strict/train.json`, `qa_pipeline/data/legal_strict/dev.json`, `qa_pipeline/data/legal_strict/test.json` |
| Final train/dev/test mirror | `qa_pipeline/data/final/train.json`, `qa_pipeline/data/final/dev.json`, `qa_pipeline/data/final/test.json` |
| Ambiguity benchmark | `evaluation/legal_clarify_eval.json` |
| Stage 1 implementation | `router/router_model.py` |
| Stage 2 router | `router/two_stage_router.py` |
| Stage 2 verifier and prompt | `router/llm_reasoning_verifier.py` |
| Feature extraction | `router/features.py`, `router/query_complexity.py`, `router/ambiguity_detector.py` |
| Graph retrieval | `rag/graph_rag_adapter.py`, `graph/neo4j_client.py`, `pipeline/hybrid_pipeline.py` |
| Paper and bibliography | `docs/AI(PM)_ver 2.3.tex`, `docs/biblio.bib`, `docs/spmpsci.bst` |

## Commands Run

- `python scripts/analyze_stage1_router.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results`
- `python scripts/analyze_ambiguity_errors.py --eval-file evaluation/legal_clarify_eval.json --predictions eval_results/clarify_two_stage.csv --output-dir results`
- `python -m compileall router graph llm scripts`
- `python -m pytest tests/ -v`
- Static inspection with `rg` over `router/`, `pipeline/`, `rag/`, `graph/`, `scripts/`, `configs/`, `eval_results/`, and `evaluation/`.

## Files Created or Modified

- `scripts/analyze_stage1_router.py`
- `scripts/analyze_ambiguity_errors.py`
- `results/stage1_confusion_matrix.json`
- `results/stage1_classification_report.json`
- `results/stage1_feature_importance.json`
- `results/stage1_diagnostics_summary.md`
- `results/latex_stage1_confusion_matrix.tex`
- `results/latex_stage1_feature_importance.tex`
- `results/ambiguity_type_metrics.json`
- `results/ambiguity_false_negatives_missing_entity.jsonl`
- `results/ambiguity_false_negatives_multi_interpretation.jsonl`
- `results/ambiguity_error_summary.md`
- `results/conversation_history_audit.md`
- `results/graph_retrieval_audit.md`
- `results/phase1_audit_report.md`

## Metrics Found

### Stage 1 Offline Diagnostics

Generated from the current checkpoint and current `configs/config.yaml`:

- Samples: `600`
- Accuracy: `0.9383`
- Macro-F1 over strict gold labels: `0.9327`
- Macro-F1 including zero-support predicted `clarify`: `0.6995`
- Weighted-F1: `0.9387`
- Confusion matrix:
  - Dense: `288 dense`, `11 graph`, `0 hybrid`, `1 clarify`
  - Graph: `14 dense`, `129 graph`, `7 hybrid`, `0 clarify`
  - Hybrid: `0 dense`, `4 graph`, `146 hybrid`, `0 clarify`
- Top feature importances by gain: `has_pronoun`, `query_length`, `conditional_depth`, `relation_chain_length`, `graph_keyword_count`.

Important consistency note: these diagnostics measure the currently deployed Stage 1 path with `graph_priority_enabled=true`. The older saved training artifact `eval_results/router_results_legal_strict_after_regex_fix.json` reports the offline classifier diagnostic used in the paper table: test accuracy `0.8050`, Macro-F1 `0.7801`. Do not replace the paper's per-class training table with the new deployed-router diagnostic unless the text explicitly explains this difference.

### Existing Strict End-to-end Benchmark

Read from `eval_results/legal_strict_full_summary.json`; full benchmark was not rerun in Phase 1.

| System | F1 | Routing Acc. | Avg latency |
|---|---:|---:|---:|
| Pure Vector | `0.3626` | `0.5000` | `1,270.7 ms` |
| Pure Graph | `0.3556` | `0.2500` | `2,283.4 ms` |
| Single-stage Router | `0.4231` | `0.9350` | `2,209.2 ms` |
| Two-stage Hybrid | `0.4235` | `0.9283` | `3,913.4 ms` |

### Existing Clarification Benchmark

Read from `eval_results/clarify_stage1_only.json`, `eval_results/clarify_two_stage.json`, and regenerated error grouping from `eval_results/clarify_two_stage.csv`.

| Variant | Route Acc. | Clarify P | Clarify R | Clarify F1 | Stage2 trigger |
|---|---:|---:|---:|---:|---:|
| Stage 1 only | `0.269` | `0.000` | `0.000` | `0.000` | `0.000` |
| Stage 1 + Stage 2 | `0.585` | `1.000` | `0.519` | `0.684` | `0.577` |

By ambiguity type:

- `incomplete_context`: recall `1.000`, false negatives `0`
- `pronoun_reference`: recall `1.000`, false negatives `0`
- `missing_entity`: recall `0.000`, false negatives `39`
- `multi_interpretation`: recall `0.000`, false negatives `36`

### Graph Quality

Read from `eval_results/post_migration_graph_quality.json`:

- Nodes: `419,251`
- Relationships: `1,239,542`
- `LegalArticle`: `70,347`
- `VectorChunk`: `199,530`
- `HAS_ARTICLE`: `70,347`
- `BELONGS_TO`: `199,530`
- `AMENDS`: `27`
- `REPEALS`: `19`

## Paper Consistency

- The current paper numbers for strict end-to-end QA match `eval_results/legal_strict_full_summary.json`.
- The clarification section matches `eval_results/clarify_stage1_only.json` and `eval_results/clarify_two_stage.json`.
- The graph statistics match `eval_results/post_migration_graph_quality.json`.
- The paper should keep the distinction between the older offline Stage 1 training report and the deployed single-stage routing behavior unless it is rewritten to explain both.

## Not Rerun in Phase 1

- Full 600-query end-to-end benchmark was not rerun because it calls retrieval and LLM generation and is costly.
- Two-stage clarification eval was not rerun; the error analysis used the saved CSV/JSON outputs.
- Neo4j migration and vector DB build were not rerun by design.
- Human evaluation of legal answer correctness was not performed.

## Top 5 Issues

1. Semantic ambiguity is the main failure mode: `missing_entity` and `multi_interpretation` have `0.000` recall and Stage 2 trigger rate `0.000`.
2. Stage 1 has no real clarify training support in the strict dataset, so clarification depends on heuristics and Stage 2.
3. Metrics can be confusing because the saved offline training report and the deployed Stage 1 path measure different things.
4. Graph legal-effect relations are sparse compared with generic `REFERENCES`, which can make graph context broad rather than legally decisive.
5. Conversation history is threaded through the architecture but is not validated by a dedicated history benchmark.

## Recommended Next Order

1. Phase 2: create a small human-reviewed ambiguity/history set focused on `missing_entity`, `multi_interpretation`, resolved history, and irrelevant history.
2. Phase 3: improve semantic ambiguity triggers and entity-linking uncertainty without retraining the full router first.
3. Phase 4: add graph retrieval diagnostics and relation-specific traversal templates for amendment, repeal, effective date, and article membership.
4. Phase 5: rerun strict benchmark, clarify benchmark, and ablation after changes; then update the paper metrics.

## Risks Before Retrain or Rebuild

- Retraining the router will invalidate the current paper's Stage 1 and end-to-end routing numbers.
- Rebuilding Chroma or Neo4j can change chunk counts, graph statistics, retrieval contexts, latency, and answer metrics.
- Stage 2 latency depends heavily on the local OpenAI-compatible endpoint, so reruns should record model, base URL class, hardware, and timestamp.
- Any paper update should cite the exact JSON artifact used for each table.

## Demo Readiness

The current state is demo-ready for strict Vietnamese legal QA, adaptive routing, graph migration evidence, and surface-context clarification examples. For the demo, avoid presenting `missing_entity` or `multi_interpretation` as solved; those should be shown as honest limitations and Phase 2 improvement targets.

## Sanity Checks

- `python -m compileall router graph llm scripts`: passed.
- `python -m pytest tests/ -v`: passed, `4` tests.
