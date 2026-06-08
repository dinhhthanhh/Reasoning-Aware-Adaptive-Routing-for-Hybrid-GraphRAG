# Phase 2 Conversation Ambiguity Benchmark Report

## Scope

This phase adds a conversation-aware ambiguity benchmark and routing-only
evaluator for the current two-stage router. It does not retrain the router,
change retrieval code, rebuild Neo4j, rebuild the vector store, or update paper
metrics.

## Added Artifacts

- Benchmark dataset: `evaluation/conversation_ambiguity_eval.json`
- Dataset builder: `scripts/build_conversation_ambiguity_eval.py`
- Evaluator: `scripts/evaluate_conversation_ambiguity.py`
- Demo script: `scripts/demo_conversation_routing.py`
- Regression tests: `tests/test_conversation_ambiguity.py`
- Benchmark outputs:
  - `results/conversation_ambiguity_summary.json`
  - `results/conversation_ambiguity_summary.md`
  - `results/conversation_ambiguity_predictions.jsonl`
  - `results/conversation_ambiguity_failures.jsonl`
- Demo outputs:
  - `results/demo_conversation_routing_output.md`
  - `results/demo_conversation_routing_output.json`

The evaluator also creates `results/conversation_ambiguity_cache.jsonl` for
reruns with `--use-cache`; this cache is an execution aid and should not be
treated as a primary result file.

## Dataset

The dataset contains 160 routing-only samples, with 20 samples per category.

| Category | Count | Purpose |
|---|---:|---|
| `answerable_with_history` | 20 | Pronoun/context-dependent query should be answerable because history resolves the legal target. |
| `clarify_without_history` | 20 | Same paired query without resolving history should ask for clarification. |
| `irrelevant_history` | 20 | History exists but does not resolve the current legal target. |
| `conflicting_history` | 20 | History mentions multiple/conflicting legal targets. |
| `missing_entity` | 20 | Query lacks the regulated entity, action, document, or legal target. |
| `multi_interpretation` | 20 | Query has multiple plausible legal interpretations. |
| `clear_dense_control` | 20 | Direct lookup should route to dense retrieval. |
| `clear_graph_or_hybrid_control` | 20 | Relation-heavy controls should route to graph or hybrid. |

The first four categories are paired conversation cases. The remaining four
categories test semantic ambiguity and clear answerable controls.

## Commands Run

```powershell
python scripts/build_conversation_ambiguity_eval.py --output evaluation/conversation_ambiguity_eval.json
python -m compileall router graph llm scripts
python -m pytest tests/ -v
python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results --limit 30
python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results --use-cache
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

Validation passed:

- `compileall`: passed
- `pytest`: 8 passed
- Limited evaluator run (`--limit 30`): completed
- Full evaluator run (`160` samples): completed
- Demo run: completed

## Limited Run Metrics

The 30-sample smoke run completed successfully before the full benchmark.

| Metric | Value |
|---|---:|
| Total samples | 30 |
| Route accuracy | 0.4667 |
| Clarify precision | 0.8462 |
| Clarify recall | 0.5000 |
| Clarify F1 | 0.6286 |
| Stage 2 trigger rate | 0.7333 |
| Stage 2 override rate | 0.3636 |
| Avg latency | 2,315.0 ms |

## Full Run Metrics

| Metric | Value |
|---|---:|
| Total samples | 160 |
| Route accuracy | 0.4750 |
| Clarify precision | 0.9070 |
| Clarify recall | 0.3900 |
| Clarify F1 | 0.5455 |
| Stage 2 trigger rate | 0.7125 |
| Stage 2 override rate | 0.3333 |
| Avg latency | 2,238.7 ms |
| History resolution accuracy | not_available |
| History non-clarify proxy | 0.8000 |

The router is conservative when it predicts `clarify`: precision is high
(`0.9070`). The main weakness is recall: many ambiguous conversation cases are
still answered through dense or graph routes.

## Metrics by Category

| Category | Route Acc. | Clarify Recall | Stage2 Rate | Prediction Pattern |
|---|---:|---:|---:|---|
| `answerable_with_history` | 0.550 | n/a | 0.800 | 4 clarify, 10 dense, 6 graph |
| `clarify_without_history` | 0.650 | 0.650 | 0.850 | 13 clarify, 7 dense |
| `irrelevant_history` | 0.300 | 0.300 | 0.800 | 6 clarify, 14 dense |
| `conflicting_history` | 0.150 | 0.150 | 0.800 | 3 clarify, 9 dense, 8 graph |
| `missing_entity` | 0.600 | 0.600 | 0.900 | 12 clarify, 8 dense |
| `multi_interpretation` | 0.250 | 0.250 | 0.600 | 5 clarify, 15 dense |
| `clear_dense_control` | 1.000 | n/a | 0.050 | 20 dense |
| `clear_graph_or_hybrid_control` | 0.300 | n/a | 0.900 | 9 graph, 10 dense, 1 hybrid |

## False Counts

| Error type | Count |
|---|---:|
| False clarification on `answerable_with_history` | 4 |
| False answer on `clarify_without_history` | 7 |
| False answer on `irrelevant_history` | 14 |
| False answer on `conflicting_history` | 17 |

## Main Findings

1. Clear dense lookup is robust. The router gets all 20 dense controls correct
   and triggers Stage 2 only once.
2. Clarification precision is strong, but recall is too low for a legal QA
   assistant. The current router asks clarification carefully, but it misses many
   cases where clarification is expected.
3. The weakest conversation cases are `conflicting_history` and
   `irrelevant_history`. The system often treats the current query as answerable
   even when history does not uniquely resolve the referent.
4. `answerable_with_history` is only partially handled. The proxy non-clarify
   rate is 0.80, but route accuracy is 0.55 and there are 4 false
   clarifications.
5. `missing_entity` improves compared with the earlier ambiguity audit, but
   `multi_interpretation` remains weak: only 5 of 20 multi-interpretation
   samples are clarified.
6. Graph/hybrid controls are still difficult. Relation-heavy controls reach only
   0.30 route accuracy, mostly because many are routed to dense retrieval.

## Demo Summary

The demo script runs seven fixed scenarios and skips retrieval/generation. It
correctly routes direct dense lookup to the vector backend and rescues a
relation-heavy document-specific query from dense retrieval to graph traversal.
It also asks clarification for no-history, irrelevant-history, missing-entity,
and multi-interpretation examples.

The important demo failure is the valid-history pronoun case:
`Văn bản đó còn hiệu lực không?` with history mentioning
`Nghị định 100/2019/NĐ-CP` is still routed to `clarify`. This matches the full
benchmark finding that explicit referent resolution is not yet exposed as a
first-class router capability.

## Recommended Phase 3 Fixes

1. Add structured conversation referent extraction for legal documents,
   articles, agencies, procedures, and legal concepts.
2. Distinguish resolving history from irrelevant or conflicting history. A
   non-empty history should not automatically make a query safer to answer.
3. Expose `resolved_referent`, `candidate_referents`, and
   `history_resolution_confidence` in router outputs and logs.
4. Improve triggers for semantic ambiguity, especially missing legal entity and
   multi-interpretation cases.
5. Add relation-specific graph/hybrid routing features for amendment, repeal,
   effective date, issuing authority, article membership, and cross-document
   dependence.
6. Calibrate Stage 2 override policy jointly on strict routing and conversation
   ambiguity data, because the current policy protects precision but misses too
   many ambiguous cases.

## Paper Impact

Do not update the main paper results from this Phase 2 run yet. This benchmark
is a diagnostic conversation stress test, not a replacement for the reported
strict 600-query QA benchmark or the 234-query ambiguity benchmark. It can be
added later as an appendix or future-work diagnostic table after the Phase 3
history-resolution fixes are implemented and rerun.
