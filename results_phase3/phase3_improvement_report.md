# Phase 3 Improvement Report

## Scope

Phase 3 improves conversation-aware ambiguity routing. It does not retrain Stage 1, rebuild Chroma, rebuild or migrate Neo4j, or update the paper.

## Code Changes

- Added `router/history_resolver.py` for deterministic referent resolution from conversation history.
- Extended `router/ambiguity_detector.py` with contextual-reference status, missing-entity and multi-interpretation signals.
- Extended `router/features.py` with metadata-only ambiguity/history fields without changing the Stage 1 feature vector.
- Updated `router/two_stage_router.py` so resolved history can unblock retrieval and unresolved/irrelevant/conflicting history can force clarification.
- Updated `router/llm_reasoning_verifier.py` to pass history-resolution context into Stage 2 prompts and parse new fields.
- Updated `scripts/evaluate_conversation_ambiguity.py` and `scripts/demo_conversation_routing.py`.
- Added `scripts/evaluate_strict_routing_only.py`.
- Added tests in `tests/test_history_resolver.py` and expanded `tests/test_conversation_ambiguity.py`.

## Commands Run

```powershell
python -m compileall router graph llm scripts
python -m pytest tests/ -v
python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results_phase3
python scripts/run_clarify_eval.py --config configs/config.yaml --eval-file evaluation/legal_clarify_eval.json --output results_phase3/clarify_eval_summary.json --csv-output results_phase3/clarify_eval_results.csv
python scripts/evaluate_strict_routing_only.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results_phase3
python scripts/demo_conversation_routing.py --config configs/config.yaml --output-dir results_phase3
python scripts/analyze_stage1_router.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results_phase3
```

## Conversation Benchmark Before/After

| Metric | Before Phase 3 | After Phase 3 |
|---|---:|---:|
| Route accuracy | 0.475 | 0.750 |
| Clarify precision | 0.907 | 0.963 |
| Clarify recall | 0.390 | 0.780 |
| Clarify F1 | 0.545 | 0.862 |
| Stage 2 trigger rate | 0.713 | 0.825 |
| Stage 2 override rate | 0.333 | 0.227 |
| Avg latency | 2,238.7 ms | 3,127.2 ms |
| History resolution accuracy | n/a | 0.850 |

## Conversation Error Counts

| Error | Before | After |
|---|---:|---:|
| False clarification on answerable history | 4 | 3 |
| False answer on no-history clarify | 7 | 3 |
| False answer on irrelevant history | 14 | 5 |
| False answer on conflicting history | 17 | 6 |

## Conversation Category Results

| Category | Before | After |
|---|---:|---:|
| `answerable_with_history` route accuracy | 0.550 | 0.600 |
| `clarify_without_history` clarify recall | 0.650 | 0.850 |
| `irrelevant_history` clarify recall | 0.300 | 0.750 |
| `conflicting_history` clarify recall | 0.150 | 0.700 |
| `missing_entity` clarify recall | 0.600 | 0.850 |
| `multi_interpretation` clarify recall | 0.250 | 0.750 |
| `clear_dense_control` route accuracy | 1.000 | 1.000 |
| `clear_graph_or_hybrid_control` route accuracy | 0.300 | 0.500 |

## Old Clarify Benchmark

| Metric | Before Phase 3 | After Phase 3 |
|---|---:|---:|
| Clarify precision | 1.000 | 1.000 |
| Clarify recall | 0.519 | 0.724 |
| Clarify F1 | 0.684 | 0.840 |
| Clarify false positives | 0 | 0 |

By type, `incomplete_context` remains 1.000, `missing_entity` improves to 1.000, and `pronoun_reference` is 0.833. `multi_interpretation` remains weak on the old template benchmark at 0.000 after the strict-sanity guard was added.

## Strict Routing Sanity

| Metric | Value |
|---|---:|
| Total samples | 600 |
| Route accuracy | 0.893 |
| Clarify false positives | 10 |
| Stage 2 trigger rate | 0.558 |
| Stage 2 override rate | 0.340 |
| Avg latency | 1,972.4 ms |

This run is routing-only and does not regenerate answers. It should be treated as a sanity check, not as a replacement for the full end-to-end benchmark already reported in the paper.

## Demo Status

The routing-only demo passed the qualitative cases:

- Direct dense lookup stays `dense_retrieval`.
- Relation-heavy query upgrades to `graph_traversal`.
- Pronoun with valid history resolves `Nghị định 100/2019/NĐ-CP` and routes to `graph_traversal`, not `clarify`.
- Pronoun without history routes to `clarify`.
- Pronoun with irrelevant history routes to `clarify`.
- Missing-entity and multi-interpretation examples route to `clarify`.

## Remaining Issues

- The old clarify benchmark still misses `multi_interpretation` templates.
- `answerable_with_history` route accuracy is 0.600, so resolved-history route selection can still choose dense instead of graph/hybrid.
- Strict routing sanity is 0.893 with 10 false clarifications; do not update paper strict metrics from this Phase 3 sanity run.
- Stage 2 trigger rate increased on the conversation benchmark, so latency rose from 2,238.7 ms to 3,127.2 ms.

## Output Files

- `results_phase3/conversation_ambiguity_summary.json`
- `results_phase3/conversation_ambiguity_summary.md`
- `results_phase3/clarify_eval_summary.json`
- `results_phase3/clarify_eval_summary.md`
- `results_phase3/clarify_eval_results.csv`
- `results_phase3/strict_routing_sanity_summary.json`
- `results_phase3/strict_routing_sanity_summary.md`
- `results_phase3/demo_conversation_routing_output.md`
- `results_phase3/demo_conversation_routing_output.json`
- `results_phase3/stage1_diagnostics_summary.md`
