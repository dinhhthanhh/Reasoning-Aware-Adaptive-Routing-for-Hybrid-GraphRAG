# Ambiguity Error Summary

This report aggregates existing clarification evaluation outputs. No router, retrieval, or LLM generation was rerun.

## Overall

- Total queries: `234`
- Expected clarify queries: `156`
- Predicted clarify queries: `81`
- Clarify precision: `1.000`
- Clarify recall: `0.519`
- Clarify F1: `0.684`
- Stage 2 trigger rate: `0.577`

## By Ambiguity Type

| Type | Total | Expected clarify | Stage2 rate | Clarify recall | False negatives |
|---|---:|---:|---:|---:|---:|
| `answerable_control` | 78 | 0 | 0.692 | 0.000 | 0 |
| `incomplete_context` | 39 | 39 | 1.000 | 1.000 | 0 |
| `missing_entity` | 39 | 39 | 0.000 | 0.000 | 39 |
| `multi_interpretation` | 36 | 36 | 0.000 | 0.000 | 36 |
| `pronoun_reference` | 42 | 42 | 1.000 | 1.000 | 0 |

## Main Failure Buckets

- Missing entity false negatives: `39`
- Multi-interpretation false negatives: `36`

The two strongest failure buckets are semantic ambiguity cases. They do not contain the same surface cues as unresolved pronouns or phrases such as `quy định này`, so Stage 2 was not triggered in the saved run.
