# Clarify Eval Summary

## Metrics

- Total samples: `234`
- Route accuracy: `0.7222`
- Clarify precision: `1.0000`
- Clarify recall: `0.7244`
- Clarify F1: `0.8401`
- Stage 2 trigger rate: `0.8120`
- Stage 2 override rate: `0.2421`
- Clarify false positives: `0`
- Clarify false negatives: `43`

## By Ambiguity Type

| Type | Total | Route Accuracy | Stage 2 Trigger | Flag Accuracy |
|---|---:|---:|---:|---:|
| `incomplete_context` | 39 | 1.000 | 1.000 | 1.000 |
| `missing_entity` | 39 | 1.000 | 1.000 | 1.000 |
| `pronoun_reference` | 42 | 0.833 | 1.000 | 1.000 |
| `multi_interpretation` | 36 | 0.000 | 0.333 | 0.028 |
| `unknown` | 78 | 0.718 | 0.744 | 0.000 |

## Notes

This run improves the original clarify benchmark F1 from `0.684` to `0.840`, while preserving zero false positives. The main remaining weakness is `multi_interpretation`, where the current conservative rescue policy avoids false clarification on answerable strict queries but misses many semantically ambiguous cases.
