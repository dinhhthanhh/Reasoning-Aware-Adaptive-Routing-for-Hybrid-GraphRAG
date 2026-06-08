# Stage 1 Router Diagnostics

This report was generated offline from the existing Stage 1 checkpoint. No model retraining, vector-store rebuild, graph migration, or Stage 2 call was performed.

## Inputs

- Dataset: `qa_pipeline\data\legal_strict\test.json`
- Model checkpoint: `data\router_training\legal_strict\router_model.pkl`
- Evaluated samples: `600`

## Metrics

- Accuracy: `0.9383`
- Macro-F1 over strict gold labels: `0.9327`
- Macro-F1 including zero-support predicted labels: `0.6995`
- Weighted-F1: `0.9387`

## Confusion Matrix

| Gold \ Pred. | dense_retrieval | graph_traversal | hybrid_reasoning | clarify |
|---|---|---|---|---|
| dense_retrieval | 288 | 11 | 0 | 1 |
| graph_traversal | 14 | 129 | 7 | 0 |
| hybrid_reasoning | 0 | 4 | 146 | 0 |
| clarify | 0 | 0 | 0 | 0 |

## Top Feature Importances

| Rank | Feature | Gain | Weight |
|---:|---|---:|---:|
| 1 | `has_pronoun` | 8.9584 | 53 |
| 2 | `query_length` | 3.5875 | 2890 |
| 3 | `conditional_depth` | 1.2074 | 429 |
| 4 | `relation_chain_length` | 1.0977 | 671 |
| 5 | `graph_keyword_count` | 0.8868 | 504 |
| 6 | `is_factoid` | 0.8165 | 11 |
| 7 | `sub_question_count` | 0.7916 | 230 |
| 8 | `complexity_level` | 0.6421 | 359 |
| 9 | `multi_hop_verb_count` | 0.6243 | 296 |
| 10 | `entity_count` | 0.5783 | 487 |
| 11 | `question_word_encoded` | 0.5735 | 1111 |
| 12 | `entity_type_count` | 0.5633 | 177 |
| 13 | `law_specificity` | 0.5001 | 392 |
| 14 | `multi_hop_score` | 0.4769 | 486 |
| 15 | `multi_entity_relation_count` | 0.4756 | 39 |

## Notes

- The strict test split contains three gold retrieval labels: `dense_retrieval`, `graph_traversal`, and `hybrid_reasoning`.
- If `clarify` appears as a predicted label, it is counted as a false positive in this strict split.
- The primary Macro-F1 above averages only labels with gold support in the strict split; the secondary Macro-F1 includes zero-support predicted labels such as `clarify`.
- Feature importance is extracted from the loaded XGBoost estimator inside the calibrated wrapper when available.
