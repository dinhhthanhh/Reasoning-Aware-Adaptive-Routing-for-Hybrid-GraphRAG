# Ablation Summary

Ablation flags are configured in `configs/config.yaml` under the `ablation:` key
and read by `router/two_stage_router.py`. Each flag disables one policy component.

| Flag | Disables | Expected effect | Measured? |
|---|---|---|---|
| `no_ambiguity_features` | Ambiguity detector features in Stage 1 | Lower clarify precision; more false negatives on ambiguous queries | Diagnostic only |
| `no_history_resolver` | `resolve_history_referents()` | Follow-up queries with pronouns cannot be unblocked by history | Conversation eval (160) |
| `no_severe_ambiguity_override` | Rule-based ambiguity → clarify override | Fewer clarify routes; more false negatives on severe ambiguity | Clarify eval (234) |
| `no_clarify_sanity_check` | Post-Stage-2 sanity that blocks unnecessary clarify when history resolves | More false clarifications on answerable-with-history queries | Conversation eval |
| `no_fallback_guard` | Pipeline fallback from failed retrieval to alternate route | More empty answers when primary route fails | E2E benchmark |
| `no_relation_features` | Legal-relation feature signals in Stage 1 | Lower graph/hybrid routing accuracy on relation-heavy queries | Stage 1 per-class report |

## Feature ablation (Stage 1 XGBoost)

From `router_model/training_report.json` → `ablation` section (5-fold CV):

| Feature set | CV accuracy | CV macro-F1 |
|---|---:|---:|
| All 16 features | 0.8517 ± 0.0249 | 0.8399 ± 0.0262 |
| (see training_report.json for individual feature removals) | varies | varies |

## Statistical significance

Ablation comparisons on end-to-end F1 have **not** been run with bootstrap tests
for all combinations. Run:

```bash
python -m evaluation.significance.bootstrap_test --a <predictions_a> --b <predictions_b>
```

after producing labelled prediction files for each ablation configuration.

## Defense talking points

- Ablation flags prove each component is **intentional**, not accidental complexity.
- The conversation ambiguity benchmark (160 queries, 8 balanced groups) is the
  primary diagnostic for history/ambiguity ablations.
- Strict 600-query benchmark does not contain `clarify` gold labels, so clarify
  ablation must be evaluated on the 234-query clarify set.
