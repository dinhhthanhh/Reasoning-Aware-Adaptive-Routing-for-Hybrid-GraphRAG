# PD-Subset Benchmark (Step 7) — Status

## Intended

Filter the `legal_strict` test set to questions whose gold article exists as a
`LegalArticle` node in the graph, then compare pure_vector / pure_graph /
two_stage on that subset using the corrected token F1 metric.

## Blocker (honest)

The gold test set identifies the answer article with **canonical** IDs:

```json
"supporting_facts": [{"law_id": "77/2026/NĐ-CP", "article_id": "Điều 18", ...}]
```

The graph's `LegalArticle` nodes (Pháp Điển) use **structural codes**:

```
article_id = "pd_007_003_0044",  law_id = "pd_007_003"
```

There is no reliable join between `77/2026/NĐ-CP · Điều 18` and a `pd_*`
structural node without a Pháp-Điển→canonical mapping. This is the same
limitation documented in `graph_known_limitations.md` (Limitation 6) and
`audit/critical_graph_risks.md` (Risk 7), and the same reason the offline strict
retrieval Hit@k is low in `official_metrics.json`.

Constructing a *clean* PD subset by gold-ID membership is therefore not possible
right now. Building it on guessed/loose matches would risk fabricated or
misleading numbers, which the rebuild prompt explicitly forbids
("DO NOT fabricate benchmark numbers").

## What can be run honestly instead (not yet executed)

A **full-corpus** three-way comparison (pure_vector vs pure_graph vs two_stage)
on all 600 legal_strict questions with the rebuilt graph and corrected metric is
feasible via:

```bash
python scripts/run_benchmark_eval.py --dataset legal_strict \
  --systems pure_vector,pure_graph,two_stage_hybrid
```

This is a long run (600 queries × 3 systems × LLM generation). It was not launched
in this session to avoid an unattended multi-hour job; it is ready to run on
request. The offline re-scored full-corpus numbers already in
`official_metrics.json` (`rescored_predictions`) remain valid in the meantime.

## Prerequisite to unblock a true PD subset

Implement a Pháp-Điển structural-code → canonical-law resolver (maps
`pd_007_003_0044` ↔ `Điều 8, Luật Hôn nhân và gia đình 2014`). Scoped as future
work. Once available, `scripts/filter_pd_subset.py` can select the subset by gold
membership and the three-way comparison becomes meaningful.

## Status: DEFERRED — blocked by PD↔canonical ID mismatch (documented future work).
The primary two-stage evidence (Step 0) is complete and unaffected.
