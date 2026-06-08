# Reproducibility Notes

This document explains how to reproduce the checks and how to locate the final
paper artifacts. It intentionally separates main benchmark results from
diagnostic/stress-test results.

## Final Paper Artifacts

```text
docs/AI(PM)_ver 2.3.tex
docs/AI(PM)_ver 2.3.pdf
docs/biblio.bib
docs/spmpsci.bst
```

Compile command used in Phase 4.1:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error "AI(PM)_ver 2.3.tex"
bibtex "AI(PM)_ver 2.3"
pdflatex -interaction=nonstopmode -halt-on-error "AI(PM)_ver 2.3.tex"
pdflatex -interaction=nonstopmode -halt-on-error "AI(PM)_ver 2.3.tex"
```

Known non-fatal warning: MiKTeX may warn that Vietnamese hyphenation patterns
are not preloaded.

## Final Results Snapshot

Curated small artifacts are copied to:

```text
docs/final_results_snapshot/
docs/final_results_snapshot/MANIFEST.md
docs/final_results_snapshot/manifest.json
```

The snapshot avoids `.env`, credentials, vector DB folders, Neo4j database
dumps, logs, model weights, and large caches.

## Main Result vs Diagnostic Results

Main strict end-to-end result:

```text
eval_results/legal_strict_full_summary.json
docs/final_results_snapshot/legal_strict_full_summary.json
```

Diagnostic results:

```text
eval_results/clarify_stage1_only.json
eval_results/clarify_two_stage.json
results_phase3/clarify_eval_summary.json
results_phase3/conversation_ambiguity_summary.json
results_phase3/strict_routing_sanity_summary.json
```

Do not replace the strict end-to-end table with routing-only sanity metrics.

## Core Phase Commands

Phase 1 graph migration and quality:

```bash
python scripts/migrate_graph.py --config configs/config.yaml
python scripts/check_neo4j_graph_quality.py --config configs/config.yaml --output eval_results/post_migration_graph_quality.json
```

Phase 2 strict benchmark:

```bash
python scripts/run_benchmark_eval.py --config configs/config_legal.yaml --dataset legal_strict --eval-file qa_pipeline/data/legal_strict/test.json --systems all --eval-answer-style
```

Phase 3 conversation/clarification diagnostics:

```bash
python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results_phase3
python scripts/evaluate_strict_routing_only.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results_phase3
python scripts/demo_conversation_routing.py --config configs/config.yaml --output-md results_phase3/demo_conversation_routing_output.md --output-json results_phase3/demo_conversation_routing_output.json
```

Phase 5 minimal checks:

```bash
python -m compileall router graph llm scripts
python -m pytest tests/ -v
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

## Caveats

- Full end-to-end benchmark needs Neo4j, Chroma, router checkpoint, and LLM
  endpoint ready.
- Conversation stress test is diagnostic and template-controlled.
- Stage 1 has no `clarify` label during training; Stage 1 clarify F1 = 0 is an
  architectural consequence, not a runtime bug.
- Phase 3 improves ambiguity handling but still requires calibration because
  strict routing-only sanity has false clarifications.
