# Reproducibility Checklist

Steps to reproduce all numbers cited in `results/final/official_metrics.json`.

## Prerequisites

- [ ] Python 3.10+, `pip install -r requirements.txt`
- [ ] `underthesea` installed (Vietnamese tokenization for F1)
- [ ] Copy `configs/config.yaml.example` → `configs/config.yaml`
- [ ] Copy `.env.example` → `.env` (Neo4j, LLM credentials)
- [ ] Neo4j 5.x running with legal graph loaded
- [ ] Chroma vector store at `data/vector_store/chroma_harrier_oss_0_6b`
- [ ] Router checkpoint at `data/router_training/legal_strict/router_model.pkl`

## Offline (no serving stack required)

These steps reproduce the corrected metrics from stored predictions:

```bash
# 1. Unit tests for metric implementations
python -m pytest tests/test_token_f1.py -v

# 2. Clean QA dataset (removes 84 placeholder records)
python scripts/regenerate_splits.py
# -> qa_pipeline/data/legal_strict_clean/
# -> data/audit_reports/qa_quality_audit.md
# -> results/final/excluded_test_ids.json

# 3. Re-score stored predictions -> official metrics
python -m evaluation.benchmark.rescore_predictions \
    --build-official \
    --exclude-ids results/final/excluded_test_ids.json
# -> results/final/official_metrics.json

# 4. Significance tests
python -m evaluation.significance.bootstrap_test
# -> results/final/significance_results.json
```

## Online (full pipeline re-run)

Required for Single-stage vs Two-stage comparison and fresh answer generation:

```bash
# Smoke tests
python -m compileall router graph llm scripts evaluation
python -m pytest tests/ -v

# Routing demo (recommended for defense)
python scripts/demo_conversation_routing.py --config configs/config.yaml

# Full 600-query benchmark (requires Neo4j + Chroma + LLM)
python scripts/run_benchmark_eval.py --config configs/config.yaml \
    --test-file qa_pipeline/data/legal_strict_clean/test.json
```

## Stage 1 router training (CV numbers)

```bash
python scripts/run_router_training.py
# -> data/router_training/legal_strict/router_model.pkl
# -> router_model/training_report.json  (CV: 0.8517 ± 0.0249)
```

## What each output file contains

| File | Contents |
|---|---|
| `results/final/official_metrics.json` | Re-scored token F1, routing, retrieval |
| `results/final/significance_results.json` | Paired bootstrap p-values |
| `router_model/training_report.json` | 5-fold CV routing metrics |
| `data/audit_reports/qa_quality_audit.md` | QA cleaning report |

## Git commit

Record the commit hash from `official_metrics.json -> _meta.git_commit` when
citing results in papers or reports.
