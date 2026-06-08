# Reproducibility Audit

Generated: 2026-06-08T23:47:01

## Git State

- Current branch: `cleanup/working-tree-content-review-pass-3c`
- Current commit before Phase 5 commit: `b762829`
- Remote URL(s), sanitized:

```text
origin	https://github.com/dinhhthanhh/Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG.git (fetch)
origin	https://github.com/dinhhthanhh/Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG.git (push)
```

- Uncommitted files before final Phase 5 commit:

```text
M .gitignore
 M README.md
 M configs/config.yaml.example
 M results/demo_conversation_routing_output.json
 M results/demo_conversation_routing_output.md
?? docs/DEMO.md
?? docs/REPRODUCIBILITY.md
?? docs/SETUP.md
?? docs/defense_evidence_pack.json
?? docs/defense_evidence_pack.md
?? docs/final_results_snapshot/
?? docs/reproducibility_audit.md
?? evaluation/README_legal_clarify_eval.md
?? evaluation/legal_clarify_eval.json
?? experiments/
?? scripts/build_legal_clarify_eval.py
?? scripts/generate_phase5_artifacts.py
```

## Recent Commits

```text
b762829 phase4.1: polish paper consistency and compile
5dccfd8 phase4: update paper with conversation ambiguity results
3e2b5f2 phase3: improve conversation ambiguity routing
e822597 phase2: add conversation ambiguity benchmark and demo
999e493 phase1: add routing diagnostics and audit reports
6d1c82b fix: add secure runtime fallback for Neo4j and LLM clients
c9a8d81 fix: make Harrier vector index build robust
057d145 docs: define public data and artifact policy
5364286 docs: add sanitized benchmark config examples
aea7c8a feat: add NER module factory
```

## Tracked/Untracked Files Worth Noting

- `configs/config.yaml.example` was already modified before Phase 5.
- `evaluation/README_legal_clarify_eval.md`, `evaluation/legal_clarify_eval.json`, `experiments/`, and `scripts/build_legal_clarify_eval.py` were already untracked before Phase 5.
- `docs/` sources are ignored by default in `.gitignore`; Phase paper/docs files are intentionally force-added when needed.
- `eval_results/` and `reports/` are ignored as generated experiment folders. The `results/` folder contains curated small diagnostic artifacts from earlier phases; ad-hoc reruns should use `results_repro/`, which is ignored.

## .gitignore Review

The current `.gitignore` excludes `.env`, local configs with credentials, vector/database folders, model weights, logs, caches, LaTeX build artifacts, `node_modules`, and generated result folders. This is appropriate for GitHub readiness.

Key sensitive/large patterns present:

- `.env`, `.env.local`, `.env.*.local`
- `configs/config.yaml`, `configs/config_legal.yaml`
- `data/`, `*.pkl`, `*.bin`, `*.pt`, `*.safetensors`, vector/Neo4j folders
- `frontend/node_modules/`, `frontend/.next/`
- `__pycache__/`, `.pytest_cache/`, LaTeX `*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`

## Snapshot Large File Policy

- No requested snapshot artifact exceeded the 5 MB copy threshold.

## Fresh Clone/Worktree Simulation

Recorded in `docs/phase5_reproducibility_report.md`. If a worktree run fails, the failure is treated as an environment/setup issue and not as a benchmark result.
