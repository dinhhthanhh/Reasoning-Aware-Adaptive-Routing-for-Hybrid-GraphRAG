# Phase 5 Reproducibility Report

Generated on 2026-06-08 for GitHub readiness, thesis defense evidence, and
lightweight reproducibility checks.

## Scope

Phase 5 did not rebuild the vector store, rerun the full 600-query
end-to-end benchmark, retrain the router, or change paper metrics. It
polished the repository for handoff by adding setup/demo documentation,
copying small evidence artifacts into a curated snapshot, checking smoke
tests, auditing the frontend/API demo path, and preparing a Git commit.

Base commit before Phase 5 commit: `b762829`.
Phase 5 commit hash: recorded by `git log` after the final commit because a
commit cannot reliably contain its own final hash.

## Files Added or Updated

- `README.md`: GitHub-facing project overview, results summary, setup links,
  demo guidance, and data policy.
- `docs/SETUP.md`: environment, dependency, Neo4j, vector store, and sanity
  setup guide.
- `docs/DEMO.md`: defense demo guide with the recommended CLI conversation
  routing demo.
- `docs/REPRODUCIBILITY.md`: explanation of main vs diagnostic results and
  commands used across phases.
- `docs/reproducibility_audit.md`: generated audit of Git state, ignored
  artifacts, and reproducibility caveats.
- `docs/defense_evidence_pack.md` and `.json`: one-place defense summary,
  benchmark map, table mapping, and FAQ.
- `docs/final_results_snapshot/`: curated small result artifacts and the
  final paper source/PDF for evidence backup.
- `docs/frontend_audit.md`: Next.js demo readiness and limitations.
- `docs/backend_api_audit.md`: FastAPI wrapper readiness and limitations.
- `scripts/generate_phase5_artifacts.py`: local utility to regenerate the
  final results snapshot and evidence pack.
- `.gitignore`: added LaTeX `*.toc`, ignored local rerun folders
  `results_repro/` and `results_demo/`, and kept a public exception for this
  Phase 5 report.
- `configs/config.yaml.example`: updated the example vector-store path to the
  Harrier vector store used by the final experiments.

Previously untracked reproducibility helpers were also prepared for commit:

- `evaluation/README_legal_clarify_eval.md`
- `evaluation/legal_clarify_eval.json`
- `experiments/run_routing_baselines.py`
- `experiments/compute_dataset_statistics.py`
- `scripts/build_legal_clarify_eval.py`

## Main Results Preservation

The strict end-to-end paper table was preserved. The main strict benchmark
remains:

| System | F1 | Routing accuracy | Avg latency |
|---|---:|---:|---:|
| Pure Vector | 0.3626 | 0.5000 | 1,270.7 ms |
| Pure Graph | 0.3556 | 0.2500 | 2,283.4 ms |
| Single-stage Router | 0.4231 | 0.9350 | 2,209.2 ms |
| Two-stage Hybrid | 0.4235 | 0.9283 | 3,913.4 ms |

The Phase 3 routing-only and conversation stress-test results remain
diagnostic evidence. They do not replace the strict end-to-end benchmark.

## Smoke Tests Run

| Check | Command | Result |
|---|---|---|
| Python compile | `python -m compileall router graph llm scripts` | PASS |
| API compile | `python -m compileall api` | PASS |
| Unit tests | `python -m pytest tests/ -v` | PASS, 16 passed |
| CLI conversation demo | `python scripts/demo_conversation_routing.py --config configs/config.yaml` | PASS |
| Limited conversation eval | `python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results_repro --limit 10 --use-cache` | PASS |
| Strict routing-only sanity | `python scripts/evaluate_strict_routing_only.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results_repro` | PASS |
| Frontend build | `npm run build` in `frontend/` | PASS |
| Fresh worktree compile | `python -m compileall router graph llm scripts` in temporary worktree | PASS |
| Fresh worktree tests | `python -m pytest tests/ -v` in temporary worktree | PASS, 16 passed |

## Smoke Test Details

The limited conversation eval on 10 cached examples produced route accuracy
`1.000`, clarify precision `1.000`, clarify recall `1.000`, clarify F1
`1.000`, and history-resolution accuracy `1.000`. This is only a small smoke
check and is not a benchmark number.

The strict routing-only sanity rerun completed on 600 examples and produced
route accuracy `0.8983`, 9 false clarifications, Stage 2 trigger rate
`0.5583`, Stage 2 override rate `0.3284`, and average latency `1924.2 ms`.
This differs slightly from the Phase 3 recorded artifact in the paper
discussion (`0.8933`, 10 false clarifications, override rate `0.3403`,
latency `1972.4 ms`). The rerun uses the LLM verifier path and should be
treated as a nondeterministic routing-only sanity check, not as a replacement
for paper metrics.

The CLI conversation demo showed the intended qualitative behavior:

- direct lookup routes to `dense_retrieval`;
- relation-heavy legal query routes to `graph_traversal`;
- "van ban do" with valid history resolves `Nghi dinh 100/2019/ND-CP` and
  routes to graph traversal;
- the same contextual-reference query without usable history routes to
  `clarify`;
- missing-entity and multi-interpretation examples route to `clarify`.

## Frontend and Backend Readiness

Frontend build passed with Next.js 15.1.6. The frontend is usable for a basic
single-turn visual demo, but the CLI demo is still recommended for thesis
defense because the frontend does not send explicit conversation history and
does not display `resolved_referent` or `history_resolution_status`.

The FastAPI backend compiles and exposes `POST /query`. It returns answer,
route, confidence, Stage 2 metadata, sources, latency, and ambiguity flag.
It does not yet expose all Phase 3 conversation-resolution fields, so it
should not be presented as the final conversation-history demo interface.

## Fresh Worktree Simulation

A temporary worktree was created outside the repository at commit `b762829`,
then removed after checks. In that worktree:

- `python -m compileall router graph llm scripts` passed;
- `python -m pytest tests/ -v` passed with 16 tests.

The current working tree also passed compileall and pytest after adding Phase
5 docs/scripts.

## Secret and Large File Policy

The repository keeps `.env`, local config files, vector stores, model weights,
Neo4j data, logs, caches, and generated benchmark folders out of Git.

The curated snapshot in `docs/final_results_snapshot/` copies only small
evidence files. The generator skips any requested artifact larger than 5 MB.
No requested snapshot artifact exceeded that threshold in this run.

## Remaining Caveats

- The frontend is optional; use the CLI demo for the clearest Phase 3 story.
- Full end-to-end benchmark reproduction still requires local vector store,
  Neo4j, router checkpoint, and LLM endpoint.
- The strict routing-only sanity result is diagnostic and nondeterministic
  enough that the paper should keep the recorded Phase 3 artifact wording.
- The main strict end-to-end table was not changed in Phase 5.
