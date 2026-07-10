# Pháp Điển-only pipeline

Active data path for the thesis: **one source** (Bộ Pháp điển điện tử), **one Neo4j database** (`phapdien`), **one Chroma collection** (`phapdien_full`).

## Verify live state (run this after any rebuild)

```bash
python scripts/phapdien_pipeline/verify_neo4j_state.py
```

Writes `build_logs/phapdien_graph_stats.json` with counts from the **live** database.

Last verified (2026-06-28): `phapdien` — 804,322 nodes, 1,968,621 edges, 0 HF, all `LegalDoc.source = phapdien`, Chroma `phapdien_full` = 68,835 vectors.

> **Note:** Markdown under `audit/` and `docs/architecture/graph_live_state_check*.md` from before the Pháp Điển + semantic rebuild describe the old `neo4j` / HF graph. Trust `verify_neo4j_state.py` and `build_logs/phapdien_graph_stats.json` instead.

## Full rebuild

```bash
python scripts/phapdien_pipeline/run_e2e.py --skip-benchmark
```

Steps: rechunk → benchmark → `build_neo4j.py --wipe` → `build_chroma.py` → verify → (optional) archive legacy HF artifacts.

Incremental structural update only (keeps semantic nodes):

```bash
python scripts/phapdien_pipeline/build_neo4j.py --replace-pd-only
```

## Cleanup legacy HF / old results

```bash
python scripts/phapdien_pipeline/cleanup_legacy.py --dry-run
python scripts/phapdien_pipeline/cleanup_legacy.py
```

Removes `results_final_unified/`, `legal_strict` benchmarks, HF corpus files, stale eval logs, and unused Chroma collections (keeps `phapdien_full` only).

Structural ingest: `build_neo4j.py` (Layer 1: LegalDoc/LegalArticle).

Semantic relations (MENTIONS, AMENDS, …): built separately via `build_semantic_kg.py` or `build_neo4j.py --build-semantics`. The current production `phapdien` database includes this semantic layer (~731k concept/actor nodes).

## Config

- Neo4j: `configs/build_kg_no_ner.yaml` → `neo4j.database: phapdien`
- Chroma: `chroma.path: data/vector_store/chroma_full`, `collection_name: phapdien_full`
- Benchmark: `qa_pipeline/data/phapdien_strict/`
