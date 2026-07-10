"""Remove legacy HuggingFace corpus, legal_strict benchmarks, and stale eval artifacts.

Keeps Pháp Điển-only assets:
  - data/processed/pd_rechunked.jsonl, phapdien_processed.jsonl
  - data/vector_store/chroma_full/phapdien_full
  - Neo4j database ``phapdien`` (not touched here)
  - qa_pipeline/data/phapdien_strict/
  - evaluation/legal_clarify_eval.json, conversation_ambiguity_eval.json
  - build_logs/phapdien_graph_stats.json

Usage:
    python scripts/phapdien_pipeline/cleanup_legacy.py --dry-run
    python scripts/phapdien_pipeline/cleanup_legacy.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Entire directories to remove
DIR_TARGETS = [
    ROOT / "results_final_unified",
    ROOT / "results_phase3",
    ROOT / "eval_results",
    ROOT / "results",
    ROOT / "data/huggingface",
    ROOT / "qa_pipeline/data/legal_strict",
    ROOT / "qa_pipeline/data/legal_strict_clean",
    ROOT / "qa_pipeline/data/final",
    ROOT / "qa_pipeline/data/checkpoints",
]

# Individual files / globs at repo root or under data/
FILE_GLOBS = [
    "eval_logs_unified_v*.txt",
    "failed_query_ids*.json",
    "scratch_check_hf.py",
]

DATA_FILE_TARGETS = [
    ROOT / "data/processed/hf_processed.jsonl",
    ROOT / "data/processed/hf_rechunked.jsonl",
    ROOT / "data/processed/relationships_final.jsonl",
    ROOT / "data/processed/final_corpus.jsonl",
    ROOT / "data/processed/core_laws_processed.jsonl",
    ROOT / "data/processed/phapdien_all.json",
]

STALE_DOC_TARGETS = [
    ROOT / "docs/architecture/graph_live_state_check.md",
    ROOT / "docs/architecture/graph_live_state_check_post_rebuild.md",
    ROOT / "docs/retrieval/corpus_coverage_audit.md",
    ROOT / "docs/retrieval/hf_law_number_audit.md",
    ROOT / "docs/retrieval/rebuild_decision.md",
    ROOT / "docs/final_results_snapshot/legal_strict_full_summary.json",
    ROOT / "docs/final_results_snapshot/legal_strict_full_summary.md",
    ROOT / "docs/final_results_snapshot/post_migration_graph_quality.json",
]

EXTRA_FILE_TARGETS = [
    ROOT / "scratch_audit_hf.py",
    ROOT / "evaluation/legal_routing_eval_unified.json",
    ROOT / "evaluation/legal_mixed_clean_fp_analysis.json",
]

EXTRA_DIR_TARGETS = [
    ROOT / "results_repro",
    ROOT / "eval_logs",
]


def _rm_path(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False
    rel = path.relative_to(ROOT)
    if dry_run:
        print(f"[dry-run] would remove {rel}")
    elif path.is_dir():
        shutil.rmtree(path)
        print(f"removed dir  {rel}")
    else:
        path.unlink()
        print(f"removed file {rel}")
    return True


def cleanup_chroma_legacy(dry_run: bool) -> None:
    try:
        import chromadb
        import yaml
    except ImportError:
        print("skip chroma cleanup (chromadb/yaml not available)")
        return

    cfg_path = ROOT / "configs/build_kg_no_ner.yaml"
    if not cfg_path.exists():
        return
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    chroma_path = ROOT / cfg["chroma"]["path"]
    keep = cfg["chroma"]["collection_name"]
    if not chroma_path.exists():
        return

    client = chromadb.PersistentClient(path=str(chroma_path))
    for coll in client.list_collections():
        if coll.name == keep:
            continue
        if dry_run:
            print(f"[dry-run] would delete chroma collection {coll.name}")
        else:
            client.delete_collection(coll.name)
            print(f"deleted chroma collection {coll.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-stale-docs", action="store_true")
    args = parser.parse_args()

    removed = 0
    for d in DIR_TARGETS:
        if _rm_path(d, args.dry_run):
            removed += 1

    for pattern in FILE_GLOBS:
        for path in ROOT.glob(pattern):
            if _rm_path(path, args.dry_run):
                removed += 1

    for path in DATA_FILE_TARGETS:
        if _rm_path(path, args.dry_run):
            removed += 1

    for path in EXTRA_FILE_TARGETS:
        if _rm_path(path, args.dry_run):
            removed += 1

    for d in EXTRA_DIR_TARGETS:
        if _rm_path(d, args.dry_run):
            removed += 1

    if not args.skip_stale_docs:
        for path in STALE_DOC_TARGETS:
            if _rm_path(path, args.dry_run):
                removed += 1

    if not args.skip_chroma:
        cleanup_chroma_legacy(args.dry_run)

    # Archive script for anything still on disk
    if not args.dry_run:
        archive_script = ROOT / "scripts/phapdien_pipeline/archive_legacy_benchmarks.py"
        if archive_script.exists():
            import subprocess

            subprocess.run([sys.executable, str(archive_script)], cwd=ROOT, check=False)

    print(f"\nDone ({removed} top-level paths processed).")
    print("Verify: python scripts/phapdien_pipeline/verify_neo4j_state.py")


if __name__ == "__main__":
    main()
