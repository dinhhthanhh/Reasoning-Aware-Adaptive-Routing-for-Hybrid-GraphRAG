"""Pháp Điển-only end-to-end pipeline.

All knowledge (vector index + Neo4j) is built exclusively from Bộ Pháp điển.
Neo4j target database: ``phapdien`` (see ``configs/build_kg_no_ner.yaml``).

Steps:
  1. Rechunk PD (canonical IDs) if pd_rechunked.jsonl missing
  2. Generate phapdien_strict benchmark
  3. Wipe + rebuild Neo4j database ``phapdien`` (structural PD graph)
  4. Rebuild Chroma collection ``phapdien_full``
  5. Optional: archive legacy HF / legal_strict artifacts
  6. Verify live Neo4j + run benchmark on phapdien_strict test set

Usage:
    python scripts/phapdien_pipeline/run_e2e.py
    python scripts/phapdien_pipeline/run_e2e.py --incremental --skip-cleanup
    python scripts/phapdien_pipeline/run_e2e.py --skip-cleanup --benchmark-limit 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "configs/build_kg_no_ner.yaml"
NEO4J_DATABASE = "phapdien"


def resolve_config() -> Path:
    primary = ROOT / "configs/config.yaml"
    return primary if primary.exists() else DEFAULT_CONFIG


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def cleanup_hf_artifacts() -> None:
    """Archive HuggingFace corpus files and legal_strict benchmark."""
    subprocess.run(
        [sys.executable, "scripts/phapdien_pipeline/archive_legacy_benchmarks.py"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rechunk", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Merge/replace :PD nodes only (no full wipe). Not for semantic-KG rebuilds.",
    )
    parser.add_argument("--benchmark-limit", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--database", default=NEO4J_DATABASE)
    args = parser.parse_args()

    config_path = resolve_config()

    py = sys.executable
    pd_rechunked = ROOT / "data/processed/pd_rechunked.jsonl"

    if not args.skip_rechunk and not pd_rechunked.exists():
        run([py, "scripts/rechunk_pd_canonical.py"], "Step 1: PD canonical rechunk")

    run(
        [py, "scripts/phapdien_pipeline/build_benchmark.py", "--test-size", str(args.test_size)],
        "Step 2: Generate phapdien_strict benchmark",
    )

    if not args.skip_neo4j:
        neo_cmd = [
            py,
            "scripts/phapdien_pipeline/build_neo4j.py",
            "--database",
            args.database,
            "--config",
            str(config_path),
        ]
        if args.incremental:
            neo_cmd.append("--replace-pd-only")
        else:
            neo_cmd.append("--wipe")
        run(neo_cmd, f"Step 3: Neo4j ingest → database '{args.database}' (Pháp Điển only)")

    if not args.skip_chroma:
        run(
            [py, "scripts/phapdien_pipeline/build_chroma.py", "--config", str(config_path)],
            "Step 4: Chroma PD-only index",
        )

    if not args.skip_cleanup:
        cleanup_hf_artifacts()

    if not args.skip_verify:
        run(
            [
                py,
                "scripts/phapdien_pipeline/verify_neo4j_state.py",
                "--config",
                str(config_path),
                "--database",
                args.database,
            ],
            "Step 5: Verify Neo4j phapdien-only state",
        )

    if not args.skip_benchmark:
        cmd = [
            py,
            "scripts/run_benchmark_eval.py",
            "--dataset",
            "phapdien_strict",
            "--systems",
            "pure_vector,pure_graph,single_stage_router,two_stage_hybrid",
            "--config",
            str(config_path),
        ]
        if args.benchmark_limit:
            cmd.extend(["--limit", str(args.benchmark_limit)])
        run(cmd, "Step 6: End-to-end benchmark")

    print("\nPháp Điển E2E pipeline complete.")


if __name__ == "__main__":
    main()
