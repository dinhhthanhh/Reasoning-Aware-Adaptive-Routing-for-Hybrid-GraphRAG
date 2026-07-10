"""Move legacy HF / legal_strict artifacts to archive/ (PD-only project).

Usage:
    python scripts/phapdien_pipeline/archive_legacy_benchmarks.py
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_ROOT = ROOT / "archive" / "qa_pipeline_legacy"


def archive_path(src: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest = ARCHIVE_ROOT / stamp / src.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    if src.is_dir():
        shutil.move(str(src), str(dest))
    elif src.is_file():
        shutil.move(str(src), str(dest))
    return dest


def main() -> None:
    targets = [
        ROOT / "qa_pipeline/data/legal_strict",
        ROOT / "qa_pipeline/data/legal_strict_clean",
        ROOT / "data/processed/hf_processed.jsonl",
        ROOT / "data/processed/hf_rechunked.jsonl",
        ROOT / "data/processed/relationships_final.jsonl",
        ROOT / "data/processed/final_corpus.jsonl",
        ROOT / "data/processed/core_laws_processed.jsonl",
    ]
    moved: list[str] = []
    for src in targets:
        if not src.exists():
            continue
        dest = archive_path(src)
        moved.append(f"{src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")
        print(f"Archived {src} -> {dest}")

    eval_dir = ROOT / "eval_results"
    if eval_dir.exists():
        for f in eval_dir.glob("legal_strict_*"):
            dest = archive_path(f)
            moved.append(f"{f.name} -> {dest.relative_to(ROOT)}")
            print(f"Archived {f} -> {dest}")

    readme = ARCHIVE_ROOT / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# Legacy QA / HF artifacts (archived)\n\n"
        "These files belonged to the old `legal_strict` + HuggingFace corpus benchmark.\n"
        "The active PD-only benchmark is `qa_pipeline/data/phapdien_strict/`.\n\n"
        f"Last archive run moved {len(moved)} paths.\n",
        encoding="utf-8",
    )
    print(f"Done. Archived {len(moved)} items under {ARCHIVE_ROOT}")


if __name__ == "__main__":
    main()
