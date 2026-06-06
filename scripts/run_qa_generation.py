"""Run QA generation pipeline: Steps 4 and 5.

Step 4: Generate QA pairs from articles using Qwen3-32B-AWQ
Step 5: Verify QA pairs with 2-level validation

Usage:
  python scripts/run_qa_generation.py
  python scripts/run_qa_generation.py --max-articles 50  # Quick test
  python scripts/run_qa_generation.py --skip-llm-verify  # Level 1 only
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
from loguru import logger

from pipeline.step04_generate_qa import process_file as generate_qa
from pipeline.step05_verify_qa import process_file as verify_qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA generation pipeline (Steps 3-4)")
    parser.add_argument("--articles", default=str(ROOT / "data" / "processed" / "hf_processed.jsonl"))
    parser.add_argument("--qa-raw", default=str(ROOT / "data" / "processed" / "qa_raw.json"))
    parser.add_argument("--qa-verified", default=str(ROOT / "data" / "processed" / "qa_verified.json"))
    parser.add_argument("--max-articles", type=int, default=None, help="Limit articles (for testing)")
    parser.add_argument("--skip-llm-verify", action="store_true", help="Skip Level 2 LLM verification")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  QA GENERATION PIPELINE — Steps 4 & 5")
    print("=" * 60)

    # Step 3: Generate QA
    print("\n🤖 Step 4: Generating QA pairs with Qwen3-32B-AWQ...")
    if args.max_articles:
        print(f"   (Limited to {args.max_articles} articles)")
    n_raw = generate_qa(args.articles, args.qa_raw, args.max_articles, args.delay)
    print(f"   ✅ {n_raw} raw QA pairs generated → {args.qa_raw}")

    # Step 4: Verify QA
    print("\n🔍 Step 5: Verifying QA pairs...")
    if args.skip_llm_verify:
        print("   (LLM verification skipped — Level 1 only)")
    total, kept = verify_qa(
        args.qa_raw, args.qa_verified, args.articles, args.skip_llm_verify, args.delay
    )
    pct = (kept / total * 100) if total > 0 else 0
    print(f"   ✅ {kept}/{total} pairs verified ({pct:.1f}%) → {args.qa_verified}")

    print("\n" + "=" * 60)
    print(f"  DONE: {n_raw} generated → {kept} verified pairs")
    print("=" * 60 + "\n")

    print("Next steps:")
    print("  1. Split QA data: python qa_pipeline/pipeline/step7_split.py")
    print("  2. Train router:  python scripts/run_router_training.py --config configs/config.yaml")
    print("  3. Run demo:      python scripts/run_pipeline.py --config configs/config.yaml --verbose")


if __name__ == "__main__":
    main()
