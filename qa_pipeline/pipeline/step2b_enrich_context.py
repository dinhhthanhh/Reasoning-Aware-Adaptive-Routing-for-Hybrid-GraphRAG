"""
Step 2b: Enrich QA samples with gold context from the article corpus.

This step joins each QA sample with the full article text from
data/processed/articles.json, creating the critical 'gold_context' field
required for RAGAS-style evaluation metrics (Context Precision/Recall,
Faithfulness, Answer Correctness).

Input:  qa_pipeline/data/checkpoints/step2_parsed.json
Corpus: data/processed/articles.json
Output: qa_pipeline/data/checkpoints/step2b_enriched.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def build_corpus_lookup(corpus_path: Path) -> Dict[str, Dict[str, Any]]:
    """Build a lookup dictionary: article_key -> full article record."""
    with corpus_path.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    lookup = {}
    for art in articles:
        key = art.get("article_key", "")
        if key:
            lookup[key] = art
    return lookup


def step2b_enrich_context(
    input_path: str | Path,
    corpus_path: str | Path,
    output_checkpoint: str | Path,
) -> None:
    """
    Step 2b: Join QA samples with full article text from corpus.

    For each QA sample:
    1. Look up the article_key in the corpus to get full content text.
    2. Populate 'gold_context' field with the full article body.
    3. Update 'evidence' with the actual article content (not just the title).
    4. Add 'supporting_facts' field listing which articles are referenced.
    """
    input_file = Path(input_path)
    corpus_file = Path(corpus_path)

    if not input_file.exists():
        print(f"❌ Error: {input_file} not found. Please run Step 2 first.")
        return
    if not corpus_file.exists():
        print(f"❌ Error: Corpus {corpus_file} not found.")
        return

    print(f"✅ Loading Step 2 checkpoint: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    print(f"✅ Loading corpus: {corpus_file}")
    corpus = build_corpus_lookup(corpus_file)
    print(f"   Corpus has {len(corpus)} articles indexed by article_key.")

    matched = 0
    unmatched = 0
    enriched_data = []

    for sample in data:
        article_key = sample.get("article_key", "")

        # --- Primary lookup: exact article_key match ---
        art = corpus.get(article_key)

        if art and art.get("content"):
            content = art["content"].strip()
            sample["gold_context"] = content

            # Upgrade evidence: if evidence is just a short title, replace it
            existing_evidence = sample.get("evidence", "")
            if len(existing_evidence) < 50:
                sample["evidence"] = content

            # Add supporting_facts (article references)
            sample["supporting_facts"] = [
                {
                    "law_id": art.get("doc_number", ""),
                    "article_id": art.get("article_id", ""),
                    "title": f"{art.get('law', '')} — {art.get('article_id', '')}",
                }
            ]
            matched += 1
        else:
            # Could not find — keep the sample but mark gold_context as empty
            sample["gold_context"] = ""
            sample["supporting_facts"] = []
            unmatched += 1

        enriched_data.append(sample)

    print(f"\n📊 Gold Context Enrichment Results:")
    print(f"   ✅ Matched: {matched}/{len(data)} ({matched/len(data)*100:.1f}%)")
    if unmatched > 0:
        print(f"   ⚠️ Unmatched: {unmatched}/{len(data)}")

    # Stats on gold_context lengths
    gc_lengths = [len(s.get("gold_context", "")) for s in enriched_data if s.get("gold_context")]
    if gc_lengths:
        print(f"\n📏 Gold Context Length Stats:")
        print(f"   Min: {min(gc_lengths)} chars")
        print(f"   Max: {max(gc_lengths)} chars")
        print(f"   Mean: {sum(gc_lengths)/len(gc_lengths):.0f} chars")

    # Save checkpoint
    out_file = Path(output_checkpoint)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Checkpoint saved to: {out_file}")


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    INPUT_PATH = ROOT_DIR / "qa_pipeline/data/checkpoints/step2_parsed.json"
    CORPUS_PATH = ROOT_DIR / "data/processed/articles.json"
    OUTPUT_CHECKPOINT = ROOT_DIR / "qa_pipeline/data/checkpoints/step2b_enriched.json"

    step2b_enrich_context(str(INPUT_PATH), str(CORPUS_PATH), str(OUTPUT_CHECKPOINT))
