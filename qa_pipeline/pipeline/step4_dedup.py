import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict, Counter

def step4_dedup(input_path: str | Path, output_checkpoint: str | Path) -> None:
    """
    Step 4: Deduplicate samples. Keep max 3 questions per article_key.
    Outputs: total removed, total remaining, top 5 article_keys, and distribution.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found. Please run Step 3 first.")
        return

    print(f"✅ Loading Step 3 checkpoint: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # Group samples by article_key
    article_groups = defaultdict(list)
    for sample in data:
        key = sample.get("article_key", "UNKNOWN")
        article_groups[key].append(sample)

    deduped_data = []
    removed_count = 0

    # Dedup logic: max 3 per key
    for key, samples in article_groups.items():
        if len(samples) > 3:
            removed_count += (len(samples) - 3)
            deduped_data.extend(samples[:3])
        else:
            deduped_data.extend(samples)

    total_remaining = len(deduped_data)
    
    # Calculate post-dedup distribution
    post_counts = Counter(sample.get("article_key", "UNKNOWN") for sample in deduped_data)
    
    top_5 = post_counts.most_common(5)
    
    distribution = Counter(post_counts.values())

    print(f"✅ Total removed (excess > 3): {removed_count}")
    print(f"✅ Total remaining samples: {total_remaining}")
    
    print("\n📊 Top 5 article_keys (post-dedup):")
    for key, count in top_5:
        print(f"  - {key}: {count} samples")
        
    print("\n📈 Distribution of samples per article_key:")
    print(f"  - Exactly 1 sample: {distribution.get(1, 0)} article_keys")
    print(f"  - Exactly 2 samples: {distribution.get(2, 0)} article_keys")
    print(f"  - Exactly 3 samples: {distribution.get(3, 0)} article_keys")

    # Save to the new checkpoint
    out_file = Path(output_checkpoint)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(deduped_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Checkpoint saved to: {out_file}")

if __name__ == "__main__":
    INPUT_PATH = "qa_pipeline/data/checkpoints/step3_filtered.json"
    OUTPUT_CHECKPOINT = "qa_pipeline/data/checkpoints/step4_deduped.json"
    step4_dedup(INPUT_PATH, OUTPUT_CHECKPOINT)
