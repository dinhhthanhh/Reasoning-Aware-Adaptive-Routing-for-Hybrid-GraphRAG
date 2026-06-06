import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

def step1_load(input_path: str | Path, output_checkpoint: str | Path) -> None:
    """
    Step 1: Load the raw JSON dataset, inspect its shape, check for anomalies,
    and save a checkpoint.
    """
    input_file = Path(input_path)
    # Check if the exact input path exists, otherwise try fallback
    if not input_file.exists():
        fallback = Path("data/processed/qa_verified.json")
        if fallback.exists():
            print(f"⚠️ Warning: {input_file} not found. Using fallback: {fallback}")
            input_file = fallback
        else:
            print(f"❌ Error: {input_file} not found!")
            return

    print(f"✅ Loading data from: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total_samples = len(data)
    
    # Extract keys and count them
    article_keys = [sample.get('article_key', 'UNKNOWN') for sample in data]
    unique_keys = len(set(article_keys))
    
    key_counter = Counter(article_keys)
    top_10_keys = key_counter.most_common(10)

    # Detect truncation: Count answers that don't end with typical punctuation
    valid_endings = ('.', '!', '?', '"', "'", ')', ']')
    truncation_count = 0
    for sample in data:
        ans = sample.get('answer', '').strip()
        if ans and not ans.endswith(valid_endings):
            truncation_count += 1

    print(f"✅ Total samples: {total_samples}")
    print(f"✅ Unique article_keys: {unique_keys}")
    print(f"⚠️ Truncated answers detected: {truncation_count}")
    print("✅ Top 10 samples per article_key:")
    for key, count in top_10_keys:
        print(f"   - {key}: {count} samples")

    # Save checkpoint
    out_file = Path(output_checkpoint)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Checkpoint saved to: {out_file}")

if __name__ == "__main__":
    # We use paths relative to the project root
    INPUT_PATH = "data/input/qa_verified.json"
    OUTPUT_CHECKPOINT = "qa_pipeline/data/checkpoints/step1_loaded.json"
    step1_load(INPUT_PATH, OUTPUT_CHECKPOINT)
