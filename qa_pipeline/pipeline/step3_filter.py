import json
from pathlib import Path
from typing import List, Dict, Any

SENTENCE_ENDINGS = [
    "(",   # mở ngoặc chưa đóng
    ",",   # dấu phẩy cuối câu
]

WORD_ENDINGS = [
    " và", 
    " hoặc", 
    " gồm", 
    " như", 
    " sau",
    " của", 
    " về", 
    " theo", 
    " tại",
    " gồm:", # bổ sung thêm trường hợp có dấu hai chấm từ list cũ dính liền
    " như:",
    " sau:"
]

def step3_filter(input_path: str | Path, output_checkpoint: str | Path) -> None:
    """
    Step 3: Filter out samples with truncated answers.
    Uses strict word-boundary aware logic for BAD_ENDINGS and min-length checks.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found. Please run Step 2 first.")
        return

    print(f"✅ Loading Step 2 checkpoint: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    filtered_data = []
    removed_samples = []

    # Rule 0: Remove exact duplicate questions
    seen_questions = set()
    unique_data = []
    dup_count = 0
    for sample in data:
        q = sample.get("question", "").strip().lower()
        if q in seen_questions:
            dup_count += 1
            continue
        seen_questions.add(q)
        unique_data.append(sample)
    
    if dup_count > 0:
        print(f"⚠️ Removed {dup_count} exact duplicate questions.")
    data = unique_data

    for sample in data:
        ans = sample.get("answer", "").strip()
        is_truncated = False
        
        # Rule 2: Minimum length check (raised to 30 for academic quality)
        if len(ans) < 30:
            is_truncated = True
        else:
            # Rule 1: Truncation signal check
            # Check sentence endings (no boundary needed)
            for bad in SENTENCE_ENDINGS:
                if ans.endswith(bad):
                    is_truncated = True
                    break
                    
            # Check word endings (must have leading space)
            if not is_truncated:
                for bad in WORD_ENDINGS:
                    if ans.endswith(bad):
                        is_truncated = True
                        break

        if is_truncated:
            removed_samples.append(sample)
        else:
            filtered_data.append(sample)

    print(f"✅ Filtered out {len(removed_samples)} truncated answers.")
    print(f"✅ Remaining clean samples: {len(filtered_data)}")
    
    # Rule 3: Print examples
    if removed_samples:
        print("\n🔍 Examples of filtered answers:")
        # Print up to 3 examples
        for i, sample in enumerate(removed_samples[:3]):
            print(f"  [{i+1}] {sample.get('answer')}")

    # Save to the new checkpoint
    out_file = Path(output_checkpoint)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Checkpoint saved to: {out_file}")

if __name__ == "__main__":
    INPUT_PATH = "qa_pipeline/data/checkpoints/step2b_enriched.json"
    OUTPUT_CHECKPOINT = "qa_pipeline/data/checkpoints/step3_filtered.json"
    step3_filter(INPUT_PATH, OUTPUT_CHECKPOINT)
