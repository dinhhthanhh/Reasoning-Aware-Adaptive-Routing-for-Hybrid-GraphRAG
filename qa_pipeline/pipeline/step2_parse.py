import json
from pathlib import Path
from typing import List, Dict, Any

def step2_parse(input_path: str | Path, output_checkpoint: str | Path) -> None:
    """
    Step 2: Parse 'article_key' (DOC_NUMBER::ARTICLE_ID) into a structured field 
    'relevant_articles': [{"law_id": doc, "article_id": article}].
    Keeps all original fields.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found. Please run Step 1 first.")
        return

    print(f"✅ Loading Step 1 checkpoint: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    parsed_count = 0
    error_count = 0

    for sample in data:
        article_key = sample.get("article_key", "")
        # Parse the 'DOC_NUMBER::ARTICLE_ID' format
        if "::" in article_key:
            doc_number, article_id = article_key.split("::", 1)
        else:
            doc_number = article_key
            article_id = ""
            error_count += 1
            
        # Add the new structured field
        sample["relevant_articles"] = [{"law_id": doc_number.strip(), "article_id": article_id.strip()}]
        parsed_count += 1

    print(f"✅ Successfully converted {parsed_count} samples.")
    if error_count > 0:
        print(f"⚠️ Warning: Found {error_count} article_keys missing the '::' delimiter.")

    # Save to the new checkpoint
    out_file = Path(output_checkpoint)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Checkpoint saved to: {out_file}")

if __name__ == "__main__":
    INPUT_PATH = "qa_pipeline/data/checkpoints/step1_loaded.json"
    OUTPUT_CHECKPOINT = "qa_pipeline/data/checkpoints/step2_parsed.json"
    step2_parse(INPUT_PATH, OUTPUT_CHECKPOINT)
