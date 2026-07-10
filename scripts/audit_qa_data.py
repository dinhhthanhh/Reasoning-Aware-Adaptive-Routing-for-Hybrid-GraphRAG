import json
from pathlib import Path
import os
import sys

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "qa_pipeline" / "data"
    
    files_to_check = [
        "legal_strict/test.json",
        "legal_strict/test_benchmark_v2.json",
        "legal_strict/test_retry_77.json",
        "legal_strict_clean/test.json"
    ]
    
    out_lines = [
        "# Test File Audit",
        "",
        "| File | Questions | Unique Laws | Sample Gold IDs |",
        "|---|---|---|---|",
    ]
    
    for rel_path in files_to_check:
        full_path = data_dir / rel_path
        if not full_path.exists():
            out_lines.append(f"| `{rel_path}` | NOT FOUND | - | - |")
            continue
            
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            q_count = len(data)
            laws = set()
            sample_ids = []
            
            for item in data:
                arts = item.get("relevant_articles", [])
                if not arts:
                    # some older formats use 'gold_articles' or 'article_key'
                    arts = item.get("gold_articles", [])
                
                for a in arts:
                    if isinstance(a, dict):
                        law_id = a.get("law_id", "")
                        art_id = a.get("article_id", "")
                        laws.add(law_id)
                        if len(sample_ids) < 3:
                            sample_ids.append(f"{law_id}::{art_id}")
                    elif isinstance(a, str):
                        # Some formats just use the raw string
                        laws.add(a.split("::")[0] if "::" in a else a)
                        if len(sample_ids) < 3:
                            sample_ids.append(a)
                            
            sample_str = "<br>".join([f"`{s}`" for s in sample_ids])
            out_lines.append(f"| `{rel_path}` | {q_count} | {len(laws)} | {sample_str} |")
        except Exception as e:
            out_lines.append(f"| `{rel_path}` | ERROR | - | - |")
            
    out_lines.append("")
    out_lines.append("## Decision rule from prompt:")
    out_lines.append("- Keep the file already used in benchmark (600 questions, `legal_strict/test.json`)")
    out_lines.append("- If `legal_strict_clean/test.json` has canonical IDs already, note this as potentially better.")
    out_lines.append("- DO NOT delete any file yet — wait for user decision.")
    
    out_dir = root / "docs" / "qa_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test_file_audit.md"
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        
    print(f"Audit written to {out_file}")

if __name__ == "__main__":
    main()
