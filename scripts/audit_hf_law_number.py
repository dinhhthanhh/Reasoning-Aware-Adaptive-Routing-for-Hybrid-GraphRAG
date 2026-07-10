import json
import re
from pathlib import Path
import random
import sys

# Define patterns based on prompt
PATTERNS = [
    (r'Số:\s*([0-9]+/[A-ZĐ\-]+)', "Số: Format"),
    (r'(\d+/VBHN-[A-Z]+)', "VBHN Format"),
    (r'(\d+/\d{4}/NĐ-CP)', "NĐ-CP"),
    (r'(\d+/\d{4}/TT-[A-ZĐ]+)', "Thông tư"),
    (r'(\d+/\d{4}/QH\d+)', "Luật QH"),
    (r'(\d+/\d{4}/QĐ-[A-ZĐ]+)', "Quyết định"),
    (r'(\d+/\d{4}/[A-ZĐ]+-[A-Z0-9Đ]+)', "General year/code"),
    (r'(\d+/[A-ZĐ]+-[A-Z0-9Đ]+)', "Short Format"),
]

def extract_law_number(text):
    text_subset = text[:1000] # Check title and header mostly
    for pat_str, pat_name in PATTERNS:
        match = re.search(pat_str, text_subset, re.IGNORECASE)
        if match:
            # Quick fix for Cyrillic 'С'
            matched_text = match.group(1).replace('\u0421', 'C').replace('\u0441', 'c')
            return matched_text.upper(), pat_name
    return None, None

def main():
    root = Path(__file__).resolve().parent.parent
    hf_path = root / "data" / "processed" / "hf_processed.jsonl"
    out_dir = root / "docs" / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hf_law_number_audit.md"
    
    if not hf_path.exists():
        print(f"HF path not found: {hf_path}")
        return

    # Reservoir sampling for 200 documents
    sample_size = 200
    sample = []
    
    with open(hf_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < sample_size:
                sample.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sample[j] = line

    results = []
    unmatched = []
    method_dist = {}
    success_count = 0

    for line in sample:
        try:
            doc = json.loads(line)
            content = doc.get("content_markdown", "") or doc.get("text", "")
            doc_id = doc.get("doc_id", "Unknown")
            title = doc.get("title", "")
            
            # Combine title and content for better extraction chance
            text_to_search = title + "\n" + content
            
            law_num, method = extract_law_number(text_to_search)
            
            if law_num:
                success_count += 1
                method_dist[method] = method_dist.get(method, 0) + 1
            else:
                unmatched.append({
                    "doc_id": doc_id,
                    "title": title
                })
                
            results.append({
                "doc_id": doc_id,
                "title": title[:50] + "..." if len(title) > 50 else title,
                "extracted": law_num,
                "method": method,
                "preview": content[:150].replace("\n", " ") + "..."
            })
            
        except Exception as e:
            pass

    rate = (success_count / len(sample)) * 100 if sample else 0
    
    out_lines = [
        "# HF Law Number Extraction Audit",
        "",
        f"**Sample size:** {len(sample)} documents",
        f"**Extraction rate:** {rate:.1f}% ({success_count}/{len(sample)})",
        "",
        "## Method Distribution",
    ]
    
    for m, count in sorted(method_dist.items(), key=lambda x: -x[1]):
        out_lines.append(f"- **{m}:** {count} documents")
        
    out_lines.append("")
    out_lines.append("## STOP A GATE CHECK")
    if rate < 60:
        out_lines.append(f"**FAILED:** Extraction rate {rate:.1f}% < 60%. STOP and report. Need different approach.")
    else:
        out_lines.append(f"**PASSED:** Extraction rate {rate:.1f}% >= 60%. Proceed to Step 1.2.")
        
    out_lines.append("")
    out_lines.append("## Sample Details (First 50)")
    out_lines.append("| Doc ID | Extracted Law | Method | Preview |")
    out_lines.append("|---|---|---|---|")
    
    for r in results[:50]:
        ext = f"`{r['extracted']}`" if r['extracted'] else "NONE"
        out_lines.append(f"| {r['doc_id']} | {ext} | {r['method'] or '-'} | {r['preview'][:50]} |")
        
    out_lines.append("")
    out_lines.append("## Top Unmatched Titles (Up to 100)")
    for r in unmatched[:100]:
        out_lines.append(f"- `[{r['doc_id']}]` {r['title']}")
        
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        
    print(f"Audit written to {out_file}")
    print(f"Extraction rate: {rate:.1f}%")

if __name__ == "__main__":
    random.seed(42) # For reproducible sampling
    main()
