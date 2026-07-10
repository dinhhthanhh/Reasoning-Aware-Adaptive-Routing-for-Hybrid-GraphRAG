import json
from collections import defaultdict, Counter
import unicodedata
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase0.legal_audit_utils import gold_articles

def check_cyrillic(text: str) -> bool:
    # Cyrillic C is U+0421
    return '\u0421' in text or '\u0441' in text

def main():
    print("=== Step 1: Scanning hf_rechunked.jsonl for canonical IDs ===")
    rechunked_path = Path("data/processed/hf_rechunked.jsonl")
    law_articles = defaultdict(set)
    total_chunks = 0
    
    with open(rechunked_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                total_chunks += 1
                cid = str(record.get("canonical_id") or record.get("doc_id") or "")
                law_num = str(record.get("law_number") or "")
                if "::" in cid:
                    law, art = cid.split("::", 1)
                    law_articles[law].add(art)
                elif law_num:
                    art = str(record.get("article_number") or "")
                    law_articles[law_num].add(art or str(record.get("chunk_index", 0)))
            except Exception:
                pass
                
    print(f"Scanned {total_chunks} chunks. Found {len(law_articles)} unique law prefixes.")

    print("\n=== Step 2: Verifying Found Laws & Generating Subset ===")
    test_path = Path("qa_pipeline/data/legal_strict/test.json")
    test_data = json.load(open(test_path, encoding="utf-8"))
    
    gold_laws = set()
    for r in test_data:
        for a in gold_articles(r):
            gold_laws.add(a["law_id"])
            
    found_gold = [law for law in gold_laws if law in law_articles]
    missing_gold = [law for law in gold_laws if law not in law_articles]
    
    print(f"Gold laws found in rechunked index: {len(found_gold)} / {len(gold_laws)}")
    print("Found laws with article counts:")
    for law in sorted(found_gold):
        print(f"  [YES] {law}: {len(law_articles[law])} chunks (Articles/Preambles)")
        
    print("\nMissing laws in rechunked index:")
    for law in sorted(missing_gold):
        print(f"  [NO]  {law}")
        
    # Generate subset
    covered_subset = []
    for r in test_data:
        g_arts = gold_articles(r)
        if any(a["law_id"] in law_articles for a in g_arts):
            covered_subset.append(r)
            
    out_path = Path("qa_pipeline/data/legal_strict/test_covered.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(covered_subset, f, ensure_ascii=False, indent=2)
    print(f"\nCreated subset {out_path} with {len(covered_subset)} / {len(test_data)} questions.")

    print("\n=== Step 3: Validating Normalization Issues ===")
    cyrillic_issues = []
    unicode_issues = []
    
    for r in test_data:
        for a in gold_articles(r):
            law = a["law_id"]
            if check_cyrillic(law):
                cyrillic_issues.append((r["id"], law))
            
            nfc_law = unicodedata.normalize("NFC", law)
            nfd_law = unicodedata.normalize("NFD", law)
            if law != nfc_law:
                unicode_issues.append((r["id"], "Not NFC", law))
            
            if "VBHN " in law or "VBHN-" in law:
                pass # Check VBHN formats later if needed
                
    if cyrillic_issues:
        print(f"[!] WARNING: Found Cyrillic characters in {len(cyrillic_issues)} gold laws!")
        for q, l in cyrillic_issues:
            print(f"    - {q}: {l} (contains Cyrillic C)")
    else:
        print("[OK] No Cyrillic characters found in gold laws.")
        
    if unicode_issues:
        print(f"[!] WARNING: Found {len(unicode_issues)} Unicode normalization issues!")
    else:
        print("[OK] All gold laws are properly NFC normalized.")

if __name__ == "__main__":
    main()
