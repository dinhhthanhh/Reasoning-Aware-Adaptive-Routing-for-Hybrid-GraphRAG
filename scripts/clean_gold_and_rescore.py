"""Strip Pháp Điển markdown formatting from gold answers and re-score all eval CSVs.

Patterns to strip:
1. Section headers:  ### 3.1. Điều 3.1.TT.7.6. Tên điều - Nghị định số XXX
2. Source citation:  *(Nguồn gốc: Điều X ... ngày DD/MM/YYYY)*
3. Trailing " - Nghị định số 42/2016/NĐ-CP" in headers (already part of header)
"""

import json
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import compute_token_f1, normalize_text


def clean_gold_answer(text: str) -> str:
    """Strip Pháp Điển formatting, keeping only substantive legal content."""
    if not text:
        return text

    # 1. Remove markdown headers: ### 3.1. Điều 3.1.TT.7.6. Title text - Nghị định ...
    #    Keep the title text but remove the ### and PD numbering prefix
    text = re.sub(
        r'^###\s+'                        # opening ###
        r'[\d.]+\s+'                      # e.g. "3.1. " or "14.5. "
        r'Điều\s+[\d.]+\.[A-ZĐa-zđ]+\.[\d.]+\.\s*'  # e.g. "Điều 3.1.TT.7.6. "
        r'(.*?)$',                        # capture the actual title
        r'\1',
        text,
        flags=re.MULTILINE
    )
    # Fallback: remove any remaining ### headers
    text = re.sub(r'^###\s+', '', text, flags=re.MULTILINE)

    # 2. Remove source citation: *(Nguồn gốc: ...)*
    text = re.sub(
        r'\*\(Nguồn gốc:.*?\)\*',
        '',
        text,
        flags=re.DOTALL
    )

    # 3. Remove " - Luật số/Nghị định số/Thông tư số XXX" suffix from titles
    #    that appeared in the header line
    text = re.sub(
        r'\s*-\s*(?:Luật số|Nghị định số|Thông tư số|Thông tư liên tịch số)\s+[\d/A-ZĐa-zđ\-]+(?:\s+ngày\s+\d+/\d+/\d+\s+của\s+[^*\n]+)?',
        '',
        text
    )

    # 4. Remove stray markdown emphasis markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**

    # 5. Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def rescore_csv(csv_path: Path, cleaned_gold_map: dict[str, str]) -> dict:
    """Re-score a CSV with cleaned gold answers. Returns summary stats."""
    if not csv_path.exists():
        return {"error": f"not found: {csv_path}"}

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    old_f1s = []
    new_f1s = []
    old_ems = []
    new_ems = []

    for row in rows:
        qid = row.get("ID", "")
        old_gt = row.get("Ground_Truth", "")
        answer = row.get("Generated_Answer", "")

        # Old scores
        old_f1s.append(float(row.get("F1", 0)))
        old_ems.append(float(row.get("EM", 0)))

        # New scores with cleaned gold
        cleaned_gt = cleaned_gold_map.get(qid, old_gt)
        scores = compute_token_f1(answer or "", cleaned_gt or "")
        new_f1s.append(float(scores["f1"]))
        new_ems.append(float(scores["exact_match"]))

    n = len(rows)
    return {
        "n": n,
        "old_f1": sum(old_f1s) / n if n else 0,
        "new_f1": sum(new_f1s) / n if n else 0,
        "delta_f1": (sum(new_f1s) - sum(old_f1s)) / n if n else 0,
        "old_em": sum(old_ems) / n if n else 0,
        "new_em": sum(new_ems) / n if n else 0,
        "old_f1_gt05": sum(1 for f in old_f1s if f > 0.5),
        "new_f1_gt05": sum(1 for f in new_f1s if f > 0.5),
    }


def main():
    # Load benchmark
    test_path = Path("qa_pipeline/data/phapdien_strict/test.json")
    data = json.loads(test_path.read_text(encoding="utf-8"))

    # Build cleaned gold map
    cleaned_gold_map: dict[str, str] = {}
    for d in data:
        qid = str(d.get("id", ""))
        raw_gold = d.get("answer") or d.get("ground_truth") or ""
        cleaned_gold_map[qid] = clean_gold_answer(raw_gold)

    # Show cleaning examples
    print("=" * 60)
    print("CLEANING EXAMPLES")
    print("=" * 60)
    for i in [0, 50, 400]:
        qid = str(data[i].get("id", ""))
        raw = data[i].get("answer") or ""
        cleaned = cleaned_gold_map[qid]
        print(f"\n[{i}] RAW (first 200): {raw[:200]}")
        print(f"[{i}] CLEAN (first 200): {cleaned[:200]}")
        print(f"[{i}] Reduction: {len(raw)} → {len(cleaned)} chars ({(1 - len(cleaned)/len(raw))*100:.1f}% removed)")

    # Re-score all CSVs
    print("\n" + "=" * 60)
    print("RE-SCORING RESULTS")
    print("=" * 60)

    csv_files = {
        "pure_vector": "eval_results/phapdien_strict_pure_vector_results.csv",
        "pure_graph": "eval_results/phapdien_strict_pure_graph_results.csv",
        "single_stage_router": "eval_results/phapdien_strict_single_stage_router_results.csv",
        "two_stage_hybrid": "eval_results/phapdien_strict_two_stage_hybrid_results.csv",
    }

    results = {}
    for sys_name, csv_path in csv_files.items():
        stats = rescore_csv(Path(csv_path), cleaned_gold_map)
        results[sys_name] = stats
        if "error" not in stats:
            print(f"\n{sys_name}:")
            print(f"  OLD F1: {stats['old_f1']:.4f}  →  NEW F1: {stats['new_f1']:.4f}  (Δ = {stats['delta_f1']:+.4f})")
            print(f"  OLD EM: {stats['old_em']:.4f}  →  NEW EM: {stats['new_em']:.4f}")
            print(f"  F1>0.5 count: {stats['old_f1_gt05']} → {stats['new_f1_gt05']}")
        else:
            print(f"\n{sys_name}: {stats['error']}")

    # Save results
    out_path = Path("results/final/cleaned_gold_rescore.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
