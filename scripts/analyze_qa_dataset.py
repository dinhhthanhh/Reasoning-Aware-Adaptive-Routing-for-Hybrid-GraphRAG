"""Full analysis of the upgraded QA dataset after pipeline fixes."""
import json
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parents[1]

# Load final data
with open(root / "qa_pipeline/data/final/train.json", "r", encoding="utf-8") as f:
    train = json.load(f)
with open(root / "qa_pipeline/data/final/dev.json", "r", encoding="utf-8") as f:
    dev = json.load(f)
with open(root / "qa_pipeline/data/final/test.json", "r", encoding="utf-8") as f:
    test = json.load(f)

all_data = train + dev + test

print("=" * 70)
print("UPGRADED QA DATASET — FULL ANALYSIS REPORT")
print("=" * 70)

print(f"\n1. DATASET SIZE")
print(f"   Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}, Total: {len(all_data)}")

print(f"\n2. FIELDS IN EACH SAMPLE")
print(f"   {list(all_data[0].keys())}")

# Routing label distribution
print(f"\n3. ROUTING LABEL DISTRIBUTION (Class Balance)")
rc = Counter(s.get("routing_label") for s in all_data)
total = len(all_data)
for label, count in rc.most_common():
    pct = count / total * 100
    print(f"   {label}: {count} ({pct:.1f}%)")
max_c = max(rc.values())
min_c = min(rc.values())
print(f"   Imbalance ratio (max/min): {max_c/min_c:.1f}x")

# Per-split distribution
print(f"\n4. PER-SPLIT DISTRIBUTION")
for name, data in [("Train", train), ("Dev", dev), ("Test", test)]:
    rc = Counter(s.get("routing_label") for s in data)
    parts = ", ".join(f"{k}: {v}" for k, v in rc.most_common())
    print(f"   {name} ({len(data)}): {parts}")

# Gold context analysis
print(f"\n5. GOLD CONTEXT (Critical Fix)")
has_gold = sum(1 for s in all_data if s.get("gold_context"))
gc_lengths = [len(s.get("gold_context", "")) for s in all_data if s.get("gold_context")]
print(f"   Has gold_context: {has_gold}/{len(all_data)} ({has_gold/len(all_data)*100:.1f}%)")
if gc_lengths:
    print(f"   Gold context length — min: {min(gc_lengths)}, max: {max(gc_lengths)}, mean: {sum(gc_lengths)/len(gc_lengths):.0f}")

# Supporting facts
has_sf = sum(1 for s in all_data if s.get("supporting_facts"))
print(f"   Has supporting_facts: {has_sf}/{len(all_data)}")

# Evidence analysis
ev_lengths = [len(s.get("evidence", "")) for s in all_data]
print(f"   Evidence length — min: {min(ev_lengths)}, max: {max(ev_lengths)}, mean: {sum(ev_lengths)/len(ev_lengths):.0f}")

# Answer analysis
print(f"\n6. ANSWER QUALITY")
ans_lengths = [len(s.get("answer", "")) for s in all_data]
print(f"   Answer length — min: {min(ans_lengths)}, max: {max(ans_lengths)}, mean: {sum(ans_lengths)/len(ans_lengths):.0f}")
empty_ans = sum(1 for s in all_data if not s.get("answer", "").strip())
print(f"   Empty answers: {empty_ans}")

# Question analysis
q_lengths = [len(s.get("question", "")) for s in all_data]
print(f"   Question length — min: {min(q_lengths)}, max: {max(q_lengths)}, mean: {sum(q_lengths)/len(q_lengths):.0f}")
unique_q = len(set(s.get("question") for s in all_data))
print(f"   Unique questions: {unique_q}/{len(all_data)}")

# Hop count
print(f"\n7. HOP COUNT DISTRIBUTION")
hc = Counter(s.get("hop_count") for s in all_data)
for h, c in sorted(hc.items()):
    print(f"   Hop {h}: {c}")

# Cross-doc
cd = sum(1 for s in all_data if s.get("is_cross_doc"))
print(f"   Cross-document: {cd}")

# Question type
print(f"\n8. QUESTION TYPE CLASSIFICATION")
qt = Counter(s.get("question_type", "unknown") for s in all_data)
for t, c in qt.most_common():
    print(f"   {t}: {c} ({c/len(all_data)*100:.1f}%)")

# Augmented tracking
print(f"\n9. PROVENANCE TRACKING")
aug = sum(1 for s in all_data if s.get("augmented"))
orig = len(all_data) - aug
print(f"   Original: {orig}")
print(f"   Augmented: {aug}")
aug_src = Counter(s.get("augmented_source", "original") for s in all_data)
for src, c in aug_src.most_common():
    print(f"   Source '{src}': {c}")

# Difficulty
print(f"\n10. DIFFICULTY SCORE")
diffs = [s.get("difficulty", 0) for s in all_data]
print(f"   Min: {min(diffs):.2f}, Max: {max(diffs):.2f}, Mean: {sum(diffs)/len(diffs):.2f}")

# BEFORE vs AFTER comparison
print(f"\n{'='*70}")
print("BEFORE vs AFTER COMPARISON")
print(f"{'='*70}")
print(f"{'Metric':<35} {'Before':<15} {'After':<15} {'Change':<15}")
print(f"{'-'*70}")

# Helper to calculate stats
total_rc = Counter(s.get("routing_label") for s in all_data)
def get_stat(label):
    n = total_rc.get(label, 0)
    p = n / total * 100 if total > 0 else 0
    return f"{n} ({p:.1f}%)"

track_a_after = get_stat("dense_retrieval")
track_b_after = get_stat("graph_traversal")
track_c_after = get_stat("hybrid_reasoning")
ratio_after = f"{max_c/min_c:.1f}x" if min_c > 0 else "inf"

print(f"{'Total samples':<35} {'940':<15} {len(all_data):<15} {'+' + str(len(all_data)-940):<15}")
print(f"{'Track A (dense)':<35} {'710 (75.5%)':<15} {track_a_after:<15}")
print(f"{'Track B (graph)':<35} {'150 (16.0%)':<15} {track_b_after:<15}")
print(f"{'Track C (hybrid)':<35} {'80 (8.5%)':<15} {track_c_after:<15}")
print(f"{'Imbalance ratio':<35} {'8.9x':<15} {ratio_after:<15}")
print(f"{'Has gold_context':<35} {'0/940':<15} {f'{has_gold}/{len(all_data)}':<15}")
print(f"{'Has supporting_facts':<35} {'0/940':<15} {f'{has_sf}/{len(all_data)}':<15}")
print(f"{'Duplicate questions':<35} {'11':<15} {f'{len(all_data)-unique_q}':<15}")
print(f"{'Question types':<35} {'None':<15} {f'{len(qt)} types':<15}")
print(f"{'Augmented flag':<35} {'None':<15} {f'{aug} tracked':<15}")
print(f"{'Min answer length':<35} {'5 chars':<15} {f'{min(ans_lengths)} chars':<15}")
print(f"{'='*70}")
