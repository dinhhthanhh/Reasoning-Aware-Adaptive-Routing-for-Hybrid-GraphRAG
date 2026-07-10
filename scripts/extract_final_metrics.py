import json
from collections import defaultdict
import numpy as np
from evaluation.metrics.bertscore_eval import compute_bertscore
import scipy.stats as st

def mean_ci(data):
    if len(data) == 0: return 0.0, 0.0, 0.0
    a = np.array(data)
    m = np.mean(a)
    if len(a) < 2: return m, m, m
    ci = st.t.interval(0.95, len(a)-1, loc=m, scale=st.sem(a))
    if np.isnan(ci[0]): return m, m, m
    return m, ci[0], ci[1]

def main():
    path = "eval_results/full_eval_log.jsonl"
    records_by_sys = defaultdict(list)
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sys = data["system"]
            records_by_sys[sys].append(data)
            
    print("="*100)
    print(f"{'System':<15} | {'Route Acc':<10} | {'F1 [95% CI]':<20} | {'EM':<6} | {'Hit@1':<6} | {'Lat(ms)':<8} | {'S2-Rate':<8} | {'BERT-F1':<8}")
    print("-" * 100)
    
    for sys, records in records_by_sys.items():
        # Keep only the last 600 records
        if len(records) > 600:
            records = records[-600:]
            
        f1s = [r["token_f1"] for r in records]
        ems = [r["exact_match"] for r in records]
        hits = [r["hit_at_1"] for r in records]
        lats = [r["latency_ms"] for r in records]
        s2s = [1.0 if r.get("stage2_invoked") else 0.0 for r in records]
        
        # Route Acc
        correct_routes = sum(1 for r in records if r.get("route_correct"))
        route_acc = correct_routes / len(records) if records else 0.0
        
        f1_m, f1_l, f1_u = mean_ci(f1s)
        em_m = np.mean(ems)
        hit_m = np.mean(hits)
        lat_m = np.mean(lats)
        s2_m = np.mean(s2s)
        
        # BERTScore
        preds = [r["answer"] for r in records]
        refs = [r["gold_answer"] for r in records]
        bert_res = compute_bertscore(preds, refs, reference_field="gold_answer")
        bert_f1 = bert_res["f1"]
        
        f1_str = f"{f1_m:.3f} [{f1_l:.3f}, {f1_u:.3f}]"
        print(f"{sys:<15} | {route_acc:<10.3f} | {f1_str:<20} | {em_m:<6.3f} | {hit_m:<6.3f} | {lat_m:<8.0f} | {s2_m*100:>5.1f}% | {bert_f1:<8.3f}")

if __name__ == "__main__":
    main()
