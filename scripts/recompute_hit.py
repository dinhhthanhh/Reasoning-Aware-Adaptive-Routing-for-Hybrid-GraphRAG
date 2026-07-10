import json
from evaluation.metrics import compute_hit_at_k, compute_mrr
from collections import defaultdict

def recompute_metrics():
    with open('eval_results/full_eval_log.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    system_metrics = defaultdict(lambda: {'h1': [], 'h3': [], 'mrr': []})
    
    for line in lines:
        entry = json.loads(line)
        sys = entry['system']
        
        # In the old logs, it was stored under retrieved_source_ids
        ret = entry.get('retrieved_source_ids', [])
        gold = entry.get('gold_source_ids', entry.get('gold_sources', []))
        
        h1 = compute_hit_at_k(ret, gold, k=1, mode='strict')
        h3 = compute_hit_at_k(ret, gold, k=3, mode='strict')
        mrr = compute_mrr(ret, gold, mode='strict')
        
        system_metrics[sys]['h1'].append(h1)
        system_metrics[sys]['h3'].append(h3)
        system_metrics[sys]['mrr'].append(mrr)
        
    for sys, metrics in system_metrics.items():
        h1_avg = sum(metrics['h1']) / len(metrics['h1'])
        h3_avg = sum(metrics['h3']) / len(metrics['h3'])
        mrr_avg = sum(metrics['mrr']) / len(metrics['mrr'])
        print(f"[{sys}] Hit@1: {h1_avg:.3f}, Hit@3: {h3_avg:.3f}, MRR: {mrr_avg:.3f}")

if __name__ == '__main__':
    recompute_metrics()
