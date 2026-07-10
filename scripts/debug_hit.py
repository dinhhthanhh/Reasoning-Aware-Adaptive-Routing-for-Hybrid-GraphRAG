import json
from evaluation.metrics.id_normalizer import normalize_legal_id, normalize_gold_article

def debug_hit_at_1():
    with open('eval_results/full_eval_log.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        entry = json.loads(line)
        if entry['system'] == 'always_on':
            gold = entry.get('gold_sources', [])
            ret = entry.get('retrieved_sources', [])
            
            gold_norm = {normalize_gold_article(g).key for g in gold if normalize_gold_article(g).is_resolvable}
            ret_norm = {normalize_legal_id(r).key for r in ret if normalize_legal_id(r).is_resolvable}
            
            if gold_norm and ret_norm:
                print(f"Query {entry['query_id']}:")
                print(f"  Raw gold: {gold}")
                print(f"  Norm gold: {gold_norm}")
                print(f"  Raw ret: {ret}")
                print(f"  Norm ret: {ret_norm}")
                print(f"  Intersection: {gold_norm.intersection(ret_norm)}")
                if gold_norm.intersection(ret_norm):
                    print("  -> MATCH!")
                break

if __name__ == '__main__':
    debug_hit_at_1()
