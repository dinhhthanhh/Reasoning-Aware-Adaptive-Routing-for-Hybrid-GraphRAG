import json
from collections import defaultdict
from evaluation.metrics.bertscore_eval import compute_bertscore

def main():
    path = "eval_results/full_eval_log.jsonl"
    preds = defaultdict(list)
    refs = defaultdict(list)
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sys = data["system"]
            preds[sys].append(data["answer"])
            refs[sys].append(data["gold_answer"])
            
    print("Computing BERTScore...")
    for sys in preds:
        print(f"System: {sys}")
        result = compute_bertscore(preds[sys], refs[sys], reference_field="gold_answer")
        print(f"  BERT-F1: {result['f1']:.3f} ± {result['f1_std']:.3f}")
        
if __name__ == "__main__":
    main()
