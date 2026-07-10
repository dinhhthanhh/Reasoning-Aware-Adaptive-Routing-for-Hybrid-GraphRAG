import json
from collections import defaultdict

def main():
    path = "eval_results/full_eval_log.jsonl"
    records_by_sys = defaultdict(list)
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sys = data["system"]
            records_by_sys[sys].append(data)
            
    # For user's authoritative systems, keep the FIRST 600 records
    # For my fixed always_on, keep the LAST 600 records
    # For single_stage, we will clear it and re-run it
    
    keep_first = {"pure_vector", "pure_graph", "pure_hybrid", "two_stage", "oracle", "oracle_stage2"}
    keep_last = {"always_on"}
    
    out_records = []
    
    for sys, records in records_by_sys.items():
        if sys in keep_first:
            if len(records) >= 600:
                out_records.extend(records[:600])
                print(f"{sys}: Kept FIRST 600 out of {len(records)}")
            else:
                out_records.extend(records)
                print(f"{sys}: Kept ALL {len(records)} (less than 600)")
        elif sys in keep_last:
            if len(records) >= 600:
                out_records.extend(records[-600:])
                print(f"{sys}: Kept LAST 600 out of {len(records)}")
            else:
                out_records.extend(records)
                print(f"{sys}: Kept ALL {len(records)} (less than 600)")
        elif sys == "single_stage":
            print(f"{sys}: DROPPED ALL {len(records)} (needs rerun)")
            
    # Sort by query_id to keep it nice? No, they were originally appended. Just write them out.
    # To maintain some order, we can group by system
    with open("eval_results/full_eval_log_cleaned.jsonl", "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print("Done! Wrote eval_results/full_eval_log_cleaned.jsonl")

if __name__ == "__main__":
    main()
