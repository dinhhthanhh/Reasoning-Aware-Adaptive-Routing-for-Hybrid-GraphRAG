"""Extract p50/p95 latency and latency decomposition from full_eval_log.jsonl."""
import json
import statistics

def extract_latency(path: str = "eval_results/full_eval_log.jsonl"):
    data: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sys = entry["system"]
            if sys not in data:
                data[sys] = {
                    "lat": [], "retr_lat": [], "gen_lat": [],
                    "s1_lat": [], "s2_lat": [],
                }
            d = data[sys]
            d["lat"].append(entry.get("latency_ms", 0))
            d["retr_lat"].append(entry.get("retrieval_latency_ms", 0) or 0)
            d["gen_lat"].append(entry.get("generation_latency_ms", 0) or 0)
            d["s1_lat"].append(entry.get("stage1_latency_ms", 0) or 0)
            d["s2_lat"].append(entry.get("stage2_latency_ms", 0) or 0)

    ORDER = ["pure_vector", "pure_graph", "pure_hybrid", "single_stage",
             "two_stage", "always_on", "oracle", "oracle_stage2"]

    print(f"\n{'System':<20} {'Mean':>8} {'p50':>8} {'p95':>8} {'Retr':>8} {'Gen':>8} {'S2Lat':>8}")
    print("-" * 80)
    for sys in ORDER:
        if sys not in data:
            continue
        d = data[sys]
        lat = sorted(x for x in d["lat"] if x > 0)
        retr = [x for x in d["retr_lat"] if x > 0]
        gen = [x for x in d["gen_lat"] if x > 0]
        s2 = [x for x in d["s2_lat"] if x > 0]
        n = len(lat)
        if not n:
            continue
        p50 = lat[int(n * 0.50)]
        p95 = lat[int(n * 0.95)]
        mean = statistics.mean(lat)
        retr_mean = statistics.mean(retr) if retr else 0
        gen_mean = statistics.mean(gen) if gen else 0
        s2_mean = statistics.mean(s2) if s2 else 0
        print(f"{sys:<20} {mean:>8.0f} {p50:>8.0f} {p95:>8.0f} {retr_mean:>8.0f} {gen_mean:>8.0f} {s2_mean:>8.0f}")

    # Hit@1 and MRR summary
    print(f"\n{'System':<20} {'Count':>6} {'F1':>8} {'Hit@1':>8} {'MRR':>8}")
    print("-" * 60)
    hits: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            sys = e["system"]
            if sys not in hits:
                hits[sys] = {"f1": [], "h1": [], "mrr": []}
            hits[sys]["f1"].append(e.get("token_f1", 0) or 0)
            hits[sys]["h1"].append(e.get("hit_at_1", 0) or 0)
            hits[sys]["mrr"].append(e.get("mrr", 0) or 0)
    for sys in ORDER:
        if sys not in hits:
            continue
        h = hits[sys]
        n = len(h["f1"])
        print(f"{sys:<20} {n:>6} {statistics.mean(h['f1']):>8.3f} {statistics.mean(h['h1']):>8.3f} {statistics.mean(h['mrr']):>8.3f}")

if __name__ == "__main__":
    extract_latency()
