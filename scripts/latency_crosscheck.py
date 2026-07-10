"""
Latency Cross-Check Script
Deep arithmetic consistency check across CSVs for the paper.
"""
import pandas as pd
import numpy as np
import os, json

base = r'c:\Users\Admin\Documents\Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG\eval_results'

pv = pd.read_csv(os.path.join(base, 'legal_strict_pure_vector_results.csv'))
pg = pd.read_csv(os.path.join(base, 'legal_strict_pure_graph_results.csv'))
ss = pd.read_csv(os.path.join(base, 'legal_strict_single_stage_router_results.csv'))
ts = pd.read_csv(os.path.join(base, 'legal_strict_two_stage_hybrid_results.csv'))

sep = "=" * 60

# -------------------------------------------------------
# 1. Route distribution verification
# -------------------------------------------------------
print(sep)
print("1. ROUTE DISTRIBUTION VERIFICATION")
print(sep)
for name, df in [('pure_vector', pv), ('pure_graph', pg), ('single_stage', ss), ('two_stage', ts)]:
    dist = dict(df['Actual_Route'].value_counts())
    total = len(df)
    print(f"  {name} (N={total}): {dist}")

# -------------------------------------------------------
# 2. Single-stage per-route latency breakdown
# -------------------------------------------------------
print()
print(sep)
print("2. SINGLE-STAGE: Latency by Actual_Route")
print(sep)
ss_route_stats = {}
for route, grp in ss.groupby('Actual_Route'):
    lat = grp['Time_ms']
    ss_route_stats[route] = {'N': len(grp), 'mean': lat.mean(), 'median': lat.median(), 'std': lat.std()}
    print(f"  {route}: N={len(grp)}, mean={lat.mean():.1f}, median={lat.median():.1f}, std={lat.std():.1f}")

# -------------------------------------------------------
# 3. Cross-file ID overlap check
# -------------------------------------------------------
print()
print(sep)
print("3. CROSS-FILE: ID overlap (same evaluation set?)")
print(sep)
ss_ids = set(ss['ID'].values)
pv_ids = set(pv['ID'].values)
ts_ids = set(ts['ID'].values)
pg_ids = set(pg['ID'].values)
print(f"  IDs in SS: {len(ss_ids)}, PV: {len(pv_ids)}, TS: {len(ts_ids)}, PG: {len(pg_ids)}")
print(f"  SS & PV: {len(ss_ids & pv_ids)}, SS & TS: {len(ss_ids & ts_ids)}, PV & PG: {len(pv_ids & pg_ids)}")
missing_in_pv = sorted(ss_ids - pv_ids)
print(f"  IDs in SS but NOT in PV ({len(missing_in_pv)}): {missing_in_pv[:10]}")
missing_in_ss = sorted(pv_ids - ss_ids)
print(f"  IDs in PV but NOT in SS ({len(missing_in_ss)}): {missing_in_ss[:10]}")

# -------------------------------------------------------
# 4. Two-stage per-route and stage2 breakdown
# -------------------------------------------------------
print()
print(sep)
print("4. TWO-STAGE: Latency by Actual_Route and Stage2")
print(sep)
ts_route_stats = {}
for route, grp in ts.groupby('Actual_Route'):
    lat = grp['Time_ms']
    ts_route_stats[route] = {'N': len(grp), 'mean': lat.mean(), 'median': lat.median(), 'std': lat.std()}
    print(f"  {route}: N={len(grp)}, mean={lat.mean():.1f}, median={lat.median():.1f}, std={lat.std():.1f}")

print()
print("  Two-stage by Stage2 flag:")
for s2, grp in ts.groupby('Stage2'):
    lat = grp['Time_ms']
    print(f"    Stage2={s2}: N={len(grp)}, mean={lat.mean():.1f}, std={lat.std():.1f}")

# -------------------------------------------------------
# 5. FINDING 1: Single-stage expected latency using two-stage per-route costs
# -------------------------------------------------------
print()
print(sep)
print("5. FINDING 1: Single-stage expected latency (using two-stage per-route means)")
print(sep)

ss_dist = dict(ss['Actual_Route'].value_counts())
ts_per_route_mean = {r: grp['Time_ms'].mean() for r, grp in ts.groupby('Actual_Route')}

print(f"  SS route distribution: {ss_dist}")
print(f"  TS per-route means (ms): {ts_per_route_mean}")

# Weighted expected latency for SS using TS per-route means
total_ss = sum(ss_dist.values())
weighted_expected = 0.0
for route, count in ss_dist.items():
    if route in ts_per_route_mean:
        weighted_expected += (count / total_ss) * ts_per_route_mean[route]
    else:
        # fallback: use SS own route mean if TS doesn't have it
        own_mean = ss_route_stats.get(route, {}).get('mean', 0)
        weighted_expected += (count / total_ss) * own_mean
        print(f"  WARNING: route '{route}' not in TS, using SS own mean={own_mean:.1f}")

ss_actual_mean = ss['Time_ms'].mean()
print()
print(f"  SS actual mean latency: {ss_actual_mean:.1f} ms")
print(f"  SS expected (TS per-route costs x SS distribution): {weighted_expected:.1f} ms")
print(f"  Difference: {ss_actual_mean - weighted_expected:.1f} ms ({(ss_actual_mean - weighted_expected)/weighted_expected*100:.1f}%)")

# -------------------------------------------------------
# 6. FINDING 2 & 3: Pure-Vector vs TS dense; Pure-Graph vs TS graph
# -------------------------------------------------------
print()
print(sep)
print("6. FINDING 2&3: Baseline vs TS same-route subsets")
print(sep)

pv_mean = pv['Time_ms'].mean()
ts_dense_grp = ts[ts['Actual_Route'] == 'dense_retrieval']['Time_ms']
ts_dense_mean = ts_dense_grp.mean()
print(f"  Pure-Vector mean (N={len(pv)}): {pv_mean:.1f} ms")
print(f"  TS dense-only  (N={len(ts_dense_grp)}): {ts_dense_mean:.1f} ms")
print(f"  Ratio TS/PV: {ts_dense_mean/pv_mean:.3f}x")
print(f"  Gap: {ts_dense_mean - pv_mean:.1f} ms")

pg_mean = pg['Time_ms'].mean()
ts_graph_grp = ts[ts['Actual_Route'] == 'graph_traversal']['Time_ms']
ts_graph_mean = ts_graph_grp.mean()
print()
print(f"  Pure-Graph mean (N={len(pg)}): {pg_mean:.1f} ms")
print(f"  TS graph-only  (N={len(ts_graph_grp)}): {ts_graph_mean:.1f} ms")
print(f"  Ratio TS/PG: {ts_graph_mean/pg_mean:.3f}x")
print(f"  Gap: {ts_graph_mean - pg_mean:.1f} ms")

# -------------------------------------------------------
# 7. Router overhead hypothesis: SS vs PV gap (fixed overhead?)
# -------------------------------------------------------
print()
print(sep)
print("7. ROUTER OVERHEAD HYPOTHESIS")
print(sep)

# Get per-route SS vs PV/PG means
ss_dense_mean = ss[ss['Actual_Route'] == 'dense_retrieval']['Time_ms'].mean()
ss_graph_mean = ss[ss['Actual_Route'] == 'graph_traversal']['Time_ms'].mean()
print(f"  SS dense mean:       {ss_dense_mean:.1f} ms")
print(f"  PV (all dense) mean: {pv_mean:.1f} ms")
print(f"  => Router overhead (dense context): {ss_dense_mean - pv_mean:.1f} ms")

print()
print(f"  SS graph mean:       {ss_graph_mean:.1f} ms")
print(f"  PG (all graph) mean: {pg_mean:.1f} ms")
print(f"  => Router overhead (graph context): {ss_graph_mean - pg_mean:.1f} ms")

# -------------------------------------------------------
# 8. Time-ordering check via ID sequence
# -------------------------------------------------------
print()
print(sep)
print("8. ID SEQUENCE / ORDERING CHECK")
print(sep)
for name, df in [('pure_vector', pv), ('pure_graph', pg), ('single_stage', ss), ('two_stage', ts)]:
    ids = df['ID'].tolist()
    try:
        id_nums = [int(str(i).replace('q', '').replace('legal_', '').split('_')[0]) for i in ids]
        print(f"  {name}: ID range [{min(id_nums)}, {max(id_nums)}], monotone={id_nums == sorted(id_nums)}")
    except Exception as e:
        print(f"  {name}: ID parse failed: {e}, sample: {ids[:3]}")

# -------------------------------------------------------
# 9. Latency percentile summary for paper
# -------------------------------------------------------
print()
print(sep)
print("9. FULL PERCENTILE SUMMARY (for paper Table verification)")
print(sep)
for name, df in [('pure_vector', pv), ('pure_graph', pg), ('single_stage', ss), ('two_stage', ts)]:
    lat = df['Time_ms']
    print(f"  {name} (N={len(df)}):")
    print(f"    mean={lat.mean():.1f}  median={lat.median():.1f}  std={lat.std():.1f}  p5={lat.quantile(0.05):.1f}  p95={lat.quantile(0.95):.1f}  min={lat.min():.1f}  max={lat.max():.1f}")

# -------------------------------------------------------
# 10. Stage2_Override consistency
# -------------------------------------------------------
print()
print(sep)
print("10. STAGE2 OVERRIDE CONSISTENCY (two_stage)")
print(sep)
print(f"  Stage2 trigger rate: {ts['Stage2'].mean():.4f}")
print(f"  Stage2_Override rate: {ts['Stage2_Override'].mean():.4f}")
print(f"  Stage2 True count: {ts['Stage2'].sum()}")
print(f"  Stage2_Override True count: {ts['Stage2_Override'].sum()}")

# Override only when Stage2=True?
override_without_s2 = ts[(ts['Stage2_Override'] == True) & (ts['Stage2'] == False)]
print(f"  Override without Stage2 trigger (should be 0): {len(override_without_s2)}")

print()
print("DONE.")
