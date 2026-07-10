"""
Per-query delta analysis and deep latency forensics
"""
import pandas as pd
import numpy as np

base = r'c:\Users\Admin\Documents\Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG\eval_results'
ss = pd.read_csv(base + r'\legal_strict_single_stage_router_results.csv')
ts = pd.read_csv(base + r'\legal_strict_two_stage_hybrid_results.csv')
pv = pd.read_csv(base + r'\legal_strict_pure_vector_results.csv')
pg = pd.read_csv(base + r'\legal_strict_pure_graph_results.csv')

sep = "=" * 60

# ---- Merge on ID ----
merged = ss[['ID','Time_ms']].rename(columns={'Time_ms':'SS_ms'}).merge(
    ts[['ID','Time_ms']].rename(columns={'Time_ms':'TS_ms'}),
    on='ID', how='inner'
)
merged = merged.merge(pv[['ID','Time_ms']].rename(columns={'Time_ms':'PV_ms'}), on='ID', how='left')
merged = merged.merge(pg[['ID','Time_ms']].rename(columns={'Time_ms':'PG_ms'}), on='ID', how='left')

print(sep)
print("A. MATCHED QUERIES")
print(sep)
print("  Matched SS+TS:", len(merged))
print("  With PV data:", merged['PV_ms'].notna().sum())
print("  With PG data:", merged['PG_ms'].notna().sum())

# ---- Per-query TS - SS delta ----
print()
print(sep)
print("B. Per-query TS - SS delta (Stage2 overhead per query)")
print(sep)
delta = merged['TS_ms'] - merged['SS_ms']
print("  mean:", round(delta.mean(), 1))
print("  median:", round(delta.median(), 1))
print("  std:", round(delta.std(), 1))
print("  p5:", round(delta.quantile(0.05), 1))
print("  p95:", round(delta.quantile(0.95), 1))
print("  queries where TS < SS:", int((delta < 0).sum()), "(" + str(round((delta < 0).mean() * 100, 1)) + "%)")

# ---- SS vs PV same-query ----
print()
print(sep)
print("C. Per-query SS - PV delta (routing overhead injected into dense queries)")
print(sep)
m_pv = merged[merged['PV_ms'].notna()].copy()
delta_ss_pv = m_pv['SS_ms'] - m_pv['PV_ms']
print("  N:", len(delta_ss_pv))
print("  mean:", round(delta_ss_pv.mean(), 1))
print("  median:", round(delta_ss_pv.median(), 1))
print("  std:", round(delta_ss_pv.std(), 1))
print("  Note: these are queries in PV (all dense) also present in SS")

# ---- TS Stage2=False vs SS ----
print()
print(sep)
print("D. TS (Stage2=False) vs SS per-query delta")
print(sep)
ts_no_s2 = ts[ts['Stage2'] == False][['ID','Time_ms']].rename(columns={'Time_ms':'TS_ns_ms'})
m_no_s2 = ss[['ID','Time_ms']].rename(columns={'Time_ms':'SS_ms'}).merge(ts_no_s2, on='ID', how='inner')
delta_no_s2 = m_no_s2['TS_ns_ms'] - m_no_s2['SS_ms']
print("  N matched:", len(delta_no_s2))
print("  mean delta:", round(delta_no_s2.mean(), 1))
print("  median delta:", round(delta_no_s2.median(), 1))
print("  std:", round(delta_no_s2.std(), 1))
print("  queries where TS_noS2 < SS:", int((delta_no_s2 < 0).sum()), "(" + str(round((delta_no_s2 < 0).mean() * 100, 1)) + "%)")

# ---- PV latency outlier check ----
print()
print(sep)
print("E. Pure-Vector latency distribution inspection (N=598 vs expected 600)")
print(sep)
all_ids = set(['legal_strict_test_' + str(i).zfill(4) for i in range(600)])
pv_ids = set(pv['ID'].values)
ss_ids = set(ss['ID'].values)
missing = sorted(all_ids - pv_ids)
print("  Missing IDs in PV:", missing[:5])
print("  PV missing count:", len(all_ids - pv_ids))

# what are the 2 missing IDs and their latency in other systems?
missing_ids = list(all_ids - pv_ids)
print()
print("  Missing IDs in SS latency:")
print(ss[ss['ID'].isin(missing_ids)][['ID', 'Time_ms', 'Actual_Route']].to_string())
print()
print("  Missing IDs in TS latency:")
print(ts[ts['ID'].isin(missing_ids)][['ID', 'Time_ms', 'Actual_Route']].to_string())

# ---- Bottleneck: are the SAME queries always slow/fast across systems? ----
print()
print(sep)
print("F. Query-level latency rank correlation across systems")
print(sep)
m_full = merged[merged['PV_ms'].notna()].copy()
from scipy.stats import spearmanr
rho_ss_ts, p_ss_ts = spearmanr(m_full['SS_ms'], m_full['TS_ms'])
rho_pv_ss, p_pv_ss = spearmanr(m_full['PV_ms'], m_full['SS_ms'])
rho_pv_ts, p_pv_ts = spearmanr(m_full['PV_ms'], m_full['TS_ms'])
print("  Spearman rho(SS, TS):", round(rho_ss_ts, 3), "p=", round(p_ss_ts, 4))
print("  Spearman rho(PV, SS):", round(rho_pv_ss, 3), "p=", round(p_pv_ss, 4))
print("  Spearman rho(PV, TS):", round(rho_pv_ts, 3), "p=", round(p_pv_ts, 4))

print()
print("DONE.")
