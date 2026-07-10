# Paper Errata

Track items that need updating in the LaTeX thesis/paper **before journal
submission**. LaTeX source files themselves are **not modified** by this cleanup
(per project constraints).

---

## Data sources

| Location | Current text (approx.) | Correction needed |
|---|---|---|
| Data section | May mention VBPL as a crawl source | **Remove VBPL.** Final corpus = HuggingFace (178,665) + Pháp Điển (70,075) = 248,740 docs. VBPL crawler archived (`archive/legacy_vbpl/`). |

---

## Metrics

| Location | Current number | Correction |
|---|---|---|
| Main F1 table | ~0.4235 "F1" | **Not standard F1.** Was keyword recall. Real Vietnamese token F1 ≈ **0.34** on cleaned test set. See `results/final/official_metrics.json`. |
| Routing accuracy | 96.67% (if cited anywhere) | **REJECT.** Use 5-fold CV: **0.8517 ± 0.0249** from `router_model/training_report.json`. |
| Hit@5 | 4.24% "sub-path normalized" | **Remove.** No backing artifact. Strict Hit@5 ≈ 2.4%; article-only upper bound ≈ 5.5%. |
| Graph F1 | 0.705 ± 0.072 (CV) | **Remove or verify.** Not in `training_report.json`. Held-out graph F1 = 0.591 (N=22). Validation graph F1 = 0.76 (N=25). |
| BERTScore | 0.8508 | Note reference field explicitly. Canonical impl uses `vinai/phobert-base` vs gold `answer`. Prior 0.8508 used `xlm-roberta-large`. |
| Two-stage vs single-stage | "Two-stage best F1" | Reframe: Δ F1 = 0.0004 on strict benchmark (not significant). Two-stage adds robustness on ambiguous/conversational cases. |

---

## Latency

Three incompatible latency snapshots exist in the paper (Snapshot P/C/Q).
Do **not** compare latency across tables from different runs. Use one snapshot
consistently or re-run all systems in a single session.

| Snapshot | Pure Vector | Two-stage | Source |
|---|---:|---:|---|
| P (canonical) | 1,271 ms | 3,913 ms | `docs/final_results_snapshot/legal_strict_full_summary.json` |
| C (unified) | 2,607 ms | 9,608 ms | archived `UNIFIED_PAPER_METRICS.json` |

---

## Ablation table (Table 5.3, RQ4)

Two cells marked `[số liệu thực nghiệm gốc]` — **do not fabricate.** Run the
ablation configs and fill with measured values, or mark as "not evaluated."

---

## GraphRAG architecture (added after graph audit)

| Location | Correction needed |
|---|---|
| GraphRAG architecture section | Qualify any multi-hop claim with "For the Pháp Điển corpus subset...". `AMENDS`/`GUIDES`/`REPEALS` coverage is <0.015%; do not claim multi-hop legal-effect reasoning over the full corpus. See `audit/` + `docs/architecture/graph_known_limitations.md`. |
| GraphRAG architecture section | Add note: "The HuggingFace corpus (149K docs) is represented at document level without article decomposition in the current implementation." |
| GraphRAG architecture section | Add note: "769,000 `REFERENCES` edges carry `type='Unknown'`; they support adjacency exploration only, not semantic cross-document reasoning." |
| Entity extraction section | Add: "Entity extraction was implemented (`ner/vi_ner.py`) but disabled in the production build due to O(N²) `CO_OCCURRED` edge generation and lack of synonym normalization. Documented as future work." |
| System comparison table | Add a Pháp Điển-subset column showing graph advantage on structured content (fill after Task 4 PD-subset benchmark). Reframe two-stage vs single-stage as "ambiguous query robustness", not "overall F1 superiority". |

---

## Test set size

After QA cleaning: strict test set is **541** records (was 600). Update any
text claiming "600-query strict benchmark" to note 59 placeholder records were
removed, or re-run on cleaned set and report n=541.

---

## Status

- [ ] Update data sources section (remove VBPL)
- [ ] Replace F1 numbers with token F1 from official_metrics.json
- [ ] Replace routing accuracy with CV number
- [ ] Remove unbacked Hit@5 and graph CV claims
- [ ] Reframe two-stage contribution
- [ ] Unify latency snapshot
- [ ] Fill or remove ablation table placeholders
- [ ] Update test set count
- [ ] Qualify GraphRAG multi-hop claim to Pháp Điển subset
- [ ] Add HF document-level (no article decomposition) note
- [ ] Add untyped REFERENCES limitation note
- [ ] Add entity-extraction-disabled note
- [ ] Add PD-subset column to system comparison table (after Task 4)
