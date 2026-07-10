# Threats to Validity

Documented limitations reviewers and examiners may raise, with mitigations.

---

## 1. Metric validity (FIXED in this cleanup)

**Threat:** Reported "F1" was keyword substring recall, not F1.
**Evidence:** `evaluation/metrics/legacy.py` line 139 returned `recall` only.
**Mitigation:** `evaluation/metrics/token_f1.py` implements real Vietnamese
token-level F1. All new numbers in `results/final/official_metrics.json`.

---

## 2. Citation / retrieval ID mismatch (PARTIALLY FIXED)

**Threat:** Hit@5 strict ≈ 2–5% because retrieved IDs use Pháp Điển structural
codes (`Điều 19.2.TT.10.15`) while gold uses VBPL document numbers
(`77/2026/NĐ-CP::Điều 18`).
**Evidence:** `evaluation/metrics/id_normalizer.py`; unresolvable rate in
official_metrics.json.
**Mitigation:** Canonical normalizer added; `article_only` mode reports recoverable
upper bound (~5.5% router, ~24% pure graph). Full fix requires indexing pipeline
to emit canonical `law::<doc>::article::<num>` IDs at retrieval time.
**Do not cite:** "Hit@5 4.24% sub-path normalized" — this number has no backing
artifact in the repo.

---

## 3. Small held-out graph evaluation set

**Threat:** Held-out test graph F1 = **0.591** (N=22 only).
**Evidence:** `router_model/training_report.json` → `held_out_test.per_class.graph_traversal`.
**Do not cite:** "graph CV F1 0.705 ± 0.072" — not found in any results file.
Validation-set graph F1 = 0.76 (N=25) is the closest backed per-class number.

---

## 4. QA dataset quality (FIXED)

**Threat:** 84/1,126 records (7.5%) had placeholder answers
(`"...đang cập nhật Nội dung => Bạn"`).
**Mitigation:** Removed by `scripts/regenerate_splits.py`. Test set: 600 → 541.
See `data/audit_reports/qa_quality_audit.md`.

---

## 5. Hand-tuned routing thresholds

**Threat:** ~15 named override reasons / thresholds in `two_stage_router.py`
risk overfitting to eval templates.
**Mitigation:** Documented in `docs/routing_design.md`. Ablation flags allow
disabling each policy component. Sensitivity analysis recommended before
journal submission.

---

## 6. VBPL data source unavailable

**Threat:** VBPL crawler produced empty output; cannot claim VBPL as corpus source.
**Mitigation:** Archived to `archive/legacy_vbpl/`. Final corpus = HuggingFace +
Pháp Điển only (248,740 docs).

---

## 7. Pháp Điển parsing gaps

**Threat:** 92/306 đề mục (30%) parsed to empty article lists despite non-empty HTML.
**Evidence:** `data/phapdien/phapdien_all.json` stats.
**Impact:** Some legal topics may be under-represented in the graph index.

---

## 8. Multiple incompatible result generations (FIXED)

**Threat:** README, UNIFIED_PAPER_METRICS, and snapshot cited different F1/latency.
**Mitigation:** Single canonical file `results/final/official_metrics.json`.
Contradictory files archived to `archive/old_benchmarks/`.

---

## 9. No significance testing on two-stage vs single-stage (PARTIAL)

**Threat:** Cannot test SS vs TS offline — stored predictions lack system labels.
**Mitigation:** `evaluation/significance/bootstrap_test.py` tests available
comparisons. SS vs TS requires a labelled pipeline re-run.

---

## 10. Frontend does not send conversation history

**Threat:** Web demo cannot demonstrate history-aware routing.
**Mitigation:** Use `scripts/demo_conversation_routing.py` for defense demos.

---

## 11. Multi-interpretation ambiguity not covered

**Threat:** The clarify benchmark includes 36 queries of type
`multi_interpretation`. The two-stage system routes all 36 incorrectly (to
retrieval rather than clarification): route accuracy = 0.000, Stage 2 trigger
rate = 0.167.
**Evidence:** `results/final/clarify_two_stage.json → by_ambiguity_type.multi_interpretation`.
**Cause:** Stage 2 trigger conditions rely on *structural* underspecification
signals (missing entity, unresolved pronoun, incomplete context), not *semantic
breadth* signals. A query like "Quyền sử dụng đất được thừa kế như thế nào?" is
syntactically self-contained and does not trigger the verifier.
**Impact:** The overall clarify F1 of 0.870 is buoyed by the three structural
types (each 100% route accuracy). Weighting `multi_interpretation` equally would
lower clarify F1 to approximately 0.72. Results should be interpreted with this
coverage gap in mind.
**Future work:** A dedicated multi-interpretation detector, or always invoking
Stage 2 for queries in high-ambiguity-risk domains (land/family/administrative
law). See `anticipated_examiner_questions.md` Q8.
