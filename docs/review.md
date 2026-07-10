# REVISION NOTES & TODO — Reasoning-aware Adaptive Routing for Hybrid GraphRAG

Target: *Discover Artificial Intelligence* (Springer Nature, journal 44163).
Base version: the `sn-jnl` manuscript (Document 2). The two-column `article`
version (Document 1) is **superseded** — its numbers (F1 0.661 / 0.606,
latency ~5.7 s) are from an old run and must not be reused anywhere.
Authoritative numbers: Two-stage **F1 = 0.636**, Oracle **0.564**, Pure Vector
**0.663** (match `f1_bar.jpg` and Vietnamese thesis Table 5.2, the designated
source of truth).

Every item below is tagged in the `.tex` files as `% TODO(Mx)` or `% NOTE(Mx)`.
Red `??` cells (`\pending`) mark numbers that must come from a re-run. The
`\pending`/`\ci` macros and the `xcolor` line in `main.tex` are review
scaffolding — delete before submission.

---

## A. MAJOR issues (science) — must fix before submission

**M1 — Oracle (0.564) and the full router (0.636) both lose to Pure Vector
(0.663) on aggregate F1.**
A reviewer's first objection: "perfect routing is worse than always-dense, so
the routing taxonomy hurts." Response implemented: candid framing in
`results.tex` (RQ1) and `limitations.tex`, plus a new per-route-class table
`tab:per_class_f1` that must show the router/oracle winning on graph- and
hybrid-class queries even if dense-class queries dominate the mean.
*Action:* group per-query F1 by gold route label for every system; fill
`tab:per_class_f1`. If graph/hybrid classes still lose after M3 is fixed, the
headline claims must be softened further.
→ Files: `sections/results.tex` (RQ1), `sections/limitations.tex`.

**M2 — Three conflicting routing-accuracy figures (0.995 vs 0.8517 vs
0.7529).**
0.995 Macro-F1 on the "paraphrased strict split" is implausible and almost
certainly single-split leakage; 5-fold CV gives **0.8517 ± 2.49% accuracy,
0.840 ± 0.026 Macro-F1** (authoritative); 0.7529 is a dev-set figure in RQ4.
Response implemented: abstract, introduction, RQ2, and conclusion now headline
the CV figure; the single-split table is retained but explicitly labelled
optimistic; new `tab:routing_cv` awaits all baselines re-run under the SAME
stratified 5-fold CV.
*Action:* re-run Majority/Keyword/3-Rule/LogReg/RF/XGBoost under identical CV;
fill `tab:routing_cv`; investigate why the single split leaks (shared
templates across the paraphrase step is the prime suspect) and state the
explanation in one sentence in RQ2.
→ Files: `main.tex` (abstract), `sections/results.tex` (RQ2).

**M3 — Pure Hybrid Hit@1 = 0.003 is the known source-ID format bug.**
Four incompatible ID formats break gold-source matching; this corrupts Hit@k
for Pure Hybrid AND every graph-touching config (Oracle, Two-stage). Response
implemented: honest framing in RQ1 + limitations ("Hit@k is a lower bound").
*Action (highest experimental priority):* normalise all retrieved and gold
source IDs to canonical `law_id::article_id` in one shared function; re-run
ALL end-to-end experiments. Expect the whole results story to shift — possibly
resolving M1.
→ Files: `sections/results.tex`, `sections/limitations.tex`.

**M4 — The chain-of-thought claim is confounded.**
Two-stage (imperfect routing + Stage-2) vs Oracle (perfect routing, NO
Stage-2) cannot isolate the CoT effect. Response implemented: new
**Oracle + Stage 2** system row (`tab:systems`, `tab:end_to_end`) and hedged
wording ("suggestive, not causal") in RQ1/limitations/conclusion.
*Action:* run gold-label routing WITH Stage-2 enabled on all 600 queries; fill
the row; state the delta in RQ1.
→ Files: `sections/experimental_setup.tex`, `sections/results.tex`.

**M5 — No confidence intervals; generator/decoding unspecified.**
Response implemented: bootstrap-CI protocol (B = 1000, 95%) declared in
`experimental_setup.tex`; `\ci{}` helper ready; implementation text now states
generator = Qwen3-32B-AWQ via vLLM, greedy decoding (temp 0, fixed seed) —
**marked TODO(M5): verify this matches the code**; latency decomposition table
`tab:latency_decomp` added.
*Action:* compute per-query bootstrap CIs for F1/EM per system; instrument
timing into Stage-1 / Stage-2 / retrieval / generation; log model + decoding
params into Appendix.
→ Files: `sections/experimental_setup.tex`, `sections/methodology.tex`,
`sections/results.tex` (RQ5), `sections/appendix.tex`.

**M6 — Token-F1 vs gold context biases against graph/hybrid answers.**
Verbatim-copying systems (dense) are rewarded; synthesising systems (graph)
penalised. Response implemented: metric-validity paragraph + BERTScore-F1
column (`zhang2020bertscore` now cited, so the bib entry is legal). Also
TODO(M6) in `methodology.tex`: document HOW the QA pairs were generated
(generator model, prompt, filtering, human check).
*Action:* run BERTScore (Vietnamese encoder) or an LLM-judge for every system;
fill the BERT-F1 column; add 2–3 sentences on dataset generation.
→ Files: `sections/experimental_setup.tex`, `sections/results.tex`,
`sections/methodology.tex`.

**M7 — Closest competitor (Adaptive-RAG) cited but never compared;
benchmark construction under-documented.**
Response implemented: comparability-gap sentence in `related_work.tex`
pointing to `limitations.tex`, which now contains the explicit
non-comparison limitation; TODO(M7) in `methodology.tex` for how the
234-query ambiguity set and 160-query conversation set were built.
*Action (optional but strong):* implement a complexity-router baseline in the
Adaptive-RAG spirit on your corpus; otherwise keep the stated limitation.
→ Files: `sections/related_work.tex`, `sections/limitations.tex`,
`sections/methodology.tex`.

**M8 — φ_amb used but never defined; τ_clar never appeared in any equation.**
Response implemented (formalised, flagged for code verification):
Eq. `ambiguity_score` defines φ_amb as a clipped weighted sum of binary
indicators ψ_i, gated to 0 when r_H = resolved; Eq. `final_route` now formally
applies τ_clar (clarify is emitted only when Stage-2 proposes it with
p_c ≥ τ_clar, else demoted to the retrieval route); Algorithm 1 rewritten to
match; Appendix Table `tab:ambiguity_features` holds the indicator/weight
slots.
*Action:* paste the exact ψ_i and w_i from `ambiguity_detector.py` into the
appendix table; confirm the gating and the demote-on-low-p_c behaviour match
the code.
→ Files: `sections/framework.tex`, `sections/appendix.tex`.

**M9 — Feature count 27 (paper) vs 17 (earlier system); list never
enumerated.**
Response implemented: `framework.tex` uses n_f with TODO(M9); Appendix Table
`tab:features` gives a full reconstructed 27-feature list **in red-flagged
form — every row must be verified against `feature_extractor.py`**, including
the exact regexes for `legal_reference_count` / `article_reference_count`
(these regexes were broken once before; the paper should not silently rely on
them).
→ Files: `sections/framework.tex`, `sections/appendix.tex`.

## B. MODERATE issues (presentation / rigor)

**M10 — "Phase 3" internal jargon** → renamed descriptively as "full policy
(resolver + severe-ambiguity override)" in RQ3.
**M11 — 8.9 ms vs 33.7 ms Stage-1 latency** → reconciled in `framework.tex`:
8.9 ms = XGBoost inference alone; 33.7 ms = full Stage-1 routing incl. feature
extraction + resolver. Keep this wording consistent everywhere.
**M12 — Citation integrity** → `he2024lightrag` author list was fabricated
(now corrected: Guo, Xia, Yu, Ao, Huang — Findings EMNLP 2025); shi2023replug
also wrong (corrected in the commented block). **Verify every remaining author
list and DOI via Crossref**; entries marked `% VERIFY DOI` first. Uncited
entries are commented out (journal permits only cited works).
**M13 — Figures are raster .jpg** → Discover AI prefers vector line art
(EPS/PDF, ≥1200 DPI for line art). Re-export the draw.io architecture/pipeline
figures and matplotlib charts (`plt.savefig('FigN.eps')` or `.pdf`), rename
`Fig1`, `Fig2`, ... per guidelines. Caption style already fixed: captions
start with the figure content and carry **no terminal period** (sn-jnl adds
"Fig. N" automatically).
**M14 — Default τ_clar = 0.80 vs RQ4 optimum 0.95** → reconciled in RQ4 text
(the two benchmarks pull in opposite directions; thresholds must be jointly
calibrated). Keep the default consistent in `framework.tex` and RQ4.
**M15 — Trigger-policy table** (`tab:trigger_policy`) added to link RQ4 to
RQ5; needs the Always-on run.

## C. Journal-compliance checklist (verified against current guidelines)

- [x] Abstract < 250 words (now ~215, numbers included)
- [x] Numbered citations `[1]` via `sn-mathphys-num`
- [x] Reference list = cited works only (uncited entries commented out)
- [ ] All references carry full DOIs — done in file, **verify each**
- [x] Mandatory declarations present: Funding, Competing interests, Ethics,
      Consent, Data availability, **Code availability**, Author contributions
- [ ] Replace "available on reasonable request" with a repository DOI/URL if
      possible (strongly preferred for a methods paper)
- [x] ≤ 3 heading levels
- [ ] Figures as vector EPS/PDF named Fig1, Fig2, ... (M13)
- [ ] Submit LaTeX as ZIP (main.tex + sections/ + biblio.bib + sn-jnl.cls +
      sn-mathphys-num.bst + figures); Springer compiles from source
- [ ] Article type in the submission system: "Research" (or "Methodology")
- [ ] Delete the review scaffolding block (`xcolor`, `\pending`, `\ci`) and
      all `% TODO/% NOTE` comments before submission

## D. Prioritised re-run plan (do in this order)

1. **Fix ID normalisation (M3)** → re-run all 7 end-to-end configs (now
   including Oracle+Stage2 and Always-on). One run produces the data for M1,
   M3, M4, M15 simultaneously.
2. **Per-query logging** in the same run: gold route class, F1/EM per query,
   component timings, retrieved source IDs → fills `tab:per_class_f1`,
   bootstrap CIs (M5), `tab:latency_decomp`.
3. **Semantic metric pass (M6)** over the saved predictions (no re-generation
   needed if predictions are logged).
4. **Router CV re-run (M2)** for all six classifiers under one stratified
   5-fold protocol.
5. **Documentation extraction (M8/M9)**: dump features, ψ/w weights, resolver
   rules, Stage-2 prompt, XGBoost params from the code into the appendix.
6. **Figures to vector (M13)**; final compile; delete scaffolding; ZIP.

## E. Code to paste into the chat so the remaining fixes can be completed

1. `evaluate_prediction` + the end-to-end runner (for per-class F1, bootstrap
   CIs, Oracle+Stage2, Always-on, latency decomposition, semantic metric).
2. Router training script (5-fold CV for all baselines; leakage check for the
   0.995 split; feature-importance export).
3. `feature_extractor.py` (exact feature list + the legal-reference regexes).
4. `ambiguity_detector.py` + HistoryResolver (exact ψ_i, w_i, rule order).
5. Stage-2 prompt template + JSON schema (and where τ_clar is applied).
6. Hybrid retrieval / ID-matching code (to write the canonical
   `law_id::article_id` normaliser).

## F. File map

```
paper/
├── main.tex                     # connector: preamble, authors, abstract, \input list
├── biblio.bib                   # corrected bibliography (DOIs, LightRAG fix)
├── figs/                        # system_architecture, system_pipeline, f1_bar, dataset_bar (.jpg → replace with vector)
└── sections/
    ├── introduction.tex         # motivation, contributions (0.636/0.564, CV 0.840)
    ├── related_work.tex         # + Think-on-Graph/GraphReader cites, Adaptive-RAG gap
    ├── methodology.tex          # problem def + notation table, dataset (TODO M6/M7), overview, implementation (TODO M5)
    ├── framework.tex            # KB stats, features (M9), ξ & φ_amb defined (M8), Stage 1/2, τ_clar in final route, Algorithm 1
    ├── experimental_setup.tex   # RQs, systems (+Oracle+Stage2), metrics (+BERTScore, bootstrap CI)
    ├── results.tex              # RQ1–RQ5 + discussion; \pending cells for re-run data
    ├── limitations.tex          # 10 candid limitations incl. M1/M4/M5/M6/M7
    ├── conclusion.tex           # synced numbers
    ├── declarations.tex         # all mandatory statements + code availability
    └── appendix.tex             # app:repro — features, ψ/w, resolver rules, prompt, hyperparams
```

Compile: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
(with `sn-jnl.cls` and `sn-mathphys-num.bst` in the project root).