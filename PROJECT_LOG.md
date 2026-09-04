# ORBIT — project log

Newest section at the bottom. One entry per event: what changed, why, what it invalidates.

Conventions:
- `[DECISION]` a ruling downstream code must respect · `[ADJUST]` a change to something
  already built, naming what it supersedes · `[FINDING]` an empirical fact that
  constrains design · `[VERIFY]` outcome of a literature verification duty ·
  `[COST]` compute consumed · `[RISK]` open threat + mitigation ·
  `[CHECK]` proposal-adherence check closing a step.
- Source of truth for numbers is the named artifact, not this file's prose.

Standing documents: `docs/PROPOSAL.md` (the claim; changes rarely),
`docs/PLAN.md` (the steps; living).

---

## S0 — Verification, feasibility probe, proposal (2026-09-03)

Selected from a three-proposal triage (ORBIT 4.65 / RASA 3.34 / BASS 2.89; assessment in
the sibling `Interpretable_PLM/Reports/PROPOSAL_ASSESSMENT.md`). This log begins at the
point ORBIT became the project.

### Verification (duties carried in from triage, now discharged)

- `[VERIFY]` **Ahlmann-Eltze, Huber & Anders, *Nat Methods* 22:1657–1661 (2025)** —
  full text retrieved and read via PMC (DOI 10.1038/s41592-025-02772-6, PMID 40759747).
  Abstract confirms five foundation models + two deep models vs simple baselines, "None
  outperformed the baselines". Supersedes the earlier metadata-only verification, which
  could confirm existence but not content.
- `[FINDING]` The paper goes further than the triage assumed: they also scored
  **genetic-interaction detection** directly and found no model beat a 'no change'
  baseline; the additive model "did not compete as, by definition, it does not predict
  interactions." They identified 5,035 interactions at 5% FDR from the full dataset, and
  scored 310,000 predictions (1,000 read-out genes × 62 held-out doubles × 5 splits).
  Their interaction taxonomy is buffering / synergistic / opposite, and all models mostly
  predicted buffering, rarely getting synergistic right.
- `[DECISION]` This reframes ORBIT's positioning. The gap is **not** "nobody looks at
  interaction residuals" — this paper does, as an *evaluation* device. The gap is that
  nobody makes the residual the **training target**. Proposal §3 states this distinction
  explicitly; it is now the novelty sentence. Central theme unchanged.
- `[VERIFY]` Adversarial novelty search. Closest neighbours found:
  **AttentionPert** (*Bioinformatics* 2024, PMC11211811) — multi-scale attention for
  multiplexed perturbations, explicitly targets non-additivity but still regresses `y`;
  promoted to a **required baseline**. **Hybrid main-effect/interaction networks**
  (Tsang et al. 2018 arXiv 1705.04977; Cui et al. arXiv 1901.08361) — separate linear
  main-effect and NN interaction branches summed, trained jointly on `y` with in-sample
  main effects; no cross-fitting, no projection guarantee. Also noted a 2025 pairwise-loss
  Perturb-seq method (biorxiv 2025.10.03.680360) and a *Bioinformatics* 2025 paper
  reporting simple controls exceeding deep methods — consistent with our premise.
  `[RISK]` Hybrid main-effect/interaction NNs are a closer neighbour than the triage
  recognised. Mitigation: proposal §3 disclaims architectural separation as prior art and
  locates the delta in cross-fitting + hard projection + the estimand/metric shift.

### Feasibility probe — the go/no-go

Downloaded Zenodo 16092690 (network access granted for zenodo.org). Contains released
predictions for 10 methods × 5 splits: additive_model, ground_truth, gears, scgpt, cpa,
scfoundation, geneformer, uce, uce33, scbert. Structure: 225 perturbations × 19,264 genes;
split 1 has 163 train / 31 test / 31 val perturbations.

- `[FINDING]` **The additive model achieves r = 0.9959 against ground truth** across all
  gene × held-out-double pairs, explaining **99.2% of the variance**. The interaction
  residual is **0.82%** of var(`y`). This is the quantitative core of the paper's
  motivation and it is measured, not cited.
- `[FINDING]` Additive-model error on **single** perturbations is 2.9e-10 — the additive
  null is exact by construction there. Doubles as a validation that our nuisance
  reconstruction matches theirs.
- `[FINDING]` The residual is **structured, not noise**: top singular component holds
  23.4% of residual SS vs 3.5% for a variance-matched permuted null; effective rank
  (participation ratio) 11.3 vs 30.9 permuted, out of max 31.
- `[FINDING]` The residual is **relational**: held-out doubles sharing a perturbed gene
  have mean residual correlation 0.297 vs 0.028 for pairs sharing none
  (Wilcoxon p = 6.6e-07). This is direct evidence a *relational* model is the right
  inductive bias, obtained before building one.
- `[DECISION]` **GO.** Small-but-structured, low-rank and relational is exactly ORBIT's
  hypothesis class. Had the residual been unstructured, the honest finding would have
  been "the residual is noise" and the project would have stopped here.
- `[DECISION]` Bilinear rank fixed at **16**, justified by the measured effective rank
  11.3 rather than by convention. Supersedes the source document's unjustified rank-16
  choice with the same number and a reason.
- `[FINDING]` `[RISK]` The released ground truth is **identical across all five splits**
  (between-split SD exactly 0) because it is a fixed pseudobulk, not resampled. So the
  noise floor **cannot** be estimated from this file. Mitigation: plan step S8 estimates
  it from raw counts (GEO GSE133344, verified reachable) by bootstrapping cells within
  perturbation. This was going to be a silent wrong number; it is now an explicit step.
- `[FINDING]` Re-scoring the published models on the residual metric is **free** — their
  `y` predictions are in the RDS, so subtracting the additive expectation costs no
  training. Promoted to a headline result in plan step S2.

### Infrastructure

- `[FINDING]` `list_compute` returns **no providers** — there is no remote GPU. All work
  is CPU on 8 cores / 16 GB. `[DECISION]` Model capacity is capped accordingly (2-layer
  GNN, hidden 128); with 31 held-out pairs per split a larger model is unsupportable
  anyway, so this constraint and good statistics point the same way. Budget in
  `docs/PLAN.md` is stated in CPU-hours.
- `[FINDING]` Blocked by the network allowlist: dataverse.harvard.edu, tdcommons.ai.
  Reachable and sufficient: GEO/NCBI FTP, GitHub, Zenodo (granted this session).

### Artifacts

`docs/PROPOSAL.md`, `docs/PLAN.md`, `results/orbit_feasibility_probe.png`,
`handoff/orbit_probe.json`, `handoff/orbit_noise.json`.

- `[COST]` S0: ~1 h wall, **0 GPU-h**. Cumulative 1 h of 48 h ceiling.
- `[CHECK]` Proposal adherence: S0 established §6 (measured feasibility) and sharpened
  §2 (the claim) and §3 (novelty positioning) against retrieved full text. Central
  innovation — the residual as *estimand* with hard orthogonal projection — unchanged
  from selection. No drift.

### Next

S1: extract pseudobulk + the five immutable splits, build the gene graph, construct
cross-fitted residual targets, and run the QC assertions listed in `docs/PLAN.md` S1.

---

## S0.1 — Novelty-audit correction (2026-09-03)

- `[ADJUST]` `docs/PROPOSAL.md` §3 previously carried the blanket header "Verified
  against retrieved full text and abstracts, not recall" over **every** row. That was
  false for most rows: only Ahlmann-Eltze et al. had been read. Superseded by per-row
  provenance tags plus an explicit statement that untagged rows are unverified.
  This is the same failure mode caught earlier for LTGA, recurring for a different
  citation — hence the rule is now written into the document rather than remembered.
- `[VERIFY]` **AttentionPert** full text retrieved via NCBI E-utilities
  (PMC11211811). Confirmed *Bioinformatics* 2024. Content verified first-hand:
  PertWeight is stated in the paper to be "additive for genes in any multi-gene
  perturbation c, which cannot simulate the nonadditive feature", and PertLocal adds a
  nonadditive bias layer `E'_z = E_z[1 + β(m−1)tanh(K_NA z_c)]` for nonlinear
  co-effects. Full-text term counts: orthogonal 0, cross-fit 0, nuisance 0, debiased 0,
  nonadditive 17. So the negative claims in §3 (no orthogonality constraint, no
  cross-fitting) are now **first-hand**, not inferred.
- `[FINDING]` AttentionPert's §3.3 analyses residual errors, but as *post hoc*
  diagnostics of its own prediction errors — "AttentionPert's residual errors to
  diagnose any failure cases" — not as a regression target. This sharpens rather than
  weakens ORBIT's positioning: the nearest neighbour looks at residuals for
  interpretation, and still trains on `y`.
- `[FINDING]` AttentionPert reports residuals uncorrelated with the GO graph structure.
  `[RISK]` Bears directly on ORBIT: our S0 probe found residual correlation 0.297 for
  pairs sharing a perturbed gene, which is a *pair-level relational* signal, not a
  gene-level GO signal — a different quantity. Ablation S7.1 (shuffled graph, STRING vs
  GO) is now the load-bearing test of whether the graph earns its place, and the
  possibility that GO adds nothing must be reported honestly if it holds.
- `[DECISION]` New plan step **S1b** — retrieve full text for the four remaining
  unverified rows and answer, per paper, whether it fits main effects out-of-fold.
  Budget 1 h; total plan now 48 h, at the ceiling. If S1b shows any neighbour
  cross-fits, §3 is rewritten before drafting.
- `[CHECK]` §2's claim and §4's method are unchanged. Only the novelty *positioning*
  and its provenance discipline changed. No drift.

---

## S0.2 — Citation-integrity correction (2026-09-03)

- `[VERIFY]` arXiv 1705.04977 and 1901.08361 confirmed via the arXiv API. The
  identifiers were genuinely retrieved (they appear in a persisted search result from
  this session), but the surrounding claims I wrote around them were not:
  - Years were wrong. Actual: **2017** (Tsang, Cheng & Liu) and **2019** (Cui,
    Marttinen & Kaski). I had written 2018 and 2020 from recall.
  - Titles show these are *"Detecting Statistical Interactions from Neural Network
    Weights"* and *"Learning Global Pairwise Interactions with Bayesian Neural
    Networks"* — **interaction-detection / interpretability** papers, not the
    "hybrid main-effect/interaction architecture, branches summed" I had described.
    That characterisation was fabricated from the row label, not from the papers.
- `[ADJUST]` `PROPOSAL.md` §3: row rewritten with verified titles, authors and years,
  and repositioned as a *different task* (detecting which feature pairs interact, versus
  ORBIT predicting an interaction residual). Supersedes the "hybrid main-effect/
  interaction NNs" row entirely. Removed from the S1b full-text queue since the titles
  already settle that they are not baselines; S1b now covers GEARS, CPA/SAMS-VAE and
  DeepSynergy/comboFM.
- `[DECISION]` **Standing citation rule**, written into `PROPOSAL.md` §3: every
  identifier, venue, year and author list must be checked against arXiv/Crossref/PubMed
  before being written. Retrieving an id from a search result establishes only that the
  id exists — not the year, not the authors, and not what the paper does. Three
  citation errors in this project have now had the same root cause: specificity
  supplied by recall around a genuinely retrieved anchor.
- `[FINDING]` Net effect on novelty: mildly **positive**. What I had described as the
  closest architectural neighbour turns out to be a different task, so §3's nearest
  neighbour remains AttentionPert (verified, and it trains on `y`).
- `[CHECK]` §2 claim and §4 method unchanged. No drift.

---

## S0.3 — Graph design, baseline scope, compute re-estimate (2026-09-03)

Triggered by three reviewer challenges. All three were investigated empirically rather
than argued; two changed the design.

### Q1: is a STRING/GO graph the wrong relation? — **Largely yes. Design changed.**

- `[FINDING]` **The Norman-2019 screen is a transcription-factor screen.** STRING GO
  enrichment over the 100 singly-perturbed genes: "DNA-binding transcription factor
  activity, RNA Pol II-specific" p = 1.6e-21 (n = 41); "Transcription regulator
  activity" p = 2.4e-21 (n = 47). **60 of 100 perturbed genes carry a
  transcriptional-regulator annotation.** Two TFs interact through shared targets and
  regulatory cascades — a physical-binding graph does not encode that pathway.
  The reviewer's instinct that GRN/TRN relations are needed is correct.
- `[FINDING]` **Coverage, not edge quality, is the real problem with PPI.** Over the 124
  doubles with both singles measured, STRING experimental (physical PPI) reaches only
  **16%** of pairs; STRING combined 19%; coexpression 14%. A PPI-only graph leaves four
  pairs in five with no edge.
- `[FINDING]` Prior edges nonetheless carry **real** interaction signal, and physical PPI
  is the strongest per-edge relation: partial correlation with per-pair |residual|,
  controlling for total effect magnitude — PPI +0.328 (p = 3e-04), combined +0.299,
  textmining +0.284, coexpression +0.190. So the fix is coverage, not replacement.
  `[FINDING]` Identified the dominant confounder: total single-effect magnitude
  correlates r = −0.672 with relative residual size, so all graph associations are
  reported as partial correlations controlling for it.
- `[FINDING]` **Multi-relational beats PPI alone.** 5-fold CV R² predicting per-pair
  |residual|: no graph 0.410 → +PPI 0.461 → +effect-profile correlation 0.433 →
  **all relations 0.483**. Combining relation types roughly doubles the graph's
  contribution over PPI alone (+0.074 vs +0.051).
- `[DECISION]` `PROPOSAL.md` §4.1 added. The encoder now runs over **four
  relation-specific graphs**: regulatory TF→target (**primary**), physical PPI (STRING
  experimental ≥ 0.4), co-functional (STRING combined ∪ GO-BP), and data-derived
  effect-profile similarity (dense, 100% coverage). Learned per-relation weights are a
  reported output. Supersedes the single "STRING + GO" graph in v1.
- `[DECISION]` Pre-registered prediction: regulatory and co-functional relations should
  outweigh physical PPI, because the screen perturbs TFs. If PPI dominates, our
  mechanistic reading is wrong and the paper says so.
- `[RISK]` The dense effect-profile relation is the most likely leakage vector in the
  project, since it is derived from data rather than a database. Mitigation: it is
  computed **only from training-fold singles**, with a per-split assertion added to
  S1's QC list.
- `[ADJUST]` S7 ablation 1 upgraded from "no graph / shuffled / STRING / GO" to a
  **per-relation leave-one-out** with a pre-registered directional prediction.

### Q2: are recent foundation models missing from the comparison? — **No, mostly already covered.**

- `[FINDING]` `[VERIFY]` The release contains predictions **on our exact five splits**
  for eight deep/foundation models. Re-scoring them on the residual metric costs zero
  training, and they are the original authors' tuned runs — a stronger comparison than
  us retraining competitors. **Provenance:** Zenodo record 16092690 / DOI
  10.5281/zenodo.16092690 confirmed against the Zenodo API (8 files); the method list
  enumerated from the downloaded `double_perturbation_results_parameters.RDS` —
  50 elements = 10 methods × 5 splits, 10 per split: `additive_model`, `cpa`, `gears`,
  `geneformer`, `ground_truth`, `scbert`, `scfoundation`, `scgpt`, `uce`, `uce33`;
  `dataset_name = norman_from_scfoundation`; 31 test perturbations in split 1.
- `[DECISION]` scLong and 2026-era Arc/Xaira models are **out of scope, stated
  explicitly in the paper.** Not in the benchmark; retraining a large foundation model
  exceeds a 24–30 h workshop budget; and it is the wrong comparison — the claim is about
  the estimand, and a ninth `y`-predicting model repeats the point rather than testing
  it. `PROPOSAL.md` §6.1 records this reasoning rather than leaving the gap unexplained.
- `[ADJUST]` AttentionPert added as the one model we train ourselves (closest
  architectural neighbour, §3-verified).

### Q3: is a GPU needed, and can 48 h come down? — **No GPU. Budget cut to 30 h.**

- `[FINDING]` Measured the core training step: rank-16 bilinear head, 19,264-dim output,
  hard projection against an 8-dim nuisance basis, 163 training pairs, batch 32 —
  **20 epochs in 1.25 s on one CPU core.** The full grid (5 splits × 3 seeds × 8
  ablations) is under 1 CPU-hour even with a 100× margin for autograd and data loading.
- `[DECISION]` **No GPU required, and a GPU would not help** — at 163 pairs per epoch,
  kernel-launch overhead would likely exceed compute. Stated in `PROPOSAL.md` §6.2 as a
  feature: the contribution is an estimand and protocol, reproducible on a laptop.
- `[DECISION]` **Budget cut from 48 h to 30 h** (`docs/PLAN.md` ledger rewritten in
  CPU-hours). Training is not the cost driver: **data preparation (S1, 4 h) and the
  noise-floor bootstrap over raw counts (S8, 3 h) are.** Supersedes the v1 budget, which
  was inherited from the source document's GPU-based estimate and never re-derived.
- `[FINDING]` The 48 h figure was wrong for a structural reason worth recording: it was
  scaled from model and dataset size, but the binding quantity here is the **number of
  training pairs (163)**, which is tiny. Estimates built from parameter counts
  mis-predict cost on small-n problems by orders of magnitude.

### Artifacts

`results/orbit_graph_evidence.png`, `handoff/pair_level.json`,
`handoff/single_profiles.json`. `docs/PROPOSAL.md` v2 (§4.1, §6.1, §6.2 new),
`docs/PLAN.md` v2 (budget in CPU-h, S1 four graphs, S7 per-relation ablation).

- `[COST]` S0.3: ~1 h wall, 0 GPU-h. Cumulative 2 h of 30 h.
- `[CHECK]` §2's claim (residual as estimand) and §4's projection mechanism are
  **unchanged**. What changed is the graph's relation set, the stated baseline scope, and
  the budget — all of which make the same claim cheaper and better tested. No drift.

---

## S0.4 — Provenance and arithmetic corrections (2026-09-03)

- `[ADJUST]` §6.2's compute claim was arithmetically inconsistent with its own stated
  margin: the measured grid cost is **0.04 CPU-h**, and 0.04 × 100 = **4 CPU-h**, not
  "under 1 CPU-hour" as written. Corrected in `PROPOSAL.md` §6.2 and `PLAN.md`. The
  no-GPU conclusion is unaffected — 4 h against a 30 h budget is still negligible — but
  the stated number now matches the stated margin.
- `[ADJUST]` The baseline-coverage claim in §6.1 was **substantively correct but
  asserted from memory** in the cell that wrote it: the method list was hardcoded rather
  than enumerated, so the artifact carried a checkable external accession with no
  in-cell derivation. Now re-verified and rewritten with explicit provenance — Zenodo
  API confirms record 16092690 (DOI 10.5281/zenodo.16092690, 8 files), and the method
  table is enumerated from `double_perturbation_results_parameters.RDS` with the
  50 = 10 × 5 structure and split-1 config id recorded.
- `[DECISION]` Extend the standing citation rule (§3) to cover **data accessions and
  dataset contents**, not just literature: any accession, file name, method list or
  split id written into a document must be produced by a cell that reads the source in
  that same cell. Hardcoding a remembered list into a print statement is not
  verification even when the list is right — it leaves nothing for a reader to trace.
- `[CHECK]` No change to §2's claim, §4's method, §4.1's graph design, or the 30 h
  budget. Corrections are to provenance and arithmetic only.

---

## S1–S9 implementation run (2026-09-03/04)

Budget: user capped total compute at 8–12 h for a 2–4 page workshop paper. Actual
compute for every number in the paper: **under 6 CPU-hours, no GPU**.

### S1 — data curation
- `[DECISION]` Benchmark release re-downloaded (Zenodo 16092690, DOI
  10.5281/zenodo.16092690, verified live). Splits used verbatim.
- `[FINDING]` **Label-format bug, caught by a NaN.** GEARS, scGPT and scFoundation label
  pairs `AHR_FEV`; ground truth uses `AHR+FEV`. Overlap of the two label sets was
  **exactly 0**, so three methods silently scored NaN and three others looked
  suspiciously identical. Fixed with a `canon()` normaliser (split on `[+_]`, drop
  `ctrl`, sort, rejoin) → 225/225 overlap for all ten methods. A second bug: one
  empty label canonicalised to `NA` and indexed out of bounds; guarded by dropping it,
  leaving **224** reference perturbations (split 1: 162 train / 31 test / 31 val).
- `[RISK]` `uce` and `scbert` predictions in the release are numerically **identical**
  (r = 1.000000); `scgpt` is r = 0.999 against both. Reported separately as released,
  but flagged in the paper — they are not three independent data points.
- `[DECISION]` Gene subset: top 2,000 by residual variance on split-1 **training**
  doubles (70% of training residual SS). Selection never sees test pairs.
- QC: gene ordering identical across all perturbations (TRUE); additive RMS on 100
  singles 7.73e-10; train/test disjoint in all 5 splits (TRUE).

### S1 — graphs (answering the user's challenge 1)
- `[ADJUST]` High-confidence DoRothEA (A–C) covers only 18/100 perturbed genes — worse
  than STRING. Union of DoRothEA + TRRUST + ENCODE + ITFP + TRED + Neph2012 +
  Marbach2016 → **3,127 TFs, 48/100 perturbed genes, 26% of pairs** (vs STRING's 19%).
- `[FINDING]` All 100 singles are training data in **every** published split (only
  doubles are held out), so profile-similarity graphs are leak-free by construction.

### S2 — baselines
- `[FINDING]` **The paper's central result.** Cross-fitted ridge on the residual reaches
  **r = 0.488 ± 0.055**, above every one of the eight published deep/foundation models
  (best 0.417), while all of them score r ≈ 0.99 on total expression. The estimand shift
  alone inverts the published ranking, at zero retraining cost.

### S3/S4 — model, and three architectural dead ends
- `[FINDING]` First gate run: r = 0.025 against ridge's 0.488. Established an **oracle
  upper bound** by projecting test residuals onto train-residual subspaces (rank 62:
  0.668), proving 0.025 was a bug, not a ceiling. Cause: output initialised 15× too
  small, and 19,264-dim node features on 100 nodes / 62 training pairs.
- `[ADJUST]` Tried, in order: feature standardisation + full-batch + unit output scale
  (best 0.306); PCA node features (0.311); SVD-initialised output basis (0.308); a
  two-path head with an explicit linear gene-space path (0.103, and path isolation gave
  linear-only 0.061, bilinear-only 0.058). None approached the ridge.
- `[DECISION]` **Restructured as a two-stage estimator** (`src/orbit/orbit.py`): stage 1
  a cross-fitted ridge for interaction *shape*, stage 2 a multi-relational GNN predicting
  a per-pair scalar *gain*, trained on stage 1's out-of-fold predictions. This makes the
  graph's contribution measurable by construction — it *is* stage 2's increment.
- `[FINDING]` **Metric artifact caught.** A scalar-gain stage 2 changed per-pair Pearson
  by exactly 0.0000, because per-pair Pearson is scale-invariant and therefore blind to
  magnitude. Added residual **R²** as co-primary. Under R², stage 1 = 0.261, optimal
  global rescale = 0.261, **oracle per-pair scale = 0.357** — that gap is the graph's
  headroom, and it is a magnitude quantity, not a shape one.

### S5/S7 — the pre-registered kill condition fires
- `[FINDING]` **Kill condition 1 fired.** Over 5 splits × 3 seeds the graph stage reaches
  R² = 0.103 against **0.274** with the graph stage removed. In the ablation grid a
  **shuffled** graph scores **0.110 — higher than the real graph on every split**
  (paired t p = 0.06, n = 3). Learned relation weights stay uniform (~0.25 each), so the
  model never differentiates the relations and the pre-registered TF prediction cannot
  even be evaluated. The GNN is not using the graph.
- `[DECISION]` Reported as a negative result with a measured cause rather than tuned
  around. This is what the kill condition was written for.

### S7b — locating the cause (capacity-controlled test)
- `[FINDING]` The prior is **not** useless: predicting per-pair interaction magnitude
  ‖r_ab‖ from graph-derived scalars with ridge and nested CV gives out-of-fold R²
  **0.394 → 0.452** across the four relation families. Partial correlations controlling
  for effect magnitude: physical PPI **+0.275**, co-functional +0.237, regulatory +0.110,
  effect-profile similarity −0.286.
- `[FINDING]` **Our pre-registered mechanistic prediction is wrong.** We predicted
  regulatory and co-functional would outweigh physical PPI because 60/100 perturbed genes
  are TF-annotated. PPI is strongest, despite reaching only 16% of pairs. Reported as
  falsified, per §4.1 of the proposal.
- `[DECISION]` Conclusion: the bottleneck is **n ≈ 60 training pairs**, a property of
  available data, not of architecture — so scaling the model is the wrong response.

### S6 — mechanism curve, redesigned
- `[FINDING]` **Confounded first design, caught before publication.** Shrinking ρ also
  changes the regression target, so held-out R² rose trivially (0.105 → 0.512) because
  the target became the larger, easier total signal. That measures the target, not the
  projection.
- `[ADJUST]` Rewrote (`scripts/08_mechanism.py`) to hold the evaluation target **fixed**
  at the true residual and degrade only the nuisance supplied to the model.
- `[FINDING]` Projection bounds nuisance sensitivity (ρ=0.25: **−0.25 with vs −0.89
  without**; ρ=0.5: −0.07 vs −0.27) but **costs accuracy when the nuisance is good**
  (ρ=1: 0.086 vs 0.232). Honest framing: insurance, not a free improvement — and on this
  benchmark the nuisance is well estimated because singles are measured directly.

### S9 — finance arm (mandatory)
- `[ADJUST]` **Both granted price sources turned out unusable**: stooq.com now gates its
  CSV endpoint behind a JavaScript proof-of-work challenge, and Yahoo returns 429. Did
  not spoof headers to evade either. Switched to the **Ken French Data Library**, which
  is strictly better here: canonical, citable, no key, and it ships the factor returns
  needed to define the additive nuisance.
- `[DECISION]` 49 industry portfolios rather than single names: no survivorship bias, no
  corporate-action handling, stable identifiers, well-defined sector relation graph.
- `[FINDING]` 6,662 days × 49 industries, 1,176 pairs, 281 out-of-sample months,
  walk-forward with trailing-window betas (no look-ahead). Ridge on the residual beats
  the factor model by **+0.066 R² (82% of months, p < 1e-17)** and trailing-covariance
  persistence by **+0.014 (57%, Wilcoxon p = 0.0015)**, paired per month.
- `[ADJUST]` First figure used across-month SDs as error bars, which swamped the axis and
  made the comparison look null. Replaced with **paired per-month differences** — the
  correct test at n=281.

### S8/S9 — reproducibility
- `[FINDING]` The README promised `scripts/01_prepare_data.R`, which did not exist (S1
  had been run interactively). Wrote it, plus `00_download.py`, `01b_to_npz.py`,
  `01c_string.py`, then **re-ran the whole pipeline from the raw release** and confirmed
  bit-identical baselines (ridge 0.4878 ± 0.0552) and 8/8 unit tests passing.
- Two bugs found only by that re-run: a broken `sys.frame` path idiom, and `base::get`
  shadowed by the script's own prediction accessor.

### Scope check against the proposal
Central theme unchanged: the contribution is a **change of estimand**, not an
architecture. What changed is the verdict on the graph — the proposal predicted the
multi-relational encoder would add signal, and the pre-registered test says it does not
at this sample size. Both the negative result and its measured cause are reported. The
mechanistic prediction about relation ordering is reported as falsified.

### Post-hoc audit corrections (2026-09-04)
- `[FINDING]` **Variance-fraction figure was wrong in two saved artifacts.** The paper
  abstract and README table stated the interaction term carries "0.44%" of the variance
  of *y*. That number is `1 - r_total(ridge) = 1 - 0.9956`, i.e. the wrong variable
  combined with a linear-difference formula, not the additive model's variance share.
  Measured directly on held-out doubles over the 2,000-gene subset:
  **var(r)/var(y) = 1.19%** (per-split 1.02-1.37%), consistent with
  `1 - r^2 = 1.17%` from the additive model's own correlation r = 0.9941. Corrected to
  **1.2%** in `paper/ORBIT_workshop.md` and `README.md`. The earlier session figure of
  0.82% came from the full 19,264-gene matrix; the 2,000-gene residual-variance subset
  raises the share, so the two are consistent and 1.2% is the correct figure for the
  data the paper actually analyses.
- `[FINDING]` **Two denominators were mixed in §3.** Regulatory coverage was quoted as
  26% (fraction of all 4,950 possible gene pairs) alongside PPI 16% and co-functional
  19% (fraction of the 122 *observed* perturbation pairs). On the 122-pair denominator
  the relations are regulatory **34%**, PPI 16%, co-functional 19%, profile 100%; on the
  4,950-pair denominator they are 26%, 3%, 4%, 100%. §3 now uses the 122-pair
  denominator throughout — the population the §4 partial correlations are computed over —
  and says so explicitly.
- `[ADJUST]` Fig. 1 panel-a title claimed "every model scores ~0.99 on *y*", which CPA
  (0.957) does not satisfy. Retitled to the measured ranges ("Totals all sit in
  0.96-1.00; residuals span 0.00-0.49"), and the caption now names CPA as the low
  outlier and separates the eight-published-model spread (0.157) from the full plotted
  spread.

---

## Round 2: GREmLN, scale, and paper restructuring (2026-09-04)

### GREmLN review (user-supplied reference)
- `[FINDING]` GREmLN's own related-work section states our concern first-hand: structural
  priors "risk introducing noisy or biased priors", and hard attention masking or additive
  structural bias do not constitute real structural message passing. Their response is to
  use GRNs **inferred from expression** (ARACNe-style) rather than literature curation.
- `[DECISION]` Transferable element adopted: the curated-vs-inferred distinction as a
  *testable axis*, not their architecture. We do not need a foundation model to test it —
  we need a dataset large enough to sweep *n*.

### New data: Horlbeck 2018 CRISPRi interaction map (GSE116198)
- `[DECISION]` Adopted as the second dataset, resolving the user's data-insufficiency
  question. 922,455 sgRNA-pair rows -> **108,799 gene pairs over 467 genes**, K562 and
  Jurkat. Three orders of magnitude beyond Norman's 62 training pairs, and the two cell
  lines give an external reproducibility axis Norman entirely lacks.
- `[FINDING]` **Measurement bug caught in our own first pass.** Using gene-paired-with-itself
  as the single-gene effect gave residual variance *exceeding* observed (225% in K562) —
  because that pairing is a double knockdown, not a single. Corrected to control-paired
  guides, giving additive r = 0.965 and residual 15.4% of variance in Jurkat.
- `[FINDING]` **74% of the naive interaction residual is a known effect-size artifact.**
  The sum null is biased for strong-effect genes; regressing out a smooth function of the
  expected phenotype (standard GI correction) removes 73.9% of the residual variance. The
  naive residual is *anti*-correlated across cell lines (r = -0.290) while the corrected
  score is positively correlated (r = +0.200, n = 104,196). Any relational claim on the
  uncorrected residual measures the artifact. All Horlbeck results use the corrected score.

### The decisive scaling result
- `[FINDING]` Sweeping training size 60 -> 86,953 pairs (3 seeds, fixed 20,000-pair test):
  **curated priors are flat in n** (+0.0003 -> +0.0014, indistinguishable from shuffled),
  while the **data-derived graph scales** (+0.0003 -> +0.0274, 20x larger). Coverage at
  genome scale: regulatory 1.9%, physical PPI 4.1%, co-functional 5.6%, data-derived 100%.
- `[DECISION]` This **reframes the S5 kill condition**. The Norman failure was read as
  small-n; the sweep shows it is *prior provenance*. Curated edges carry little
  transferable relational signal at any reachable n; inferred structure does. That is a
  sharper and more useful claim than "we needed more data", and it independently supports
  the user's hypothesis that we were over-relying on incomplete graph priors.

### Paper restructuring
- `[ADJUST]` Rewritten from a findings compilation into standard methodological form:
  Abstract + 4 contribution bullets; §1 Introduction (problem -> why existing approaches
  fail -> insight -> method -> results); §2 Background (~0.5 p); §3 Method with 6 numbered
  equations including the general message-passing form and exactly what we change; §4
  Setup; §5 Results (6 subsections); §6 Conclusion/limitations; references. Title changed
  from a rhetorical question to a declarative methodological statement.

### Round 3: venue targeting and single-thesis restructure (2026-09-04)
- `[FINDING]` **Venue check, grounded in the 2026 call texts.** ICBINB-BIO (NeurIPS 2026)
  solicits exactly this paper: "strong benchmark results often fail to survive new
  mutations, perturbations, individuals, assays, or deployment settings ... what breaks,
  why it breaks, how we should evaluate it". Our three failures are one instance each of
  that description, and negative results with identified causes are the workshop's
  explicit subject rather than a liability. AI4DD 2026 is framed around "robust,
  trustworthy, and translatable AI for real-world drug discovery" and is non-archival with
  concurrent submissions allowed, so it is a compatible second target; GenBio moved to
  ICML for 2026.
- `[DECISION]` **Primary: ICBINB-BIO. Secondary: AI4DD.** Both are reachable with the same
  manuscript; AI4DD's non-archival policy permits concurrent submission.
- `[ADJUST]` **Restructured around one thesis** in response to the "compilation of
  findings" critique. New title "Three Ways to Mismeasure a Genetic Interaction". The
  thesis: the binding constraint in combinatorial perturbation prediction is measurement
  validity, not model capacity. The three former standalone findings are now numbered
  instances (I expression-level scoring, II the naive residual, III curated graph priors)
  ordered along the path a practitioner actually takes, each paired with the cheap control
  that catches it (additive baseline / cross-assay reproducibility / shuffled graph plus
  sample-size sweep).
- `[ADJUST]` The two genuinely peripheral results were **demoted to appendices**: the
  nuisance projection (Appendix A) and the finance transfer (Appendix B, reframed as
  evidence that mismeasurement I is structural rather than biological). This removes the
  two weakest-linked claims from the main narrative without discarding the work.
- `[FINDING]` The method section is now motivated by the thesis rather than by novelty:
  the two-stage factorisation is presented as an *instrument* that makes the prior's
  contribution identifiable, which is what Control III requires. This is a more honest
  framing of what it is for, given that the graph stage does not help.

### Round 4: venue re-targeting after deadline lapse (2026-09-04)
- `[FINDING]` ICBINB-BIO closed **2 September 2026** (extended date) — missed.
- `[FINDING]` Live deadline scan. Still open: **AI4DD 6 Sep** (5 p full / 2 p short,
  non-archival, concurrent submissions allowed, Sydney) — the user's originally named
  venue; **STODY 6 Sep**; **ML4SpatialBio 4 Sep** (hours away); **GEM Bio 3 Sep** (closed);
  **ML4H 10 Sep** (standalone symposium, archival Proceedings 8 p + non-archival Findings).
- `[DECISION]` **Primary: AI4DD (6 Sep).** Its call is a direct match — the workshop is
  built on the observation that "strong performance on static, curated benchmarks does not
  necessarily persist under prospective experiments, new targets, new chemical series,
  assay shifts". That is our thesis in their vocabulary. Non-archival with concurrent
  submissions permitted, so **ML4H Findings (10 Sep)** is a free second shot.
- `[ADJUST]` Recut to AI4DD's **5-page** limit: abstract compressed 422→~290 w, four
  contribution bullets folded to one sentence, §5.4 merged into the conclusion, the
  learning-curve table removed (Fig. 2a plots it; per-*n* values cited to the CSV), the
  mechanism figure demoted to Appendix A1, and §1's "why not small data" paragraph moved to
  where its evidence sits (§5.3). §1 reframed to open on target discovery.
- `[FINDING]` **Two self-inflicted bugs caught by re-reading.** Blind string edits (a) left
  a dangling "Gains are in held-out $R^2$..." clause after the table was cut, and (b)
  over-applied figure renumbering so §5.3 pointed at Fig. A1a for the scale result. Both
  fixed by rewriting §5.3 as a block. Lesson: renumber by rewriting the section, not by
  global replace.
- `[VERIFY]` Audit finding: reference [2] checked first-hand against the bioRxiv API —
  doi:10.1101/2025.07.03.663009, title confirmed, v3 dated 2026-03-10, corresponding
  author A. Califano. Citation now carries the DOI and version.
- `[ADJUST]` Audit finding: page count was reported as "~4.0 pages" against my own script's
  "~4.7". Corrected; length is now reported as a range across words-per-page assumptions
  with the caveat that only compiling the NeurIPS template settles it.
