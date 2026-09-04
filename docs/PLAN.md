# ORBIT — execution plan

**Version:** v1.0 (2026-09-03). Living document — revise in place, log every change in
`PROJECT_LOG.md` with an `[ADJUST]` entry naming what it supersedes.

**Budget:** **30 CPU-hours ceiling. No GPU required.** Measured: the core training step
runs 20 epochs in 1.25 s on one core (163 training pairs per split, 19,264-dim output,
rank-16 head, hard projection). The full grid measures at 0.04 CPU-h; with a 100×
margin for autograd, message passing and data loading, budget **≈ 4 CPU-h**. The budget is dominated by data preparation, the noise-floor bootstrap
over raw counts, and analysis — **not** by training. Revised down from 48 h after
measurement; see `PROJECT_LOG.md` S0.3.

**Discipline:** each step ends with (a) named artifacts, (b) a `PROJECT_LOG.md` entry,
and (c) an explicit **proposal check** — restate which §2 claim the step served, and
confirm no drift. A step that cannot name its claim is cut.

---

## S0 — Verification and go/no-go  ✅ DONE (2026-09-03)

Discharge the novelty duties; measure whether the estimand is learnable before building.

- Retrieved and read the full text of Ahlmann-Eltze et al. 2025 (*Nat Methods*).
- Downloaded Zenodo 16092690: 10 methods × 5 splits of released predictions.
- Measured: additive r = 0.9959; residual = 0.82% of variance; effective rank 11.3
  vs 30.9 permuted; shared-gene residual correlation 0.297 vs 0.028 (p = 6.6e-07).
- Adversarial novelty search: closest neighbours are AttentionPert (implicit
  non-additivity, predicts `y`) and hybrid main-effect/interaction NNs (no cross-fitting,
  no projection). Gap holds.

**Artifacts:** `docs/PROPOSAL.md`, `results/orbit_feasibility_probe.png`,
`handoff/orbit_probe.json`, `handoff/orbit_noise.json`.
**Verdict: GO.** Residual is structured, low-rank and relational.
**Proposal check:** established §6 and fixed §2's claim in measurable terms.

---

## S1 — Data curation and immutable splits  (est. 3 h CPU, 0 GPU)

**Deliverables**
- `data/norman_pseudobulk.parquet` — gene × perturbation matrix (19,264 × 225),
  extracted from the released benchmark RDS so our `y` is byte-identical to theirs.
- `data/splits/split_{1..5}.json` — the five published train/test/val perturbation
  lists, copied verbatim. **Immutable; checksummed; never regenerated.**
- `data/graphs/` — **four relation-specific graphs** (see `PROPOSAL.md` §4.1), not one
  merged PPI graph:
  - `regulatory.npz` — TF→target edges. **Primary relation**: 60/100 perturbed genes are
    transcriptional regulators (p = 1.6e-21), so regulatory relations are the
    mechanistically right ones. Source: a curated human TF-target set (TRRUST /
    DoRothEA-style), restricted to our gene set; provenance and version recorded.
  - `ppi.npz` — STRING experimental channel, score ≥ 0.4. Strongest per-edge signal
    (partial r = +0.328) but only 16% pair coverage.
  - `cofunctional.npz` — STRING combined ≥ 0.4 ∪ GO-BP shared-term.
  - `profile_sim.npz` — single-effect-profile correlation. Dense (100% coverage),
    **computed only from training-fold singles** — leakage assertion required.
- `data/residual_targets.npz` — cross-fitted `r̃` per split (the estimand).
- `results/fig_s1_data_qc.png` — 4 panels: (i) residual magnitude distribution,
  (ii) per-gene residual SS Lorenz curve, (iii) graph degree distribution,
  (iv) train/test perturbation overlap confirming doubles are held out.

**QC that catches silent bugs** (from the comp-bio conventions):
- Assert gene name ordering identical across all 225 perturbation vectors.
- Assert single perturbations reconstruct the additive model to < 1e-8 (measured 2.9e-10
  in S0) — this validates our nuisance construction against theirs.
- Assert no test perturbation appears in its own split's train list.
- Deduplicate graph edges on sorted node-pair content, not identifier string.
- Assert graph node set ⊆ expression gene set; **report pair-level coverage per
  relation** (measured: PPI 16%, combined 19%, profile-sim 100%).
- **Leakage assertion for `profile_sim.npz`:** for every split, assert the correlation
  matrix was computed using only that split's training singles. This relation is dense
  and data-derived, so it is the most likely source of silent leakage in the project.

**Proposal check:** serves §9 (pre-registered split) and §4 (nuisance stage).

---

## S1b — Finish the novelty audit  (est. 1 h, 0 GPU)

`PROPOSAL.md` §3 marks three rows as **unverified** (GEARS, CPA/SAMS-VAE,
DeepSynergy/comboFM). Retrieve full text for each and
answer one question per paper: **does it fit main effects out-of-fold?** If any does,
ORBIT's cross-fitting delta shrinks and §3 must be rewritten before the paper is drafted.

**Also:** verify every identifier, venue, year and author list in `PROPOSAL.md` §3
against arXiv/Crossref/PubMed, per the standing citation rule. Three citation errors so
far all came from recall supplying detail around a retrieved anchor.

**Deliverables:** `docs/novelty_audit.md` — one row per method, with the retrieved
evidence quoted and a first-hand / unverified tag. Update `PROPOSAL.md` §3 in place.
**Proposal check:** protects §3, the novelty claim. Cheap, and it gates the paper's
central sentence.

---

## S2 — Baselines first  (est. 2 h CPU, 0 GPU)

Baselines before the model, so the bar exists before we can be tempted to move it.

**Deliverables**
- `scripts/baselines.py` + `results/baselines.csv` with, per split, on the **residual**
  metric of §9:
  1. **Zero** (predict `r̂ = 0`) — the additive model. The number to beat.
  2. **Cross-fitted ridge** on `r̃` from `u_ab,c` features, no graph. **Kill condition 1.**
  3. **Mean residual** (predict the training-set mean residual per gene).
  4. **Published models re-scored on the residual metric** — the eight methods
     enumerated from `double_perturbation_results_parameters.RDS` (`scgpt`,
     `scfoundation`, `geneformer`, `uce`, `uce33`, `scbert`, `gears`, `cpa`). Their
     predictions are already in the release, so this is free: we subtract the additive expectation from *their* `y`
     predictions and score on `r`. **This is a headline result** — it says how well
     current SOTA recovers interactions when graded on interactions.
- `results/fig_s2_baselines.png` — forest plot, residual Pearson with bootstrap CIs,
  all methods, sorted.

**Why this matters:** if the published models score ≈ 0 on the residual metric while
scoring 0.99 on `y`, that single panel motivates the entire paper.

**Proposal check:** establishes §7 kill conditions 1 and 4 quantitatively.

---

## S3 — Model implementation  (est. 4 h CPU, 0 GPU)

**Deliverables**
- `models/orbit.py` — three composable pieces, each independently ablatable:
  - `RelationalEncoder`: 2-layer **multi-relational** GNN over the four graphs of §4.1,
    hidden 128, with a **learned per-relation weight** that is logged and reported.
    Small by design — 163 training pairs cannot support a large model.
  - `BilinearInteractionHead`: rank-16 low-rank bilinear + hard-concrete gate.
  - `NuisanceProjection`: `(I − P_M)` via batched `lstsq`, differentiable, applied to
    the output. Unit-tested: assert `‖P_M(I−P_M)v‖ < 1e-6` and that projecting a vector
    already in `M` yields ≈ 0.
- `configs/orbit_base.yaml` — Hydra config; all hyperparameters in one file.
- `scripts/train.py` — single entry point; saves per-pair predictions, not just aggregates.
- Unit tests: `scripts/test_projection.py`, `scripts/test_crossfit_leakage.py`
  (asserts no test perturbation influenced any nuisance estimate).

**Proposal check:** implements §4 exactly; rank 16 justified by the measured
effective rank 11.3, not by convention.

---

## S4 — Gate run: ORBIT vs ridge  (est. 3 h, GPU 0 / CPU)

**The 6-hour kill switch, run before anything else.** Single split, single seed.

**Pass rule (pre-registered):** ORBIT beats cross-fitted ridge on residual Pearson by a
margin exceeding the paired bootstrap interval over held-out pairs.
**Stop rule:** if ridge matches, halt feature work and write the paper as *"the estimand
is what mattered, not the architecture"* — reporting the ridge result as the headline.
That is a legitimate and publishable finding, and it is cheaper to discover now.

**Deliverables:** `results/gate.csv`, `results/fig_s4_gate.png`, log entry with verdict.
**Proposal check:** §7 kill condition 1, resolved before compute is spent.

---

## S5 — Main benchmark  (est. 12 h)

Only if S4 passes.

**Deliverables**
- `results/main_results.csv` — ORBIT vs all S2 baselines, 5 splits × 3 seeds, residual
  Pearson + interaction-sign accuracy + AUPRC for non-additive pairs, paired bootstrap CIs.
- `results/interaction_subtypes.csv` — precision by buffering / synergistic / opposite,
  following the benchmark's taxonomy. The published finding is that all models mostly
  predict buffering and rarely get synergistic right; **ORBIT must be tested on exactly
  that weakness.**
- `results/fig_s5_main.png` — (a) forest plot vs baselines; (b) subtype precision
  grouped bars; (c) predicted-vs-observed residual scatter for the best split.

**Proposal check:** the primary test of §2's claim on the §9 metric and split.

---

## S6 — Mechanism evidence: the nuisance-degradation curve  (est. 5 h)

**The experiment that distinguishes ORBIT from a regularisation trick.** Deliberately
corrupt the nuisance estimate by shrinking `τ̂` toward zero by ρ ∈ {1, 0.75, 0.5, 0.25, 0}
and plot held-out residual risk for three model variants.

**Pre-registered predictions** (Proposition 1):
- **Hard projection: flat.** Risk independent of δ.
- **Soft penalty (L⊥, λ tuned): smoothly degrading**, slope set by λ.
- **Same architecture trained on total `y`: steeply degrading.**

Three predicted curve shapes from one construction. If the projected curve is not
flatter than the penalised one, **kill condition 3 fires** and we say so.

**Deliverables:** `results/degradation_curve.csv`,
`results/fig_s6_mechanism.png` (the paper's Figure 3).
**Proposal check:** direct test of §4 Proposition 1 — the mechanism, not the headline.

---

## S7 — Ablations  (est. 8 h, single seed, labelled as such)

| # | Ablation | Isolates |
|---|---|---|
| 1 | **Per-relation leave-one-out**: no graph / shuffled / regulatory-only / PPI-only / cofunctional-only / profile-sim-only / all | Which relation carries interaction signal? Pre-registered prediction: regulatory + cofunctional > physical PPI, because the screen perturbs TFs. Measured priors: PPI partial r = +0.328 but 16% coverage; all-relations CV R² 0.483 vs 0.461 PPI-only |
| 2 | In-sample vs cross-fitted nuisance | Does cross-fitting matter, or is it ceremony? |
| 3 | Hard projection vs soft penalty vs neither | The central design choice |
| 4 | Rank 4 / 16 / full | Is rank 16 (eff. rank 11.3) right? |
| 5 | Gate removed (dense) | Is sparsity load-bearing? |
| 6 | Capacity-matched `y`-trained control, scored on residual | **Kill condition 4** |
| 7 | Nonlinear basis expansion in `M` | Probes Proposition 1's (A1) linearity scope limit |

**Deliverables:** `results/ablations.csv`, `results/fig_s7_ablations.png`.
**Proposal check:** ablation 6 is the capacity confound; ablation 7 is the honest
scope limit from §4. Neither is optional.

---

## S8 — Noise floor  (est. 2 h)

Estimate the replicate-derived noise floor on the interaction residual from the raw
Norman counts (GEO GSE133344 — the released pseudobulk is not resampled, so it cannot
supply this; discovered in S0). Bootstrap cells within perturbation to get a residual
estimation SD, and report predictive R² against that floor.

**Deliverables:** `results/noise_floor.csv`, panel in `fig_s5_main.png`.
**Proposal check:** §7 kill condition 2, and a number the field does not publish.

---

## S9 — Finance transfer  (est. 4 h, OPTIONAL — cut first under pressure)

CSI300 or S&P 500 daily returns. Residualise against market/sector/style factors
estimated **strictly point-in-time** (expanding window, no lookahead), build a dynamic
correlation relation graph, predict pair-level residual co-movement with the identical
projection machinery. Metrics: IC, RankIC, ICIR, and turnover — no strategy claim
without a cost-aware backtest (§8).

**Deliverables:** `results/finance_transfer.csv`, `results/fig_s9_finance.png`.
**Cut rule:** if S1–S8 consume > 40 GPU-h, cut this and note the biology-only scope in
the paper. The biology track is never cut.

---

## S10 — Paper  (est. 3 h, 0 GPU)

4-page workshop short paper + appendix.

**Figure plan** (final, ordered):
1. **The estimand problem** — additive r = 0.9959, residual 0.8%, published models
   re-scored on residual ≈ 0. Panels from S0 + S2.
2. **ORBIT recovers interactions** — forest plot + subtype precision. From S5.
3. **The mechanism** — three degradation curves. From S6.
4. **Ablations + noise floor.** From S7/S8.
5. *(Appendix)* Finance transfer, if run.

**Deliverables:** `docs/paper.md`, `docs/paper.pdf`, `results/compute_table.csv`
(GPU-h and parameter counts per experiment, as §5.5 of the source requires),
one explicit negative/failure analysis.

---

## Budget ledger

| Step | Est. (CPU-h) | Actual | Cum. |
|---|---|---|---|
| S0 verification + probe | 1 | **1** | 1 |
| S0.3 graph + compute probe | 1 | **1** | 2 |
| S1 data + 4 relation graphs | 4 | — | |
| S1b novelty audit | 1 | — | |
| S2 baselines (8 free + AttentionPert + ridge) | 3 | — | |
| S3 implementation | 5 | — | |
| S4 **gate** | 1 | — | |
| S5 main benchmark | 3 | — | |
| S6 mechanism curve | 2 | — | |
| S7 ablations (incl. per-relation) | 3 | — | |
| S8 noise floor (raw counts, CPU-bound) | 3 | — | |
| S9 finance (optional) | 2 | — | |
| S10 paper | 3 | — | |
| **Total** | **32** | | **≤ 30 h target, 2 h slack in S9** |

Training is not the cost driver; **S1 (data) and S8 (noise-floor bootstrap) are.**

**Cut order under pressure:** S9 finance → S7 ablations 4,5 → S8 noise floor.
**Never cut:** S4 gate, S6 mechanism curve, S2 baselines. Those three *are* the paper.

## Standing rules

1. No test-set look before S4's pass rule is fixed in writing. It is (§9).
2. Splits are immutable. Any change invalidates every prior number and requires a
   `[DECISION]` log entry.
3. Three seeds for headline comparisons; single-seed ablations labelled as such.
4. Save per-pair predictions, never only aggregates.
5. Report the residual metric as primary and `y` metrics as secondary — always both, so
   the estimand shift is visible rather than hidden.
6. After each step, re-read `PROPOSAL.md` §2 and §7 and record the proposal check.
