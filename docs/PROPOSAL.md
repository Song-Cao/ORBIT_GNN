# ORBIT — Orthogonalised Residual Bilinear Interaction Targeting

**Status:** v1.0, 2026-09-03. Standing reference document.
**Venue target:** NeurIPS 2026 workshop (primary AI4DD; fallback GDDL).
**Budget ceiling:** 30 CPU-hours. **No GPU required** — see §6.2.
**Rule:** every plan step is checked back against §2 (the claim) and §7 (kill conditions).
If a step does not serve the claim in §2, it is cut regardless of how interesting it is.

---

## 1. The problem, stated precisely

Predicting how a cell responds to a *combination* of perturbations is a central task in
AIDD and cell modelling. The field's standard framing is: given perturbations
{a, b} and context c, predict the post-perturbation expression profile `y_ab,c`.

That framing has just been shown to be uninformative. Ahlmann-Eltze, Huber & Anders
(*Nat Methods* 22:1657–1661, 2025) benchmarked five foundation models and two other deep
models against deliberately simple baselines and found **none outperformed the
baselines**. We reproduced the arithmetic behind that result from their released
predictions (Zenodo 16092690) and it is stark:

> On held-out Norman-2019 double perturbations, an **additive model** — the sum of the
> two single-perturbation effects — achieves **r = 0.9959** against ground truth across
> all gene × perturbation pairs. It explains **99.2%** of the variance in `y`.

This is the diagnosis: **the target is dominated by main effects.** A model predicting
`y` is graded almost entirely on how well it reproduces `μ_c + τ_a + τ_b`, a quantity
that requires no interaction modelling, no graph, and no learned representation. The
interaction — the only part that is biologically surprising and the only part a
combination model should be judged on — is **0.8% of the variance**, and it is
invisible in the headline metric. Reporting Pearson-δ on `y` and calling it perturbation
prediction is measuring the wrong thing.

The same paper closes the loop: when they scored the models on genetic-interaction
detection directly, **none beat a 'no change' baseline**, and the additive model "did not
compete as, by definition, it does not predict interactions."

So the field has a benchmark on which the honest baseline wins the headline metric and
*nothing* wins the mechanistic one. That is not a modelling failure. It is an
**estimand** failure.

## 2. The claim

ORBIT's contribution is a change of estimand, not an architecture.

> **Claim.** Learning the cross-fitted, orthogonalised interaction residual `r` as the
> direct regression target — rather than the total outcome `y` — yields better
> interaction recovery than predicting `y` at matched capacity, and the advantage is
> attributable to the orthogonalisation rather than to added capacity or regularisation.

Formally, decompose
```
    y_{S,c} = μ_c + Σ_{a∈S} τ_{a,c} + r_{S,c} + ε
```
and take **`r`, not `y`, as the estimand.** The nuisance components (μ, τ) are fitted on
held-out folds and *projected out of the model's output space*, so the interaction head
is structurally incapable of scoring points on main effects.

This is falsifiable in three independent ways (§7), and it makes a prediction that
predicting-`y` models cannot make: performance on interaction recovery should be
insensitive to nuisance-estimation error, because the projection removes it exactly.

## 3. Why this is not already done

**Provenance marking is per row, not blanket.** Rows tagged *(full text read)* were
retrieved and searched in this session — their claims, including the negative ones, are
first-hand. Rows without that tag are characterised from **recall plus retrieved
bibliographic metadata only**, and their mechanistic details are **unverified** pending a
read. Nothing in the second class may be used as a novelty argument in a submission
until read. This distinction is enforced because an earlier draft of this table asserted
verified-sounding negatives about a method whose abstract had never been retrieved.

| Closest work | What it does | Why ORBIT is not it |
|---|---|---|
| **Ahlmann-Eltze et al. 2025** (*Nat Methods* 22:1657–1661; full text read) | Benchmarks models predicting `y`; scores interactions *post hoc* by thresholding deviation from additive expectation | Diagnoses the problem; proposes no method. Interaction deviation is an *evaluation* device, never a training target. ORBIT makes it the training target. |
| **GEARS** (*Nat Biotech* 2023) — *metadata only, unverified* | GNN over GO/coexpression graphs; predicts `y`; reports a derived GI score | Estimand is `y`. Additive expectation is a comparator, not a fitted-and-projected nuisance. |
| **AttentionPert** (*Bioinformatics* 40(7), 2024; PMC11211811, full text read) | Two encoders: PertWeight (global, attention-based) and PertLocal. The paper states PertWeight "is additive for genes in any multi-gene perturbation c, which cannot simulate the nonadditive feature", so PertLocal adds a **nonadditive bias layer** `E'_z = E_z[1 + β(m−1)tanh(K_NA z_c)]` to learn nonlinear co-effects | Targets non-additivity **architecturally** and regresses `y`. Full text contains **zero** occurrences of "orthogonal", "cross-fit", "nuisance", or "debiased": there is no fitted-and-projected nuisance stage and no cross-fitting. Its "residual" analysis is *post hoc* diagnostics of its own prediction errors (§3.3), not a training target. **Closest neighbour in biology — required baseline.** |
| **CPA / SAMS-VAE** — *recall only, unverified* | Latent additive composition of perturbation embeddings | These *impose* additivity; ORBIT *removes* it to expose what additivity misses. |
| **Double/debiased ML** (Chernozhukov et al. 2018; R-learner, Nie & Wager 2021) — *classical, cited as prior art we disclaim* | Cross-fitting + Neyman-orthogonal scores for low-dim causal parameters | Classical and **disclaimed as ours**. All are scalar/low-dimensional treatment effects. ORBIT's delta: a 19,264-dimensional structured outcome, a graph-relational interaction head, and a hard projection rather than a score-function correction. |
| **Statistical-interaction detection in NNs** — Tsang, Cheng & Liu, "Detecting Statistical Interactions from Neural Network Weights", arXiv 1705.04977 (2017); Cui, Marttinen & Kaski, "Learning Global Pairwise Interactions with Bayesian Neural Networks", arXiv 1901.08361 (2019). *Titles, authors and years verified via the arXiv API; method internals not yet read.* | Identify/quantify which feature pairs interact, from a trained network's weights or via a Bayesian NN | **Different task.** These *detect* interactions among input features as an interpretability output; ORBIT *predicts* a high-dimensional interaction residual as its regression target. Neither defines an estimand net of cross-fitted main effects. Relevant as prior art on "interaction" terminology, not as a baseline. |
| **DeepSynergy, comboFM** — *recall only, unverified* | Drug-synergy prediction vs fixed analytic nulls (Bliss/Loewe) | Fixed analytic null, scalar outcome, no cross-fitting, no orthogonality. |

**Verification duty carried forward.** The GEARS, CPA/SAMS-VAE and DeepSynergy/comboFM
rows above are **not yet verified from full text.** Each must be read before the novelty sentence is finalised —
in particular, whether any of them fits main effects out-of-fold. This is a tracked
plan step (S1b), not an aspiration.

**Standing citation rule.** Every identifier in this document (arXiv id, DOI, PMID,
venue, year, author list) must be checked against the issuing service — arXiv API,
Crossref, or PubMed — before it is written, not after. Recalled years and author
attributions have already been wrong twice in this project: the two interaction-detection
papers above were first cited as "Tsang et al. 2018" and "Cui et al. 2020" when the arXiv
records say 2017 and 2019, and were mischaracterised as hybrid main-effect/interaction
architectures before their titles were read. Retrieving an identifier from a search
result establishes only that the identifier exists.

**Residual novelty:** the specific combination of (i) a cross-fitted *fitted* additive
null, (ii) a hard orthogonal projection of the model output onto the nuisance complement,
(iii) a graph-relational low-rank interaction head, and (iv) evaluation on
interaction recovery as the primary metric — applied to a high-dimensional
transcriptomic outcome. Each ingredient is known; the combination and the estimand shift
are the contribution. **We claim the estimand and the mechanism evidence, not the parts.**

## 4. Method

**Nuisance stage (cross-fitted).** Partition perturbations into K=5 folds. For fold k,
estimate `μ̂^(−k)_c` and `τ̂^(−k)_{a,c}` from data excluding fold k. Construct the
target
```
    r̃_{ab,c} = y_{ab,c} − μ̂^(−k)_c − τ̂^(−k)_{a,c} − τ̂^(−k)_{b,c}
```
Cross-fitting matters: with in-sample nuisance, the residual absorbs nuisance
overfitting and the interaction head learns that instead. Experiment 6 tests exactly this.

**Interaction head.** Relation-aware GNN encoder `z_a = GNN_ψ(a, G)` over a
**multi-relational** gene graph (§4.1). Low-rank bilinear interaction with a learned gate:
```
    r̂_{ab,c} = g_{ab,c} · D_ω( z_a^T B(e_c) z_b , u_{ab,c} )
    B(e_c) = U diag(w(e_c)) V^T,   rank 16
    u_{ab,c} = [z_a, z_b, z_a⊙z_b, |z_a−z_b|, e_c]
```
Rank 16 is justified by measurement, not convention: the observed residual matrix has
**effective rank 11.3** (§6), so rank 16 covers it with margin.

### 4.1 The graph: why not PPI

A protein-protein interaction network is the wrong primary relation for this task, and
we measured this rather than assuming it.

**The screen is a transcription-factor screen.** Of the 100 singly-perturbed genes in
Norman-2019, **60 carry a transcriptional-regulator annotation** (STRING GO enrichment:
"DNA-binding transcription factor activity, RNA Pol II-specific", p = 1.6e-21; n = 41
for that term alone). Two TFs interact genetically through shared *targets* and
regulatory cascades, which a physical-binding graph does not encode.

**PPI coverage is the binding constraint, not PPI quality.** Among the 124 double
perturbations with both singles measured:

| Relation | Pair coverage | Partial `r` with \|residual\|, effect magnitude controlled |
|---|---|---|
| STRING experimental (physical PPI) | **16%** | **+0.328** (p = 3e-04) |
| STRING combined | 19% | +0.299 (p = 0.001) |
| STRING textmining | — | +0.284 (p = 0.001) |
| STRING coexpression | 14% | +0.190 (p = 0.035) |
| Single-effect-profile correlation (data-derived) | **100%** | −0.233 (p = 0.009) |

So prior edges *do* carry genuine interaction signal — physical PPI has the **strongest**
per-edge association, and it survives controlling for the dominant confounder (pairs with
large single effects have smaller relative residuals, r = −0.672). But PPI reaches fewer
than one pair in five. **Coverage, not edge quality, is what limits a PPI-only graph.**

**Multi-relational beats PPI alone.** Cross-validated R² predicting per-pair
\|residual\| (5-fold, n = 124):

| Features | CV R² | Δ vs no graph |
|---|---|---|
| effect magnitude only (no graph) | 0.410 | — |
| + STRING physical PPI | 0.461 | +0.051 |
| + effect-profile correlation | 0.433 | +0.024 |
| **+ all relations** | **0.483** | **+0.074** |

**Design decision.** The encoder operates over `R` relation-specific graphs:
1. **Regulatory (primary).** TF→target edges. Given that 60% of perturbed genes are
   transcriptional regulators, this is the mechanistically appropriate relation for
   genetic interaction between TFs.
2. **Physical PPI** (STRING experimental, score ≥ 0.4) — sparse but the strongest
   per-edge signal.
3. **Co-functional** (STRING combined + GO-BP shared-term).
4. **Data-derived co-expression / effect-profile similarity** — dense, 100% coverage,
   computed **only from training-fold singles** so it cannot leak held-out doubles.

Relation-specific message passing with a learned per-relation weight, so the model can
report *which* relation carries the interaction signal. That per-relation weight is a
readable output, and with (4) providing complete coverage the model is never forced to
fall back on an absent edge.

**Falsifiable prediction this makes:** the regulatory and co-functional relations should
receive larger learned weights than physical PPI, because the screen perturbs TFs.
If physical PPI dominates instead, our mechanistic reading is wrong and we say so.

**Orthogonalisation as a hard constraint.** Let `M` span the nuisance directions
(control, single-perturbation, batch). The model output is
```
    r̃_θ := (I − P̂_M) r_θ ,    P̂_M = M(M^T M)^{−1} M^T
```
computed inside the forward pass per batch and differentiated through.

**Proposition 1 (nuisance-error invariance).** Under (A1) `M` a fixed
finite-dimensional subspace of L²(P) with nuisance error δ ∈ M; (A2) cross-fitting, so δ
is independent of evaluation-fold data; (A3) `r_θ ∈ M^⊥` for all θ; (A4) `E[ε|W] = 0` —
the risk satisfies `R(θ;δ) = R(θ;0) + E[δ²]`, so **argmin_θ R is exactly independent of
δ.** Proof: the cross term `E[⟨r_θ, δ⟩]` vanishes by (A3).

*Scope, stated honestly.* (A1) assumes the nuisance is a linear span. Nonlinear nuisance
lies outside `M` and is not removed. Relaxing (A3) to approximate orthogonality gives
Neyman-orthogonality with excess risk O(‖δ‖²) rather than O(‖δ‖).

## 5. Dual-domain transfer (finance)

Not a reskin — the *same estimand*. Cross-sectional equity returns decompose as
market + sector + style factors + idiosyncratic residual, and residualising against
factors estimated **strictly point-in-time** is cross-fitted orthogonalisation as
practised in quantitative research. We predict the residual (pair-level co-movement
beyond factor exposure) on a dynamic relation graph, with the identical projection
machinery. This tests whether the estimand shift generalises beyond biology — the
mechanism claim is domain-agnostic, so it should.

## 6. Feasibility, measured (not estimated)

Probe run 2026-09-03 on the released benchmark predictions, before any training:

| Quantity | Value | Consequence |
|---|---|---|
| Additive model r vs ground truth | **0.9959** | Confirms the estimand problem quantitatively |
| Interaction residual, frac. of var(y) | **0.82%** | The target is small — noise floor is the key risk |
| Additive-model error on **singles** | 2.9e-10 | Additive null verified exact by construction |
| Residual top singular component | **23.4%** (permuted null: 3.5%) | Residual is **structured**, not noise |
| Residual effective rank | **11.3** (permuted: 30.9) | Low-rank → rank-16 bilinear head is well-specified |
| Residual corr., pairs sharing a gene | **0.297** vs 0.028 (p = 6.6e-07) | Residual is **relational** → a graph should help |
| Held-out doubles per split | 31 (163 train perts, 5 splits) | Small; report paired bootstrap CIs over pairs |

**This is the go decision.** The estimand is small but structured, low-rank, and shares
information across pairs with a common perturbed gene — exactly the inductive bias a
relational bilinear model encodes. Had the residual been unstructured, ORBIT would be
dead and the honest finding would be "the residual is noise"; it is not.

## 6.1 Baselines: what we compare against

The released benchmark contains predictions on **our exact five splits** for eight
deep/foundation models, plus the additive model and ground truth. Re-scoring these on
the interaction-residual metric costs **zero training** — we subtract the additive
expectation from their released `y` predictions.

*Provenance (verified, not recalled).* Zenodo record **16092690**, DOI
`10.5281/zenodo.16092690`, title "Code and data for the benchmark conducted in
'Deep-learning-based gene perturbation effect prediction does not yet outperform simple
linear baselines'" — confirmed against the Zenodo API, 8 files. The method list below
is **enumerated from the downloaded file** `double_perturbation_results_parameters.RDS`
(50 elements = 10 methods × 5 splits, verified 10 per split), not from recall:

| Enumerated from the release | Role here |
|---|---|
| `scgpt`, `scfoundation`, `geneformer`, `uce`, `uce33`, `scbert` | Foundation-model baselines, free |
| `gears`, `cpa` | Task-specific deep baselines, free |
| `additive_model` | The null / nuisance reference |
| `ground_truth` | Evaluation target |

Recorded metadata from the same file: `dataset_name = norman_from_scfoundation`,
`test_train_config_id = ede703be5dd78-c8355c044d3ba` (split 1), 31 test perturbations
per split — the same identifiers our split files must reproduce byte-for-byte.

So the foundation-model comparison is not merely present, it is *stronger* than
retraining would allow: these are the authors' own tuned runs on identical splits,
which removes any suspicion that we under-trained a competitor.

**Additionally trained by us:** AttentionPert (the closest architectural neighbour,
verified in §3 as targeting non-additivity while regressing `y`) and a cross-fitted
ridge (kill condition 1).

**Explicitly out of scope, with reasons stated in the paper:** scLong and 2026-era
models from Arc Institute / Xaira. These are not in the released benchmark, and
retraining a large foundation model on our splits is both outside a 24–30 h workshop
budget and *the wrong comparison* — our claim is about the **estimand**, and it is tested
by showing that models predicting `y` fail on interaction recovery. Adding a ninth
`y`-predicting model does not test the claim; it repeats it. We will say this rather
than implying the omission went unnoticed.

## 6.2 Compute: no GPU is required

Measured, not estimated. The core training step — rank-16 bilinear head, 19,264-dim
output, hard projection against an 8-dim nuisance basis, 163 training pairs — runs
**20 epochs in 1.25 s** on one CPU core. The problem is small because the *data* is
small: 163 training pairs per split, not 163,000.

The full grid (5 splits × 3 seeds × 8 ablations) measures at **0.04 CPU-hours** for the
core arithmetic. Allowing a **100× margin** for autograd, relation-specific message
passing, data loading and evaluation gives **≈ 4 CPU-hours** — the figure we budget
against. (An earlier draft of this section mis-stated that as "under 1 CPU-hour", which
is inconsistent with its own stated margin; 0.04 × 100 = 4.)

Even at 4 h, training is not the cost driver. The 30 h budget is dominated by data
preparation, the noise-floor bootstrap over raw counts, and analysis.

**A GPU would not shorten this run.** At batch 32 and 163 pairs per epoch, kernel-launch
overhead would likely exceed the compute. We state this in the paper: the contribution
is an estimand and an evaluation protocol, and it is deliberately reproducible on a
laptop.

## 7. Kill conditions (pre-registered)

Written before results. Any one of these terminates or redirects the project.

1. **Linear-model parity.** If cross-fitted **ridge** on `r̃` matches ORBIT within the
   paired bootstrap interval, the graph and bilinear head are decorative. Finding
   becomes "the estimand mattered, not the architecture" — reported as such, not buried.
2. **Noise floor.** If held-out residual predictive R² does not exceed zero against a
   replicate-derived noise floor, the residual is unlearnable at this sample size. Report
   the noise floor — the field does not currently publish it.
3. **Projection is inert.** If the hard-projection model's nuisance-degradation curve is
   *not* flatter than the soft-penalty model's, Proposition 1 has no empirical purchase
   and the central mechanism claim fails.
4. **Capacity confound.** If a capacity-matched model trained on `y` and *evaluated* on
   interaction recovery matches ORBIT, the gain is capacity, not estimand.

## 8. What we will not claim

- Not that attention weights or gates are causal explanations.
- Not biological mechanism from predictive residual accuracy alone.
- Not that cross-fitting, orthogonal scores, low-rank bilinear forms, hard-concrete
  gates, or graph-based interaction prediction are ours. They are not.
- Not main-conference novelty until the exact method survives a broader search and a
  baseline reproduction.
- Not a finance strategy claim without a point-in-time, cost-aware backtest.

## 9. Primary metric and split, pre-registered

- **Primary metric:** Pearson correlation between predicted and observed *interaction
  residual* on held-out doubles, computed per perturbation pair and aggregated with a
  paired bootstrap over pairs.
- **Primary split:** the five published `norman_from_scfoundation` test/train splits
  from Zenodo 16092690, used verbatim so numbers are directly comparable to the
  published benchmark.
- **Secondary:** interaction-sign accuracy; AUPRC for non-additive pairs; per-subtype
  precision (buffering / synergistic / opposite, following the benchmark's taxonomy).
- **Reported but not primary:** metrics on total `y`, to show explicitly that ORBIT is
  *not* optimising the quantity the field currently reports.
