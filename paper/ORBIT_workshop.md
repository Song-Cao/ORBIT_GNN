# The Interaction Residual is the Estimand: Why Perturbation Models that Score 0.99 Tell You Nothing

**Abstract.**
Deep models for combinatorial genetic perturbation are routinely evaluated on predicted
expression, where they reach Pearson *r* ≈ 0.99. We show this number is uninformative:
the additive expectation built from single-gene effects alone attains *r* = 0.995, and the
interaction term it omits carries 1.2% of the variance of the target. The reported
agreement is almost entirely main effects, which every model inherits for free. We
re-score eight published deep and foundation models (GEARS, CPA, scGPT, scFoundation,
Geneformer, UCE, scBERT, UCE-33) on the *interaction residual* — what remains after the
additive expectation is removed — using the released predictions and splits of a recent
benchmark, so the comparison is exact and costs no retraining. On this estimand the
ranking inverts: a cross-fitted ridge on single-perturbation effect profiles reaches
*r* = 0.488 ± 0.055, above every deep model (best 0.417). We then ask whether relational
priors help, and pre-register the test. Prior graphs *do* carry information about
interaction magnitude (out-of-fold *R*² 0.394 → 0.452 when four relation families are
added to an effect-magnitude control; physical PPI partial *r* = +0.275), but a
multi-relational GNN trained on the ~60 training double perturbations available per split
extracts none of it — it does no better than with a *shuffled* graph (*R*² 0.110 vs 0.102)
and degrades the ridge it is built on (0.299 → 0.102). The bottleneck is the
number of training pairs, not the architecture or the prior. Finally we show the estimand
shift is domain-general: on 281 out-of-sample months of US industry portfolios, the same
construction applied to Fama–French factor residuals beats a persistence baseline
(+0.014 residual *R*², *p* = 0.0015, 57% of months) and the factor model itself (+0.066,
82% of months). Our contribution is an evaluation protocol and a negative result with a
measured cause, both of which we argue should precede further architectural work on this
task.

---

## 1. The problem

For a perturbation set *S* in context *c*, write

    y_{S,c} = μ_c + Σ_{a∈S} τ_{a,c} + r_{S,c} + ε ,

separating the context mean, the additive single-perturbation main effects, and the
interaction residual *r* — the only term that requires knowing the perturbations were
applied *together*.

On the Norman et al. double-perturbation benchmark, *r* carries **1.2%** of the variance
of *y* on held-out doubles (var(*r*)/var(*y*) measured directly over the 2,000-gene
subset; equivalently 1 - *r*^2 = 1.17% from the additive model's correlation). The additive expectation *A* = μ + τ_a + τ_b correlates with the observed profile
at *r* = 0.995 and reproduces held-out singles to an RMS of 7.7 × 10⁻¹⁰ (it is exact
there by construction). So a model reported at *r* = 0.99 on *y* has demonstrated that it
learned the main effects, which are directly measurable from single-perturbation
experiments and need no model at all.

This is not a modelling failure; it is a **measurement** failure. The published number is
dominated by a quantity nobody needs predicted. Our starting claim is that the interaction
residual, not the expression profile, is the estimand a combinatorial perturbation model
should be judged on — and that simply changing what is scored, with no new architecture,
already changes the field's conclusions.

## 2. Re-scoring the published models

We use the released predictions and the five published test/train splits of the
benchmark of Ahlmann-Eltze et al. (Nat. Methods 22:1657–1661, 2025; data DOI
10.5281/zenodo.16092690), so nothing is retrained and no split is redrawn. Each method's
prediction of *y* is converted to a prediction of the interaction by subtracting the same
additive expectation the benchmark supplies: r̂ = ŷ − A. Genes are the top 2,000 by
residual variance on split-1 *training* doubles (70% of training residual sum of squares);
the selection never sees test pairs.

**Metric.** Per-pair Pearson correlation between predicted and observed residual on
held-out doubles, averaged over pairs and splits. Because per-pair Pearson is
scale-invariant, we also report residual *R*² = 1 − ‖r̂ − r‖²/‖r‖², which *is* sensitive
to magnitude. The two together separate getting the interaction's shape right from
getting its size right, and we report both throughout.

**Result (Fig. 1a).** All eight methods score *r* = 0.957-0.992 on *y*, a range of 0.035. On the residual they spread over 0.16 and are all beaten by a cross-fitted ridge
on symmetric single-profile pair features:

| method | *r* on residual | *r* on total *y* |
|---|---|---|
| **cross-fitted ridge (ours, no graph)** | **0.488 ± 0.055** | 0.996 |
| UCE | 0.417 | 0.987 |
| scBERT | 0.417 | 0.987 |
| UCE-33 | 0.417 | 0.987 |
| scGPT | 0.387 | 0.986 |
| GEARS | 0.385 | 0.987 |
| Geneformer | 0.379 | 0.992 |
| scFoundation | 0.297 | 0.982 |
| CPA | 0.260 | 0.957 |
| mean training residual | 0.219 | — |
| additive model | 0.000 | 0.995 |

The additive model scores exactly 0 on the residual by definition and 0.995 on *y*, which
is the whole argument in two numbers. We note that UCE and scBERT predictions in the
release are numerically identical (*r* = 1.000000) and scGPT is near-identical to both
(0.999); we report them separately as released but they should not be read as three
independent data points.

## 3. Does a relational prior help? A pre-registered test

The natural next step is a graph. Genetic interaction between two perturbations should
depend on how the genes relate, and we made a falsifiable prediction before running
anything: because 60 of the 100 perturbed genes carry a transcription-factor annotation,
*regulatory* and *co-functional* relations should outweigh physical PPI.

**Four relation families,** each a dense weighted graph over the 100 perturbed genes.
Coverage below is the fraction of the 122 observed perturbation pairs a relation reaches
(the population the tests in §4 are computed over); coverage over all 4,950 possible gene
pairs is lower for every relation:

- **regulatory** — TF→target edges from the union of DoRothEA, TRRUST, ENCODE, ITFP,
  TRED, Neph2012 and Marbach2016 (3,127 TFs; 48/100 perturbed genes have a regulon),
  scored by target-set Jaccard overlap or direct regulation. Reaches 34% of pairs.
- **physical PPI** — STRING experimental channel. 16% of pairs.
- **co-functional** — STRING combined score. 19% of pairs.
- **effect-profile similarity** — correlation of single-perturbation profiles, computed
  from *training-fold* singles only. 100% of pairs.

**Model.** A two-stage estimator. Stage 1 is the cross-fitted ridge above (shape). Stage 2
is a multi-relational GNN with learned per-relation weights and a low-rank bilinear
interaction head, predicting a per-pair scalar gain on the stage-1 prediction
(magnitude), trained against stage 1's *out-of-fold* predictions and projected onto the
orthogonal complement of a nuisance subspace so no part of the output can express main
effects. Rank 16 is set by measurement: the residual matrix has effective rank 11.3
against 30.9 for a variance-matched permuted null.

**Kill condition, declared in advance.** If the graph stage does not beat the no-graph
ridge, we report that.

**It does not (Fig. 1b).** Over 5 splits × 3 seeds the graph stage reaches residual
*R*² = 0.103 against 0.274 for the same estimator with the graph stage removed; in the
ablation grid (3 splits) the corresponding figures are 0.102 and 0.299. Decisively, a **shuffled** graph scores 0.110 — higher than the real
graph on every split tested (mean difference +0.008, paired *t* *p* = 0.06, *n* = 3
splits), so the real edges contribute nothing the permuted edges do not. Every
single-relation variant lands at 0.10-0.11. Learned relation weights stay at ~0.25 each, i.e. uniform: the model
never differentiates the relations, so our TF prediction cannot even be evaluated. The
GNN is not using the graph.

## 4. Why it fails, and what the graph does carry

A GNN that ties a shuffled graph is uninformative about whether the *prior* is useless or
whether the *estimator* cannot reach it. These have opposite implications, so we separate
them with a capacity-controlled test: predict a scalar per-pair quantity — the interaction
magnitude ‖r_ab‖ — from a handful of graph-derived scalar features with ridge and nested
cross-validated regularisation. Five features and 122 unique pooled training pairs is a
regime where a linear model is well-posed, so a null here is evidence about information.

**The graph does carry signal (Fig. 1c, Fig. 2b).** Out-of-fold *R*² rises from 0.394
(effect magnitude alone) to 0.452 with all four relation families. Controlling for effect
magnitude, physical PPI has partial *r* = +0.275 with interaction strength, co-functional
+0.237, regulatory +0.110, and effect-profile similarity −0.286.

Two things follow. First, **the failure in §3 is a sample-size failure, not a prior
failure**: the information is present and linearly accessible, but ~60 training pairs per
split cannot fit a GNN that finds it. Second, our pre-registered mechanistic prediction is
**wrong in its ordering** — physical PPI is the strongest relation, not regulatory, despite
the perturbed genes being predominantly TFs and despite regulatory having the widest
coverage. We had committed to reporting this outcome and we do: the mechanistic reading
that TF-TF genetic interaction should run through shared regulons is not supported here.

## 5. The nuisance projection

Proposition 1 of our design claims that constraining the output to the orthogonal
complement of the nuisance subspace *M* makes the risk minimiser independent of nuisance
estimation error δ ∈ *M*. Testing this requires care: degrading the nuisance also changes
the regression target, so a naive sweep shows *R*² rising simply because the target becomes
the larger, easier total signal. We therefore hold the evaluation target fixed at the true
residual and degrade only the nuisance supplied to the model,
A_ρ = mean(A) + ρ(A − mean(A)).

**Result (Fig. 2a).** Projection substantially bounds the damage. At ρ = 0.25 the
unprojected estimator collapses to *R*² = −0.89 while the projected one holds at −0.25; at
ρ = 0.5, −0.27 versus −0.07. But when the nuisance is *well* estimated (ρ = 1) projection
costs accuracy: 0.086 versus 0.232, and the ablation in Fig. 1b shows the same
(0.200 without projection versus 0.102 with, 3 splits). The projection is insurance, not a free
improvement — worth its premium only when the nuisance is poorly estimated, which on this
benchmark it is not, because single-perturbation effects are measured directly.

## 6. Domain transfer: equity factor residuals

The estimand shift is not specific to biology. In asset pricing the additive model is a
factor model, R_it = α_i + β_i′f_t + e_it, and the analogue of the interaction residual is
pairwise residual co-movement r_ij = mean_t(e_it e_jt) — co-movement the factor model does
not explain. As in biology, the additive part dominates and a model scored on raw
covariance looks excellent while saying nothing about the interaction.

**Setup.** 49 Ken French industry portfolios, daily, 2000-01 to 2026-06 (6,662 days,
1,176 pairs). Fama–French 5-factor betas from a 252-day trailing window; residuals formed
on the following 21 days with those trailing betas, so the target month never informs the
features. Walk-forward with a 24-window training history, 281 out-of-sample months. Pair
features mirror the biology arm: sector prior (↔ regulatory), factor-loading similarity
(↔ co-functional), trailing residual correlation (↔ profile similarity), volatility
product (↔ effect magnitude), and trailing covariance (persistence).

**Result (Fig. 1d).** Paired per-month comparisons: the cross-fitted ridge on the residual
beats the factor model itself by +0.066 residual *R*² (82% of months, *p* < 10⁻¹⁷) and the
trailing-covariance persistence baseline by +0.014 (57% of months, *p* = 0.0015, Wilcoxon).
The same estimand shift, the same pair-feature construction, and the same conclusion —
that the interaction is small, structured, and reachable by a well-regularised linear model
— hold in a domain with no genes in it.

## 7. What we claim

1. **Evaluation.** Reporting expression-level correlation for combinatorial perturbation
   models is uninformative; the additive baseline attains 0.995. The interaction residual
   should be scored, and it costs nothing to score it on already-released predictions.
   Doing so inverts the published ranking and puts a cross-fitted ridge on top.
2. **A measured negative result.** Relational priors carry real information about
   interaction magnitude (out-of-fold *R*² 0.394 → 0.452), but a multi-relational GNN
   cannot extract it from ~60 training pairs and ties a shuffled graph. We locate the
   bottleneck in the number of held-out double perturbations, which is a property of
   available data, not of the architecture — so scaling the model is the wrong response.
3. **Generality.** The estimand shift transfers to equity factor residuals with the same
   construction and the same qualitative result.

**Limitations.** One dataset (Norman), one cell type, 31 test doubles per split; the
2,000-gene subset captures 70% of training residual sum of squares, not all of it; the
finance arm uses industry portfolios, not single names, so it avoids survivorship and
corporate-action handling at the cost of cross-sectional breadth. We did not train scLong
or 2026-era models from Arc/Xaira — they are absent from the release and retraining them
exceeds this workshop's scope. The projection result is a bound on nuisance sensitivity,
not an accuracy improvement; on this benchmark the nuisance is well estimated and the
projection costs performance.

**What would change our mind.** A screen with ~10³ held-out double perturbations. Our
capacity-controlled test says the relational information is there; the GNN's failure is
about *n*, and only more pairs can distinguish "GNNs cannot use this prior" from "GNNs
could not use this prior *here*".

## Reproducibility

All code, tables, and figures are in this repository. `scripts/01`–`08` run end to end on
8 CPU cores with no GPU; total compute for every number reported here is under 6 CPU-hours.
Data are fetched by script from Zenodo (10.5281/zenodo.16092690), STRING, GitHub-hosted
regulon releases, and the Ken French Data Library. Splits are the published ones, used
verbatim.

---

## Figures

**Figure 1. The estimand shift, its consequences, and its transfer.**
(**a**) Held-out Pearson correlation for eight published models plus a cross-fitted ridge,
scored on total expression *y* (light bars) and on the interaction residual *r* (dark
bars). Every method's total-expression score falls in 0.957-0.996 (CPA is the lowest at
0.957); on the residual the same methods span 0.000-0.488 and the ridge, which uses no
graph and no pretraining, ranks first. Among the eight published models alone the
residual spread is 0.157. The additive model is exactly
0 on *r* by construction. Mean over 5 published splits, 31 held-out doubles each.
(**b**) Ablations of the two-stage estimator, residual *R*² (mean ± s.d. over 3 splits).
Removing the graph stage is the best configuration; a shuffled graph matches the real one.
(**c**) Nested out-of-fold *R*² for predicting per-pair interaction magnitude ‖r_ab‖ from
graph-derived scalar features, adding one relation family at a time on top of an
effect-magnitude control (dotted line). 122 unique pooled training pairs, ridge with
nested cross-validated penalty. The relations do carry information.
(**d**) Finance arm: paired per-month gain in residual co-movement *R*² over 281
out-of-sample months, versus the factor model itself and versus a trailing-covariance
persistence baseline. Error bars are 95% CIs of the paired mean; *p*-values are Wilcoxon
signed-rank.

**Figure 2. Mechanism.**
(**a**) Nuisance-degradation curve. The evaluation target is held fixed at the true
interaction residual while the nuisance supplied to the model is shrunk toward its mean by
ρ. With the hard projection onto the nuisance complement (blue) held-out *R*² degrades far
more slowly than without it (orange), but projection also costs accuracy when the nuisance
is accurate (ρ = 1, left edge). Mean ± s.d. over 5 splits.
(**b**) Partial correlation of each relation family with per-pair interaction magnitude,
controlling for effect magnitude, annotated with the fraction of the 122 observed
perturbation pairs each relation reaches. Physical PPI is the strongest relation despite reaching only 16% of pairs; this
contradicts our pre-registered prediction that regulatory relations would dominate.
