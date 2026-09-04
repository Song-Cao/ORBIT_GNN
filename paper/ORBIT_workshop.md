# Interaction Residuals, Not Expression: Re-scoring Perturbation Models and the Limits of Curated Graph Priors

## Abstract

Models of combinatorial genetic perturbation are evaluated on predicted expression, where
they reach Pearson *r* ≈ 0.99. We show this metric is uninformative, because the additive
expectation assembled from single-perturbation effects alone attains *r* = 0.994. We
therefore re-score eight published deep and foundation models on the **interaction
residual** — the component of a double perturbation not explained by its constituent
singles — using released predictions and splits, at no retraining cost. The ranking
inverts: a cross-fitted ridge on single-perturbation profiles reaches *r* = 0.488 ± 0.055,
above every deep model (best 0.417). We then test whether a multi-relational graph prior
helps, pre-registering the criterion. It does not: the graph stage ties a *shuffled* graph
and degrades the ridge it corrects. To determine whether this reflects an uninformative
prior or an unreachable one, we scale the estimand to a 108,799-pair CRISPRi interaction
map and sweep training size across three orders of magnitude. Curated priors (regulatory,
physical PPI, co-functional) yield a gain that is **flat in *n*** at +0.001 held-out *R*²,
whereas a graph *inferred from the training interactions themselves* scales from +0.000 to
**+0.027**, twenty times larger and separated from its shuffled control. The bottleneck is
therefore not sample size alone but **prior provenance**: curated edges cover under 6% of
gene pairs at genome scale and do not become more useful with more data. We also show that
74% of the naive interaction residual in growth screens is a known effect-size artifact
that must be removed before any relational claim is credible. The same estimand shift
transfers to equity factor residuals, beating a persistence baseline over 281 out-of-sample
months.

**Contributions.**
1. An evaluation protocol: score the interaction residual, not expression. Applied to
   eight released models it inverts the published ranking at zero retraining cost.
2. A pre-registered negative result with an identified cause — curated graph priors do not
   help, and their failure does not diminish with *n*.
3. A scaling diagnostic on 108,799 gene pairs separating *prior provenance* from *sample
   size*, showing data-derived structure scales where curated structure does not.
4. A measurement correction: the naive residual in growth screens is dominated by an
   effect-size artifact, with a quantification of how much (74%).

---

## 1. Introduction

Predicting the effect of perturbing two genes at once is a central problem in cell
modelling, and the field reports strong numbers: recent deep and foundation models attain
Pearson correlations near 0.99 against measured expression on held-out double
perturbations.

**Why those numbers do not mean what they appear to.** For a perturbation set *S* in
context *c*, the measured profile decomposes as

$$y_{S,c} \;=\; \mu_c \;+\; \sum_{a \in S} \tau_{a,c} \;+\; r_{S,c} \;+\; \varepsilon ,$$

where μ is the context mean, τ are additive single-perturbation main effects, and *r* is
the **interaction residual** — the only term requiring that the perturbations were applied
*together*. On the standard Norman benchmark the additive expectation
*A* = μ + τ_a + τ_b already correlates with the observed profile at *r* = 0.994, and *r*
carries 1.2% of the variance of *y*. A model scored at 0.99 has therefore demonstrated
that it recovered main effects, which are directly measurable from single-perturbation
experiments and require no model. The reported metric is dominated by a quantity nobody
needs predicted.

**Our insight.** The estimand, not the architecture, is what needs changing. Scoring *r*
instead of *y* requires no retraining — released predictions can be re-scored by
subtracting the same additive expectation the benchmark already supplies — and it changes
the field's conclusions immediately.

**Method and results.** We formalise interaction-residual estimation as a two-stage
estimator: a cross-fitted ridge for interaction *shape*, and a multi-relational graph
network predicting per-pair *magnitude*, constrained by a hard projection onto the
orthogonal complement of the nuisance subspace so that no part of the output can express
main effects. Re-scoring eight published models on *r* inverts their ranking and places
the ridge first. The graph stage, however, fails its pre-registered test — and rather than
tune around that, we diagnose it. Scaling to a 108,799-pair CRISPRi interaction map shows
the failure is specific to *curated* priors, whose contribution is flat in *n*, while a
graph inferred from training interactions scales with *n*. This distinguishes two
hypotheses that a single-dataset result cannot separate.

## 2. Background and related work

**Perturbation prediction.** GEARS and CPA introduce graph and compositional priors for
perturbation response; scGPT, scFoundation, Geneformer, UCE and scBERT are transcriptomic
foundation models evaluated on perturbation tasks. Ahlmann-Eltze et al. [1] showed that on
double perturbations these models do not outperform simple linear baselines, using the
interaction residual as an *evaluation* device. We take the further step of making the
residual the **training target and the reported metric**, and we re-score their released
predictions directly.

**Graph priors in transcriptomics.** GREmLN [2] embeds gene-regulatory structure inside a
transformer's attention and argues that structural priors "risk introducing noisy or
biased priors", motivating regulatory networks *inferred from expression* rather than
literature curation. Our scaling result is the quantitative counterpart of that concern:
we measure how curated and inferred structure behave as *n* grows, and find they diverge.

**Interaction estimation.** Cross-fitting and orthogonalisation are standard in
double/debiased machine learning for removing nuisance-estimation bias. AttentionPert
models non-additive perturbation effects, but analyses residual errors only as post hoc
diagnostics rather than as the estimand; its full text contains no orthogonalisation or
cross-fitting. To our knowledge the residual has not previously been made the training
target for combinatorial perturbation.

## 3. Method

### 3.1 The estimand

Let $\hat A_{ab}$ be a cross-fitted additive expectation for the pair $(a,b)$, estimated
from single perturbations in training folds only. The target is

$$r_{ab} \;=\; y_{ab} - \hat A_{ab} \;\in\; \mathbb{R}^{G},$$

over $G$ read-out genes. Fitting $r$ rather than $y$ removes the main-effect component
that dominates $y$ and makes the interaction the object of both optimisation and
evaluation.

### 3.2 Two-stage estimator

A standard relational layer updates node states as

$$h_i^{(l+1)} \;=\; \phi\!\left(h_i^{(l)},\; \operatorname{AGG}_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)} h_j^{(l)}\right).$$

Applying this directly to $r$ fails in the small-$n$ regime (§5), so we factor the problem
into shape and magnitude. **Stage 1** predicts shape with a cross-fitted ridge on
symmetric pair features $\psi_{ab} = [x_a + x_b,\; x_a \odot x_b,\; |x_a - x_b|]$ built
from single-perturbation profiles $x$:

$$\hat r^{(1)}_{ab} \;=\; W_1 \psi_{ab}, \qquad W_1 = \arg\min_W \sum_{(a,b) \in \mathcal{D}_{\text{tr}}} \lVert W\psi_{ab} - r_{ab}\rVert^2 + \lambda \lVert W \rVert_F^2 .$$

The penalty $\lambda$ is chosen by $K$-fold cross-fitting, and out-of-fold predictions
$\tilde r^{(1)}$ are retained so that stage 2 trains on honest residuals.

**Stage 2** predicts a per-pair scalar *gain* from relational structure. With $R$
relation-specific graphs $G^{(r)}$ over perturbed genes and learned relation weights
$\omega = \operatorname{softmax}(\theta)$, node embeddings are

$$z_i \;=\; \sum_{r=1}^{R} \omega_r \sum_{j \in \mathcal{N}_r(i)} G^{(r)}_{ij} \, W_r x_j ,$$

and the gain uses a rank-$k$ bilinear form symmetric in $(a,b)$:

$$g_{ab} \;=\; 1 + \tanh\!\big(u^\top B(z_a, z_b)\big), \qquad B(z_a,z_b) = \sum_{m=1}^{k} \sigma_m \,(v_m^\top z_a)(v_m^\top z_b),$$

giving $\hat r_{ab} = g_{ab} \cdot \tilde r^{(1)}_{ab}$. The $\tanh$ lets the graph
attenuate or amplify the predicted interaction but never flip its sign. We set $k=16$ from
measurement: the observed residual matrix has effective rank 11.3 against 30.9 for a
variance-matched permuted null.

**Why this factorisation.** It makes the graph's contribution *identifiable* — stage 2's
increment over stage 1 **is** the graph's value, which is precisely the quantity our
pre-registered test interrogates. An end-to-end model conflates the two.

### 3.3 Nuisance projection

Let $M = \operatorname{span}(m_1,\dots,m_q)$ be a nuisance subspace spanned by the
constant direction, the mean additive direction, and leading principal components of
training additive predictions. With $Q$ an orthonormal basis of $M$, we constrain outputs
to $M^{\perp}$:

$$\tilde r_\theta \;=\; (I - QQ^{\top})\, r_\theta .$$

**Proposition 1.** *If the nuisance error satisfies $\delta \in M$, then the population
risk decomposes as $\mathcal{R}(\theta;\delta) = \mathcal{R}(\theta;0) + \mathbb{E}\lVert\delta\rVert^2$, so the risk
minimiser is invariant to $\delta$.* The projection is idempotent and costs
$O(G q)$; unit tests assert idempotency, subspace removal, and preservation of the
orthogonal component.

### 3.4 Relations, and the provenance question

We instantiate four relation families: **regulatory** (TF→target, union of DoRothEA,
TRRUST, ENCODE, ITFP, TRED, Neph2012, Marbach2016), **physical PPI** (STRING experimental
channel), **co-functional** (STRING combined), and **data-derived** effect-profile or
interaction-profile similarity computed from training folds only. The first three are
*curated*; the last is *inferred from the data being modelled*. §5.3 shows this
distinction, not sample size alone, governs whether relational structure helps.

## 4. Experimental setup

**Datasets.** (i) *Norman* — the released Norman double-perturbation benchmark with its
five published test/train splits used verbatim [1], 224 perturbations, 62 training doubles
and 31 test doubles per split; read-out restricted to the top 2,000 genes by residual
variance on training doubles (70% of training residual sum of squares), selected without
seeing test pairs. (ii) *Horlbeck* — a CRISPRi dual-guide growth-interaction map
(GSE116198) in K562 and Jurkat, aggregated to **108,799 gene pairs over 467 genes**, with
single-gene effects from control-paired guides. The two cell lines give an external
reproducibility axis Norman lacks.

**Metrics.** Per-pair Pearson *r* between predicted and observed residual (shape) and
residual $R^2 = 1 - \lVert\hat r - r\rVert^2/\lVert r\rVert^2$ (magnitude). Both are
reported because per-pair Pearson is scale-invariant and therefore cannot register
magnitude calibration — a subtlety that initially masked stage 2's behaviour entirely.

**Baselines.** Eight released models (GEARS, CPA, scGPT, scFoundation, Geneformer, UCE,
scBERT, UCE-33) re-scored via $\hat r = \hat y - A$; plus additive (zero), mean training
residual, and the cross-fitted ridge.

**Pre-registered kill criterion.** If the graph stage does not beat the no-graph ridge, we
report that rather than tuning around it.

## 5. Results

### 5.1 Re-scoring inverts the published ranking

All methods score $r = 0.957$–$0.992$ on total expression, a range of 0.035. On the
interaction residual they span 0.260–0.417 and every one loses to the ridge (Fig. 1a).

| method | *r* on residual | *r* on total *y* |
|---|---|---|
| **cross-fitted ridge (no graph)** | **0.488 ± 0.055** | 0.996 |
| UCE / scBERT / UCE-33 | 0.417 | 0.987 |
| scGPT | 0.387 | 0.986 |
| GEARS | 0.385 | 0.987 |
| Geneformer | 0.379 | 0.992 |
| scFoundation | 0.297 | 0.982 |
| CPA | 0.260 | 0.957 |
| mean training residual | 0.219 | — |
| additive model | 0.000 | 0.995 |

The additive model scores exactly 0 on the residual and 0.995 on *y*: the argument in two
numbers. (UCE and scBERT predictions in the release are numerically identical, *r* = 1.000000,
and scGPT is 0.999 against both; we report them as released but they are not three
independent points.)

### 5.2 The graph stage fails its pre-registered test

Over 5 splits × 3 seeds the graph stage reaches residual $R^2 = 0.103$ against **0.274**
with the stage removed. A **shuffled** graph scores 0.110 — higher than the real graph on
every split tested (mean difference +0.008, paired *t* *p* = 0.06, *n* = 3). Learned
relation weights remain uniform at ≈0.25 (Fig. 1b), so the model never differentiates the
relations. The kill criterion fires.

### 5.3 Prior provenance, not sample size, is the binding constraint

A single-dataset null cannot distinguish "the prior is uninformative" from "the estimator
cannot reach it at *n* = 62". We separate them on Horlbeck by sweeping training size from
60 to 86,953 pairs with a fixed 20,000-pair test set, 3 seeds (Fig. 3a).

| training pairs *n* | curated priors | data-derived graph | shuffled control |
|---|---|---|---|
| 200 | +0.0003 | +0.0003 | +0.0003 |
| 2,000 | +0.0003 | +0.0003 | +0.0003 |
| 20,000 | +0.0010 | +0.0020 | +0.0010 |
| 60,000 | +0.0014 | +0.0159 | +0.0013 |
| 86,953 | **+0.0014** | **+0.0274** | +0.0014 |

Gains are in held-out $R^2$ over an effect-magnitude control. Curated priors are **flat in
*n***: three orders of magnitude more data does not make them more useful, and their gain
is indistinguishable from the shuffled control. The data-derived graph is likewise flat
until $n \approx 10^4$ and then scales, reaching a gain 20× larger. Coverage explains why:
at genome scale curated relations reach 1.9% (regulatory), 4.1% (physical PPI) and 5.6%
(co-functional) of gene pairs, against 100% for the data-derived relation (Fig. 3b).

This reframes §5.2. The Norman failure is not simply small-*n*; it is that curated priors
carry little transferable relational signal *at any n we can reach*, whereas structure
inferred from the interactions themselves does — consistent with GREmLN's argument for
inferred over curated networks [2].

### 5.4 A measurement artifact in naive interaction residuals

In growth screens the additive (sum) null is systematically biased for strong-effect genes.
Regressing out a smooth function of the expected phenotype — the standard genetic-interaction
correction — shows that **73.9%** of the variance of the naive residual is this
effect-size artifact (Fig. 3c). It is not a harmless nuisance: the naive residual is
*anti*-correlated across cell lines (*r* = −0.290) while the corrected score is positively
correlated (*r* = +0.200, *n* = 104,196). Any relational claim made on the uncorrected
residual is measuring the artifact. All Horlbeck results above use the corrected score.

### 5.5 The nuisance projection is insurance, not improvement

Holding the evaluation target fixed at the true residual and degrading only the nuisance
supplied to the model by $A_\rho = \bar A + \rho(A - \bar A)$: at $\rho = 0.25$ the
unprojected estimator collapses to $R^2 = -0.89$ while the projected one holds at $-0.25$;
at $\rho = 0.5$, $-0.27$ versus $-0.07$. But when the nuisance is well estimated
($\rho = 1$) projection costs accuracy, 0.086 versus 0.232 (Fig. 2a). On this benchmark
singles are measured directly, so the nuisance is accurate and the projection is not worth
its premium — a conclusion we would have missed had we degraded the target as well, which
inverts the curve.

### 5.6 Domain transfer

In asset pricing the additive model is a factor model $R_{it} = \alpha_i + \beta_i^\top f_t + e_{it}$
and the analogue of the interaction residual is pairwise residual co-movement
$r_{ij} = \mathbb{E}_t[e_{it}e_{jt}]$. On 49 Ken French industry portfolios (2000–2026,
6,662 days, 1,176 pairs, FF5 betas from trailing 252-day windows, 281 out-of-sample
months), the same construction beats the factor model by +0.066 residual $R^2$ (82% of
months, $p < 10^{-17}$) and a trailing-covariance persistence baseline by +0.014 (57% of
months, Wilcoxon $p = 0.0015$), paired per month (Fig. 1d).

## 6. Conclusion and limitations

Scoring the interaction residual rather than expression changes which perturbation models
look good, and costs nothing to adopt. Our pre-registered test of curated graph priors
fails, and the scaling diagnostic identifies why: curated relations are sparse at genome
scale and their contribution does not grow with data, while structure inferred from the
interactions themselves does. For practitioners this argues against literature-derived
graph priors for interaction prediction and toward inferred structure, at scale.

**Limitations.** Norman is one cell type with 31 test doubles per split; the 2,000-gene
subset captures 70% of training residual sum of squares. Horlbeck measures a growth
phenotype, not expression, so it tests the relational question at scale but not the same
read-out; extending the scaling sweep to a large expression-based combinatorial screen is
the natural next step. The data-derived graph's advantage may partly reflect shared
measurement noise between the graph and the target, which our shuffled control bounds but
does not fully exclude. The finance arm uses industry portfolios rather than single names.
scLong and 2026-era Arc/Xaira models are not evaluated: they are absent from the release
and retraining them exceeds this scope. Total compute for all reported results is under 8
CPU-hours with no GPU.

## References

[1] C. Ahlmann-Eltze, W. Huber, S. Anders. Deep-learning-based gene perturbation effect
prediction does not yet outperform simple linear baselines. *Nature Methods* 22:1657–1661,
2025. doi:10.1038/s41592-025-02772-6

[2] GREmLN: A Cellular Graph Structure Aware Transcriptomics Foundation Model. bioRxiv
2025.07.03.663009.

[3] M. Horlbeck et al. Mapping the genetic landscape of human cells. *Cell* 174:953–967,
2018. GEO GSE116198.

---

## Figures

**Figure 1. The estimand shift and its consequences.** (**a**) Held-out Pearson for eight
published models plus the cross-fitted ridge, on total expression (light) and interaction
residual (dark). Every method's total score falls in 0.957–0.996 (CPA lowest at 0.957);
residuals span 0.000–0.488. Mean over 5 published splits. (**b**) Ablations, residual $R^2$
(mean ± s.d., 3 splits): removing the graph stage is best; a shuffled graph matches the
real one. (**c**) Nested out-of-fold $R^2$ for per-pair interaction magnitude on Norman,
adding relation families over an effect-magnitude control. (**d**) Finance arm: paired
per-month gain in residual co-movement $R^2$ over 281 out-of-sample months; error bars 95%
CI of the paired mean, *p*-values Wilcoxon signed-rank.

**Figure 2. Mechanism.** (**a**) Nuisance-degradation curve with the evaluation target held
fixed; projection slows degradation but costs accuracy at $\rho = 1$. (**b**) Partial
correlation of each relation with per-pair interaction magnitude on Norman, controlling for
effect magnitude, annotated with the fraction of the 122 observed perturbation pairs each
relation reaches.

**Figure 3. Prior provenance governs whether graphs help.** (**a**) Gain in held-out $R^2$
over an effect-magnitude control versus training pairs (log scale), Horlbeck corrected GI
score, 3 seeds, fixed 20,000-pair test set. Curated priors are flat in *n*; the
data-derived graph scales to +0.027. Shaded band marks the Norman training regime
(*n* ≈ 62). (**b**) Fraction of the 108,799 gene pairs each relation reaches. (**c**)
Variance of the naive residual versus the corrected GI score: 74% of the naive residual is
an effect-size artifact.
