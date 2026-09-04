# Three Ways to Mismeasure a Genetic Interaction

## Abstract

Every reported advance in combinatorial perturbation prediction rests on a *measurement* of
what an interaction is. We show that three measurements in current use are each dominated
by something other than interaction, and that each failure is exposed by a control the
field does not routinely run. **(I)** Models reach Pearson *r* ≈ 0.99 on held-out double
perturbations, but the additive expectation from single-gene effects alone attains
*r* = 0.994. Re-scoring eight released models on the **interaction residual** — no
retraining, using their own published predictions and splits — inverts the ranking: a
cross-fitted ridge reaches 0.488 ± 0.055 against 0.417 for the best deep model. *Control:
the additive baseline.* **(II)** Subtracting an additive null is not enough. In growth
screens that null is biased for strong-effect genes, and **73.9%** of the naive residual's
variance is this artifact; the naive residual *anti*-correlates between two cell lines
(*r* = −0.290) where the corrected score correlates positively (+0.200, *n* = 104,196).
*Control: cross-assay reproducibility.* **(III)** On a 108,799-pair CRISPRi map, curated
priors (regulatory, PPI, co-functional) give a gain **flat in *n*** at +0.0014 held-out
*R*², indistinguishable from a shuffled graph, while structure *inferred from the training
interactions* scales to **+0.0274**; curated edges reach under 6% of gene pairs at genome
scale. *Control: shuffled graph plus a sample-size sweep.* The binding constraint is
measurement validity, not model capacity.

**Contributions.** (1) a re-scoring protocol that inverts the published ranking of eight
perturbation models at zero retraining cost; (2) quantification of an effect-size artifact
that dominates naive interaction residuals, validated against cross-assay reproducibility;
(3) a sample-size sweep separating prior *provenance* from prior *reachability*; and (4)
three cheap, reusable controls, each of which reverses a conclusion the field draws.

---

## 1. Introduction

Combinatorial perturbation prediction is a workhorse of computational target discovery:
which two genes, knocked down together, produce a non-additive phenotype worth pursuing.
The reported numbers are strong — recent deep and foundation models attain Pearson
correlations near 0.99 against measured expression on held-out double perturbations. Yet
this is precisely the setting where strong benchmark performance is known not to survive
contact with a new assay, and we show why: the benchmark metric does not measure the
quantity that target selection depends on.

**The problem is not the models; it is what "interaction" was taken to mean.** For a
perturbation set *S* in context *c*,

$$y_{S,c} \;=\; \mu_c \;+\; \sum_{a \in S} \tau_{a,c} \;+\; r_{S,c} \;+\; \varepsilon ,$$

where μ is the context mean, τ are additive single-perturbation main effects, and $r$ — the
**interaction residual** — is the only term requiring that the perturbations were applied
*together*. On Norman the additive expectation $A = \mu + \tau_a + \tau_b$ already
correlates with the observed profile at $r = 0.994$, and $r$ carries just 1.2% of the
variance of $y$. A model scored at 0.99 has therefore shown that it recovered main effects
— directly measurable from single-perturbation experiments, requiring no model at all.

**Our thesis.** This is one instance of a general pattern: a quantity is reported as
evidence about interaction while being dominated by something else. We identify three such
mismeasurements along the path a practitioner actually takes — score expression, then score
the residual, then condition on a graph prior — and show that each is caught by a simple
control, and that each control reverses the conclusion.

## 2. Background

**Perturbation prediction.** GEARS and CPA introduce graph and compositional priors;
scGPT, scFoundation, Geneformer, UCE and scBERT are transcriptomic foundation models
evaluated on perturbation tasks. Ahlmann-Eltze et al. [1] showed these models do not
outperform linear baselines on double perturbations, using the interaction residual as an
*evaluation* device. We make the residual the reported metric, re-score their released
predictions, and then audit the residual itself — where failures (II) and (III) appear.

**Graph priors.** GREmLN [2] embeds gene-regulatory structure inside transformer attention
and argues that structural priors "risk introducing noisy or biased priors", motivating
networks *inferred from expression* over literature curation. Our sweep is the
quantitative counterpart: we measure how curated and inferred structure behave as *n*
grows and find they diverge by a factor of twenty.

**Interaction estimation.** Cross-fitting and orthogonalisation are standard in
double/debiased ML for removing nuisance-estimation bias. AttentionPert models non-additive
effects but treats residual errors as post hoc diagnostics only. The effect-size bias of
additive nulls is known in the classical genetic-interaction literature; failure (II)
quantifies what ignoring it costs an ML pipeline.

## 3. Method: the estimand and its three controls

### 3.1 The target

Let $\hat A_{ab}$ be a cross-fitted additive expectation for pair $(a,b)$, estimated from
single perturbations in training folds only. The target is

$$r_{ab} \;=\; y_{ab} - \hat A_{ab} \;\in\; \mathbb{R}^{G}$$

over $G$ read-outs. **Control I** is $\hat A$ itself: any metric on which the additive
baseline scores well is not measuring interaction.

### 3.2 Correcting the target (Control II)

For growth phenotypes the additive null is biased for strong-effect genes, so the naive
residual conflates interaction with effect size. We regress out a smooth function of the
expected phenotype $e_{ab} = s_a + s_b$,

$$\hat r^{\text{GI}}_{ab} \;=\; r_{ab} - \big(\beta_1 e_{ab} + \beta_2 e_{ab}^2 + \beta_3 e_{ab}^3 + \beta_4 s_a s_b + \beta_5 |s_a - s_b|\big),$$

and validate the correction not by fit but by **cross-assay reproducibility**: the same
gene pairs measured in an independent cell line should agree. A target that
anti-correlates across cell lines is measuring the assay, not the biology.

### 3.3 An instrument for testing prior value (Control III)

To ask whether a relational prior helps, its contribution must be *identifiable*. A
standard relational layer updates node states as

$$h_i^{(l+1)} \;=\; \phi\!\left(h_i^{(l)},\; \operatorname{AGG}_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)} h_j^{(l)}\right),$$

but end-to-end training conflates the prior's value with the estimator's capacity, so we
factor the problem. **Stage 1** predicts interaction *shape* with a cross-fitted ridge on
symmetric pair features $\psi_{ab} = [x_a + x_b,\; x_a \odot x_b,\; |x_a - x_b|]$ from
single-perturbation profiles $x$:

$$W_1 \;=\; \arg\min_W \sum_{(a,b) \in \mathcal{D}_{\text{tr}}} \lVert W\psi_{ab} - r_{ab}\rVert^2 + \lambda \lVert W \rVert_F^2 ,$$

with $\lambda$ chosen by $K$-fold cross-fitting and out-of-fold predictions
$\tilde r^{(1)}$ retained. **Stage 2** predicts a per-pair scalar *gain* from $R$
relation-specific graphs $G^{(r)}$ with learned weights
$\omega = \operatorname{softmax}(\theta)$:

$$z_i = \sum_{r=1}^{R} \omega_r \sum_{j \in \mathcal{N}_r(i)} G^{(r)}_{ij} W_r x_j, \qquad
g_{ab} = 1 + \tanh\!\big(u^\top B(z_a,z_b)\big),$$

with $B(z_a,z_b) = \sum_{m=1}^{k} \sigma_m (v_m^\top z_a)(v_m^\top z_b)$ symmetric in
$(a,b)$, giving $\hat r_{ab} = g_{ab} \cdot \tilde r^{(1)}_{ab}$. The $\tanh$ lets the graph
attenuate or amplify but never flip sign; $k=16$ is set by the residual matrix's measured
effective rank (11.3, against 30.9 for a variance-matched permuted null).

Stage 2's increment over stage 1 **is** the prior's value. Control III pairs this with two
falsifiers: a **shuffled graph** (same capacity, no real edges) and a **sweep over training
size**, which separates an uninformative prior from an unreachable one. Stage-2 outputs are
additionally constrained to a nuisance subspace's orthogonal complement (Appendix A).

### 3.4 Relations, curated and inferred

Four families: **regulatory** (TF→target; union of DoRothEA, TRRUST, ENCODE, ITFP, TRED,
Neph2012, Marbach2016), **physical PPI** (STRING experimental), **co-functional** (STRING
combined), all curated from literature; and **data-derived** interaction-profile similarity
computed from training pairs only. §5.3 shows this curated/inferred distinction, not sample
size, governs whether relational structure helps.

## 4. Experimental setup

**Datasets.** (i) *Norman* — released Norman double-perturbation benchmark, five published
splits used verbatim [1]; 224 perturbations, 62 training and 31 test doubles per split;
read-out restricted to the top 2,000 genes by residual variance on training doubles (70%
of training residual sum of squares), selected without seeing test pairs. (ii) *Horlbeck* —
CRISPRi dual-guide growth-interaction map (GSE116198) [3] in K562 and Jurkat, aggregated to
**108,799 gene pairs over 467 genes**, single-gene effects from control-paired guides. The
two cell lines supply the independent assay that Control II requires.

**Metrics.** Per-pair Pearson *r* (shape) and residual
$R^2 = 1 - \lVert\hat r - r\rVert^2/\lVert r\rVert^2$ (magnitude). Both are needed: per-pair
Pearson is scale-invariant and therefore structurally blind to magnitude calibration.

**Baselines.** Eight released models (GEARS, CPA, scGPT, scFoundation, Geneformer, UCE,
scBERT, UCE-33) re-scored as $\hat r = \hat y - A$; plus additive (zero), mean training
residual, and the cross-fitted ridge.

**Pre-registration.** Before running stage 2 we committed to reporting a null if the graph
stage failed to beat the no-graph ridge, and to reporting our mechanistic prediction
(that regulatory relations would dominate, since 60% of perturbed genes are TF-annotated)
as falsified if PPI dominated instead. Both outcomes occurred and both are reported.

## 5. Results

### 5.1 Mismeasurement I — expression-level scoring cannot see interaction

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

The additive model scores exactly 0 on the residual and 0.995 on $y$: the argument in two
numbers. Re-scoring needs no retraining, so this control is available to anyone releasing
predictions. (UCE and scBERT predictions in the release are numerically identical
($r = 1.000000$) and scGPT is 0.999 against both, so they are not three independent points.)

### 5.2 Mismeasurement II — the naive residual is mostly effect size

Applying the correction of §3.2 to the Horlbeck map removes **73.9%** of the variance of
the naive residual (Fig. 2c). Control II shows this is not a matter of taste: across the two
cell lines the naive residual is anti-correlated ($r = -0.290$) while the corrected score
is positively correlated ($r = +0.200$, $n = 104{,}196$). A target that disagrees with
itself across assays cannot support a relational claim; all Horlbeck results below use the
corrected score.

### 5.3 Mismeasurement III — curated priors do not improve with data

On Norman the graph stage fails its pre-registered test: over 5 splits × 3 seeds it reaches
residual $R^2 = 0.103$ against **0.274** with the stage removed, and a shuffled graph
scores 0.110 — higher than the real graph on every split tested (paired *t* $p = 0.06$,
$n = 3$). Learned relation weights stay uniform at ≈0.25 (Fig. 1b): the model never
differentiates the relations. Our mechanistic prediction is also falsified — physical PPI
is the strongest single relation (partial $r = +0.275$ controlling for effect magnitude)
despite the perturbed genes being predominantly TFs and regulatory having the widest
coverage (Fig. A1b).

At $n \approx 62$ such a null is uninterpretable: it cannot separate an uninformative prior
from an unreachable one. We therefore sweep training size on Horlbeck, 3 seeds, fixed
20,000-pair test set, reporting gains in held-out $R^2$ over an effect-magnitude control
(Fig. 2a). **Curated priors are flat in *n***: +0.0003 at $n = 2{,}000$ rising only to
+0.0014 at $n = 86{,}953$, sitting exactly on their shuffled control across three orders of
magnitude. The data-derived graph is flat until $n \approx 10^4$ and then scales — +0.0159
at $n = 60{,}000$ and **+0.0274** at $n = 86{,}953$, twenty times larger and well clear of
its own shuffled control. Coverage explains the asymmetry: curated relations reach 1.9%
(regulatory), 4.1% (physical PPI) and 5.6% (co-functional) of gene pairs, against 100% for
the inferred relation (Fig. 2b). Per-*n* values are in
`results/tables/horlbeck_learning_curve.csv`.

Norman alone could not license this reading. The prior was not merely unreachable at small
$n$: curated edges carry little transferable relational signal *at any $n$ we can reach*,
while inferred structure does — the quantitative form of GREmLN's argument [2].

## 6. Conclusion and limitations

Three measurements in routine use — expression-level correlation, the naive additive
residual, and curated graph priors — are each dominated by something other than
interaction, and each is caught by a control that costs almost nothing to run. What
survives all three is a cross-fitted linear estimator on the correctly-defined residual
plus inferred rather than curated structure: nothing here supports adding literature graph
priors to an interaction predictor. We suggest these controls become standard: report the additive baseline alongside any expression
metric, validate an interaction target against an independent assay, and pair any graph
prior with a shuffled control and a sample-size sweep.

**Limitations.** Norman is one cell type with 31 test doubles per split, and the 2,000-gene
subset captures 70% of training residual sum of squares. Horlbeck measures growth, not
expression, so failures (II) and (III) are established at scale but on a different read-out
from failure (I); extending the sweep to a large expression-based combinatorial screen is
the natural next step, and to our knowledge none exists at $10^5$ pairs. The data-derived
graph's advantage may partly reflect measurement noise shared between graph and target —
the shuffled control bounds this but does not exclude it, and a held-out-assay version of
the inferred graph would settle it. We do not extrapolate to drug-combination synergy.
All reported results take under 8 CPU-hours, no GPU.

## References

[1] C. Ahlmann-Eltze, W. Huber, S. Anders. Deep-learning-based gene perturbation effect
prediction does not yet outperform simple linear baselines. *Nature Methods*
22:1657–1661, 2025. doi:10.1038/s41592-025-02772-6

[2] GREmLN: A Cellular Graph Structure Aware Transcriptomics Foundation Model. bioRxiv,
doi:10.1101/2025.07.03.663009 (v3, 10 March 2026; corresponding author A. Califano).
Verified against the bioRxiv API.

[3] M. Horlbeck et al. Mapping the genetic landscape of human cells. *Cell* 174:953–967,
2018. GEO GSE116198.

---

## Appendix A — The nuisance projection

Constraining stage-2 outputs to $M^{\perp}$, where $M$ is the nuisance subspace, gives
$\tilde r_\theta = (I - QQ^{\top}) r_\theta$ for orthonormal $Q$. If the nuisance error
satisfies $\delta \in M$ then $\mathcal{R}(\theta;\delta) = \mathcal{R}(\theta;0) +
\mathbb{E}\lVert\delta\rVert^2$, so the risk minimiser is invariant to $\delta$.

Testing this requires holding the evaluation target fixed at the true residual and
degrading only the nuisance supplied to the model, $A_\rho = \bar A + \rho(A - \bar A)$;
degrading the target as well inverts the curve, because the target becomes the larger,
easier total signal. With the target fixed (Fig. 2a): at $\rho = 0.25$ the unprojected
estimator collapses to $R^2 = -0.89$ while the projected one holds at $-0.25$; at
$\rho = 0.5$, $-0.27$ versus $-0.07$. But at $\rho = 1$ projection costs accuracy, 0.086
versus 0.232. The projection is insurance whose premium is only worth paying when the
nuisance is poorly estimated; on these benchmarks singles are measured directly, so it is
not. We report it because the same reasoning applies wherever the additive null must be
*estimated* rather than measured.

## Appendix B — The failure mode is not specific to biology

In asset pricing the additive model is a factor model
$R_{it} = \alpha_i + \beta_i^\top f_t + e_{it}$, and the analogue of the interaction
residual is pairwise residual co-movement $r_{ij} = \mathbb{E}_t[e_{it}e_{jt}]$ — exactly
the quantity a factor model does not explain, and exactly the one that is invisible if you
score raw covariance. On 49 Ken French industry portfolios (2000–2026, 6,662 days, 1,176
pairs, FF5 betas from trailing 252-day windows, 281 out-of-sample months) the same
construction beats the factor model by +0.066 residual $R^2$ (82% of months,
$p < 10^{-17}$) and a trailing-covariance persistence baseline by +0.014 (57% of months,
Wilcoxon $p = 0.0015$), paired per month (Fig. 1d). Mismeasurement I is therefore
structural, not biological: wherever an additive model explains most of the variance,
scoring the total hides whether the interaction was learned.

---

## Main figures

**Figure 1. Mismeasurement I, and the instrument.** (**a**) Held-out Pearson for eight
published models plus the cross-fitted ridge, on total expression (light) and interaction
residual (dark); totals fall in 0.957–0.996 (CPA lowest), residuals span 0.000–0.488, mean
over 5 published splits. (**b**) Ablations, residual $R^2$ (mean ± s.d., 3 splits): removing
the graph stage is best, and a shuffled graph matches the real one. (**c**) Nested
out-of-fold $R^2$ for per-pair interaction magnitude, adding relation families over an
effect-magnitude control. (**d**) Appendix B: paired per-month gain in residual co-movement
$R^2$ over 281 out-of-sample months; error bars 95% CI of the paired mean, Wilcoxon
*p*-values.

**Figure 2. Mismeasurements II and III.** (**a**) Gain in held-out $R^2$ over an
effect-magnitude control versus training pairs (log scale); Horlbeck corrected interaction
score, 3 seeds, fixed 20,000-pair test set. Curated priors are flat in *n* and sit on their
shuffled control; the data-derived graph scales to +0.027. The shaded band marks the Norman
training regime ($n \approx 62$), where the distinction is invisible. (**b**) Fraction of
the 108,799 gene pairs each relation reaches. (**c**) 74% of the naive residual's variance
is an effect-size artifact.

**Appendix figure A1. Mechanism.** (**a**) Nuisance-degradation curve with the evaluation
target held fixed; projection slows degradation but costs accuracy at $\rho = 1$.
(**b**) Partial correlation of each relation with per-pair interaction magnitude on Norman,
controlling for effect magnitude, annotated with the fraction of the 122 observed
perturbation pairs each relation reaches. Physical PPI is strongest, falsifying our
pre-registered prediction that regulatory relations would dominate.
