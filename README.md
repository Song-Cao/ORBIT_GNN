# ORBIT — the interaction residual is the estimand

Code and results for the NeurIPS-workshop short paper
**"The Interaction Residual is the Estimand: Why Perturbation Models that Score 0.99
Tell You Nothing"** (`paper/ORBIT_workshop.md`).

## The claim in three numbers

| | |
|---|---|
| Additive model, Pearson *r* on total expression *y* | **0.995** |
| Fraction of the variance of *y* carried by the interaction term | **0.44%** |
| Best published deep model vs. a cross-fitted ridge, on the interaction residual | 0.417 vs **0.488** |

Reporting expression-level correlation for combinatorial perturbation models is
uninformative, because the additive expectation built from single-perturbation effects
already attains 0.995. Scoring the *interaction residual* instead — at no retraining cost,
using released predictions — inverts the published ranking.

## Headline results

1. **Re-scoring eight published models** (GEARS, CPA, scGPT, scFoundation, Geneformer,
   UCE, scBERT, UCE-33) on the residual: all score ~0.99 on *y*, spread over 0.16 on *r*,
   and all lose to a cross-fitted ridge (`results/tables/baselines.csv`).
2. **A pre-registered negative result.** Prior graphs carry real information about
   interaction magnitude (out-of-fold *R*² 0.394 → 0.452; physical PPI partial
   *r* = +0.275), but a multi-relational GNN on ~60 training pairs per split extracts none
   of it and ties a *shuffled* graph. The bottleneck is the number of held-out doubles
   (`results/tables/ablations.csv`, `graph_value_*.csv`).
3. **Domain transfer.** The same estimand shift on Fama–French factor residuals beats
   persistence over 281 out-of-sample months (+0.014 *R*², *p* = 0.0015) and the factor
   model itself (+0.066, 82% of months) (`results/tables/finance_paired.csv`).

## Reproduce

No GPU. Under 6 CPU-hours end to end on 8 cores.

```bash
pip install -r requirements.txt

Rscript scripts/01_prepare_data.R        # pseudobulk, splits, graphs  (needs the Zenodo release)
python  scripts/02_baselines.py          # re-score the 8 published models  -> baselines.csv
python  scripts/04_experiments.py --stage main       # 5 splits x 3 seeds
python  scripts/04_experiments.py --stage ablations  # incl. shuffled-graph control
python  scripts/07_graph_value.py        # capacity-controlled test of the prior
python  scripts/08_mechanism.py          # nuisance-degradation curve
python  scripts/05_finance_data.py       # Ken French panel
python  scripts/06_finance_experiment.py # finance arm
pytest  tests/                           # projection + leakage assertions
```

## Data

| source | what | access |
|---|---|---|
| Zenodo `10.5281/zenodo.16092690` | Norman double-perturbation benchmark: ground truth, additive model, 8 methods' predictions, 5 splits | script |
| STRING v12 | physical PPI (`escore`) and co-functional (`score`) channels | API |
| DoRothEA, TRRUST, ENCODE, ITFP, TRED, Neph2012, Marbach2016 | TF→target regulons (union: 3,127 TFs) | GitHub releases |
| Ken French Data Library | 49 industry portfolios + FF5 factors, daily | script |

Splits are the published ones, used verbatim, so numbers are directly comparable to the
benchmark.

## Layout

```
src/orbit/        model.py (encoder, bilinear head, nuisance projection)
                  backbone.py (cross-fitted ridge, pair features)
                  orbit.py (two-stage estimator)
                  data.py, metrics.py
scripts/          01-08, one concern each, all CLI-driven
tests/            projection algebra + train/test leakage assertions
results/tables/   every number in the paper
results/figures/  fig1_main.png, fig2_mechanism.png
docs/             PROPOSAL.md, PLAN.md
paper/            ORBIT_workshop.md
```

## Honest limitations

One dataset, one cell type, 31 test doubles per split. The 2,000-gene subset captures 70%
of training residual sum of squares. The finance arm uses industry portfolios rather than
single names. scLong and 2026-era Arc/Xaira models are not evaluated — they are absent
from the release and retraining exceeds this scope. The nuisance projection bounds
sensitivity to nuisance error but costs accuracy when the nuisance is well estimated,
which on this benchmark it is.
