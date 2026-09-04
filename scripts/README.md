# Scripts

Run from the repository root, in order. No GPU; under 6 CPU-hours total.

| script | what |
|---|---|
| `00_download.py` | benchmark release (Zenodo 16092690) + regulon sources |
| `01_prepare_data.R` | pseudobulk matrices, splits verbatim, regulatory + profile graphs, QC assertions |
| `01b_to_npz.py` | TSV → compressed npz, asserts label alignment and no surviving NAs |
| `01c_string.py` | STRING physical (`escore`) and co-functional (`score`) channels |
| `02_baselines.py` | re-score the 8 published models on the residual; fitted baselines |
| `04_experiments.py` | `--stage {gate,main,mechanism,ablations}` — the two-stage estimator |
| `07_graph_value.py` | capacity-controlled test of whether the prior carries signal |
| `08_mechanism.py` | nuisance-degradation curve with the evaluation target held fixed |
| `05_finance_data.py` | Ken French 49 industry portfolios + FF5 factors |
| `06_finance_experiment.py` | walk-forward finance arm, 281 out-of-sample months |

## Not used in the paper

`03_train.py` — the original single-stage end-to-end trainer. **Superseded** by the
two-stage estimator in `src/orbit/orbit.py`, driven by `04_experiments.py`. It reached
residual r = 0.025 against the ridge's 0.488; the diagnosis of why (output scale, and
19,264-dim node features on 62 training pairs) is in `PROJECT_LOG.md` under S3/S4. Kept
because the ablation flags it defines (`--no_projection`, `--shuffle_graph`,
`--target_total`) document the configurations we tested, and `--target_total` is the
capacity-matched "train on y instead of r" control.

`08_mechanism.py` supersedes `04_experiments.py --stage mechanism`, which is confounded
(see `results/tables/README.md`).
