# Results tables

Every number in `paper/ORBIT_workshop.md` comes from a file here.

| file | what | used in |
|---|---|---|
| `baselines.csv` | 8 published models + fitted baselines on the interaction residual and on total expression, mean over 5 splits | Table in §2, Fig. 1a |
| `baselines_per_split.csv` | the same, per split | — |
| `main_results.csv` | two-stage estimator, 5 splits × 3 seeds, with/without the graph stage | §3, Fig. 1b |
| `ablations.csv` | 9 configurations × 3 splits incl. the shuffled-graph control | §3, Fig. 1b |
| `graph_vs_shuffled.csv` | paired test, real vs shuffled vs no graph | §3 |
| `graph_value_nested.csv` | nested out-of-fold R² for predicting per-pair interaction magnitude | §4, Fig. 1c |
| `graph_value_partial.csv` | partial correlation of each relation with ‖r_ab‖, effect magnitude controlled | §4, Fig. 2b |
| `mechanism_corrected.csv` | nuisance-degradation curve, evaluation target held fixed | §5, Fig. 2a |
| `finance_per_window.csv` | per-month R² for each finance baseline, 281 out-of-sample months | §6 |
| `finance_paired.csv` | paired per-month comparisons with Wilcoxon p | §6, Fig. 1d |
| `gate.csv` | the S4 gate run (kill condition 1) | §3 |

## Superseded

`mechanism_SUPERSEDED_confounded.csv` — the first nuisance-degradation sweep. **Do not
cite.** It varied ρ in the evaluation target as well as in the nuisance supplied to the
model, so R² rose with degradation simply because the target became the larger, easier
total signal. Kept for transparency; `mechanism_corrected.csv` is the reported result.
