#!/usr/bin/env python
"""S7b - Does relational structure carry information about the interaction residual?

The main experiment shows a full GNN stage *degrades* held-out performance. That could
mean either (i) the graph carries no information, or (ii) it carries information but a
GNN cannot extract it from ~60 training pairs. Those have very different implications,
so we separate them with a capacity-controlled test.

Protocol. Predict a *scalar* per-pair quantity -- the magnitude of the interaction
residual, ||r_pair|| -- from a handful of graph-derived scalar features, using ridge with
nested cross-validated regularisation. Six features and n ~ 190 pairs pooled over splits
is a regime where a linear model is well-posed, so a null result here is evidence about
the *information*, not about optimiser capacity.

Feature blocks, added cumulatively:
  effect magnitude  : ||x_a|| + ||x_b||         (the confounder; always included)
  + regulatory      : TF-target overlap between the perturbed genes
  + physical PPI    : STRING experimental channel
  + co-functional   : STRING combined score
  + profile         : training-fold effect-profile correlation

Reports out-of-fold R^2 for each nested model, plus partial correlations of each relation
with ||r_pair|| controlling for effect magnitude.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from orbit.data import DATA                                    # noqa: E402

TAB = ROOT / "results" / "tables"


def load_graphs(split):
    REG = pd.read_csv(DATA / "graphs" / "regulatory_matrix.csv", index_col=0).values
    Z = np.load(DATA / "graphs" / "string.npz", allow_pickle=True)
    PS = pd.read_csv(DATA / "graphs" / f"profilesim_split{split}.csv", index_col=0).values
    return REG, Z["escore"], Z["score"], PS


def ridge_oof_r2(X, y, k=5, seed=0, alphas=np.logspace(-3, 4, 12)):
    """Out-of-fold R^2 with the penalty chosen inside each training fold."""
    n = len(X)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    oof = np.zeros(n)
    for f in folds:
        t = np.setdiff1d(np.arange(n), f)
        mu, sd = X[t].mean(0), X[t].std(0) + 1e-9
        Xt = np.hstack([np.ones((len(t), 1)), (X[t] - mu) / sd])
        Xf = np.hstack([np.ones((len(f), 1)), (X[f] - mu) / sd])
        # inner leave-one-out selection of alpha on the training fold
        best, bw = np.inf, None
        for a in alphas:
            A = Xt.T @ Xt + a * np.eye(Xt.shape[1]); A[0, 0] -= a
            w = np.linalg.solve(A, Xt.T @ y[t])
            H = Xt @ np.linalg.solve(A, Xt.T)
            r = y[t] - Xt @ w
            loo = np.mean((r / (1 - np.clip(np.diag(H), 0, 1 - 1e-6))) ** 2)
            if loo < best:
                best, bw = loo, w
        oof[f] = Xf @ bw
    return float(1 - ((oof - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def partial_corr(x, y, z):
    """Correlation of x and y after linearly removing z from both."""
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ex", ROOT / "scripts" / "04_experiments.py")
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

    feats, ys = [], []
    for split in range(1, 6):
        d = ex.prep(split)
        REG, PPI, COF, PS = load_graphs(split)
        X = d["X"]
        norms = np.linalg.norm(X, axis=1)
        # pool TRAIN pairs across splits (test pairs are never used here)
        for (a, b), r in zip(d["p_tr"], d["R_tr"]):
            feats.append([norms[a] + norms[b], REG[a, b], PPI[a, b], COF[a, b], PS[a, b]])
            ys.append(np.linalg.norm(r))
    F = np.array(feats, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    # de-duplicate pairs that recur across splits
    _, uniq = np.unique(F.round(8), axis=0, return_index=True)
    F, y = F[np.sort(uniq)], y[np.sort(uniq)]
    print(f"unique training pairs pooled over 5 splits: {len(F)}")

    blocks = [("effect magnitude", [0]),
              ("+ regulatory", [0, 1]),
              ("+ physical PPI", [0, 1, 2]),
              ("+ co-functional", [0, 1, 2, 3]),
              ("+ profile sim.", [0, 1, 2, 3, 4])]
    rows = []
    for name, cols in blocks:
        r2 = ridge_oof_r2(F[:, cols], y)
        rows.append(dict(model=name, n_features=len(cols), oof_r2=r2))
        print(f"  {name:20s} OOF R^2 = {r2:+.4f}")

    print("\npartial correlation with ||r_pair||, controlling for effect magnitude:")
    pcs = []
    for j, nm in [(1, "regulatory"), (2, "physical PPI"), (3, "co-functional"), (4, "profile sim.")]:
        pc = partial_corr(F[:, j], y, F[:, [0]])
        cov = float((F[:, j] != 0).mean())
        pcs.append(dict(relation=nm, partial_r=pc, pair_coverage=cov))
        print(f"  {nm:16s} partial r = {pc:+.3f}   coverage = {100*cov:.0f}% of pairs")

    TAB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TAB / "graph_value_nested.csv", index=False)
    pd.DataFrame(pcs).to_csv(TAB / "graph_value_partial.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
