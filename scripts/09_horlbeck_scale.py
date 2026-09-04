#!/usr/bin/env python
"""S9b - The scaling diagnostic: does relational structure help as n grows, and does it
matter whether the structure is CURATED or DATA-DERIVED?

Norman gives ~62 training pairs, which cannot separate "the prior is uninformative" from
"the estimator cannot reach it". Horlbeck 2018 (GSE116198) gives 108,799 gene pairs, so
we sweep training size across three orders of magnitude and compare:
  - curated priors      (regulatory union, STRING physical + co-functional)
  - data-derived graph  (interaction-profile similarity, built from TRAINING pairs only)
  - shuffled controls   (both, to separate signal from added capacity)

Target is the CORRECTED genetic-interaction score, not the naive residual: in growth
screens the sum null is biased for strong-effect genes and ~74% of the naive residual is
that artifact (see --report-artifact).

Outputs: results/tables/horlbeck_learning_curve.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ridge_fit(X, y, alpha):
    Xm, ym = X.mean(0), y.mean()
    Xc = X - Xm
    b = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return b, ym - Xm @ b


def r2(yt, yp):
    return 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


def best_r2(Xtr, ytr, Xte, yte, alphas=(1e-2, 1, 10, 100, 1e3, 1e4)):
    out = []
    for a in alphas:
        b, c = ridge_fit(Xtr, ytr, a)
        out.append(r2(yte, Xte @ b + c))
    return max(out)


def profile_sim(gi, ia, ib, tr_idx, ng, min_obs=5):
    """Interaction-profile similarity between genes, from TRAINING pairs only.

    This is the data-derived relation: each gene's vector of measured interactions with
    other genes, correlated between genes. Leak-free because only training pairs enter.
    """
    Pm = np.zeros((ng, ng), np.float32)
    Cn = np.zeros((ng, ng), np.float32)
    Pm[ia[tr_idx], ib[tr_idx]] = gi[tr_idx]
    Pm[ib[tr_idx], ia[tr_idx]] = gi[tr_idx]
    Cn[ia[tr_idx], ib[tr_idx]] = 1
    Cn[ib[tr_idx], ia[tr_idx]] = 1
    obs = Cn.sum(1) >= min_obs
    ok = np.where(obs)[0]
    V = np.where(Cn[obs] > 0, Pm[obs], 0.0)
    Vn = V - V.mean(1, keepdims=True)
    den = np.linalg.norm(Vn, axis=1, keepdims=True)
    den[den == 0] = 1
    C = np.zeros((ng, ng), np.float32)
    C[np.ix_(ok, ok)] = (Vn / den) @ (Vn / den).T
    return C[ia, ib].reshape(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_test", type=int, default=20000)
    args = ap.parse_args()

    d = np.load(ROOT / "data/processed/horlbeck_gi.npz", allow_pickle=True)
    gi, F_eff, F_str, F_reg = d["gi_J"], d["F_eff"], d["F_str"], d["F_reg"]
    ia, ib, genes = d["ia"], d["ib"], d["genes"]
    ng = len(genes)
    F_cur = np.hstack([F_eff, F_str, F_reg])          # effect magnitude + curated priors

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pm = rng.permutation(len(gi))
        te, pool = pm[: args.n_test], pm[args.n_test:]
        for n in [200, 600, 2000, 6000, 20000, 60000, len(pool)]:
            tr = pool[:n]
            ps = profile_sim(gi, ia, ib, tr, ng)
            ps_sh = ps.copy(); rng.shuffle(ps_sh)
            F_sh = F_str.copy(); rng.shuffle(F_sh)
            cfg = {
                "effect magnitude":          F_eff,
                "+ literature graph":        F_cur,
                "+ literature (shuffled)":   np.hstack([F_eff, F_sh, F_reg]),
                "+ data-derived graph":      np.hstack([F_cur, ps]),
                "+ data-derived (shuffled)": np.hstack([F_cur, ps_sh]),
            }
            for name, F in cfg.items():
                rows.append(dict(seed=seed, n=n, config=name,
                                 r2=best_r2(F[tr], gi[tr], F[te], gi[te])))
            print(f"  seed {seed} n {n} done")

    L = pd.DataFrame(rows)
    out = ROOT / "results/tables/horlbeck_learning_curve.csv"
    L.to_csv(out, index=False)
    piv = L.groupby(["n", "config"]).r2.mean().unstack()
    base = piv["effect magnitude"]
    print("\ngain in held-out R^2 over the effect-magnitude control:")
    print(pd.DataFrame({
        "curated":      piv["+ literature graph"] - base,
        "data-derived": piv["+ data-derived graph"] - base,
        "shuffled":     piv["+ data-derived (shuffled)"] - base,
    }).round(4).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
