#!/usr/bin/env python
"""S9b - The finance arm: same estimand shift, different domain.

Task. For each held-out month and each industry pair (i, j), predict the *residual
co-movement* r_{ij} = mean_t(e_it * e_jt) over that month, where e is the residual of a
point-in-time Fama-French 5-factor regression estimated on a trailing window that ends
before the test month. This is the finance analogue of the interaction residual: what is
left after the additive (factor) model has taken its share.

Why this is the same problem. Raw pairwise covariance is dominated by shared factor
exposure, just as double-perturbation expression is dominated by single-gene main
effects. A model scored on raw covariance looks excellent while being uninformative
about the interaction; the estimand shift is what exposes the difference.

Design (strictly out-of-sample, no look-ahead):
  - rolling: fit factor betas on a 252-day trailing window, form residuals on the next
    21 trading days, aggregate to a per-pair target.
  - the "graph" is the same four-relation construction as biology, transferred:
      sector      : same Fama-French industry group (analogue of the regulatory prior)
      beta        : similarity of trailing factor loadings (analogue of co-functional)
      corr        : trailing residual correlation (analogue of profile similarity)
      vol         : trailing volatility product (the effect-magnitude control)
  - baselines: zero (the factor model itself), trailing residual covariance (persistence),
    and cross-fitted ridge on the same pair features -- the direct analogue of the
    biology arm's stage 1.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbit.backbone import RidgeBackbone                      # noqa: E402

RAW = ROOT / "data" / "raw" / "finance"
TAB = ROOT / "results" / "tables"
WIN, HOR = 252, 21


def load():
    exc = pd.read_parquet(RAW / "industry_excess_returns.parquet")
    fac = pd.read_parquet(RAW / "factors.parquet")
    idx = exc.index.intersection(fac.index)
    return exc.loc[idx], fac.loc[idx]


def factor_residuals(R: np.ndarray, F: np.ndarray):
    """OLS of each asset on the factors; returns residuals and betas."""
    X = np.hstack([np.ones((len(F), 1)), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    return R - X @ B, B


def build_windows(exc, fac):
    """Rolling out-of-sample windows: betas from the trailing year, target from the
    following month. No information from the target month enters the features."""
    R, F = exc.values, fac.values
    n = R.shape[1]
    iu, ju = np.triu_indices(n, 1)
    out = []
    for t0 in range(WIN, len(R) - HOR, HOR):
        Rtr, Ftr = R[t0 - WIN:t0], F[t0 - WIN:t0]
        Rte, Fte = R[t0:t0 + HOR], F[t0:t0 + HOR]
        e_tr, B = factor_residuals(Rtr, Ftr)
        # test residuals use the TRAILING betas: strictly out of sample
        Xte = np.hstack([np.ones((len(Fte), 1)), Fte])
        e_te = Rte - Xte @ B
        # target: realised residual co-movement over the test month
        y = (e_te[:, iu] * e_te[:, ju]).mean(0)
        # features, all from the trailing window only
        Ctr = np.corrcoef(e_tr.T)
        vol = e_tr.std(0)
        beta = B[1:].T                                    # (n_assets, n_factors)
        bsim = beta @ beta.T / (np.linalg.norm(beta, axis=1)[:, None] *
                                np.linalg.norm(beta, axis=1)[None, :] + 1e-12)
        cov_tr = (e_tr[:, iu] * e_tr[:, ju]).mean(0)      # persistence baseline
        out.append(dict(date=exc.index[t0], y=y, corr=Ctr[iu, ju], bsim=bsim[iu, ju],
                        volprod=vol[iu] * vol[ju], cov_tr=cov_tr, iu=iu, ju=ju))
    return out


def pair_features(w, sector_same):
    """Same four relation families as the biology arm, transferred to finance."""
    return np.column_stack([
        sector_same,          # sector prior      <-> regulatory prior
        w["bsim"],            # factor-loading similarity <-> co-functional
        w["corr"],            # trailing residual corr    <-> profile similarity
        w["volprod"],         # effect-magnitude control  <-> effect magnitude
        w["cov_tr"],          # persistence
        w["corr"] * w["volprod"],
    ]).astype(np.float32)


def r2(p, o):
    return float(1 - ((p - o) ** 2).sum() / (o ** 2).sum())


def main():
    exc, fac = load()
    ws = build_windows(exc, fac)
    print(f"windows: {len(ws)}  pairs per window: {len(ws[0]['y'])}")

    # sector prior: Ken French industry names grouped into broad sectors
    names = list(exc.columns)
    groups = {
        "cyclical": ["Autos", "BldMt", "Cnstr", "Rubbr", "Steel", "Mach", "ElcEq", "Aero",
                     "Ships", "Guns", "Gold", "Mines", "Coal", "FabPr", "LabEq", "Paper",
                     "Boxes", "Txtls", "Chems"],
        "tech":     ["Hardw", "Softw", "Chips", "LabEq", "Telcm", "BusSv"],
        "health":   ["Drugs", "MedEq", "Hlth"],
        "consumer": ["Food", "Soda", "Beer", "Smoke", "Toys", "Fun", "Books", "Hshld",
                     "Clths", "Meals", "Rtail", "Whlsl", "PerSv"],
        "energy":   ["Oil", "Enrgy"],
        "finance":  ["Banks", "Insur", "RlEst", "Fin"],
        "utils":    ["Util"],
        "transport":["Trans"],
    }
    g_of = {}
    for g, members in groups.items():
        for m in members:
            g_of.setdefault(m, g)
    grp = np.array([g_of.get(n, "other") for n in names])
    iu, ju = ws[0]["iu"], ws[0]["ju"]
    sector_same = (grp[iu] == grp[ju]).astype(np.float32)
    print("same-sector pairs:", int(sector_same.sum()), "/", len(sector_same),
          f"({100*sector_same.mean():.0f}%)")

    # walk-forward: train on all windows strictly before the test window
    rows = []
    n_tr_min = 24
    for k in range(n_tr_min, len(ws)):
        Xtr = np.vstack([pair_features(ws[j], sector_same) for j in range(k - n_tr_min, k)])
        ytr = np.concatenate([ws[j]["y"] for j in range(k - n_tr_min, k)])
        Xte = pair_features(ws[k], sector_same)
        yte = ws[k]["y"]
        # standardise the target scale per window set (returns are in percent)
        rb = RidgeBackbone(alphas=(1e0, 1e1, 1e2, 1e3, 1e4)).fit(
            Xtr, ytr[:, None].astype(np.float32))
        p_ridge = rb.predict(Xte).ravel()
        rows.append(dict(
            date=ws[k]["date"],
            zero=r2(np.zeros_like(yte), yte),                    # the factor model itself
            persistence=r2(ws[k]["cov_tr"], yte),                # trailing covariance
            ridge=r2(p_ridge, yte),
            # raw-covariance framing: score the SAME predictions on total covariance
            raw_zero=r2(np.zeros_like(yte), yte + 0),            # placeholder, filled below
            alpha=rb.alpha_))
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "finance_per_window.csv", index=False)
    agg = df[["zero", "persistence", "ridge"]].mean()
    print("\n=== finance arm, residual co-movement R^2 (mean over "
          f"{len(df)} out-of-sample months) ===")
    for k, v in agg.items():
        print(f"  {k:14s} {v:+.4f}")
    df[["zero", "persistence", "ridge"]].agg(["mean", "std"]).to_csv(
        TAB / "finance_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
