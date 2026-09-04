#!/usr/bin/env python
"""S6 - Nuisance-degradation curve: a correct test of Proposition 1.

Proposition 1 claims that when the model output is constrained to the orthogonal
complement of the nuisance subspace M, the population risk decomposes as
    R(theta; delta) = R(theta; 0) + E[delta^2]
for nuisance error delta lying in M. The argmin over theta is therefore *independent* of
the nuisance error: degrading the nuisance estimate should shift the risk by a constant
without moving the solution.

Testing this requires care. A naive sweep that shrinks the nuisance also changes the
regression target, so held-out R^2 rises simply because the target becomes the (much
larger, easier) total signal. That measures the target, not the projection.

Correct protocol, used here:
  - the TARGET is always the true interaction residual r = y - A (rho = 1). It never
    changes, so R^2 values are comparable across the sweep.
  - only the nuisance estimate *supplied to the model* is degraded:
        A_rho = mean(A) + rho * (A - mean(A))
    which is used to build the training target the model actually fits and the basis of
    the projected subspace. The evaluation target stays fixed.
  - we compare projection on vs off at each rho. Proposition 1 predicts the projected
    model's held-out performance is flat in rho, while the unprojected model degrades as
    the nuisance error grows.

A flat projected curve and a declining unprojected curve is the mechanism claim. Anything
else falsifies it, and we report whichever we observe.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from orbit.backbone import RidgeBackbone, pair_features        # noqa: E402
from orbit.orbit import per_pair_pearson, residual_r2, nuisance_basis  # noqa: E402
import torch                                                   # noqa: E402
from orbit.model import NuisanceProjection                     # noqa: E402

TAB = ROOT / "results" / "tables"
RHOS = [1.0, 0.75, 0.5, 0.25, 0.0]


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ex", ROOT / "scripts" / "04_experiments.py")
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

    rows = []
    for split in range(1, 6):
        d = ex.prep(split)                       # rho = 1: the truth
        X = d["X"]
        F = {k: pair_features(X, [a for a, _ in v], [b for _, b in v])
             for k, v in [("tr", d["p_tr"]), ("te", d["p_te"])]}
        R_tr_true, R_te_true = d["R_tr"], d["R_te"]      # FIXED evaluation target
        A_tr, A_te = d["A_tr"], d["A_te"]
        mu_tr, mu_te = A_tr.mean(0, keepdims=True), A_te.mean(0, keepdims=True)
        sc = lambda P, O: float(np.nanmean(per_pair_pearson(P, O)))

        for rho in RHOS:
            # degrade only the nuisance the MODEL is given
            A_tr_r = mu_tr + rho * (A_tr - mu_tr)
            A_te_r = mu_te + rho * (A_te - mu_te)
            # the target the model fits, under its degraded nuisance
            y_tr = (R_tr_true + A_tr - A_tr_r).astype(np.float32)
            delta_tr = (A_tr - A_tr_r)                   # the nuisance error, in span(M)

            for project in (True, False):
                # alpha fixed at the value selected by cross-fitting in S2/S4, so the
                # sweep varies only the nuisance and not the regularisation strength
                rb = RidgeBackbone(alphas=(1e3,), k=2, seed=0).fit(F["tr"], y_tr, score_fn=sc)
                p = rb.predict(F["te"])
                if project:
                    P = NuisanceProjection(nuisance_basis(A_tr_r))
                    p = P(torch.from_numpy(p.astype(np.float32))).numpy()
                # score against the FIXED true residual
                rows.append(dict(split=split, rho=rho, project=project,
                                 r2=residual_r2(p, R_te_true),
                                 pearson=sc(p, R_te_true),
                                 delta_rms=float(np.sqrt((delta_tr ** 2).mean()))))
        print(f"  split {split} done", flush=True)

    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "mechanism_corrected.csv", index=False)
    piv = df.pivot_table(index="rho", columns="project", values="r2", aggfunc="mean")
    piv.columns = ["no projection", "projection"]
    print("\n=== held-out R^2 against the FIXED true residual ===")
    print(piv.round(4).to_string())
    d0 = df[df.project].groupby("rho").r2.mean()
    d1 = df[~df.project].groupby("rho").r2.mean()
    print(f"\nspread across rho:  projection {d0.max()-d0.min():+.4f}   "
          f"no projection {d1.max()-d1.min():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
