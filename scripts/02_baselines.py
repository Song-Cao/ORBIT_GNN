#!/usr/bin/env python
"""S2 - Baselines on the interaction-residual metric.

Two families:
  (a) Free re-scoring of the eight published deep/foundation models. Their released
      predictions of y are converted to interaction predictions by subtracting the
      additive expectation: r_hat = y_hat - A. No training required.
  (b) Fitted baselines: zero (= the additive model), training-mean residual, and a
      cross-fitted ridge on pair features (kill condition 1 in the proposal).

Writes results/tables/baselines.csv with per-pair Pearson and bootstrap CIs.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbit.data import load_split, DATA                      # noqa: E402
from orbit.metrics import (per_pair_pearson, bootstrap_ci,    # noqa: E402
                           interaction_sign_accuracy, delta_pearson)

PUBLISHED = ["gears", "cpa", "scgpt", "scfoundation", "geneformer", "uce", "uce33", "scbert"]


def load(split):
    z = np.load(DATA / "processed" / f"split_{split}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def pair_index(perts, pg):
    gi = {g: i for i, g in enumerate(pg)}
    out = []
    for j, p in enumerate(perts):
        p = str(p)
        if p.endswith("+ctrl") or "+" not in p:
            continue
        a, b = p.split("+")
        if a in gi and b in gi:
            out.append((j, gi[a], gi[b], p))
    return out


def ridge_fit(X, Yt, alpha):
    """Closed-form multi-output ridge with intercept, no sklearn dependency."""
    Xc = np.hstack([np.ones((len(X), 1)), X])
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    A[0, 0] -= alpha
    return np.linalg.solve(A, Xc.T @ Yt)


def ridge_pred(W, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ W


def pair_features(S1, ia, ib):
    """Symmetric features from the two single-perturbation effect profiles."""
    za, zb = S1[:, ia].T, S1[:, ib].T
    return np.hstack([za + zb, za * zb, np.abs(za - zb)])


def main():
    pg = [l.strip() for l in open(DATA / "graphs" / "perturbed_genes.txt") if l.strip()]
    rows = []
    for split in range(1, 6):
        m = load(split)
        sp = load_split(split)
        perts = [str(p) for p in m["perts"]]
        pos = {p: i for i, p in enumerate(perts)}
        idx = pair_index(perts, pg)
        R = m["Y"] - m["A"]                                  # estimand, genes x perts

        # single-perturbation profiles restricted to the selected genes
        S1 = np.zeros((m["Y"].shape[0], len(pg)), dtype=np.float32)
        for k, g in enumerate(pg):
            key = f"{g}+ctrl"
            if key in pos:
                S1[:, k] = m["Y"][:, pos[key]]

        tr = [(j, a, b, p) for (j, a, b, p) in idx if p in set(map(str, np.atleast_1d(sp["train"])))]
        te = [(j, a, b, p) for (j, a, b, p) in idx if p in set(map(str, np.atleast_1d(sp["test"])))]
        if not te:
            continue
        Rtr = R[:, [j for j, *_ in tr]].T
        Rte = R[:, [j for j, *_ in te]].T
        Yte = m["Y"][:, [j for j, *_ in te]].T

        # ---- (a) published models, free ----
        for meth in PUBLISHED:
            key = "pred_" + meth
            if key not in m:
                continue
            Pte = m[key][:, [j for j, *_ in te]].T
            r_res = per_pair_pearson(Pte - m["A"][:, [j for j, *_ in te]].T, Rte)
            r_tot = per_pair_pearson(Pte, Yte)
            rows.append(dict(split=split, method=meth, family="published",
                             r_residual=np.nanmean(r_res), r_total=np.nanmean(r_tot),
                             sign_acc=interaction_sign_accuracy(
                                 Pte - m["A"][:, [j for j, *_ in te]].T, Rte),
                             n_pairs=len(te)))

        # ---- (b) fitted baselines ----
        # zero = the additive model itself
        rows.append(dict(split=split, method="additive (zero)", family="baseline",
                         r_residual=0.0, r_total=np.nanmean(per_pair_pearson(
                             m["A"][:, [j for j, *_ in te]].T, Yte)),
                         sign_acc=0.5, n_pairs=len(te)))
        # training-mean residual
        mean_r = np.tile(Rtr.mean(0), (len(te), 1))
        rows.append(dict(split=split, method="mean residual", family="baseline",
                         r_residual=np.nanmean(per_pair_pearson(mean_r, Rte)),
                         r_total=np.nan,
                         sign_acc=interaction_sign_accuracy(mean_r, Rte), n_pairs=len(te)))
        # cross-fitted ridge on pair features  (kill condition 1)
        Xtr = pair_features(S1, [a for _, a, _, _ in tr], [b for _, _, b, _ in tr])
        Xte = pair_features(S1, [a for _, a, _, _ in te], [b for _, _, b, _ in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        best, best_a = None, None
        for alpha in [1e1, 1e2, 1e3, 1e4]:
            n = len(Xtr); k = 5
            rng = np.random.default_rng(0); order = rng.permutation(n)
            oof = np.zeros_like(Rtr)
            for f in np.array_split(order, k):
                t = np.setdiff1d(order, f)
                W = ridge_fit((Xtr[t] - mu) / sd, Rtr[t], alpha)
                oof[f] = ridge_pred(W, (Xtr[f] - mu) / sd)
            sc = np.nanmean(per_pair_pearson(oof, Rtr))
            if best is None or sc > best:
                best, best_a = sc, alpha
        W = ridge_fit((Xtr - mu) / sd, Rtr, best_a)
        Pr = ridge_pred(W, (Xte - mu) / sd)
        rows.append(dict(split=split, method="ridge (cross-fitted)", family="baseline",
                         r_residual=np.nanmean(per_pair_pearson(Pr, Rte)),
                         r_total=np.nanmean(per_pair_pearson(Pr + m["A"][:, [j for j, *_ in te]].T, Yte)),
                         sign_acc=interaction_sign_accuracy(Pr, Rte),
                         n_pairs=len(te), alpha=best_a))
        print(f"split {split}: {len(te)} test doubles, ridge alpha={best_a:g}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "baselines_per_split.csv", index=False)
    agg = (df.groupby(["method", "family"])
             .agg(r_residual=("r_residual", "mean"), r_residual_sd=("r_residual", "std"),
                  r_total=("r_total", "mean"), sign_acc=("sign_acc", "mean"),
                  n_splits=("split", "nunique"))
             .reset_index().sort_values("r_residual", ascending=False))
    agg.to_csv(out / "baselines.csv", index=False)
    print("\n=== residual metric (mean over 5 splits) ===")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return agg


if __name__ == "__main__":
    main()
