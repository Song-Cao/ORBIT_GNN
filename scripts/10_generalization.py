#!/usr/bin/env python
"""S10 - The two experiments that decide whether the scaling result means anything.

Experiment A (COLD START). Warm-start evaluation holds out random gene *pairs*, so every
test gene has already been screened against ~370 partners. Target discovery does not work
that way: you ask about a gene you have not screened. We therefore hold out whole GENES
and test pairs involving them. This is also the fairest possible test of curated priors --
in cold start the inferred graph is undefined by construction, so literature edges are the
only relational information available.

Experiment B (CROSS-CELL-LINE TRANSFER). The inferred graph is built from the same
measurements as the target, so its advantage could be shared measurement noise. Building
the graph in Jurkat and evaluating in K562 removes that confound: shared noise cannot
transfer across independent screens, but real interaction structure can.

Outputs: results/tables/generalization.csv, results/tables/gi_spectrum.csv
"""
import argparse, gzip
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/horlbeck/GSE116198_sgRNA_pair_phenotypes.txt.gz"


# ---------------------------------------------------------------- utilities
def ridge_fit(X, y, alpha):
    Xm, ym = X.mean(0), y.mean()
    Xc = X - Xm
    b = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return b, ym - Xm @ b


def r2(yt, yp):
    return 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


def best_r2(Xtr, ytr, Xte, yte, alphas=(1e-2, 1, 10, 100, 1e3, 1e4)):
    return max(r2(yte, Xte @ (bc := ridge_fit(Xtr, ytr, a))[0] + bc[1]) for a in alphas)


# ---------------------------------------------------------------- data
def singles_and_pairs():
    """Gene-pair phenotypes and single-gene effects, both cell lines.

    Single-gene effect = gene paired with a non-targeting control. Using gene-paired-with-
    itself instead is a double knockdown and inflates the residual past the observed
    variance -- a trap we fell into on the first pass.
    """
    df = pd.read_csv(RAW, sep="\t", skiprows=4, header=None, low_memory=False)
    sp = df[0].str.split(r"\+\+", regex=True)
    ga = sp.str[0].map(lambda x: x.split("_")[0])
    gb = sp.str[1].map(lambda x: x.split("_")[0])
    d = pd.DataFrame({"ga": ga, "gb": gb, "J": df[9].astype(float), "K": df[18].astype(float)})
    d = d.dropna(subset=["ga", "gb"])
    neg = lambda s: s.str.lower().eq("negative")
    s = d[neg(d.ga) ^ neg(d.gb)].copy()
    s["g"] = np.where(neg(s.ga), s.gb, s.ga)
    S = s.groupby("g")[["J", "K"]].median()
    S = S.drop(index=[i for i in S.index if str(i).lower() == "negative"], errors="ignore")
    key = [tuple(sorted(t)) for t in zip(d.ga, d.gb)]
    d["A"] = [k[0] for k in key]
    d["B"] = [k[1] for k in key]
    gp = d[(d.A != d.B) & ~neg(d.A) & ~neg(d.B)].groupby(["A", "B"])[["J", "K"]].median().reset_index()
    return gp[gp.A.isin(S.index) & gp.B.isin(S.index)].reset_index(drop=True), S


def corrected_gi(obs, sa, sb):
    """Standard GI correction: the sum null is biased for strong-effect genes, so regress
    out a smooth function of the expected phenotype. 74% of the naive residual is this."""
    m = ~(np.isnan(obs) | np.isnan(sa) | np.isnan(sb))
    e = (sa + sb).astype(np.float64)
    D = np.column_stack([e, e ** 2, e ** 3, sa * sb, np.abs(sa - sb)])
    resid = obs - e
    b, c = ridge_fit(D[m], resid[m], 1e-6)
    out = np.full(len(obs), np.nan)
    out[m] = resid[m] - (D[m] @ b + c)
    return out


def eff_features(sa, sb):
    return np.column_stack([sa + sb, sa * sb, np.abs(sa - sb),
                            np.abs(sa) + np.abs(sb), np.minimum(sa, sb), np.maximum(sa, sb)])


# ---------------------------------------------------------------- inferred structure
def profile_sim_feat(gi, ia, ib, tr, ng, min_obs=5):
    """Interaction-profile similarity between genes, from TRAINING pairs only."""
    P = np.zeros((ng, ng), np.float32); C = np.zeros((ng, ng), np.float32)
    P[ia[tr], ib[tr]] = gi[tr]; P[ib[tr], ia[tr]] = gi[tr]
    C[ia[tr], ib[tr]] = 1;      C[ib[tr], ia[tr]] = 1
    obs = C.sum(1) >= min_obs
    ok = np.where(obs)[0]
    V = np.where(C[obs] > 0, P[obs], 0.0)
    Vn = V - V.mean(1, keepdims=True)
    den = np.linalg.norm(Vn, axis=1, keepdims=True); den[den == 0] = 1
    M = np.zeros((ng, ng), np.float32)
    M[np.ix_(ok, ok)] = (Vn / den) @ (Vn / den).T
    return M[ia, ib].reshape(-1, 1), M


def lowrank_feat(gi, ia, ib, tr, ng, k=16, iters=25):
    """Rank-k symmetric completion of the interaction matrix from training entries.

    The explicit model behind profile similarity: if the GI matrix is low-rank, held-out
    entries are predictable from a gene's latent factors. Undefined for genes with no
    training interactions -- which is exactly why cold start is hard.
    """
    G = np.zeros((ng, ng), np.float32); msk = np.zeros((ng, ng), bool)
    G[ia[tr], ib[tr]] = gi[tr]; G[ib[tr], ia[tr]] = gi[tr]
    msk[ia[tr], ib[tr]] = True; msk[ib[tr], ia[tr]] = True
    X = np.where(msk, G, 0.0).astype(np.float64)
    Xk = X
    for _ in range(iters):
        w, V = np.linalg.eigh(X)
        idx = np.argsort(-np.abs(w))[:k]
        Xk = (V[:, idx] * w[idx]) @ V[:, idx].T
        X = np.where(msk, G, Xk)
    return Xk[ia, ib].reshape(-1, 1)


def spectrum(gi, ia, ib, ng):
    G = np.zeros((ng, ng)); G[ia, ib] = gi; G[ib, ia] = gi
    w = np.abs(np.linalg.eigvalsh(G))[::-1]
    p = w / w.sum()
    return w, float(np.exp(-(p * np.log(p + 1e-300)).sum()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()

    gp, S = singles_and_pairs()
    genes = sorted(set(gp.A) | set(gp.B))
    gi_idx = {g: i for i, g in enumerate(genes)}
    ng = len(genes)
    ia = gp.A.map(gi_idx).values
    ib = gp.B.map(gi_idx).values

    st = np.load(ROOT / "data/graphs/horlbeck_string.npz", allow_pickle=True)
    sgl = list(st["genes"]); si = {g: i for i, g in enumerate(sgl)}
    sa_i = np.array([si.get(g, -1) for g in gp.A]); sb_i = np.array([si.get(g, -1) for g in gp.B])
    F_str = np.column_stack([np.where((sa_i >= 0) & (sb_i >= 0),
                                      st[c][np.clip(sa_i, 0, None), np.clip(sb_i, 0, None)], 0.0)
                             for c in ["escore", "score", "ascore", "tscore", "dscore"]])
    RG = pd.read_csv(ROOT / "data/graphs/horlbeck_regulatory.csv", index_col=0)
    ri = {g: i for i, g in enumerate(RG.index)}; RM = RG.values.astype(np.float32)
    F_reg = np.array([RM[ri[x], ri[y]] if (x in ri and y in ri) else 0.0
                      for x, y in zip(gp.A, gp.B)], np.float32).reshape(-1, 1)
    CUR = np.hstack([F_str, F_reg])

    lines = {}
    for cl in ["J", "K"]:
        sa = S[cl].reindex(gp.A).values.astype(float)
        sb = S[cl].reindex(gp.B).values.astype(float)
        lines[cl] = dict(gi=corrected_gi(gp[cl].values.astype(float), sa, sb),
                         eff=eff_features(sa, sb))

    w, erank = spectrum(np.nan_to_num(lines["J"]["gi"]), ia, ib, ng)
    pd.DataFrame({"index": np.arange(1, ng + 1), "abs_eigenvalue": w}).to_csv(
        ROOT / "results/tables/gi_spectrum.csv", index=False)
    print(f"GI matrix ({ng}x{ng}): effective rank {erank:.1f} of {ng}; "
          f"top-16 eigenvalues carry {w[:16].sum()/w.sum():.1%} of spectral mass\n")

    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)

        # ---- Experiment A: warm vs cold, within Jurkat
        gi = lines["J"]["gi"]; EFF = lines["J"]["eff"]
        val = ~np.isnan(gi)
        for regime in ["warm", "cold"]:
            if regime == "warm":
                idx = np.where(val)[0]; pm = rng.permutation(idx)
                te, tr = pm[:20000], pm[20000:]
            else:
                held = set(rng.choice(ng, size=int(0.2 * ng), replace=False))
                inv = np.isin(ia, list(held)) | np.isin(ib, list(held))
                tr = np.where(val & ~inv)[0]; te = np.where(val & inv)[0]
            ps, _ = profile_sim_feat(gi, ia, ib, tr, ng)
            lr = lowrank_feat(gi, ia, ib, tr, ng, k=a.rank)
            cfg = {"effect magnitude": EFF,
                   "+ curated graph": np.hstack([EFF, CUR]),
                   "+ inferred graph": np.hstack([EFF, ps]),
                   "+ low-rank completion": np.hstack([EFF, lr]),
                   "+ curated & inferred": np.hstack([EFF, CUR, ps, lr])}
            for nm, F in cfg.items():
                rows.append(dict(seed=seed, experiment=f"A:{regime}", method=nm,
                                 n_train=len(tr), n_test=len(te),
                                 r2=best_r2(F[tr], gi[tr], F[te], gi[te])))

        # ---- Experiment B: graph from Jurkat, target K562
        giK = lines["K"]["gi"]; EFFK = lines["K"]["eff"]
        vb = ~np.isnan(giK) & ~np.isnan(gi)
        idx = np.where(vb)[0]; pm = rng.permutation(idx)
        te, tr = pm[:20000], pm[20000:]
        ps_own, _ = profile_sim_feat(giK, ia, ib, tr, ng)      # K562 structure (same assay)
        lr_own = lowrank_feat(giK, ia, ib, tr, ng, k=a.rank)
        ps_x, _ = profile_sim_feat(gi, ia, ib, tr, ng)          # Jurkat structure (transfer)
        lr_x = lowrank_feat(gi, ia, ib, tr, ng, k=a.rank)
        cfg = {"effect magnitude": EFFK,
               "+ curated graph": np.hstack([EFFK, CUR]),
               "+ inferred (same assay)": np.hstack([EFFK, ps_own, lr_own]),
               "+ inferred (transferred)": np.hstack([EFFK, ps_x, lr_x])}
        for nm, F in cfg.items():
            rows.append(dict(seed=seed, experiment="B:K562", method=nm,
                             n_train=len(tr), n_test=len(te),
                             r2=best_r2(F[tr], giK[tr], F[te], giK[te])))
        print(f"  seed {seed} done")

    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results/tables/generalization.csv", index=False)
    for exp in R.experiment.unique():
        sub = R[R.experiment == exp]
        g = sub.groupby("method").r2.agg(["mean", "std"])
        base = g.loc["effect magnitude", "mean"]
        g["gain"] = g["mean"] - base
        print(f"\n=== {exp}   (train {sub.n_train.iloc[0]:,} / test {sub.n_test.iloc[0]:,}) ===")
        print(g.round(4).to_string())
    print(f"\nwrote results/tables/generalization.csv")


if __name__ == "__main__":
    main()
