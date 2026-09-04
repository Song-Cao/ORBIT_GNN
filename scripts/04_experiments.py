#!/usr/bin/env python
"""S4-S7 - Gate, main benchmark, mechanism curve and ablations.

One entry point so every compared configuration shares the identical pipeline.

    python scripts/04_experiments.py --stage gate
    python scripts/04_experiments.py --stage main       # 5 splits x 3 seeds
    python scripts/04_experiments.py --stage mechanism  # nuisance-degradation curve
    python scripts/04_experiments.py --stage ablations
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbit.data import load_split, load_relations, DATA          # noqa: E402
from orbit.orbit import ORBITEstimator, per_pair_pearson, residual_r2  # noqa: E402

TAB = ROOT / "results" / "tables"
RUNS = ROOT / "results" / "runs"
ALL_RELS = ("regulatory", "ppi", "cofunctional", "profile")


def prep(split, relations=ALL_RELS, shrink_tau=1.0, shuffle_graph=False, seed=0):
    """Assemble one split. ``shrink_tau`` scales the additive nuisance toward zero,
    which is the nuisance-degradation manipulation of experiment S6."""
    z = np.load(DATA / "processed" / f"split_{split}.npz", allow_pickle=True)
    m = {k: z[k] for k in z.files}
    sp = load_split(split)
    perts = [str(p) for p in m["perts"]]
    pos = {p: i for i, p in enumerate(perts)}
    pg = [l.strip() for l in open(DATA / "graphs" / "perturbed_genes.txt") if l.strip()]
    gi = {g: i for i, g in enumerate(pg)}

    # degrade the nuisance estimate: A -> mean + shrink * (A - mean)
    A = m["A"].astype(np.float64)
    if shrink_tau != 1.0:
        mu = A.mean(1, keepdims=True)
        A = mu + shrink_tau * (A - mu)
    R = (m["Y"] - A).astype(np.float32)

    X = np.zeros((len(pg), m["Y"].shape[0]), dtype=np.float32)
    for k, g in enumerate(pg):
        key = f"{g}+ctrl"
        if key in pos:
            X[k] = m["Y"][:, pos[key]]

    def pairs(labels):
        idx, ab = [], []
        for p in labels:
            p = str(p)
            if "+" not in p or p.endswith("+ctrl") or p not in pos:
                continue
            a, b = p.split("+")
            if a in gi and b in gi:
                idx.append(pos[p]); ab.append((gi[a], gi[b]))
        return idx, ab

    itr, abtr = pairs(sp["train"]); iva, abva = pairs(sp["val"]); ite, abte = pairs(sp["test"])
    adjs, names = load_relations(split, relations)
    if shuffle_graph:
        rng = np.random.default_rng(seed)
        adjs = [A_[rng.permutation(A_.shape[0])][:, rng.permutation(A_.shape[1])] for A_ in adjs]
    return dict(X=X, adjs=adjs, rel_names=names,
                A_tr=A[:, itr].T.astype(np.float32),
                R_tr=R[:, itr].T, R_va=R[:, iva].T, R_te=R[:, ite].T,
                Y_te=m["Y"][:, ite].T, A_te=A[:, ite].T.astype(np.float32),
                p_tr=abtr, p_va=abva, p_te=abte,
                labels_te=[perts[i] for i in ite], m=m, ite=ite)


def run_one(split, seed=0, relations=ALL_RELS, use_graph=True, project=True,
            shrink_tau=1.0, shuffle_graph=False, global_scale=True, tag=""):
    d = prep(split, relations, shrink_tau, shuffle_graph, seed)
    est = ORBITEstimator(relations=relations, use_graph=use_graph, project=project,
                         seed=seed, global_scale=global_scale)
    t0 = time.time()
    out = est.fit_predict(d["X"], d["adjs"], d["A_tr"], d["p_tr"], d["p_va"], d["p_te"],
                          d["R_tr"], d["R_va"], d["R_te"])
    row = dict(tag=tag, split=split, seed=seed, relations="+".join(relations),
               use_graph=use_graph, project=project, shrink_tau=shrink_tau,
               shuffle_graph=shuffle_graph, global_scale=global_scale,
               alpha=out["alpha"], scale=out["global_scale"],
               s1_pearson=out["stage1"]["pearson"], s1_r2=out["stage1"]["r2"],
               pearson=out["final"]["pearson"], r2=out["final"]["r2"],
               epochs=out.get("epochs", 0), wall_s=round(time.time() - t0, 1))
    if out.get("relation_weights"):
        for n, w in zip(d["rel_names"], out["relation_weights"]):
            row["w_" + n] = w
    if "pred" in out and tag in ("main", "gate"):
        RUNS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RUNS / f"pred_{tag}_s{split}_seed{seed}.npz",
                            pred=out["pred"], obs=d["R_te"],
                            pairs=np.array(d["labels_te"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gate", "main", "mechanism", "ablations"])
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    rows = []

    if a.stage == "gate":
        # kill condition 1: does the graph stage beat the no-graph ridge on split 1?
        rows.append(run_one(1, 0, use_graph=False, tag="gate"))
        rows.append(run_one(1, 0, use_graph=True, tag="gate"))
        df = pd.DataFrame(rows); df.to_csv(TAB / "gate.csv", index=False)
        print(df[["use_graph", "s1_pearson", "s1_r2", "pearson", "r2"]].to_string(index=False))

    elif a.stage == "main":
        for s in range(1, 6):
            for sd in range(3):
                rows.append(run_one(s, sd, use_graph=True, tag="main"))
                print(f"  split {s} seed {sd}: r2={rows[-1]['r2']:+.4f}", flush=True)
            rows.append(run_one(s, 0, use_graph=False, tag="ridge"))
        df = pd.DataFrame(rows); df.to_csv(TAB / "main_results.csv", index=False)
        g = df.groupby("use_graph").agg(pearson=("pearson", "mean"), r2=("r2", "mean"))
        print(g.to_string())

    elif a.stage == "mechanism":
        # S6: shrink the nuisance estimate and watch held-out risk
        for rho in [1.0, 0.75, 0.5, 0.25, 0.0]:
            for proj in [True, False]:
                r = run_one(1, 0, project=proj, shrink_tau=rho, tag="mechanism")
                rows.append(r)
                print(f"  rho={rho:4.2f} project={proj!s:5s} r2={r['r2']:+.4f}", flush=True)
        df = pd.DataFrame(rows); df.to_csv(TAB / "mechanism.csv", index=False)

    elif a.stage == "ablations":
        cfgs = [
            ("all relations", dict(relations=ALL_RELS)),
            ("no graph (ridge only)", dict(use_graph=False)),
            ("shuffled graph", dict(shuffle_graph=True)),
            ("regulatory only", dict(relations=("regulatory",))),
            ("ppi only", dict(relations=("ppi",))),
            ("cofunctional only", dict(relations=("cofunctional",))),
            ("profile only", dict(relations=("profile",))),
            ("no projection", dict(project=False)),
            ("no global scale", dict(global_scale=False)),
        ]
        for name, kw in cfgs:
            for s in (1, 2, 3):
                r = run_one(s, 0, tag="ablation", **kw); r["config"] = name
                rows.append(r)
            print(f"  {name:24s} r2={np.mean([x['r2'] for x in rows[-3:]]):+.4f}", flush=True)
        df = pd.DataFrame(rows); df.to_csv(TAB / "ablations.csv", index=False)
        print(df.groupby("config").agg(r2=("r2", "mean"), pearson=("pearson", "mean"))
                .sort_values("r2", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
