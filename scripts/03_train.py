#!/usr/bin/env python
"""S3/S4/S5 - Train ORBIT on the interaction residual.

Usage:
    python scripts/03_train.py --split 1 --seed 0 [--variant name] [ablation flags]

Every ablation is a flag on this one entry point so the compared runs share the exact
training loop. Predictions are always saved per pair, not only aggregates.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbit.model import ORBIT                                   # noqa: E402
from orbit.data import load_split, load_relations, DATA         # noqa: E402
from orbit.metrics import per_pair_pearson                      # noqa: E402

RELS = ("regulatory", "ppi", "cofunctional", "profile")


def build(split, relations=RELS):
    z = np.load(DATA / "processed" / f"split_{split}.npz", allow_pickle=True)
    m = {k: z[k] for k in z.files}
    sp = load_split(split)
    perts = [str(p) for p in m["perts"]]
    pos = {p: i for i, p in enumerate(perts)}
    pg = [l.strip() for l in open(DATA / "graphs" / "perturbed_genes.txt") if l.strip()]
    gi = {g: i for i, g in enumerate(pg)}

    def pairs(labels):
        out = []
        for p in labels:
            p = str(p)
            if "+" not in p or p.endswith("+ctrl") or p not in pos:
                continue
            a, b = p.split("+")
            if a in gi and b in gi:
                out.append((pos[p], gi[a], gi[b], p))
        return out

    R = m["Y"] - m["A"]                                # the estimand, genes x perts
    # node features: single-perturbation effect profile of each perturbed gene.
    # Available for every split because all 100 singles are training data.
    X = np.zeros((len(pg), m["Y"].shape[0]), dtype=np.float32)
    for k, g in enumerate(pg):
        key = f"{g}+ctrl"
        if key in pos:
            X[k] = m["Y"][:, pos[key]]
    adjs, names = load_relations(split, relations)
    return dict(m=m, R=R, X=X, adjs=adjs, rel_names=names, pg=pg,
                tr=pairs(sp["train"]), va=pairs(sp["val"]), te=pairs(sp["test"]))


def nuisance_basis(d, idx):
    """Basis of the nuisance subspace M, in gene space.

    Columns: (1) global mean direction, (2) mean single-perturbation effect direction,
    (3-4) leading PCs of the additive predictions on training pairs. These are the
    directions in which nuisance-estimation error lives; the model output is projected
    onto their orthogonal complement.
    """
    A = d["m"]["A"][:, [j for j, *_ in idx]].T          # (n_pairs, n_genes)
    cols = [np.ones(A.shape[1]), A.mean(0)]
    Ac = A - A.mean(0)
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    cols += [Vt[0], Vt[1]]
    B = np.stack(cols, 1).astype(np.float32)
    return torch.from_numpy(B)


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rels = tuple(args.relations.split(",")) if args.relations else RELS
    d = build(args.split, rels)
    ng = d["R"].shape[0]
    tr, va, te = d["tr"], d["va"], d["te"]

    X = torch.from_numpy(d["X"])
    adjs = [torch.from_numpy(A) for A in d["adjs"]]
    if args.shuffle_graph:                              # ablation: destroy graph structure
        g = torch.Generator().manual_seed(args.seed)
        adjs = [A[torch.randperm(A.shape[0], generator=g)][:, torch.randperm(A.shape[1], generator=g)]
                for A in adjs]
    if args.no_graph:                                   # ablation: identity only
        adjs = [torch.eye(X.shape[0]) for _ in adjs]

    basis = None if args.no_projection else nuisance_basis(d, tr)
    model = ORBIT(n_nodes=X.shape[0], d_in=X.shape[1], n_out=ng, nuisance_basis=basis,
                  d_hidden=args.hidden, rank=args.rank, n_relations=len(adjs),
                  n_layers=args.layers, dropout=args.dropout, gate=not args.no_gate,
                  project=not args.no_projection)

    def tensors(idx):
        ia = torch.tensor([a for _, a, _, _ in idx])
        ib = torch.tensor([b for _, _, b, _ in idx])
        if args.target_total:                           # ablation: predict y, not r
            tgt = d["m"]["Y"][:, [j for j, *_ in idx]].T
        else:
            tgt = d["R"][:, [j for j, *_ in idx]].T
        return ia, ib, torch.from_numpy(np.ascontiguousarray(tgt))

    ia_tr, ib_tr, y_tr = tensors(tr)
    ia_va, ib_va, y_va = tensors(va)
    ia_te, ib_te, y_te = tensors(te)
    # observed residual on test is always the estimand, whatever the training target
    r_te = torch.from_numpy(np.ascontiguousarray(d["R"][:, [j for j, *_ in te]].T))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state, patience = -np.inf, None, 0
    t0 = time.time()
    n = len(tr)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            out, g = model(X, adjs, ia_tr[b], ib_tr[b])
            loss = nn.functional.mse_loss(out, y_tr[b])
            if g is not None and args.l1_gate > 0:
                loss = loss + args.l1_gate * g.abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv, _ = model(X, adjs, ia_va, ib_va)
            sv = float(np.nanmean(per_pair_pearson(pv.numpy(), y_va.numpy())))
        if sv > best:
            best, best_state, patience = sv, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt, gt_ = model(X, adjs, ia_te, ib_te)
        pred = pt.numpy()
        # if trained on y, convert to an interaction prediction for scoring
        if args.target_total:
            pred = pred - d["m"]["A"][:, [j for j, *_ in te]].T
        r = per_pair_pearson(pred, r_te.numpy())
        relw = model.enc.relation_weights.detach().numpy().tolist()

    res = dict(variant=args.variant, split=args.split, seed=args.seed,
               r_residual=float(np.nanmean(r)), val_best=float(best),
               epochs_run=ep + 1, wall_s=round(time.time() - t0, 1),
               n_train=len(tr), n_test=len(te),
               relations=list(d["rel_names"]), relation_weights=relw,
               n_params=sum(p.numel() for p in model.parameters()))
    outd = ROOT / "results" / "runs"
    outd.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_s{args.split}_seed{args.seed}"
    np.savez_compressed(outd / f"pred_{tag}.npz", pred=pred, obs=r_te.numpy(),
                        per_pair_r=r, pairs=np.array([p for *_, p in te]))
    with open(outd / f"res_{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    return res


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="orbit")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--l1_gate", type=float, default=1e-3)
    p.add_argument("--relations", default="")
    # ablation switches
    p.add_argument("--no_projection", action="store_true")
    p.add_argument("--no_gate", action="store_true")
    p.add_argument("--no_graph", action="store_true")
    p.add_argument("--shuffle_graph", action="store_true")
    p.add_argument("--target_total", action="store_true",
                   help="train on y instead of the residual (capacity-matched control)")
    return p.parse_args()


if __name__ == "__main__":
    run(cli())
