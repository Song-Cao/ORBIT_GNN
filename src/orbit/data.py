"""Data loading for ORBIT.

All arrays come from the published benchmark release (Zenodo 16092690,
DOI 10.5281/zenodo.16092690), converted once by scripts/01_prepare_data.R.
Splits are used verbatim so numbers are directly comparable to the published benchmark.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"


def load_split(split: int) -> dict:
    """Perturbation label lists for one of the five published splits."""
    with open(DATA / "splits" / f"split_{split}.json") as f:
        d = json.load(f)
    for k in ("train", "test", "val"):
        d[k] = [str(x) for x in np.atleast_1d(d[k])]
    return d


def load_matrices(split: int) -> dict:
    """Y (ground truth), A (additive null), baseline predictions; genes x perturbations."""
    z = np.load(DATA / "processed" / f"split_{split}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _row_normalise(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    A = np.abs(A)
    s = A.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return A / s


def load_relations(split: int, which=("regulatory", "ppi", "cofunctional", "profile")) -> tuple:
    """Return (list of row-normalised dense adjacencies, relation names).

    - regulatory   : union of DoRothEA/TRRUST/ENCODE/ITFP/TRED/Neph/Marbach TF->target,
                     scored by target-set Jaccard overlap or direct regulation
    - ppi          : STRING experimental channel (physical interaction)
    - cofunctional : STRING combined score
    - profile      : single-perturbation effect-profile correlation, computed from
                     TRAINING singles only for this split (leakage guard)
    """
    import pandas as pd
    out, names = [], []
    for w in which:
        if w == "regulatory":
            M = pd.read_csv(DATA / "graphs" / "regulatory_matrix.csv", index_col=0).values
        elif w in ("ppi", "cofunctional"):
            z = np.load(DATA / "graphs" / "string.npz", allow_pickle=True)
            M = z["escore"] if w == "ppi" else z["score"]
        elif w == "profile":
            M = pd.read_csv(DATA / "graphs" / f"profilesim_split{split}.csv", index_col=0).values
        else:
            raise ValueError(f"unknown relation {w!r}")
        out.append(_row_normalise(np.asarray(M, dtype=np.float32)))
        names.append(w)
    return out, names


class ResidualDataset:
    """Held-out double perturbations with the interaction residual as target.

    The target is r = Y - A where A is the additive expectation built from
    single-perturbation effects fitted on the training folds only. This is the estimand:
    predicting r is not predicting y.
    """

    def __init__(self, split: int, subset: str = "train", genes_subset: np.ndarray | None = None):
        sp = load_split(split)
        m = load_matrices(split)
        perts = [str(p) for p in m["perts"]]
        pos = {p: i for i, p in enumerate(perts)}
        pg = [l.strip() for l in open(DATA / "graphs" / "perturbed_genes.txt") if l.strip()]
        gi = {g: i for i, g in enumerate(pg)}

        # doubles only, and only those whose both constituent genes were measured singly
        keep, ia, ib = [], [], []
        for p in sp[subset]:
            if p.endswith("+ctrl") or "+" not in p:
                continue
            a, b = p.split("+")
            if a in gi and b in gi and p in pos:
                keep.append(pos[p]); ia.append(gi[a]); ib.append(gi[b])
        self.idx = np.array(keep, dtype=int)
        self.ia = np.array(ia, dtype=int)
        self.ib = np.array(ib, dtype=int)
        self.perts = [perts[i] for i in self.idx]

        gsub = slice(None) if genes_subset is None else genes_subset
        self.Y = m["Y"][gsub][:, self.idx].T.astype(np.float32)   # (n_pairs, n_genes)
        self.A = m["A"][gsub][:, self.idx].T.astype(np.float32)
        self.R = self.Y - self.A                                   # the estimand
        self.baselines = {k[5:]: m[k][gsub][:, self.idx].T.astype(np.float32)
                          for k in m.files if k.startswith("pred_")} if hasattr(m, "files") else {}

    def __len__(self):
        return len(self.idx)
