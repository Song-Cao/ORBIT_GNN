#!/usr/bin/env python
"""S1b - Convert the per-split TSVs written by 01_prepare_data.R into compressed npz,
then delete the TSVs. Asserts every method aligns to the ground-truth column order and
that no NAs survive the label canonicalisation."""
import glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / "data" / "processed"
METHS = ["cpa", "gears", "geneformer", "scbert", "scfoundation", "scgpt", "uce", "uce33"]

genes = [l.strip() for l in open(P / "genes_selected.txt") if l.strip()]
for s in range(1, 6):
    Y = pd.read_csv(P / f"Y_split{s}.tsv", sep="\t", index_col=0)
    A = pd.read_csv(P / f"A_split{s}.tsv", sep="\t", index_col=0)
    assert list(Y.columns) == list(A.columns), f"split {s}: Y/A column mismatch"
    d = {"Y": Y.values.astype(np.float32), "A": A.values.astype(np.float32),
         "perts": np.array(list(Y.columns)), "genes": np.array(genes)}
    for m in METHS:
        f = P / f"pred_{m}_split{s}.tsv"
        if not f.exists():
            continue
        M = pd.read_csv(f, sep="\t", index_col=0)
        assert list(M.columns) == list(Y.columns), f"split {s}/{m}: column mismatch"
        assert not M.isna().any().any(), f"split {s}/{m}: NAs survived canonicalisation"
        d["pred_" + m] = M.values.astype(np.float32)
    np.savez_compressed(P / f"split_{s}.npz", **d)
    si = [i for i, p in enumerate(Y.columns) if p.endswith("+ctrl")]
    rms = float(np.sqrt(((d["Y"][:, si] - d["A"][:, si]) ** 2).mean()))
    print(f"split {s}: {Y.shape[0]} genes x {Y.shape[1]} perts, "
          f"{sum(k.startswith('pred_') for k in d)} methods, additive RMS on singles {rms:.3g}")
    assert rms < 1e-8

for f in glob.glob(str(P / "*.tsv")):
    os.remove(f)
print("npz written, TSVs removed")
