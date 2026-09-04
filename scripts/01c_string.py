#!/usr/bin/env python
"""S1c - Fetch the STRING relation channels for the perturbed genes.

Two channels are used as separate relations: `escore` (experimental, i.e. physical
interaction evidence) and `score` (combined, i.e. co-functional association). STRING
occasionally returns 504 under load, so the request is retried with backoff.
"""
import time, urllib.parse, urllib.request
from pathlib import Path
import numpy as np

G = Path(__file__).resolve().parents[1] / "data" / "graphs"
pg = [l.strip() for l in open(G / "perturbed_genes.txt") if l.strip()]
q = urllib.parse.urlencode({"identifiers": "%0d".join(pg), "species": 9606,
                            "caller_identity": "orbit_s1_graphs"})
url = "https://string-db.org/api/tsv/network?" + q
raw = None
for k in range(5):
    try:
        raw = urllib.request.urlopen(url, timeout=240).read().decode()
        break
    except Exception as e:
        print(f"  attempt {k}: {type(e).__name__} {getattr(e, 'code', '')}")
        time.sleep(15 + 10 * k)
if raw is None:
    raise SystemExit("STRING unreachable after 5 attempts")

rows = [l.split("\t") for l in raw.strip().split("\n")]
ci = {c: i for i, c in enumerate(rows[0])}
idx = {g: i for i, g in enumerate(pg)}
n = len(pg)
CH = {k: np.zeros((n, n), dtype=np.float32) for k in ["score", "escore", "ascore", "tscore"]}
for r in rows[1:]:
    a, b = r[ci["preferredName_A"]], r[ci["preferredName_B"]]
    if a in idx and b in idx:
        for k, M in CH.items():
            v = float(r[ci[k]])
            M[idx[a], idx[b]] = M[idx[b], idx[a]] = v
ut = np.triu_indices(n, 1)
for k, M in CH.items():
    print(f"  {k:8s} {int((M[ut] > 0).sum()):4d} edges ({(M[ut] > 0).mean()*100:.0f}% of pairs)")
np.savez(G / "string.npz", genes=np.array(pg), **CH)
print("saved string.npz")
