#!/usr/bin/env python
"""S0 - Fetch the benchmark release and the regulon sources. Idempotent."""
import json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

WANT = {"double_perturbation_results_predictions.RDS",
        "double_perturbation_results_parameters.RDS"}
rec = json.load(urllib.request.urlopen("https://zenodo.org/api/records/16092690", timeout=60))
print("Zenodo record 16092690, DOI", rec["doi"], "-", len(rec["files"]), "files")
for f in rec["files"]:
    if f["key"] in WANT and not os.path.exists(f["key"]):
        print("  downloading", f["key"], f"({f['size']/1e6:.0f} MB)")
        urllib.request.urlretrieve(f["links"]["self"], f["key"])

REGULONS = {
    "dorothea_hs.rda": "https://raw.githubusercontent.com/saezlab/dorothea/master/data/dorothea_hs.rda",
    "tftargets.rda":   "https://raw.githubusercontent.com/slowkow/tftargets/master/data/tftargets.rda",
}
for name, url in REGULONS.items():
    p = RAW / name
    if not p.exists():
        print("  downloading", name)
        urllib.request.urlretrieve(url, p)
print("done")
