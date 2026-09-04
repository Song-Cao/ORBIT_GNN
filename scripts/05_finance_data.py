#!/usr/bin/env python
"""S9a - Build the finance panel for the cross-domain transfer.

The finance arm tests the *same estimand* as the biology arm: predict what is left after
removing additive effects, rather than the raw outcome.

  Biology : y_{S,c} = mu_c + sum_{a in S} tau_{a,c} + r_{S,c}
  Finance : R_{i,t} = alpha_i + beta_i' f_t + e_{i,t},  and the estimand is the pairwise
            interaction  r_{ij,t} = e_{i,t} * e_{j,t}  -- residual co-movement that the
            factor model does not explain.

Both are "the part the additive model cannot reach", and in both the additive part
dominates the variance, which is exactly why the estimand shift matters.

Data (Ken French Data Library, Dartmouth):
  - 49 Industry Portfolios, daily value-weighted returns  -> the cross-section
  - Fama-French 5 factors + RF, daily                     -> the additive nuisance

Industry portfolios rather than single stocks: no survivorship bias, no corporate-action
handling, stable identifiers over the full sample, and a well-defined relation graph.
"""
from __future__ import annotations
import io, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "finance"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


def get_zip_csv(name: str) -> str:
    b = urllib.request.urlopen(f"{BASE}/{name}", timeout=180).read()
    z = zipfile.ZipFile(io.BytesIO(b))
    return z.read(z.namelist()[0]).decode("latin-1")


def parse_ff_block(txt: str, start_hint: str = None) -> pd.DataFrame:
    """Ken French CSVs carry prose headers and several stacked blocks. Take the first
    block of 8-digit-dated rows, which is the daily value-weighted series."""
    rows, cols = [], None
    for line in txt.split("\n"):
        p = [x.strip() for x in line.split(",")]
        if cols is None:
            if len(p) > 3 and p[0] == "" and any(c and not c[0].isdigit() for c in p[1:]):
                cols = [c for c in p[1:] if c]
            continue
        if p[0][:8].isdigit() and len(p[0]) == 8:
            rows.append([p[0]] + p[1:len(cols) + 1])
        elif rows:
            break                                    # first block only
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").replace([-99.99, -999], np.nan)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    ind = parse_ff_block(get_zip_csv("49_Industry_Portfolios_daily_CSV.zip"))
    fac = parse_ff_block(get_zip_csv("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"))
    print("industries:", ind.shape, ind.index.min().date(), "->", ind.index.max().date())
    print("factors:   ", fac.shape, list(fac.columns))

    # modern sample: post-2000, both series complete
    idx = ind.index.intersection(fac.index)
    idx = idx[idx >= "2000-01-03"]
    ind, fac = ind.loc[idx], fac.loc[idx]
    ind = ind.dropna(axis=1)                         # keep fully observed industries
    # excess returns
    rf = fac["RF"]
    exc = ind.sub(rf, axis=0)
    exc.to_parquet(RAW / "industry_excess_returns.parquet")
    fac[[c for c in fac.columns if c != "RF"]].to_parquet(RAW / "factors.parquet")
    print(f"\npanel: {exc.shape[0]} days x {exc.shape[1]} industries "
          f"({exc.index.min().date()} to {exc.index.max().date()})")
    print("n pairs:", exc.shape[1] * (exc.shape[1] - 1) // 2)
    print("mean |excess return| (pct/day):", round(float(exc.abs().mean().mean()), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
