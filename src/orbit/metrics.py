"""Evaluation metrics for ORBIT.

Primary metric (pre-registered, proposal Sec. 9): Pearson correlation between predicted
and observed *interaction residual* on held-out doubles, computed per perturbation pair
and aggregated with a paired bootstrap over pairs.
"""
from __future__ import annotations
import numpy as np


def per_pair_pearson(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Pearson r per row (perturbation pair). pred/obs: (n_pairs, n_genes)."""
    p = pred - pred.mean(1, keepdims=True)
    o = obs - obs.mean(1, keepdims=True)
    num = (p * o).sum(1)
    den = np.sqrt((p ** 2).sum(1) * (o ** 2).sum(1))
    den[den == 0] = np.nan
    return num / den


def delta_pearson(pred: np.ndarray, obs: np.ndarray) -> float:
    """Pooled Pearson over all gene x pair entries."""
    return float(np.corrcoef(pred.ravel(), obs.ravel())[0, 1])


def interaction_sign_accuracy(pred: np.ndarray, obs: np.ndarray, q: float = 0.9) -> float:
    """Sign agreement on the entries where the true interaction is large.

    Restricting to the top-|r| decile avoids scoring sign on numerical noise, where sign
    is undefined in practice.
    """
    thr = np.quantile(np.abs(obs), q)
    m = np.abs(obs) >= thr
    if m.sum() == 0:
        return np.nan
    return float((np.sign(pred[m]) == np.sign(obs[m])).mean())


def auprc_nonadditive(pred: np.ndarray, obs: np.ndarray, q: float = 0.9) -> float:
    """AUPRC for retrieving strongly non-additive gene x pair entries."""
    from sklearn.metrics import average_precision_score
    y = (np.abs(obs) >= np.quantile(np.abs(obs), q)).astype(int).ravel()
    s = np.abs(pred).ravel()
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap mean and CI over independent units (perturbation pairs)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Paired bootstrap on a - b over shared units. Returns mean diff, CI and p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    rng = np.random.default_rng(seed)
    n = len(a)
    d = a - b
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "p": float(min(p, 1.0)), "n": int(n)}
