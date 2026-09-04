"""Closed-form cross-fitted ridge backbone for the interaction residual.

ORBIT is a *two-stage* estimator of the same estimand:

  Stage 1 (this module). A linear map from symmetric gene-space pair features to the
  interaction residual, fitted in closed form with cross-fitted alpha selection. This
  is the strongest purely-linear predictor of the residual and it uses no graph.

  Stage 2 (``model.ORBIT``). A graph-gated low-rank bilinear correction fitted to the
  *out-of-fold* residual of stage 1. Because stage 1 is cross-fitted, stage 2 never sees
  in-sample stage-1 fits, so the correction cannot be an artefact of stage-1 overfitting.

Separating the stages makes the central question measurable: stage 2's increment over
stage 1 *is* the graph's contribution, which is exactly what kill condition 1 asks about.
"""
from __future__ import annotations
import numpy as np


def pair_features(X: np.ndarray, ia, ib) -> np.ndarray:
    """Symmetric gene-space features for perturbation pairs.

    ``X`` is (n_nodes, n_genes) single-perturbation effect profiles.
    Symmetry matters: the perturbation set {a, b} is unordered, so every block is
    invariant under swapping a and b.
    """
    xa, xb = X[ia], X[ib]
    return np.hstack([xa + xb, xa * xb, np.abs(xa - xb)]).astype(np.float32)


def _solve(Xc: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1], dtype=Xc.dtype)
    A[0, 0] -= alpha                       # do not penalise the intercept
    return np.linalg.solve(A, Xc.T @ Y)


class RidgeBackbone:
    """Multi-output ridge with K-fold cross-fitted alpha selection."""

    def __init__(self, alphas=(1e1, 1e2, 1e3, 1e4, 1e5), k: int = 5, seed: int = 0):
        self.alphas, self.k, self.seed = tuple(alphas), k, seed
        self.alpha_ = None
        self.W_ = None
        self.mu_ = None
        self.sd_ = None

    def _design(self, F):
        Z = (F - self.mu_) / self.sd_
        return np.hstack([np.ones((len(Z), 1), dtype=Z.dtype), Z])

    def fit(self, F: np.ndarray, Y: np.ndarray, score_fn=None):
        """Select alpha by K-fold CV, then refit on all training pairs.

        Also stores ``oof_`` — the out-of-fold predictions at the selected alpha. Stage 2
        trains on ``Y - oof_``, which is what makes the two-stage estimator cross-fitted.
        """
        self.mu_, self.sd_ = F.mean(0), F.std(0) + 1e-8
        Xc = self._design(F)
        n = len(F)
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(n), self.k)
        best, best_a, best_oof = -np.inf, None, None
        for a in self.alphas:
            oof = np.zeros_like(Y)
            for f in folds:
                t = np.setdiff1d(np.arange(n), f)
                W = _solve(Xc[t], Y[t], a)
                oof[f] = Xc[f] @ W
            sc = score_fn(oof, Y) if score_fn is not None else -np.mean((oof - Y) ** 2)
            if sc > best:
                best, best_a, best_oof = sc, a, oof
        self.alpha_, self.oof_, self.cv_score_ = best_a, best_oof, best
        self.W_ = _solve(Xc, Y, best_a)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        return self._design(F) @ self.W_
