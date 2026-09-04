"""ORBIT: two-stage estimator of the cross-fitted interaction residual.

Design follows the measured structure of the estimand (see results/tables/probe.csv):

  Stage 1 -- shape.  A cross-fitted multi-output ridge maps symmetric gene-space pair
  features to the interaction residual. This recovers the *direction* of the residual in
  gene space and needs no graph. It is also the kill-condition-1 baseline: whatever
  stage 2 adds on top is the graph's contribution, by construction.

  Stage 2 -- magnitude.  A multi-relational GNN predicts a per-pair scalar gain applied
  to the stage-1 prediction. This is the quantity the graph demonstrably carries: prior
  edges predict per-pair interaction *strength* (partial r = +0.33 for physical PPI,
  controlling for effect magnitude), not the gene-level shape. Stage 2 is trained against
  stage 1's *out-of-fold* predictions, so the correction cannot absorb stage-1
  overfitting.

Both stages predict the residual r = y - (mu + tau_a + tau_b), never the total outcome y,
and the stage-2 output is projected onto the orthogonal complement of the nuisance
subspace (Proposition 1).

Metric note. Per-pair Pearson is scale-invariant, so it is blind to magnitude by
construction and cannot register stage 2. We therefore report Pearson (shape) *and*
residual R^2 (magnitude-sensitive) as co-primary, which is the honest way to expose what
each stage does.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .backbone import RidgeBackbone, pair_features
from .model import MultiRelationalEncoder, NuisanceProjection


def nuisance_basis(A_train: np.ndarray) -> torch.Tensor:
    """Basis of the nuisance subspace M in gene space.

    Columns: the constant direction, the mean additive-prediction direction, and the two
    leading principal directions of the additive predictions on training pairs. Nuisance
    (mu, tau) estimation error lives in this span; the model output is projected onto its
    orthogonal complement so that error cannot enter the interaction prediction.
    """
    cols = [np.ones(A_train.shape[1]), A_train.mean(0)]
    _, _, Vt = np.linalg.svd(A_train - A_train.mean(0), full_matrices=False)
    cols += [Vt[0], Vt[1]]
    return torch.from_numpy(np.stack(cols, 1).astype(np.float32))


def per_pair_pearson(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    p = pred - pred.mean(1, keepdims=True)
    o = obs - obs.mean(1, keepdims=True)
    den = np.sqrt((p ** 2).sum(1) * (o ** 2).sum(1))
    den[den == 0] = np.nan
    return (p * o).sum(1) / den


def residual_r2(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(1.0 - ((pred - obs) ** 2).sum() / (obs ** 2).sum())


class GainNet(nn.Module):
    """Multi-relational GNN predicting a per-pair scalar gain on the stage-1 prediction."""

    def __init__(self, n_nodes, d_in, n_relations, d_hidden=32, n_layers=2, dropout=0.1):
        super().__init__()
        self.enc = MultiRelationalEncoder(n_nodes, d_in, d_hidden, n_relations,
                                          n_layers, dropout)
        self.head = nn.Sequential(nn.Linear(3 * d_hidden, d_hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_hidden, 1))

    @property
    def relation_weights(self):
        return self.enc.relation_weights

    def forward(self, x, adjs, ia, ib):
        z = self.enc(x, adjs)
        za, zb = z[ia], z[ib]
        # symmetric in (a, b)
        ctx = torch.cat([za + zb, za * zb, (za - zb).abs()], dim=-1)
        # gain in (0, 2) centred at 1: stage 2 can attenuate or amplify, not flip sign
        return 1.0 + torch.tanh(self.head(ctx))


class ORBITEstimator:
    """The full two-stage estimator."""

    def __init__(self, relations=("regulatory", "ppi", "cofunctional", "profile"),
                 d_hidden=32, n_pc=16, epochs=800, lr=1e-2, wd=1e-3, patience=25,
                 project=True, use_graph=True, seed=0, global_scale=True):
        self.__dict__.update(locals()); del self.self

    def _encoder_features(self, X):
        Xr = X - X.mean(0)
        U, S, _ = np.linalg.svd(Xr, full_matrices=False)
        Xp = U[:, :self.n_pc] * S[:self.n_pc]
        return ((Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)).astype(np.float32)

    def fit_predict(self, X, adjs, A_train, pairs_tr, pairs_va, pairs_te,
                    R_tr, R_va, R_te):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        F = {k: pair_features(X, [a for a, _ in v], [b for _, b in v])
             for k, v in [("tr", pairs_tr), ("va", pairs_va), ("te", pairs_te)]}
        sc = lambda P, O: float(np.nanmean(per_pair_pearson(P, O)))

        # ---- stage 1 ----
        rb = RidgeBackbone(seed=self.seed).fit(F["tr"], R_tr, score_fn=sc)
        p1 = {k: rb.predict(F[k]) for k in F}
        # global magnitude calibration, fitted on the out-of-fold stage-1 predictions
        s_glob = 1.0
        if self.global_scale:
            o = rb.oof_
            s_glob = float((o * R_tr).sum() / (o * o).sum())
            p1 = {k: v * s_glob for k, v in p1.items()}
        out = dict(alpha=rb.alpha_, global_scale=s_glob,
                   stage1=dict(pearson=sc(p1["te"], R_te), r2=residual_r2(p1["te"], R_te)))
        if not self.use_graph:
            out.update(final=out["stage1"], relation_weights=None, epochs=0)
            out["pred"] = p1["te"]
            return out

        # ---- stage 2 ----
        Xe = torch.from_numpy(self._encoder_features(X))
        Adj = [torch.from_numpy(A) for A in adjs]
        net = GainNet(Xe.shape[0], Xe.shape[1], len(Adj), self.d_hidden, dropout=0.1)
        proj = NuisanceProjection(nuisance_basis(A_train)) if self.project else None
        I = {k: (torch.tensor([a for a, _ in v]), torch.tensor([b for _, b in v]))
             for k, v in [("tr", pairs_tr), ("va", pairs_va), ("te", pairs_te)]}
        OOF = torch.from_numpy(rb.oof_ * s_glob)
        Y = torch.from_numpy(R_tr)
        P1 = {k: torch.from_numpy(v) for k, v in p1.items()}

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, pat = -np.inf, None, 0
        for ep in range(self.epochs):
            net.train(); opt.zero_grad()
            corr = OOF * net(Xe, Adj, *I["tr"])
            if proj is not None:
                corr = proj(corr)
            nn.functional.mse_loss(corr, Y).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if ep % 10 == 0:
                net.eval()
                with torch.no_grad():
                    pv = P1["va"] * net(Xe, Adj, *I["va"])
                    if proj is not None:
                        pv = proj(pv)
                    # select on the magnitude-sensitive metric: that is stage 2's job
                    sv = residual_r2(pv.numpy(), R_va)
                if sv > best:
                    best, best_state, pat = sv, {k: v.clone() for k, v in net.state_dict().items()}, 0
                else:
                    pat += 1
                    if pat >= self.patience:
                        break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pt = P1["te"] * net(Xe, Adj, *I["te"])
            if proj is not None:
                pt = proj(pt)
            pred = pt.numpy()
            relw = net.relation_weights.numpy().tolist()
        out.update(final=dict(pearson=sc(pred, R_te), r2=residual_r2(pred, R_te)),
                   relation_weights=relw, epochs=ep + 1, val_r2=float(best), pred=pred)
        return out
