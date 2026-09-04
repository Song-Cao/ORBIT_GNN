"""ORBIT model components.

The contribution is the *estimand*: we regress the cross-fitted interaction residual
r = y - (mu + tau_a + tau_b) rather than the total outcome y, and we project the model
output onto the orthogonal complement of the nuisance subspace so the interaction head
cannot score points on main effects.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class NuisanceProjection(nn.Module):
    """Hard orthogonal projection onto the complement of a nuisance subspace.

    Given a basis ``M`` (n_out x k) spanning the nuisance directions, applies
    ``(I - P_M) v`` with ``P_M = M (M^T M)^-1 M^T``. The basis is orthonormalised once
    at construction (QR), so the projection is ``v - Q (Q^T v)`` — O(n_out * k) per
    sample rather than forming the n_out x n_out projector.

    Proposition 1 (proposal Sec. 4): with a fixed finite-dimensional M, cross-fitted
    nuisance error delta in M, and model output constrained to M-perp, the risk
    decomposes as R(theta; delta) = R(theta; 0) + E[delta^2], so the argmin is exactly
    independent of the nuisance error.
    """

    def __init__(self, basis: torch.Tensor):
        super().__init__()
        # Orthonormal basis of the nuisance span; buffer so it moves with .to(device)
        Q, _ = torch.linalg.qr(basis)
        self.register_buffer("Q", Q)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (batch, n_out) -> remove the component lying in span(Q)
        return v - (v @ self.Q) @ self.Q.T

    @torch.no_grad()
    def check_idempotent(self, n_probe: int = 8, tol: float = 1e-5) -> dict:
        """Self-test: P(I-P) = 0 and (I-P) is idempotent."""
        v = torch.randn(n_probe, self.Q.shape[0], device=self.Q.device)
        pv = self.forward(v)
        # component of the projected vector still inside span(Q) must vanish
        leak = (pv @ self.Q).abs().max().item()
        # projecting twice equals projecting once
        idem = (self.forward(pv) - pv).abs().max().item()
        # a vector already in span(Q) must project to ~0
        inM = self.forward(self.Q.T[:n_probe] @ torch.eye(self.Q.shape[0], device=self.Q.device))
        return {"leak": leak, "idempotent_err": idem,
                "in_subspace_residual": inM.abs().max().item(),
                "pass": leak < tol and idem < tol}


class MultiRelationalEncoder(nn.Module):
    """Message passing over R relation-specific dense adjacency matrices.

    Each relation gets its own linear transform; a learned softmax weight over relations
    is exposed as ``relation_weights`` so the model reports *which* relation carries the
    interaction signal (proposal Sec. 4.1 makes a falsifiable prediction about this).
    """

    def __init__(self, n_nodes: int, d_in: int, d_hidden: int, n_relations: int,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_relations = n_relations
        self.inp = nn.Linear(d_in, d_hidden)
        self.rel = nn.ModuleList([
            nn.ModuleList([nn.Linear(d_hidden, d_hidden, bias=False) for _ in range(n_relations)])
            for _ in range(n_layers)
        ])
        self.self_loop = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.rel_logit = nn.Parameter(torch.zeros(n_relations))
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    @property
    def relation_weights(self) -> torch.Tensor:
        return torch.softmax(self.rel_logit, dim=0)

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        # x: (n_nodes, d_in); adjs: list of (n_nodes, n_nodes) row-normalised
        h = self.act(self.inp(x))
        w = self.relation_weights
        for layer in range(len(self.rel)):
            msg = 0.0
            for r, A in enumerate(adjs):
                msg = msg + w[r] * (A @ self.rel[layer][r](h))
            h = self.norm[layer](self.self_loop[layer](h) + msg)
            h = self.drop(self.act(h))
        return h


class BilinearInteractionHead(nn.Module):
    """Interaction head with two paths, both predicting in gene space.

    (1) *Linear pair path.* A ridge-like map from symmetric gene-space pair features
        [x_a + x_b, x_a * x_b, |x_a - x_b|] to the residual. This is the strong,
        essentially linear component of the interaction signal, and it carries no graph
        information -- switching off path (2) reduces the model to this path.

    (2) *Graph-gated bilinear path.* A rank-``rank`` bilinear form on the multi-relational
        node embeddings, expanded onto a learned gene-space basis and multiplied by a
        pair gate. This is where relational structure enters: the graph decides *how
        much* non-additivity a pair shows and modulates its shape.

    ``rank`` is set from measurement, not convention: the observed residual matrix has
    effective rank ~11.3 (proposal Sec. 6), so rank 16 covers it with margin.
    """

    def __init__(self, d: int, d_gene: int, n_out: int, rank: int = 16,
                 gate: bool = True, linear_path: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_gate = gate
        self.use_linear = linear_path
        if linear_path:
            # 3 symmetric gene-space feature blocks -> residual; the ridge-like path
            self.lin = nn.Linear(3 * d_gene, n_out, bias=False)
            nn.init.zeros_(self.lin.weight)          # start at the additive null
        self.U = nn.Linear(d, rank, bias=False)
        self.V = nn.Linear(d, rank, bias=False)
        self.basis = nn.Linear(rank, n_out, bias=False)   # learned gene-space basis
        if gate:
            self.gate = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d, 1))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, za, zb, xa=None, xb=None):
        # symmetric in (a, b): the perturbation set {a, b} is unordered
        h = self.U(za) * self.V(zb) + self.U(zb) * self.V(za)
        g = None
        if self.use_gate:
            ctx = torch.cat([za + zb, za * zb, (za - zb).abs(), za * zb], dim=-1)
            g = torch.sigmoid(self.gate(ctx))
            h = h * g
        out = self.basis(h) * self.scale
        if self.use_linear and xa is not None:
            feat = torch.cat([xa + xb, xa * xb, (xa - xb).abs()], dim=-1)
            out = out + self.lin(feat)
        return out, g


class ORBIT(nn.Module):
    """Full model: multi-relational encoder -> two-path interaction head -> projection.

    The output is projected onto the orthogonal complement of the nuisance subspace, so
    no part of the prediction can express main effects (Proposition 1).
    """

    def __init__(self, n_nodes: int, d_in: int, d_gene: int, n_out: int,
                 nuisance_basis: torch.Tensor | None, d_hidden: int = 64,
                 rank: int = 16, n_relations: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, gate: bool = True, project: bool = True,
                 linear_path: bool = True):
        super().__init__()
        self.enc = MultiRelationalEncoder(n_nodes, d_in, d_hidden, n_relations,
                                          n_layers, dropout)
        self.head = BilinearInteractionHead(d_hidden, d_gene, n_out, rank, gate,
                                            linear_path, dropout)
        self.proj = (NuisanceProjection(nuisance_basis)
                     if (project and nuisance_basis is not None) else None)

    def forward(self, x, adjs, ia, ib, xg=None):
        """``x`` node features for the encoder; ``xg`` gene-space features for the
        linear path (defaults to ``x`` when they are the same array)."""
        z = self.enc(x, adjs)
        xg = x if xg is None else xg
        out, g = self.head(z[ia], z[ib], xg[ia], xg[ib])
        if self.proj is not None:
            out = self.proj(out)
        return out, g
