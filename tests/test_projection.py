"""Unit tests for the hard nuisance projection (Proposition 1's mechanism)."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orbit.model import NuisanceProjection


def test_projection_removes_subspace():
    torch.manual_seed(0)
    n_out, k = 200, 4
    B = torch.randn(n_out, k)
    P = NuisanceProjection(B)
    # a vector lying entirely in span(B) must project to ~0
    v_in = (B @ torch.randn(k, 8)).T
    assert P(v_in).abs().max() < 1e-4, "vector in M did not project to zero"


def test_projection_idempotent():
    torch.manual_seed(1)
    P = NuisanceProjection(torch.randn(150, 3))
    v = torch.randn(16, 150)
    pv = P(v)
    assert (P(pv) - pv).abs().max() < 1e-5, "(I-P) is not idempotent"
    # projected output carries no component in span(Q)
    assert (pv @ P.Q).abs().max() < 1e-4, "leakage into nuisance subspace"


def test_projection_preserves_orthogonal_part():
    torch.manual_seed(2)
    n_out, k = 120, 5
    B = torch.randn(n_out, k)
    P = NuisanceProjection(B)
    v = torch.randn(4, n_out)
    pv = P(v)
    # the removed part must lie in span(Q) => norms decompose (Pythagoras)
    assert torch.allclose((v ** 2).sum(1), (pv ** 2).sum(1) + ((v - pv) ** 2).sum(1),
                          rtol=1e-4), "orthogonal decomposition violated"


def test_gradient_flows_through_projection():
    B = torch.randn(60, 2)
    P = NuisanceProjection(B)
    v = torch.randn(3, 60, requires_grad=True)
    P(v).pow(2).sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


if __name__ == "__main__":
    for fn in [test_projection_removes_subspace, test_projection_idempotent,
               test_projection_preserves_orthogonal_part,
               test_gradient_flows_through_projection]:
        fn(); print(f"PASS {fn.__name__}")
