"""Assert the data pipeline cannot leak held-out doubles into training inputs."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbit.data import DATA, load_split


def test_train_test_disjoint():
    for s in range(1, 6):
        sp = load_split(s)
        assert not (set(sp["train"]) & set(sp["test"])), f"split {s}: train/test overlap"
        assert not (set(sp["val"]) & set(sp["test"])), f"split {s}: val/test overlap"


def test_test_pairs_are_doubles():
    for s in range(1, 6):
        sp = load_split(s)
        for p in sp["test"]:
            assert "+" in p and not p.endswith("+ctrl"), f"{p} is not a held-out double"


def test_node_features_use_singles_only():
    """Node features are single-perturbation profiles; singles are never in the test set."""
    for s in range(1, 6):
        sp = load_split(s)
        singles = {p for p in sp["train"] if p.endswith("+ctrl")}
        assert not (singles & set(sp["test"])), "a single leaked into test"


def test_profile_graph_symmetric_and_zero_diag():
    import pandas as pd
    for s in range(1, 6):
        M = pd.read_csv(DATA / "graphs" / f"profilesim_split{s}.csv", index_col=0).values
        assert np.allclose(M, M.T, atol=1e-6), f"split {s}: profile graph not symmetric"
        assert np.abs(np.diag(M)).max() < 1e-9, f"split {s}: nonzero diagonal"


if __name__ == "__main__":
    for fn in [test_train_test_disjoint, test_test_pairs_are_doubles,
               test_node_features_use_singles_only,
               test_profile_graph_symmetric_and_zero_diag]:
        fn(); print(f"PASS {fn.__name__}")
