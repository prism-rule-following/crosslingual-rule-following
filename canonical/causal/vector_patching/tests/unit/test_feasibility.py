import numpy as np

from canonical.causal.vector_patching.feasibility import patch_feasibility, top_k_layers
import pandas as pd


def test_patch_feasibility_recovers_known_separating_direction():
    rng = np.random.default_rng(0)
    d = 16
    u_true = np.zeros(d)
    u_true[0] = 1.0

    n = 200
    X_don = rng.normal(size=(n, d)) * 0.1
    y_don = rng.integers(0, 2, n)
    X_don[y_don == 1, 0] += 5.0

    X_rec = rng.normal(size=(n, d)) * 0.1
    y_rec = rng.integers(0, 2, n)
    X_rec[y_rec == 1, 0] += 5.0

    result = patch_feasibility(X_don, y_don, X_rec, y_rec, {"good": u_true, "noise": np.eye(d)[1]})
    assert result["good"]["auc_in_recipient"] > 0.9
    assert abs(result["good"]["cohens_d"]) > abs(result["noise"]["cohens_d"])
    # ranked descending by |cohens_d|
    names = list(result.keys())
    assert names[0] == "good"


def test_patch_feasibility_no_overflow_at_fp16_scale():
    # magnitudes that overflow fp16 in squared-sum reductions if not upcast,
    # with real within-group variance so cohens_d is well-defined
    rng = np.random.default_rng(1)
    n, d = 20, 4096
    X_don = (300.0 + rng.normal(size=(n, d))).astype(np.float16)
    X_rec = (-300.0 + rng.normal(size=(n, d))).astype(np.float16)
    y_don = np.array([1] * (n // 2) + [0] * (n // 2))
    y_rec = np.array([1] * (n // 2) + [0] * (n // 2))
    u = np.zeros(d, dtype=np.float16)
    u[0] = 1.0

    result = patch_feasibility(X_don, y_don, X_rec, y_rec, {"u": u})
    for value in result["u"].values():
        assert np.isfinite(value)


def test_top_k_layers():
    grid = pd.DataFrame(
        {
            "language_donor": ["en"] * 4,
            "language_recipient": ["yo"] * 4,
            "direction": ["dom_donor"] * 4,
            "layer": [0, 1, 2, 3],
            "cohens_d": [0.1, 2.0, -3.0, 0.5],
        }
    )
    top = top_k_layers(grid, k=2)
    assert set(top["layer"]) == {1, 2}
