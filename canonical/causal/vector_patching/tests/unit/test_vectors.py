import numpy as np
import pandas as pd

from canonical.causal.vector_patching.vectors import average_vector, dom_vector


def _toy_activations() -> pd.DataFrame:
    # 2 layers, 3-dim. held ids score high on dim 0, failed ids score low.
    rows = []
    for i in range(4):
        rows.append(
            {"canonical_id": f"held_{i}", "activation": np.array([[10.0, 0, 0], [20.0, 0, 0]], dtype=np.float16)}
        )
    for i in range(4):
        rows.append(
            {"canonical_id": f"failed_{i}", "activation": np.array([[1.0, 0, 0], [2.0, 0, 0]], dtype=np.float16)}
        )
    return pd.DataFrame(rows)


def test_dom_vector_basic():
    acts = _toy_activations()
    held_ids = {f"held_{i}" for i in range(4)}
    failed_ids = {f"failed_{i}" for i in range(4)}
    v = dom_vector(acts, held_ids, failed_ids)
    assert v.shape == (2, 3)
    assert v.dtype == np.float32
    np.testing.assert_allclose(v[0], [9.0, 0, 0], atol=1e-4)
    np.testing.assert_allclose(v[1], [18.0, 0, 0], atol=1e-4)


def test_dom_vector_excludes_unknown_ids():
    acts = _toy_activations()
    # only 2 of the 4 held ids are "known" (in held_ids); rest excluded
    held_ids = {"held_0", "held_1"}
    failed_ids = {"failed_0", "failed_1"}
    v = dom_vector(acts, held_ids, failed_ids)
    np.testing.assert_allclose(v[0], [9.0, 0, 0], atol=1e-4)


def test_dom_vector_no_fp16_overflow_at_large_magnitude():
    # residual-stream-scale magnitudes (hundreds) that would overflow fp16
    # squared-sum reductions if not upcast before arithmetic
    rows = []
    for i in range(3):
        rows.append({"canonical_id": f"held_{i}", "activation": np.full((1, 4096), 300.0, dtype=np.float16)})
    for i in range(3):
        rows.append({"canonical_id": f"failed_{i}", "activation": np.full((1, 4096), -300.0, dtype=np.float16)})
    acts = pd.DataFrame(rows)
    v = dom_vector(acts, {f"held_{i}" for i in range(3)}, {f"failed_{i}" for i in range(3)})
    assert np.isfinite(v).all()
    np.testing.assert_allclose(v[0], np.full(4096, 600.0), rtol=1e-2)


def test_average_vector():
    vectors = {"a": np.array([[1.0, 1.0]]), "b": np.array([[3.0, 3.0]])}
    avg = average_vector(vectors, ["a", "b"])
    np.testing.assert_allclose(avg, [[2.0, 2.0]])
