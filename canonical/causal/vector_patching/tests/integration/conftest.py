"""Fixtures for integration tests against a real toy TransformerLens model.

Model: "attn-only-1l" (same tiny model used by canonical/causal's own EAP
integration tests). Unlike that suite, nothing here depends on the `eap`
package's hardcoded CUDA paths -- this hook is our own code, CPU-safe.
"""

import pytest

transformer_lens = pytest.importorskip("transformer_lens", reason="transformer_lens not installed")

TOY_MODEL_NAME = "attn-only-1l"


@pytest.fixture(scope="session")
def tiny_model():
    model = transformer_lens.HookedTransformer.from_pretrained(TOY_MODEL_NAME, device="cpu")
    return model
