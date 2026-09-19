"""Integration test for intervention='mean' against the real toy model.

Requires CUDA (see integration/conftest.py): eap.utils.compute_mean_activations,
called internally by eap.evaluate.evaluate_graph whenever intervention='mean',
hardcodes device='cuda' for the tensor it accumulates means into, with no way
to override it. This module is skipped entirely on a machine without CUDA
(e.g. run it on a Colab GPU runtime instead).
"""

import torch

from canonical.causal.activation_patching.circuit_verifier import CircuitVerifier


def test_verify_sufficiency_with_mean_intervention_and_neutral_baseline(
    tiny_model, tiny_graph, tiny_dataloader, tiny_metric, neutral_dataloader
):
    verifier = CircuitVerifier(tiny_model, tiny_graph, tiny_dataloader, [tiny_metric])

    sufficiency = verifier.verify_sufficiency(
        intervention="mean", intervention_dataloader=neutral_dataloader
    )

    assert torch.isfinite(sufficiency).all()
