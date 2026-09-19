"""Integration tests against the real "attn-only-1l" toy model.

Unlike the unit tests, these exercise genuine forward passes through real
model weights (downloaded from HuggingFace on first use) and, for the
CircuitVerifier test, a real (hand-picked, not attributed -- see
integration/conftest.py::tiny_graph) circuit on the toy model's Graph. They
require `transformer_lens` and `eap` (EAP-IG) to be installed -- see
integration/conftest.py, which skips this whole module via
`pytest.importorskip` if either is missing.
"""

import torch

from canonical.causal.activation_patching.circuit_verifier import CircuitVerifier
from canonical.causal.activation_patching.evaluation_metrics import logit_difference


def test_tiny_model_loads_and_runs_a_forward_pass(tiny_model):
    tokens = tiny_model.to_tokens("The cat sat on the mat")

    logits = tiny_model(tokens)

    assert logits.shape[0] == 1
    assert logits.shape[-1] == tiny_model.cfg.d_vocab
    assert torch.isfinite(logits).all()


def test_logit_difference_against_real_model_logits(tiny_model):
    tokens = tiny_model.to_tokens("The cat sat. The cat")
    logits = tiny_model(tokens)
    input_length = torch.tensor([tokens.shape[-1]])
    correct = tiny_model.to_tokens(" sat", prepend_bos=False)[0, 0].item()
    wrong = tiny_model.to_tokens(" ran", prepend_bos=False)[0, 0].item()
    labels = torch.tensor([[correct, wrong]])

    result = logit_difference(logits, None, input_length, labels, mean=False, loss=False)

    assert result.shape == (1,)
    assert torch.isfinite(result).all()
    # attn-only-1l's whole specialty is induction: "The cat sat. The cat" should
    # favour repeating " sat" over the unrelated " ran" -- not just "some finite
    # number", a specific, checkable behavioural claim about this exact model.
    assert result.item() > 0


def test_evaluate_with_edge_mask_actually_applies_the_mask(
    tiny_model, tiny_graph, tiny_dataloader, tiny_metric
):
    """The whole verification framework is only meaningful if masking self.graph.in_graph
    actually changes what evaluate_with_edge_mask computes. This directly proves that --
    a bug that makes masking a no-op (e.g. restoring the saved mask before, instead of
    after, running evaluate_graph) would leave every isfinite()/shape check elsewhere
    passing, since the model still runs fine -- it would just silently be evaluating the
    same unmasked graph every time.
    """
    verifier = CircuitVerifier(tiny_model, tiny_graph, tiny_dataloader, [tiny_metric])

    full_circuit_result = verifier.evaluate_with_edge_mask()

    real_edges = tiny_graph.in_graph.nonzero(as_tuple=False)
    single_edge = real_edges[0]
    verifier.graph.in_graph[:] = False
    verifier.graph.in_graph[single_edge[0], single_edge[1]] = True
    verifier.graph.prune()

    single_edge_result = verifier.evaluate_with_edge_mask()

    assert not torch.allclose(full_circuit_result, single_edge_result)
    # evaluate_with_edge_mask must also restore the graph's mask afterwards
    assert torch.equal(verifier.graph.in_graph, verifier.in_graph_mask)


def test_circuit_verifier_runs_end_to_end_on_a_real_toy_circuit(
    tiny_model, tiny_graph, tiny_dataloader, tiny_metric
):
    verifier = CircuitVerifier(tiny_model, tiny_graph, tiny_dataloader, [tiny_metric])

    assert not torch.allclose(verifier.clean_data_eval, verifier.corrupt_data_eval), (
        "clean/corrupt baselines are degenerate -- normalise_metric would divide by ~0"
    )

    sufficiency = verifier.verify_sufficiency()
    assert torch.isfinite(sufficiency).all()

    necessity = verifier.verify_necessity()
    assert torch.isfinite(necessity).all()
    # sufficiency keeps only the circuit clean (4 edges); necessity ablates only
    # those same 4 edges and keeps everything else clean -- these are different
    # masks over a model with far more than 4 edges, so a real masking mechanism
    # should not produce identical scores.
    assert not torch.allclose(sufficiency, necessity)

    completeness = verifier.verify_completeness(steps=2)
    assert torch.isfinite(completeness["incompleteness_vector"]).all()
    assert isinstance(completeness["incompleteness_median"], float)
    assert len(completeness["greedy_path"]) == 2

    minimality = verifier.verify_minimality(
        tiny_model, tiny_graph, tiny_dataloader, [tiny_metric]
    )
    assert isinstance(minimality, dict)
    assert len(minimality) == 4  # tiny_graph's hand-picked circuit has exactly 4 edges
    diff_vectors = [entry["minimality_diff_vector"] for entry in minimality.values()]
    for diff_vector in diff_vectors:
        assert torch.isfinite(diff_vector).all()
    # knocking out different edges one at a time should not produce identical
    # effects across the board (that would suggest the knockout loop isn't
    # actually mutating the graph between iterations)
    assert not all(torch.allclose(diff_vectors[0], other) for other in diff_vectors[1:])
