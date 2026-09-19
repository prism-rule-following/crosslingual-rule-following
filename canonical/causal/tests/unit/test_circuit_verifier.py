"""Unit tests for CircuitVerifier: pure logic, no real model/eap graph needed.

evaluate_graph/evaluate_baseline are mocked out (patched in the
circuit_verifier module's own namespace, since it does `from eap.evaluate
import evaluate_graph, evaluate_baseline`) so these run fast and
deterministically, without a real model, real weights, or GPU. A minimal
FakeGraph/FakeEdge stand in for eap's real Graph -- CircuitVerifier's own
book-keeping (mask save/restore, id lookup, argument threading) doesn't
depend on eap's actual connectivity/pruning rules, only on the same
attributes eap's Graph exposes.
"""

from unittest.mock import MagicMock

import pytest
import torch

import canonical.causal.activation_patching.circuit_verifier as cv_module
from canonical.causal.activation_patching.circuit_verifier import CircuitVerifier


class FakeEdge:
    def __init__(self, matrix_index):
        self.matrix_index = matrix_index


class FakeGraph:
    """Minimal stand-in for eap.graph.Graph."""

    def __init__(self, in_graph, real_edge_mask, edges, n_backward):
        self.in_graph = in_graph
        self.real_edge_mask = real_edge_mask
        self.edges = edges
        self.n_backward = n_backward
        self.prune_calls = 0

    def prune(self):
        # Deliberately a no-op: eap's real connectivity-based pruning isn't
        # CircuitVerifier's concern, just that prune() gets called.
        self.prune_calls += 1


def _make_graph(n_forward=2, n_backward=3):
    in_graph = torch.zeros(n_forward, n_backward, dtype=torch.bool)
    in_graph[0, 0] = True  # matches edge "a->b" below
    in_graph[1, n_backward - 1] = True  # matches edge "a->logits" below
    real_edge_mask = torch.ones(n_forward, n_backward, dtype=torch.bool)
    edges = {
        "a->b": FakeEdge((0, 0)),
        # -1 is eap's real shorthand for "the logits column" -- valid for
        # tensor indexing, but not literally equal to (1, n_backward - 1).
        "a->logits": FakeEdge((1, -1)),
    }
    return FakeGraph(in_graph, real_edge_mask, edges, n_backward)


@pytest.fixture
def mocked_evaluate(monkeypatch):
    """Patches evaluate_graph/evaluate_baseline; returns the mocks so tests
    can assert on how/with-what they were called."""
    baseline_mock = MagicMock(side_effect=[torch.tensor(1.0), torch.tensor(0.0)])
    graph_mock = MagicMock(return_value=torch.tensor(0.5))
    monkeypatch.setattr(cv_module, "evaluate_baseline", baseline_mock)
    monkeypatch.setattr(cv_module, "evaluate_graph", graph_mock)
    return baseline_mock, graph_mock


# --- ids2names ----------------------------------------------------------------


def test_ids2names_normalises_negative_matrix_index(mocked_evaluate):
    """Regression test: edge.matrix_index == (1, -1) must resolve to the literal
    positive index (1, n_backward - 1), matching what torch.nonzero() returns
    elsewhere (verify_minimality, verify_completeness's named_path).
    """
    graph = _make_graph()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])

    assert verifier.ids2names[(1, graph.n_backward - 1)] == "a->logits"
    assert verifier.ids2names[(0, 0)] == "a->b"


# --- evaluate_with_edge_mask ----------------------------------------------------


def test_evaluate_with_edge_mask_forwards_args_and_restores_mask(mocked_evaluate):
    baseline_mock, graph_mock = mocked_evaluate
    graph = _make_graph()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])

    graph.in_graph[0, 1] = True  # mutate the mask before calling
    sentinel_dataloader = object()

    result = verifier.evaluate_with_edge_mask(
        intervention="mean", intervention_dataloader=sentinel_dataloader
    )

    graph_mock.assert_called_once_with(
        "model",
        graph,
        "loader",
        ["m"],
        intervention="mean",
        intervention_dataloader=sentinel_dataloader,
    )
    assert result == torch.tensor(0.5)
    # must restore the mask to what it was at __init__ time, undoing our mutation
    assert torch.equal(graph.in_graph, verifier.in_graph_mask)


def test_evaluate_with_edge_mask_restores_mask_even_if_evaluate_graph_raises(mocked_evaluate):
    """The mask must be restored no matter what -- a crashed forward pass, an OOM,
    a bad intervention_dataloader -- not just on the happy path.
    """
    _, graph_mock = mocked_evaluate
    graph = _make_graph()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])

    graph_mock.side_effect = RuntimeError("simulated forward-pass crash")
    graph.in_graph[0, 1] = True  # mutate before calling, as every verify_* does

    with pytest.raises(RuntimeError):
        verifier.evaluate_with_edge_mask()

    assert torch.equal(graph.in_graph, verifier.in_graph_mask)


def test_verify_necessity_restores_mask_even_if_prune_raises(mocked_evaluate):
    """Guards the setup step too: if graph.prune() itself raises (before
    evaluate_with_edge_mask is even reached), the mask must still be restored.
    """
    _, graph_mock = mocked_evaluate
    graph = _make_graph()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])

    original_in_graph = graph.in_graph.clone()
    graph.prune = MagicMock(side_effect=RuntimeError("simulated prune failure"))

    with pytest.raises(RuntimeError):
        verifier.verify_necessity()

    assert torch.equal(graph.in_graph, original_in_graph)
    graph_mock.assert_not_called()  # never even got to evaluate_graph


def test_evaluate_with_edge_mask_rejects_invalid_intervention(mocked_evaluate):
    _, graph_mock = mocked_evaluate
    graph = _make_graph()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])

    with pytest.raises(ValueError):
        verifier.evaluate_with_edge_mask(intervention="not-a-real-intervention")

    graph_mock.assert_not_called()


# --- normalise_metric (regression test for the @classmethod/@staticmethod bug) --


def test_normalise_metric_computes_ratio():
    result = CircuitVerifier.normalise_metric(
        torch.tensor(6.0), torch.tensor(2.0), torch.tensor(10.0)
    )
    assert torch.allclose(result, torch.tensor(0.5))  # (6-2)/(10-2)


def test_normalise_metric_mean_agg_reduces():
    circuit = torch.tensor([6.0, 10.0])
    corrupt = torch.tensor([2.0, 2.0])
    clean = torch.tensor([10.0, 10.0])

    result = CircuitVerifier.normalise_metric(circuit, corrupt, clean, mean_agg=True)

    # (6-2)/(10-2)=0.5, (10-2)/(10-2)=1.0 -> mean 0.75
    assert torch.allclose(result, torch.tensor(0.75))


# --- verify_necessity: mask complement ------------------------------------------


def test_verify_necessity_complements_the_circuit_mask(mocked_evaluate):
    _, graph_mock = mocked_evaluate
    graph = _make_graph()
    original_in_graph = graph.in_graph.clone()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])

    seen_masks = []

    def record(model, graph_arg, dataloader, metrics, **kwargs):
        seen_masks.append(graph_arg.in_graph.clone())
        return torch.tensor(0.5)

    graph_mock.side_effect = record

    verifier.verify_necessity()

    expected_complement = graph.real_edge_mask & ~original_in_graph
    assert torch.equal(seen_masks[0], expected_complement)
    assert graph.prune_calls == 1


# --- verify_minimality: int keys + negative-index lookup (regression test) -----


def test_verify_minimality_uses_int_keys_matching_ids2names(mocked_evaluate):
    _, graph_mock = mocked_evaluate
    graph = _make_graph()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])
    graph_mock.return_value = torch.tensor(0.3)

    knockout = verifier.verify_minimality("model", graph, "loader", ["m"])

    assert set(knockout.keys()) == {(0, 0), (1, graph.n_backward - 1)}
    assert knockout[(1, graph.n_backward - 1)]["name"] == "a->logits"
    assert knockout[(0, 0)]["name"] == "a->b"


# --- intervention/intervention_dataloader threading through completeness -------


def test_verify_completeness_threads_intervention_and_dataloader(mocked_evaluate):
    _, graph_mock = mocked_evaluate
    graph = _make_graph()
    verifier = CircuitVerifier(model="model", graph=graph, dataloader="loader", metrics=["m"])

    seen_kwargs = []

    def record(*args, **kwargs):
        seen_kwargs.append(kwargs)
        return torch.tensor(0.5)

    graph_mock.side_effect = record
    sentinel_dataloader = object()

    verifier.verify_completeness(
        steps=1, intervention="mean", intervention_dataloader=sentinel_dataloader
    )

    assert seen_kwargs, "evaluate_graph was never called"
    assert all(kw["intervention"] == "mean" for kw in seen_kwargs)
    assert all(kw["intervention_dataloader"] is sentinel_dataloader for kw in seen_kwargs)
