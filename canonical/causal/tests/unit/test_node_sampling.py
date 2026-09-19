"""Unit tests for node_sampling.py sampling strategies."""

import torch

from canonical.causal.activation_patching.node_sampling import (
    sample_functional_nodes,
    sample_greedy,
    sample_random_nodes,
)


# --- sample_random_nodes -----------------------------------------------------


def test_sample_random_nodes_ablates_requested_fraction():
    in_graph_mask = torch.ones(4, 4, dtype=torch.bool)
    real_edge_mask = torch.zeros(4, 4, dtype=torch.bool)
    real_edge_mask[0, 1] = real_edge_mask[1, 2] = True
    real_edge_mask[2, 3] = real_edge_mask[3, 0] = True

    result = sample_random_nodes(in_graph_mask, real_edge_mask, frac=0.5)

    assert result.shape == in_graph_mask.shape
    assert result.dtype == torch.bool
    ablated = (~result) & real_edge_mask
    assert ablated.sum().item() == 2  # round(0.5 * 4 real edges)
    # non-real-edge entries are untouched
    assert result[~real_edge_mask].all()
    # original mask must be untouched (clone semantics)
    assert in_graph_mask.all()


def test_sample_random_nodes_frac_zero_is_noop():
    in_graph_mask = torch.ones(3, 3, dtype=torch.bool)
    real_edge_mask = torch.eye(3, dtype=torch.bool)

    result = sample_random_nodes(in_graph_mask, real_edge_mask, frac=0.0)

    assert torch.equal(result, in_graph_mask)


def test_sample_random_nodes_frac_one_ablates_all_real_edges():
    in_graph_mask = torch.ones(3, 3, dtype=torch.bool)
    real_edge_mask = torch.eye(3, dtype=torch.bool)

    result = sample_random_nodes(in_graph_mask, real_edge_mask, frac=1.0)

    assert not result[real_edge_mask].any()
    assert result[~real_edge_mask].all()


def test_sample_random_nodes_reproducible_with_seeded_generator():
    in_graph_mask = torch.ones(5, 5, dtype=torch.bool)
    real_edge_mask = torch.ones(5, 5, dtype=torch.bool)

    first = sample_random_nodes(
        in_graph_mask, real_edge_mask, frac=0.4, generator=torch.Generator().manual_seed(42)
    )
    second = sample_random_nodes(
        in_graph_mask, real_edge_mask, frac=0.4, generator=torch.Generator().manual_seed(42)
    )

    assert torch.equal(first, second)


# --- sample_functional_nodes -------------------------------------------------


def test_sample_functional_nodes_zeroes_whole_rows():
    in_graph_mask = torch.ones(4, 4, dtype=torch.bool)

    result = sample_functional_nodes(in_graph_mask, [1, 3])

    assert not result[1].any()
    assert not result[3].any()
    assert result[0].all()
    assert result[2].all()
    # original mask must be untouched (clone semantics)
    assert in_graph_mask.all()


# --- sample_greedy ------------------------------------------------------------


class FakeVerifier:
    """Stand-in for CircuitVerifier: incompleteness_score is the sum of fixed,
    per-edge weights over whatever edges are currently True in chosen_sample.
    Since the score is additive, the greedy loop should always pick the
    highest-weight remaining candidate at each step -- easy to assert against.
    """

    def __init__(self, edge_weights):
        self.edge_weights = edge_weights
        self.calls = []

    def incompleteness_score(
        self, chosen_sample, circuit_mask, intervention="patching", intervention_dataloader=None
    ):
        self.calls.append(intervention)
        total = sum(
            self.edge_weights[tuple(idx)]
            for idx in chosen_sample.nonzero(as_tuple=False).tolist()
        )
        return torch.tensor([float(total)])


def _toy_circuit_mask():
    circuit_mask = torch.zeros(3, 3, dtype=torch.bool)
    circuit_mask[0, 1] = True
    circuit_mask[1, 2] = True
    circuit_mask[2, 0] = True
    return circuit_mask


def test_sample_greedy_picks_highest_weight_edges_in_order():
    circuit_mask = _toy_circuit_mask()
    verifier = FakeVerifier({(0, 1): 1.0, (1, 2): 5.0, (2, 0): 3.0})

    sample_tensor, path = sample_greedy(verifier, circuit_mask, steps=2, intervention="mean")

    assert path == [((1, 2), 5.0), ((2, 0), 8.0)]
    assert sample_tensor[1, 2] and sample_tensor[2, 0]
    assert not sample_tensor[0, 1]
    # 3 candidates probed in step 1, 2 remaining candidates probed in step 2
    assert len(verifier.calls) == 5
    assert all(call == "mean" for call in verifier.calls)


def test_sample_greedy_stops_early_once_all_candidates_are_added():
    circuit_mask = _toy_circuit_mask()
    verifier = FakeVerifier({(0, 1): 1.0, (1, 2): 5.0, (2, 0): 3.0})

    sample_tensor, path = sample_greedy(verifier, circuit_mask, steps=10)

    # only 3 real candidates exist, so the loop must stop after 3 steps, not 10
    assert len(path) == 3
    assert sample_tensor[0, 1] and sample_tensor[1, 2] and sample_tensor[2, 0]


def test_sample_greedy_respects_explicit_candidates_override():
    circuit_mask = _toy_circuit_mask()
    verifier = FakeVerifier({(0, 1): 1.0, (1, 2): 5.0, (2, 0): 3.0})
    candidates = torch.tensor([[0, 1], [2, 0]])  # excludes (1, 2) even though it's the best

    sample_tensor, path = sample_greedy(
        verifier, circuit_mask, steps=2, candidates=candidates
    )

    assert not sample_tensor[1, 2]
    assert sample_tensor[0, 1] and sample_tensor[2, 0]
    assert len(path) == 2
