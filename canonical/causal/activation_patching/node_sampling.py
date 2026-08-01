"""Sampling strategies for completness verification of circuits."""

from typing import List
import torch


def sample_random_nodes(
    in_graph_mask, real_edge_mask, frac=0.3, generator=None
):  # generator for reproducibility
    """Sampling a subset of randomly selected edges to ablate."""
    idx = (in_graph_mask & real_edge_mask).nonzero(as_tuple=False)
    length = idx.shape[0]
    number2ablate = int(round(frac * length))  # how many edges to pick for patching

    chosen = idx[torch.randperm(length, generator=generator)[:number2ablate]]

    sample = in_graph_mask.clone()
    sample[chosen[:, 0], chosen[:, 1]] = False
    return sample


def sample_functional_nodes(
    in_graph_mask: torch.Tensor, rows2ablate: List[int]
) -> torch.Tensor:
    """Sample an established class of nodes, e.g. a full attention layer.

    Each row corresponds to a node (e.g. attn.result_out in TransformerLens); zeroing
    the whole row ablates all of that node's connections.
    """
    sample = in_graph_mask.clone()
    sample[rows2ablate] = False
    return sample


def greedy_sampling():
    """Sampling the nodes that are specifically responsible for the incompleteness score.
    This on is the hardest and should be used as a control method after sampling random or functional nodes.
    In the original paper this method showed drastically different results from random/functional.
    Expensive! Requires many runs.
    """
    pass
