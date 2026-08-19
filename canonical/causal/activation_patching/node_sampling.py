"""Sampling strategies for completness verification of circuits."""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader

from canonical.causal.activation_patching.utils import InterventionType

if TYPE_CHECKING:
    from canonical.causal.activation_patching.circuit_verifier import CircuitVerifier


def sample_random_nodes(
    in_graph_mask: torch.Tensor,
    real_edge_mask: torch.Tensor,
    frac: float = 0.3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:  # generator for reproducibility
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


def sample_greedy(
    verifier: "CircuitVerifier",
    circuit_mask: torch.Tensor,
    steps: int = 10,
    candidates: Optional[torch.Tensor] = None,
    reduce: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    intervention: InterventionType = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
) -> Tuple[torch.Tensor, List[Tuple[Tuple[int, int], float]]]:
    """Sampling the nodes that are specifically responsible for the incompleteness score.
    This on is the hardest and should be used as a control method after sampling random or functional nodes.
    In the original paper this method showed drastically different results from random/functional.
    Expensive! Requires many runs.

    `verifier` is the CircuitVerifier instance whose incompleteness_score is used to
    score each candidate edge (it already carries model/graph/dataloader/metrics).
    """
    sample_tensor = torch.zeros_like(
        circuit_mask
    )  # running mask == membership structure
    cand = (
        circuit_mask.nonzero(as_tuple=False) if candidates is None else candidates
    ).tolist()
    path = []
    for _ in range(steps):
        best, best_s = None, -float("inf")
        for i, j in cand:
            if sample_tensor[i, j]:
                continue
            sample_tensor[i, j] = True
            s = reduce(
                verifier.incompleteness_score(
                    sample_tensor,
                    circuit_mask,
                    intervention=intervention,
                    intervention_dataloader=intervention_dataloader,
                )
            ).item()
            sample_tensor[i, j] = False
            if s > best_s:
                best_s, best = s, (i, j)
        if best is None:
            break  # nothing left to add
        sample_tensor[best[0], best[1]] = True  # add the best edge this step
        path.append((best, best_s))
    return sample_tensor, path
