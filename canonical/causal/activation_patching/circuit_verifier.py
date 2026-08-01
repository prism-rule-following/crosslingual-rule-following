from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from eap.evaluate import evaluate_graph, evaluate_baseline
from canonical.causal.activation_patching.evaluation_metrics import MetricFn
from canonical.causal.activation_patching.node_sampling import sample_greedy
from canonical.causal.activation_patching.utils import (
    InterventionType,
    validate_intervention,
)

if TYPE_CHECKING:
    from eap.graph import Graph
    from transformer_lens import HookedTransformer


class CircuitVerifier:
    """Circuit verifier class."""

    def __init__(
        self,
        model: "HookedTransformer",
        graph: "Graph",
        dataloader: DataLoader,
        metrics: List[MetricFn],
    ) -> None:
        self.model, self.graph, self.dataloader = model, graph, dataloader
        self.metrics = metrics
        # Saving original edge masks
        # shows whether an edge is in graph
        self.in_graph_mask = graph.in_graph.clone()
        # shows whether an edge is in graph AND real, e.g. not 'm10->m2'
        self.real_mask = graph.real_edge_mask.clone()
        # important whether the edge is in graph AND real
        self.real_in_graph_mask = (graph.in_graph & graph.real_edge_mask).clone()
        # static edge_id -> edge_name lookup, doesn't change over the verifier's lifetime.
        # edge.matrix_index uses Python negative-index shorthand for the logits column
        # (e.g. (1, -1)), valid for real tensor indexing but not equal to the literal
        # positive index (e.g. (1, 25)) that torch.nonzero() returns -- normalise so
        # lookups by literal index (as used in verify_minimality/verify_completeness)
        # actually match for edges going straight to logits.
        self.ids2names: Dict[Tuple[int, int], str] = {}
        for name, edge in graph.edges.items():
            fwd, bwd = edge.matrix_index
            if bwd < 0:
                bwd += graph.n_backward
            self.ids2names[(fwd, bwd)] = name

        # Evaluation results useful for many functions:
        # evaluation on clean and corrupt data
        self.clean_data_eval = evaluate_baseline(
            model, dataloader, metrics, run_corrupted=False
        )
        self.corrupt_data_eval = evaluate_baseline(
            model, dataloader, metrics, run_corrupted=True
        )

        # results to return TODO: wrap them into a pydantic class
        self.results = {}

    @staticmethod
    def normalise_metric(
        circuit: torch.Tensor,
        corrupt: torch.Tensor,
        clean: torch.Tensor,
        mean_agg: bool = False,
    ) -> torch.Tensor:
        """Function normalises verification score."""
        metric_tensor = (circuit - corrupt) / (clean - corrupt)
        return metric_tensor.mean() if mean_agg else metric_tensor

    def evaluate_with_edge_mask(
        self,
        intervention: InterventionType = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Evaluates graph while masking certain edges.
        Function comes up nearly in every verification procedure.
        """
        validate_intervention(intervention)
        try:
            return evaluate_graph(
                self.model,
                self.graph,
                self.dataloader,
                self.metrics,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
            )
        finally:
            # setting the originally saved mask back, even if evaluate_graph raised
            self.graph.in_graph.copy_(self.in_graph_mask)

    def verify_sufficiency(
        self,
        intervention: InterventionType = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
        mean_agg: bool = False,
    ) -> torch.Tensor:
        """Function verifies sufficiency - patch everything BUT the circuit.
        The close the score is to 1, the more sufficient the better the discovered circuit recovers the behaviour.
        """
        circuit_clean = self.evaluate_with_edge_mask(
            intervention=intervention, intervention_dataloader=intervention_dataloader
        )
        # TODO: add classical adherence
        return CircuitVerifier.normalise_metric(
            circuit_clean,
            self.corrupt_data_eval,
            self.clean_data_eval,
            mean_agg=mean_agg,
        )

    def verify_necessity(
        self,
        intervention: InterventionType = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
        mean_agg: bool = False,
    ) -> torch.Tensor:
        """Function verifies necessity - patch the circuit, keep everything else."""
        self.graph.in_graph = self.graph.real_edge_mask & ~self.graph.in_graph
        try:
            self.graph.prune()
            circuit_complement_clean = self.evaluate_with_edge_mask(
                intervention=intervention, intervention_dataloader=intervention_dataloader
            )
        finally:
            # restore even if prune() itself raises, before evaluate_with_edge_mask's
            # own (evaluate_graph-failure) restore would ever run
            self.graph.in_graph.copy_(self.in_graph_mask)
        # TODO: add classical adherence
        return CircuitVerifier.normalise_metric(
            circuit_complement_clean,
            self.corrupt_data_eval,
            self.clean_data_eval,
            mean_agg=mean_agg,
        )

    def verify_minimality(
        self,
        model: "HookedTransformer",
        graph: "Graph",
        dataloader: DataLoader,
        metrics: List[MetricFn],
        intervention: InterventionType = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Function verifies the minimality by knocking out each edge of the circuit."""
        knockout_dict = {}
        in_graph_ids = torch.nonzero((self.graph.in_graph & self.graph.real_edge_mask))
        # looping through all the edges
        for out_, in_ in in_graph_ids:
            edge_id = (int(out_), int(in_))
            try:
                self.graph.in_graph[out_, in_] = False
                self.graph.prune()
                circuit_knockout = self.evaluate_with_edge_mask(
                    intervention=intervention,
                    intervention_dataloader=intervention_dataloader,
                )
            finally:
                # restore before the next iteration (or on exit), even if prune()
                # or evaluation raised partway through this edge's knockout
                self.graph.in_graph.copy_(self.in_graph_mask)
            # the metric is essentially the difference between full circuit clean and 1-node-out
            diff_vector = self.clean_data_eval - circuit_knockout
            # saving the results per each node
            knockout_dict[edge_id] = {
                "minimality_diff_vector": diff_vector,
                "name": self.ids2names[edge_id],
            }
            # TODO: add classical adherence
        return knockout_dict

    def incompleteness_score(
        self,
        chosen_sample: torch.Tensor,
        circuit_mask: torch.Tensor,
        intervention: InterventionType = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
    ) -> torch.Tensor:
        """|F(C\\K) - F(M\\K)|, per-example.
        F(C\\K): keep circuit-minus-K clean, corrupt the rest  -> mask = Cmask & ~K
        F(M\\K): keep all-real-minus-K clean, corrupt only K   -> mask = real_mask & ~K
        Large gap => the full model recovers where the circuit can't (backups) => incomplete.
        """
        self.graph.in_graph = circuit_mask & ~chosen_sample
        try:
            self.graph.prune()
            F_C_K = self.evaluate_with_edge_mask(
                intervention=intervention, intervention_dataloader=intervention_dataloader
            )
        finally:
            self.graph.in_graph.copy_(self.in_graph_mask)

        self.graph.in_graph = self.graph.real_edge_mask & ~chosen_sample
        try:
            self.graph.prune()
            F_M_K = self.evaluate_with_edge_mask(
                intervention=intervention, intervention_dataloader=intervention_dataloader
            )
        finally:
            self.graph.in_graph.copy_(self.in_graph_mask)

        return (F_C_K - F_M_K).abs()

    def verify_completeness(
        self,
        steps: int,
        candidates: Optional[torch.Tensor] = None,
        reduce: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
        intervention: InterventionType = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Function verifies completeness - patch out a set of edges out of the circuit AND out of the models.
        Additionally, the completeness detects hydra effect.
        """
        circuit_mask = (self.graph.in_graph & self.graph.real_edge_mask).clone()

        star_subset, path = sample_greedy(
            self,
            circuit_mask,
            steps,
            candidates=candidates,
            reduce=reduce,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
        )

        incompleteness_vec = self.incompleteness_score(
            star_subset,
            circuit_mask,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
        )
        named_path = [(self.ids2names.get(tuple(e), tuple(e)), s) for e, s in path]

        return {
            "star_subset": star_subset,  # worst-case subset (edge mask)
            "incompleteness_vector": incompleteness_vec,  # per-example |F(C\K*) - F(M\K*)|
            "incompleteness_median": reduce(incompleteness_vec).item(),
            "greedy_path": named_path,  # ordered [(edge_name, score), ...]
        }
