import torch

from eap.evaluate import evaluate_graph, evaluate_baseline


class CircuitVerifier:
    """Circuit verifier class."""

    def __init__(self, model, graph, dataloader, metrics):
        self.model, self.graph, self.dataloader = model, graph, dataloader
        self.metrics = metrics
        # Saving original edge masks
        # shows whether an edge is in graph
        self.in_graph_mask = graph.in_graph.clone()
        # shows whether an edge is in graph AND real, e.g. not 'm10->m2'
        self.real_mask = graph.real_edge_mask.clone()
        # important whether the edge is in graph AND real
        self.real_in_graph_mask = (graph.in_graph & graph.real_edge_mask).clone()

        # Evaluation results useful for many functions:
        # evaluation on clean and corrupt data
        self.clean_data_eval = evaluate_baseline(
            model, dataloader, metrics, run_corrupted=False
        )
        self.corrupt_data_eval = evaluate_baseline(
            model, dataloader, metrics, run_corrupted=True
        )
        self.F_C = self._eval(self.Cmask)
        self.F_M = self._eval(self.real)

        # results to return TODO: wrap them into a pydantic class
        self.results = {}

    @classmethod
    def normalise_metric(circuit, corrupt, clean, mean_agg=False):
        """Function normalises verification score."""
        metric_tensor = (circuit - corrupt) / (clean - corrupt)
        return metric_tensor.mean() if mean_agg else metric_tensor

    def evaluate_with_edge_mask(
        self, intervention="patching", intervention_dataloader=None
    ):
        """Evaluates graph while masking certain edges.
        Function comes up nearly in every verification procedure.
        """
        masked_evaluation = evaluate_graph(
            self.model,
            self.graph,
            self.dataloader,
            self.metrics,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
        )
        # setting the originally saved mask back
        self.graph.in_graph.copy_(self.in_graph_mask)
        return masked_evaluation

    @classmethod
    def verify_sufficiency(
        self, intervention="patching", intervention_dataloader=None, mean_agg=False
    ):
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
        self, intervention="patching", intervention_dataloader=None, mean_agg=False
    ):
        """Function verifies necessity - patch the circuit, keep everything else."""
        self.graph.in_graph = self.graph.real_edge_mask & ~self.graph.in_graph
        self.graph.prune()
        circuit_complement_clean = self.evaluate_with_edge_mask(
            intervention=intervention, intervention_dataloader=intervention_dataloader
        )
        # TODO: add classical adherence
        return CircuitVerifier.normalise_metric(
            circuit_complement_clean,
            self.corrupt_data_eval,
            self.clean_data_eval,
            mean_agg=mean_agg,
        )

    def verify_minimality(
        self,
        model,
        graph,
        dataloader,
        metrics,
        intervention="patching",
        intervention_dataloader=None,
    ):
        """Function verifies the minimality by knocking out each edge of the circuit."""
        knockout_dict = {}
        in_graph_ids = torch.nonzero((self.graph.in_graph & self.graph.real_edge_mask))
        ids2names = {edge.matrix_index: name for name, edge in self.graph.edges.items()}
        # looping through all the edges
        for out_, in_ in in_graph_ids:
            self.graph.in_graph[out_, in_] = False
            self.graph.prune()
            circuit_knockout = self.evaluate_with_edge_mask(
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
            )
            # the metric is essentially the difference between full circuit clean and 1-node-out
            diff_vector = self.clean_data_eval - circuit_knockout
            # saving the results per each node
            knockout_dict[(out_, in_)] = {
                "minimality_diff_vector": diff_vector,
                "name": ids2names[(out_, in_)],
            }
            # TODO: add classical adherence
        return knockout_dict

    def verify_completeness(self):
        """Function verifies completeness - patch out a set of edges out of the circuit AND out of the models.
        Additionally, the completeness detects hydra effect.
        """
        pass
