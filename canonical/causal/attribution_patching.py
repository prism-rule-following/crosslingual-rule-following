from functools import partial
from typing import List, Optional, Tuple

import pandas as pd
import torch
from eap.attribute import attribute
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph
from kneed import KneeLocator, find_shape
from tqdm import tqdm
from transformer_lens import HookedTransformer

from causal.model import EAPConfig, EAPEdges, EAPNodes, EAPResult, EAPResults
from config.dataset_config import CrossLingualRuleFollowingDataset


def collate_EAP(batch: List[dict]):
    """EAP view. Requires correct_idx/incorrect_idx (run build_indices first,
    and filter out categories that can't produce them, e.g. banned_word)."""
    if not batch:
        raise ValueError("collate_EAP received an empty batch.")

    def _absent(x):
        return x is None or (isinstance(x, float) and pd.isna(x))

    missing = [r["id"] for r in batch if _absent(r.get("correct_idx"))]
    if missing:
        raise ValueError(
            f"collate_EAP: {len(missing)} row(s) lack correct_idx/incorrect_idx "
            f"(e.g. {missing[:3]}). Run build_indices and/or filter categories."
        )
    clean = [r["clean"] for r in batch]
    corrupted = [r["corrupted"] for r in batch]
    labels = torch.tensor(
        [[r["correct_idx"], r["incorrect_idx"]] for r in batch], dtype=torch.long
    )
    return clean, corrupted, labels


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    index = torch.arange(batch_size, device=logits.device)
    return logits[index, input_length - 1]


def logit_difference(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean=True,
    loss=True,
):
    logits = get_logit_positions(logits=logits, input_length=input_length)
    last_token_logits = torch.gather(logits, -1, labels.to(logits.device))
    results = last_token_logits[:, 0] - last_token_logits[:, 1]
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


def graph_to_eap_result(
    graph: Graph, circuit_performance: float, circuit_faithfulness: float
) -> EAPResult:
    nodes = [
        EAPNodes(
            name=name,
            **({"score": float(node.score)} if graph.nodes_scores is not None else {}),
        )
        for name, node in graph.nodes.items()
        if node.in_graph
    ]
    edges = [
        EAPEdges(name=name, score=edge.score.item())
        for name, edge in graph.edges.items()
        if edge.in_graph
    ]
    return EAPResult(
        n_nodes=len(nodes),
        n_edges=len(edges),
        nodes=nodes,
        edges=edges,
        circuit_performance=circuit_performance,
        circuit_faithfulness=circuit_faithfulness,
    )


def find_knee(n_edges: List[int], circuit_faithfulness: List[float]) -> KneeLocator:
    direction, curve = find_shape(n_edges, circuit_faithfulness)
    return KneeLocator(n_edges, circuit_faithfulness, curve=curve, direction=direction)


def _load_model(config: EAPConfig, device: torch.device) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        config.model_id,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device,
        dtype=torch.float16,
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    return model


def _compute_baselines(model: HookedTransformer, dataset_loader) -> Tuple[float, float]:
    baseline = (
        evaluate_baseline(
            model, dataset_loader, partial(logit_difference, loss=False, mean=False)
        )
        .mean()
        .item()
    )
    corrupted_baseline = (
        evaluate_baseline(
            model,
            dataset_loader,
            partial(logit_difference, mean=False, loss=False),
            run_corrupted=True,
        )
        .mean()
        .item()
    )
    return baseline, corrupted_baseline


def _sweep_n_edges(
    config: EAPConfig,
    model: HookedTransformer,
    graph: Graph,
    dataset_loader,
    baseline: float,
    corrupted_baseline: float,
) -> List[EAPResult]:
    """Sweep a range of n_edges, evaluating faithfulness at each point
    (Pareto-curve pattern, see hannamw/EAP-IG's own pareto.py)."""
    start = config.n_edge_start
    end = int(len(graph.edges) * config.n_edge_end_proportion)
    step = max(1, (end - start) // config.n_edge_steps)
    n_edges_candidates = list(range(start, end + 1, step))

    results_sweep = []
    for n_edges in tqdm(n_edges_candidates):
        graph.apply_greedy(n_edges=n_edges, absolute=True)
        circuit_performance = (
            evaluate_graph(
                model,
                graph,
                dataset_loader,
                partial(logit_difference, loss=False, mean=False),
            )
            .mean()
            .item()
        )
        circuit_faithfulness = (circuit_performance - corrupted_baseline) / (
            baseline - corrupted_baseline
        )
        results_sweep.append(
            graph_to_eap_result(
                graph,
                circuit_performance=circuit_performance,
                circuit_faithfulness=circuit_faithfulness,
            )
        )
    return results_sweep


def _select_best_circuit(
    results_sweep: List[EAPResult],
) -> Tuple[EAPResult, KneeLocator]:
    """Pick the elbow of the n_edges-vs-faithfulness curve; fall back to the
    highest-faithfulness sweep point if no exact elbow match is found."""
    n_edges = [r.n_edges for r in results_sweep]
    circuit_faithfulness = [r.circuit_faithfulness for r in results_sweep]
    elbow = find_knee(n_edges=n_edges, circuit_faithfulness=circuit_faithfulness)

    elbow_matches = [r for r in results_sweep if r.n_edges == elbow.elbow]
    best_circuit = (
        elbow_matches[0]
        if elbow_matches
        else max(results_sweep, key=lambda r: r.circuit_faithfulness)
    )
    return best_circuit, elbow


def _save_circuit_image(graph: Graph, config: EAPConfig) -> Optional[str]:
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        print("No pygraphviz installed; skipping circuit image.")
        return None

    circuit_image_path = config.image_file_name("circuit")
    graph.to_image(circuit_image_path)
    return circuit_image_path


def _save_knee_plot(elbow: KneeLocator, config: EAPConfig) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("No matplotlib installed; skipping knee plot.")
        return None

    knee_image_path = config.image_file_name("knee")
    elbow.plot_knee(xlabel="n_edges", ylabel="circuit_faithfulness")
    plt.savefig(knee_image_path, bbox_inches="tight")
    plt.close()
    return knee_image_path


def run(config: EAPConfig) -> EAPResults:
    global_device = (
        torch.get_default_device() if hasattr(torch, "get_default_device") else None
    )
    model = None
    graph = None
    device = None

    try:
        dataset = CrossLingualRuleFollowingDataset(config.dataset_config)
        dataset_loader = dataset.to_dataloader(
            batch_size=config.batch_size, collate_fn=collate_EAP
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)

        model = _load_model(config, device)

        graph = Graph.from_model(model)
        attribute(
            model,
            graph,
            dataset_loader,
            partial(logit_difference, loss=True, mean=True),
            method=config.method.value,
            ig_steps=config.ig_steps,
        )

        baseline, corrupted_baseline = _compute_baselines(model, dataset_loader)
        results_sweep = _sweep_n_edges(
            config, model, graph, dataset_loader, baseline, corrupted_baseline
        )
        best_circuit, elbow = _select_best_circuit(results_sweep)

        # Restore graph to the winning (elbow-selected) circuit before rendering -
        # after the sweep, graph still reflects whichever n_edges was tested last.
        graph.apply_greedy(n_edges=best_circuit.n_edges, absolute=True)
        circuit_image_path = _save_circuit_image(graph, config)
        knee_image_path = _save_knee_plot(elbow, config)

        return EAPResults(
            baseline=baseline,
            corrupted_baseline=corrupted_baseline,
            metadata={
                "config": config.model_dump_json(exclude_none=True),
            },
            results=results_sweep,
            best=best_circuit,
            circuit_image_path=circuit_image_path,
            knee_image_path=knee_image_path,
        )
    finally:
        # Always runs, whether `run` returned normally or raised partway through.
        del model
        del graph
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
        if global_device is not None:
            torch.set_default_device(global_device)
