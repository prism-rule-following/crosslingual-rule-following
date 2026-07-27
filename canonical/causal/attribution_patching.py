from functools import partial

import pandas as pd
import torch
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import StrEnum
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from config.dataset_config import (
    CrossLingualRuleFollowingDataset,
    DatasetConfig,
    DataCategories,
    DatasetLanguage,
)


class EAPContinuationMode(StrEnum):
    normal = "normal"  # start_with, ack_invert, bold_html, language, single_word
    teacher_forced = "teacher_forced"  # banned_word, include_word, word_count


class EAPMetrics(StrEnum):
    logit_diff = "logit_diff"


class EAPMethods(StrEnum):
    EAP = "EAP"
    EAP_IG_inputs = "EAP-IG-inputs"
    EAP_IG_activations = "EAP-IG-activations"
    clean_corrupted = "clean-corrupted"


class EAPDatasetConfig(DatasetConfig):
    category: List[DataCategories] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="exactly one category to attribute over",
    )

    language: List[DatasetLanguage] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="language of the rows this config attributes over",
    )


class EAPConfig(BaseModel):
    mode: EAPContinuationMode = Field(
        default=EAPContinuationMode.normal,
        description=(
            "how the metric reads model output: 'normal' scores the first "
            "response token (start_with, ack_invert, bold_html, language, "
            "single_word); 'teacher_forced' scores a forced multi-token "
            "response (banned_word, include_word, word_count)"
        ),
    )
    model_id: str = Field(
        ..., description="HuggingFace model id of the model to attribute"
    )
    dataset_config: EAPDatasetConfig = Field(
        ..., description="dataset config, restricted to a single category per run"
    )
    batch_size: int = Field(
        default=10, description="number of examples per dataloader batch"
    )
    metrics: EAPMetrics = Field(
        default=EAPMetrics.logit_diff,
        description="which metric to attribute/evaluate with respect to",
    )
    method: EAPMethods = Field(
        default=EAPMethods.EAP_IG_activations,
        description="attribution method to run: EAP, EAP-IG-inputs, EAP-IG-activations, or clean-corrupted",
    )
    steps: int = Field(
        default=10, description="number of integrated-gradients interpolation steps"
    )
    n_edges: int = Field(
        default=20, description="number of top edges to keep via apply_greedy"
    )

    @property
    def image_file_name(self):
        model_id = self.model_id.split("/")[1]
        return f"{model_id}_{self.dataset_config.language[0].value}_{self.method}_{self.dataset_config.category[0].value}.png"


class EAPNodes(BaseModel):
    name: str = Field(..., description="node name, e.g. 'input', 'm11', 'a0.h11'")
    score: Optional[float] = Field(
        default=None,
        description="node's attribution score, if node-level scoring was used",
    )


class EAPEdges(BaseModel):
    name: str = Field(..., description="edge name, formatted as '[parent]->[child]'")
    score: Optional[float] = Field(
        default=None, description="edge's attribution score from the attribution method"
    )


class EAPResults(BaseModel):
    nodes: List[EAPNodes] = Field(
        default_factory=list, description="nodes included in the pruned circuit"
    )
    edges: List[EAPEdges] = Field(
        default_factory=list, description="edges included in the pruned circuit"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="run metadata: model cfg plus faithfulness numbers (baseline, circuit_performance)",
    )
    baseline: float = Field(
        default=0, description="metric on the full, unablated model"
    )
    corrupted_baseline: float = Field(
        default=0,
        description="metric on the model with everything corrupted (the floor reference)",
    )
    circuit_faithfulness: float = Field(
        default=0,
        description="metric on the pruned circuit alone, everything outside it ablated",
    )


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

    logits = logits[index, input_length - 1]
    return logits


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


def graph_to_response(
    g: Graph,
    baseline: float,
    corrupted_baseline: float,
    circuit_faithfulness: float,
    metadata: Optional[dict] = None,
) -> EAPResults:
    nodes = [
        EAPNodes(
            name=name,
            **({"score": float(node.score)} if g.nodes_scores is not None else {}),
        )
        for name, node in g.nodes.items()
        if node.in_graph
    ]
    edges = [
        EAPEdges(name=name, score=edge.score.item())
        for name, edge in g.edges.items()
        if edge.in_graph
    ]
    return EAPResults(
        nodes=nodes,
        edges=edges,
        baseline=baseline,
        corrupted_baseline=corrupted_baseline,
        circuit_faithfulness=circuit_faithfulness,
    )


def run(hyperparameter: str) -> EAPResults:
    global_device = (
        torch.get_default_device() if hasattr(torch, "get_default_device") else None
    )
    model = None
    g = None
    device = None

    try:
        config = EAPConfig(**hyperparameter)
        dataset = CrossLingualRuleFollowingDataset(config.dataset_config)
        dataset_loader = dataset.to_dataloader(
            batch_size=config.batch_size, collate_fn=collate_EAP
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)

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

        g = Graph.from_model(model)
        attribute(
            model,
            g,
            dataset_loader,
            partial(logit_difference, loss=True, mean=True),
            method=config.method.value,
            ig_steps=config.steps,
        )

        g.apply_greedy(n_edges=config.n_edges, absolute=True)
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

        circuit_performance = (
            evaluate_graph(
                model,
                g,
                dataset_loader,
                partial(logit_difference, loss=False, mean=False),
            )
            .mean()
            .item()
        )

        try:
            import pygraphviz

            gx = g.to_image(config.image_file_name)
            # TODO: Upload file to huggingface
        except ImportError:
            print("No pygraphviz installed; skipping this part")

        print(g.count_included_nodes(), g.count_included_edges())

        return graph_to_response(
            g,
            baseline=baseline,
            corrupted_baseline=corrupted_baseline,
            circuit_faithfulness=circuit_performance,
            metadata={
                "config": config.model_dump_json(exclude_none=True),
            },
        )
    finally:
        # Always runs, whether `run` returned normally or raised partway through.
        del model
        del g
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
        if device is not None:
            torch.set_default_device(global_device)
