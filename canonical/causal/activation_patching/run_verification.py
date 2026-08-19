"""Runs the full circuit verification suite and saves a JSON report."""

import argparse
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader

from canonical.causal.activation_patching.circuit_verifier import CircuitVerifier
from canonical.causal.activation_patching.utils import (
    METRIC_REGISTRY,
    VALID_INTERVENTIONS,
    InterventionType,
    resolve_metrics,
    to_serialisable,
    validate_intervention,
)

if TYPE_CHECKING:
    # import only when necessary
    from eap.graph import Graph
    from transformer_lens import HookedTransformer

DATETIME_FORMAT = "%d-%m-%Y_%H-%M-%S"


def run_verification(
    model: "HookedTransformer",
    graph: "Graph",
    dataloader: DataLoader,
    metric_names: List[str],
    model_name: str,
    dataset: str,
    steps: int = 10,
    out_path: str = "verification_results.json",
    intervention: InterventionType = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    checker: Optional[Callable[[str], float]] = None,
    tokenizer: Any = None,
    internal_cache: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
    target_internal_cache: Optional[
        Union[Dict[str, torch.Tensor], torch.Tensor]
    ] = None,
) -> Dict[str, Any]:
    """Run sufficiency, necessity, completeness and minimality checks in order.

    `metric_names` is a list of metric names, e.g. ["logit_diff", "adherence", "fidelity"]

    `model_name` and `dataset` (e.g. an HF dataset id/link) are recorded as run metadata
    in the resulting file.

    `intervention` must be one of "mean", "patching" or "zero" and is used for every
    verification step; `intervention_dataloader` is required by EAP when intervention
    is "mean"

    Results are saved to `out_path` as JSON and also returned.
    """
    validate_intervention(intervention)
    metrics = resolve_metrics(
        metric_names,
        checker=checker,
        tokenizer=tokenizer,
        internal_cache=internal_cache,
        target_internal_cache=target_internal_cache,
    )
    start_time = datetime.now()

    verifier = CircuitVerifier(model, graph, dataloader, metrics)

    runs = {}
    runs["sufficiency"] = verifier.verify_sufficiency(
        intervention=intervention, intervention_dataloader=intervention_dataloader
    )
    runs["necessity"] = verifier.verify_necessity(
        intervention=intervention, intervention_dataloader=intervention_dataloader
    )
    runs["completeness"] = verifier.verify_completeness(
        steps=steps,
        intervention=intervention,
        intervention_dataloader=intervention_dataloader,
    )
    runs["minimality"] = verifier.verify_minimality(
        model,
        graph,
        dataloader,
        metrics,
        intervention=intervention,
        intervention_dataloader=intervention_dataloader,
    )

    report = {
        "meta": {
            "model_name": model_name,
            "dataset": dataset,
            "datetime": start_time.strftime(DATETIME_FORMAT),
            "intervention": intervention,
        },
        "runs": runs,
    }

    with open(out_path, "w") as f:
        json.dump(to_serialisable(report), f, indent=2)

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full circuit verification suite (sufficiency, "
        "necessity, completeness, minimality) from the command line."
    )
    parser.add_argument(
        "model_name",
        help="TransformerLens model name, e.g. 'attn-only-1l' or 'gpt2-small'",
    )
    parser.add_argument(
        "graph_path", help="Path to a saved circuit, i.e. eap Graph.to_json(...) output"
    )
    parser.add_argument(
        "dataset_csv", help="CSV with clean/corrupted/label columns for evaluation"
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset identifier/link recorded in the report's meta field, e.g. an HF dataset link",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--metrics", nargs="+", default=["logit_diff"], choices=list(METRIC_REGISTRY)
    )
    parser.add_argument(
        "--intervention", default="patching", choices=list(VALID_INTERVENTIONS)
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Greedy search steps for completeness verification",
    )
    parser.add_argument(
        "--neutral-n-sentences",
        type=int,
        default=64,
        help="Size of the neutral baseline sample; only used when --intervention mean",
    )
    parser.add_argument("--neutral-batch-size", type=int, default=16)
    parser.add_argument("--out", dest="out_path", default="verification_results.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = _build_arg_parser().parse_args(argv)

    from transformer_lens import HookedTransformer
    from eap.graph import Graph

    from canonical.causal.activation_patching.dataloaders import (
        build_clean_corrupted_dataloader,
        build_neutral_dataloader,
    )

    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    graph = Graph.from_json(args.graph_path)
    dataloader = build_clean_corrupted_dataloader(
        args.dataset_csv, batch_size=args.batch_size
    )

    intervention_dataloader = None
    if args.intervention == "mean":
        intervention_dataloader = build_neutral_dataloader(
            n_sentences=args.neutral_n_sentences, batch_size=args.neutral_batch_size
        )

    return run_verification(
        model,
        graph,
        dataloader,
        args.metrics,
        args.model_name,
        args.dataset,
        steps=args.steps,
        out_path=args.out_path,
        intervention=args.intervention,
        intervention_dataloader=intervention_dataloader,
    )


if __name__ == "__main__":
    main()
