"""Main script for running the probes."""

from canonical.probing.utils import create_results_path, save_clf_with_skops, split_by_layer, upload_repo_to_hf
from canonical.probing.probes_dataset_creation_script import create_canonical_dataset
from canonical.probing.train_probes import training
from canonical.probing.evaluate_probes import evaluate
from canonical.probing.plot_probes import accuracies_from_evaluation_results, plot_accuracy_per_layer
from canonical.probing.control_experiments import (
    p_value_control,
    train_on_shuffled_labels,
    weights_vs_diff_of_means,
)
from canonical.probing.config import RunConfig

from typing import Any, Callable, List, Optional


def main(
    jsonl_in_hf: str,
    hf_repo_ix: str,
    classifiers: List[Callable],
    n_layers: int,
    dataset_name: str,
    hf_repo_type: str = "dataset",
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    remove_local: bool = False,
    upload_to_hf: bool = False,
    language: str = "en",
    results_folder: Optional[str] = None,
):
    """Main function for running training, evaluation and visualisation of the probes."""
    # init config
    run_cfg = RunConfig(
        language=language,
        n_layers=n_layers,
        dataset_name=dataset_name,
        results_folder=results_folder,
    )

    # create necessary folders
    create_results_path(run_cfg)

    # loading and splitting the data
    dataset = create_canonical_dataset(
        jsonl_in_hf,
        hf_repo_ix,
        hf_repo_type=hf_repo_type,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
    )

    # training
    trained_classifiers = training(
        run_cfg,
        classifiers,
        split_by_layer(dataset.train_x),
        dataset.train_y,
    )
    trained_probes_path = save_clf_with_skops(run_cfg, trained_classifiers)
    if upload_to_hf:
        upload_repo_to_hf(trained_probes_path, run_cfg, remove_local=remove_local)

    # evaluation
    validation_evals, validation_eval_path = evaluate(
        run_cfg,
        trained_classifiers,
        split_by_layer(dataset.test_x),
        dataset.test_y,
        save_path_prefix="Valid",
    )
    held_out_evals, held_eval_path = evaluate(
        run_cfg,
        trained_classifiers,
        split_by_layer(dataset.held_x),
        dataset.held_y,
        save_path_prefix="Held",
    )
    # run control experiments
    shuffled_results = train_on_shuffled_labels(
        run_cfg,
        jsonl_in_hf,
        hf_repo_ix,
        classifiers,
        hf_repo_type=hf_repo_type,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
    )
    p_value_results = p_value_control(
        run_cfg,
        jsonl_in_hf,
        hf_repo_ix,
        classifiers,
        hf_repo_type=hf_repo_type,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
        load_normal_eval_scores=f"{run_cfg.eval_path}/ValidEval_{run_cfg.language}.json",
    )
    weights_results = weights_vs_diff_of_means(
        run_cfg,
        jsonl_in_hf,
        hf_repo_ix,
        classifiers,
        hf_repo_type=hf_repo_type,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
        trained_clfs_folder=trained_probes_path,
    )
    # visualisation
    validation_accuracies, validation_legend = accuracies_from_evaluation_results(validation_evals)
    plot_accuracy_per_layer(run_cfg, validation_accuracies, validation_legend, save_path_prefix="Valid")
    held_accuracies, held_legend = accuracies_from_evaluation_results(held_out_evals)
    plot_accuracy_per_layer(run_cfg, held_accuracies, held_legend, save_path_prefix="Held")


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full probing pipeline: train, evaluate, visualise."
    )
    parser.add_argument("--jsonl-in-hf", required=True)
    parser.add_argument("--hf-repo-ix", required=True)
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument(
        "--activations-in-hf",
        default=None,
        help="If given, download precomputed activations instead of extracting them.",
    )
    parser.add_argument("--y-in-hf", default=None)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Required when activations aren't precomputed, to extract them on the fly.",
    )
    parser.add_argument("--hook-name", default="hook_resid_post")
    parser.add_argument("--pos-slice", type=int, default=-1)
    parser.add_argument(
        "--classifiers", required=True, help="JSON object mapping classifier name to kwargs."
    )
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--remove-local", action="store_true")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--language", default="en")
    parser.add_argument("--results-folder", default=None)
    return parser


def cli():
    import json

    from canonical.probing.utils import build_classifiers

    args = _build_arg_parser().parse_args()
    classifiers = build_classifiers(json.loads(args.classifiers))

    model = None
    if args.model_name:
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(args.model_name)

    main(
        args.jsonl_in_hf,
        args.hf_repo_ix,
        classifiers,
        args.n_layers,
        args.dataset_name,
        hf_repo_type=args.hf_repo_type,
        activations_in_hf=args.activations_in_hf,
        y_in_hf=args.y_in_hf,
        model=model,
        hook_name=args.hook_name,
        pos_slice=args.pos_slice,
        remove_local=args.remove_local,
        upload_to_hf=args.upload_to_hf,
        language=args.language,
        results_folder=args.results_folder,
    )


if __name__ == "__main__":
    cli()
