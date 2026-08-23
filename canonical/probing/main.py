"""Main script for running the probes."""

from canonical.probing.utils import create_results_path
from canonical.probing.probes_dataset_creation_script import create_from_xy_text
from canonical.probing.train_probes import training
from canonical.probing.evaluate_probes import evaluate
from canonical.probing.plot_probes import plot_accuracy_per_layer
from canonical.probing.control_experiments import train_on_shuffled_labels
from canonical.probing.config import RunConfig

from typing import Callable, List, Optional


def main(
    activations_in_hf: str,
    y_in_hf: str,
    text_index_in_hf: str,
    hf_repo_ix: str,
    classifiers: List[Callable],
    n_layers: int,
    dataset_name: str,
    hf_repo_type: str = "dataset",
    remove_local: bool = False,
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
    dataset = create_from_xy_text(
        activations_in_hf,
        y_in_hf,
        text_index_in_hf,
        hf_repo_ix,
        hf_repo_type=hf_repo_type,
    )

    # training
    probe_clfs, trained_probes_path = training(
        run_cfg,
        classifiers,
        dataset.train_x,
        dataset.train_y,
        remove_local=remove_local,
    )

    # evaluation
    validation_evals, validation_eval_path = evaluate(
        run_cfg,
        probe_clfs,
        dataset.test_x,
        dataset.test_y,
        save_path_prefix="Valid",
    )
    held_out_evals, held_eval_path = evaluate(
        run_cfg,
        probe_clfs,
        dataset.held_x,
        dataset.held_y,
        save_path_prefix="Held",
    )
    # run control experiments
    shuffled_results = train_on_shuffled_labels(
        run_cfg,
        activations_in_hf,
        y_in_hf,
        text_index_in_hf,
        hf_repo_ix,
        hf_repo_type,
        classifiers,
    )
    # visualisation
    plot_accuracy_per_layer(run_cfg, validation_evals, save_path_prefix="Valid")
    plot_accuracy_per_layer(run_cfg, held_out_evals, save_path_prefix="Held")


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full probing pipeline: train, evaluate, visualise."
    )
    parser.add_argument("--activations-in-hf", required=True)
    parser.add_argument("--y-in-hf", required=True)
    parser.add_argument("--text-index-in-hf", required=True)
    parser.add_argument("--hf-repo-ix", required=True)
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument(
        "--classifiers", required=True, help="Comma-separated classifier names."
    )
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--remove-local", action="store_true")
    parser.add_argument("--language", default="en")
    parser.add_argument("--results-folder", default=None)
    return parser


def cli():
    from canonical.probing.utils import build_classifiers

    args = _build_arg_parser().parse_args()
    classifiers = build_classifiers(args.classifiers.split(","))
    main(
        args.activations_in_hf,
        args.y_in_hf,
        args.text_index_in_hf,
        args.hf_repo_ix,
        classifiers,
        args.n_layers,
        args.dataset_name,
        hf_repo_type=args.hf_repo_type,
        remove_local=args.remove_local,
        language=args.language,
        results_folder=args.results_folder,
    )


if __name__ == "__main__":
    cli()
