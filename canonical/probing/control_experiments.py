"""
1. Train on shuffled labels
2. p-value, subsampling
3. cross-validation
4. Analysis (not here): multiple comparisons (easy for us across langs, rule cats etc.)
"""

import random
from typing import Callable, List

import numpy as np
from canonical.probing.config import RunConfig
from canonical.probing.evaluate_probes import evaluate
from canonical.probing.probes_dataset_creation_script import create_from_xy_text
from canonical.probing.pydantic_models import ShuffledLabelsResults
from canonical.probing.train_probes import training


def train_on_shuffled_labels(
    cfg: RunConfig,
    activations_in_hf: str,
    y_in_hf: str,
    text_index_in_hf: str,
    hf_repo_ix: str,
    hf_repo_type: str,
    classifiers: List[Callable],
) -> ShuffledLabelsResults:
    """Control training on shuffled labels."""
    # loading and shuffling the data
    dataset = create_from_xy_text(
        activations_in_hf,
        y_in_hf,
        text_index_in_hf,
        hf_repo_ix,
        hf_repo_type=hf_repo_type,
    )
    np.random.shuffle(dataset.train_y)

    # training on shuffled labels
    shuffled_path_prefix = "ShuffledLabels"
    shuffled_clfs, shuffled_train_path = training(
        cfg,
        classifiers,
        dataset.train_x,
        dataset.train_y,
        save_path_prefix=shuffled_path_prefix,
    )

    # evaluating shuffled labels
    shuffled_eval_results, shuffled_eval_path = evaluate(
        cfg,
        shuffled_clfs,
        dataset.test_x,
        dataset.test_y,
        save_path_prefix=shuffled_path_prefix,
    )
    return ShuffledLabelsResults(
        shuffled_eval_results=shuffled_eval_results,
        shuffled_eval_path=shuffled_eval_path,
        shuffled_train_path=shuffled_train_path,
    )


def p_value_control():
    pass


def cross_validation_training():
    pass


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Control experiment: train probes on shuffled labels."
    )
    parser.add_argument("--activations-in-hf", required=True)
    parser.add_argument("--y-in-hf", required=True)
    parser.add_argument("--text-index-in-hf", required=True)
    parser.add_argument("--hf-repo-ix", required=True)
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument(
        "--classifiers", required=True, help="Comma-separated classifier names."
    )
    parser.add_argument("--language", required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--results-folder", default=None)
    return parser


def main():
    from canonical.probing.utils import build_classifiers, create_results_path

    args = _build_arg_parser().parse_args()

    cfg = RunConfig(
        language=args.language,
        n_layers=args.n_layers,
        dataset_name=args.dataset_name,
        results_folder=args.results_folder,
    )
    create_results_path(cfg)
    classifiers = build_classifiers(args.classifiers.split(","))
    results = train_on_shuffled_labels(
        cfg,
        args.activations_in_hf,
        args.y_in_hf,
        args.text_index_in_hf,
        args.hf_repo_ix,
        args.hf_repo_type,
        classifiers,
    )
    print(results)


if __name__ == "__main__":
    main()
