"""Probes training."""

from typing import Callable, Dict, List

import numpy as np
from canonical.probing.config import RunConfig
from sklearn.base import clone


def training(
    run_config: RunConfig,
    classifiers: List[Callable],
    train_X: Dict[int, np.ndarray],
    train_y: np.ndarray,
) -> Dict[str, Dict[int, object]]:
    """Train the probes. Pure: fits and returns classifiers, no file I/O."""
    # check clfs
    valid_clfs = [clf for clf in classifiers if hasattr(clf, "fit")]
    if len(valid_clfs) == 0:
        raise ValueError(
            f"Classifiers do not have a fit() method. Pass valid classifiers."
        )

    # do the training
    trained_classifiers = {}
    for clf in valid_clfs:
        name = type(clf).__name__
        trained_classifiers[name] = {}
        try:
            # training per layer
            for layer in train_X:
                layer_clf = clone(clf)
                layer_clf.fit(train_X[layer], train_y)
                trained_classifiers[name][layer] = layer_clf
        except Exception as e:
            raise ValueError(f"An error occurred while training the classifier: {e}")

    return trained_classifiers


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train probes per layer on activations."
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Local filepath for double-rule data, unused otherwise.",
    )
    parser.add_argument("--classifiers", required=True, help="JSON object mapping classifier name to kwargs.")
    parser.add_argument("--llm", default=None, type=str)
    parser.add_argument("--hook", default="hook_resid_post", type=str)
    parser.add_argument("--pos-slice", default=-1, type=int)
    parser.add_argument("--activations-hf", type=str, default=None)
    parser.add_argument("--y-hf", type=str, default=None)
    parser.add_argument("--jsonl-in-hf", type=str)
    parser.add_argument("--hf-repo-ix", type=str)
    parser.add_argument("--hf-repo-type", default="dataset", type=str)
    parser.add_argument("--language", required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--results-folder", default=None)
    parser.add_argument("--remove-local", action="store_true")
    parser.add_argument("--save-path-prefix", default="")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--train-double-rule-data", action="store_true")
    return parser


def main():
    import json

    from canonical.probing.probes_dataset_creation_script import (
        create_canonical_dataset,
        opposite_statuses_rules,
    )
    from canonical.probing.utils import (
        build_classifiers,
        create_results_path,
        save_clf_with_skops,
        split_by_layer,
        upload_repo_to_hf,
    )

    args = _build_arg_parser().parse_args()

    model = None
    if args.llm:
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(args.llm)

    if args.train_double_rule_data:
        dataset = opposite_statuses_rules(
            args.data_path,
            model,
            hook_name=args.hook,
            pos_slice=args.pos_slice,
        )
        train_x, train_y = dataset.doublerule_x, dataset.doublerule_y
    else:
        dataset = create_canonical_dataset(
            args.jsonl_in_hf,
            args.hf_repo_ix,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_hf,
            y_in_hf=args.y_hf,
            model=model,
            hook_name=args.hook,
            pos_slice=args.pos_slice,
        )
        train_x, train_y = dataset.train_x, dataset.train_y

    run_config = RunConfig(
        language=args.language,
        n_layers=args.n_layers,
        dataset_name=args.dataset_name,
        results_folder=args.results_folder,
    )
    create_results_path(run_config)
    classifiers = build_classifiers(json.loads(args.classifiers))

    trained_classifiers = training(
        run_config,
        classifiers,
        split_by_layer(train_x),
        train_y,
    )
    trained_probes_path = save_clf_with_skops(
        run_config, trained_classifiers, save_path_prefix=args.save_path_prefix
    )
    if args.upload_to_hf:
        upload_repo_to_hf(
            trained_probes_path, run_config, remove_local=args.remove_local
        )
    print(trained_probes_path)


if __name__ == "__main__":
    main()
