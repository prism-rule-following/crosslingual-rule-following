"""Probes training."""

from typing import Callable, Dict, List, Tuple

import numpy as np
from canonical.probing.config import RunConfig
from canonical.probing.utils import make_trained_probes_metadata, upload_repo_to_hf
from sklearn.base import clone


def training(
    run_config: RunConfig,
    classifiers: List[Callable],
    train_X: Dict[int, np.ndarray],
    train_y: np.ndarray,
    remove_local: bool = False,
    save_path_prefix: str = "",
    upload_to_hf: bool = False,
) -> Tuple[Dict[str, Dict[int, object]], str]:
    """Train the probes."""
    import skops.io as sio

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
                sio.dump(
                    layer_clf,
                    f"{run_config.trained_probes_path}/{save_path_prefix}{name}_layer_{layer}_{run_config.language}.skops",
                )

        except Exception as e:
            raise ValueError(f"An error occurred while training the classifier: {e}")
    # make and save metadata file
    make_trained_probes_metadata(run_config, trained_classifiers)

    # uploading the probes repo to hf
    if upload_to_hf:
        upload_repo_to_hf(
            run_config.trained_probes_path,
            run_config,
            remove_local=remove_local,
            save_path_prefix=save_path_prefix,
        )

    return trained_classifiers, run_config.trained_probes_path


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train probes per layer on activations.")
    parser.add_argument(
        "--train-x-dir", required=True, help="Directory of layer_{N}.npy activation files."
    )
    parser.add_argument("--train-y-path", required=True, help="Path to a .npy file of labels.")
    parser.add_argument(
        "--classifiers", required=True, help="Comma-separated classifier names."
    )
    parser.add_argument("--language", required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--results-folder", default=None)
    parser.add_argument("--remove-local", action="store_true")
    parser.add_argument("--save-path-prefix", default="")
    parser.add_argument("--upload-to-hf", action="store_true")
    return parser


def main():
    import glob
    import os

    from canonical.probing.utils import build_classifiers, create_results_path

    args = _build_arg_parser().parse_args()

    train_X = {
        int(os.path.basename(path).split("_")[1].split(".")[0]): np.load(path)
        for path in glob.glob(os.path.join(args.train_x_dir, "layer_*.npy"))
    }
    train_y = np.load(args.train_y_path)

    run_config = RunConfig(
        language=args.language,
        n_layers=args.n_layers,
        dataset_name=args.dataset_name,
        results_folder=args.results_folder,
    )
    create_results_path(run_config)
    classifiers = build_classifiers(args.classifiers.split(","))
    _, trained_probes_path = training(
        run_config,
        classifiers,
        train_X,
        train_y,
        remove_local=args.remove_local,
        save_path_prefix=args.save_path_prefix,
        upload_to_hf=args.upload_to_hf,
    )
    print(trained_probes_path)


if __name__ == "__main__":
    main()
