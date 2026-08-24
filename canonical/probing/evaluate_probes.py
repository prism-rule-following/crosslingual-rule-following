"""Script for evaluating the probes."""

import json
from datetime import datetime as dt
from typing import Dict, Tuple

from canonical.probing.config import RunConfig
from canonical.probing.utils import upload_repo_to_hf
from sklearn.metrics import classification_report, roc_auc_score


def evaluate(
    cfg: RunConfig,
    probe_clfs,
    test_X,
    test_y,
    save_path_prefix: str = "",
    upload_to_hf: bool = False,
) -> Tuple[Dict[str, Dict[int, Dict]], str]:
    """Evaluate the probes.
    Computes classification report for each layer.
    """
    # computing evaluations
    evaluation_results = {}
    for model_name, layer_dict in probe_clfs.items():
        evaluation_results[model_name] = {}
        for layer, clf in layer_dict.items():
            predictions = clf.predict(test_X[layer])
            report = classification_report(test_y, predictions, output_dict=True)
            report["roc_auc"] = roc_auc_score(test_y, clf.predict_proba(test_X[layer])[:, 1])
            evaluation_results[model_name][layer] = report
    # saving the results
    try:
        with open(f"{cfg.eval_path}/{save_path_prefix}Eval_{cfg.language}.json", "w") as f:
            json.dump(evaluation_results, f)
    except Exception as e:
        print(f"Error occurred while saving results: {e}")
        raise

    # uploading to hf
    if upload_to_hf:
        upload_repo_to_hf(cfg.eval_path, cfg, repo_type="dataset")

    return evaluation_results, cfg.eval_path


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained probes.")
    parser.add_argument("--trained-probes-path", required=True)
    parser.add_argument(
        "--test-x-dir", required=True, help="Directory of layer_{N}.npy activation files."
    )
    parser.add_argument("--test-y-path", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--results-folder", default=None)
    parser.add_argument("--save-path-prefix", default="")
    parser.add_argument("--upload-to-hf", action="store_true")
    return parser


def main():
    import glob
    import os

    import numpy as np
    from canonical.probing.utils import create_results_path, load_trained_clfs

    args = _build_arg_parser().parse_args()

    cfg = RunConfig(
        language=args.language,
        n_layers=args.n_layers,
        dataset_name=args.dataset_name,
        results_folder=args.results_folder,
    )
    create_results_path(cfg)

    probe_clfs = load_trained_clfs(
        args.trained_probes_path, cfg.language, save_path_prefix=args.save_path_prefix
    )

    test_X = {
        int(os.path.basename(path).split("_")[1].split(".")[0]): np.load(path)
        for path in glob.glob(os.path.join(args.test_x_dir, "layer_*.npy"))
    }
    test_y = np.load(args.test_y_path)

    _, eval_path = evaluate(
        cfg,
        probe_clfs,
        test_X,
        test_y,
        save_path_prefix=args.save_path_prefix,
        upload_to_hf=args.upload_to_hf,
    )
    print(eval_path)


if __name__ == "__main__":
    main()
