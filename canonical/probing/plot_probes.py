"""Plot functions to visualise the probes results."""

from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from canonical.probing.config import RunConfig
from canonical.probing.utils import safe_roc_auc


def accuracies_from_evaluation_results(evaluation_results) -> Tuple[List[Dict[int, float]], List[str]]:
    """Flattens an evaluation_results dict into parallel (accuracies, legend) lists."""
    accuracies, legend = [], []
    for model_name, layer_dict in evaluation_results.items():
        layer_dict = {int(layer): report for layer, report in layer_dict.items()}
        accuracies.append({layer: report["accuracy"] for layer, report in layer_dict.items()})
        legend.append(model_name)
    return accuracies, legend


def plot_accuracy_per_layer(
    cfg: RunConfig,
    accuracies: List[Dict[int, float]],
    legend: List[str],
    save_path_prefix: str = "",
):
    """Plots accuracy per layer for as many {layer: accuracy} series as given, on the same axes."""
    plt.figure(figsize=(10, 6))
    all_layers = set()
    for layer_accuracies, label in zip(accuracies, legend):
        layer_accuracies = {int(layer): acc for layer, acc in layer_accuracies.items()}
        layers = sorted(layer_accuracies.keys())
        values = [layer_accuracies[layer] for layer in layers]
        plt.plot(layers, values, marker="o", label=label)
        all_layers.update(layers)

    plt.title("Probe Accuracy per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.xticks(sorted(all_layers))
    plt.grid()
    plt.legend()

    # save
    plt.savefig(f"{cfg.vis_path}/{save_path_prefix}AccuracyPerLayer{cfg.language}.png")

    # show
    plt.show()


def plot_auroc_curves(
    cfg: RunConfig,
    y_trues: List[np.ndarray],
    y_scores: List[np.ndarray],
    legend: List[str],
    save_path_prefix: str = "",
):
    """Plots ROC curves for as many (y_true, y_score) pairs as given, on the same axes."""
    plt.figure(figsize=(10, 6))
    for y_true, y_score, label in zip(y_trues, y_scores, legend):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = safe_roc_auc(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.legend()

    # save
    plt.savefig(f"{cfg.vis_path}/{save_path_prefix}AUROC{cfg.language}.png")

    # show
    plt.show()


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Visualisations for probing results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cfg_args(sp):
        sp.add_argument("--language", required=True)
        sp.add_argument("--n-layers", type=int, required=True)
        sp.add_argument("--dataset-name", required=True)
        sp.add_argument("--results-folder", default=None)
        sp.add_argument("--save-path-prefix", default="")

    accuracy = subparsers.add_parser("accuracy")
    accuracy.add_argument(
        "--eval-json-path", required=True, help="Path to a saved *Eval_{language}.json file."
    )
    add_cfg_args(accuracy)

    auroc = subparsers.add_parser("auroc")
    auroc.add_argument(
        "--y-true-paths", required=True, nargs="+", help="One .npy path per curve."
    )
    auroc.add_argument(
        "--y-score-paths", required=True, nargs="+", help="One .npy path per curve."
    )
    auroc.add_argument("--legend", required=True, nargs="+", help="One label per curve.")
    add_cfg_args(auroc)

    return parser


def main():
    import json

    import numpy as np
    from canonical.probing.utils import create_results_path

    args = _build_arg_parser().parse_args()

    cfg = RunConfig(
        language=args.language,
        n_layers=args.n_layers,
        dataset_name=args.dataset_name,
        results_folder=args.results_folder,
    )
    create_results_path(cfg)

    if args.command == "accuracy":
        with open(args.eval_json_path) as f:
            evaluation_results = json.load(f)
        accuracies, legend = accuracies_from_evaluation_results(evaluation_results)
        plot_accuracy_per_layer(cfg, accuracies, legend, save_path_prefix=args.save_path_prefix)
    elif args.command == "auroc":
        y_trues = [np.load(path) for path in args.y_true_paths]
        y_scores = [np.load(path) for path in args.y_score_paths]
        plot_auroc_curves(
            cfg, y_trues, y_scores, args.legend, save_path_prefix=args.save_path_prefix
        )


if __name__ == "__main__":
    main()
