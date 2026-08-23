"""Plot functions to visualise the probes results."""

from matplotlib import pyplot as plt

from canonical.probing.config import RunConfig


def plot_accuracy_per_layer(
    cfg: RunConfig, evaluation_results, save_path_prefix: str = ""
):
    """Plots the accuracy per layer."""
    plt.figure(figsize=(10, 6))
    all_layers = set()
    for model_name, layer_dict in evaluation_results.items():
        layers = sorted(layer_dict.keys())
        accuracies = [layer_dict[layer]["accuracy"] for layer in layers]
        plt.plot(layers, accuracies, marker="o", label=model_name)
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


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Plot accuracy per layer from a saved evaluation.")
    parser.add_argument("--eval-json-path", required=True, help="Path to a saved *Eval_{language}.json file.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--results-folder", default=None)
    parser.add_argument("--save-path-prefix", default="")
    return parser


def main():
    import json

    from canonical.probing.utils import create_results_path

    args = _build_arg_parser().parse_args()

    cfg = RunConfig(
        language=args.language,
        n_layers=args.n_layers,
        dataset_name=args.dataset_name,
        results_folder=args.results_folder,
    )
    create_results_path(cfg)
    with open(args.eval_json_path) as f:
        evaluation_results = json.load(f)

    plot_accuracy_per_layer(cfg, evaluation_results, save_path_prefix=args.save_path_prefix)


if __name__ == "__main__":
    main()
