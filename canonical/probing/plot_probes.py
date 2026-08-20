"""Plot functions to visualise the probes results."""

from matplotlib import pyplot as plt

from canonical.probing.config import RunConfig


def plot_accuracy_per_layer(
    cfg: RunConfig, evaluation_results, save_path_prefix: str = ""
):
    """Plots the accuracy per layer."""
    layers = list(evaluation_results.keys())
    accuracies = [evaluation_results[layer]["accuracy"] for layer in layers]

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, accuracies, marker="o")
    plt.title("Probe Accuracy per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.xticks(layers)
    plt.grid()

    # save
    plt.savefig(f"{cfg.vis_path}/{save_path_prefix}AccuracyPerLayer{cfg.language}")

    # show
    plt.show()
