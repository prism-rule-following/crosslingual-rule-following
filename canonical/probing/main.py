"""Main script for running the probes."""

from canonical.probing.utils import load_from_hf, check_XY, create_results_path
from canonical.probing.train_probes import training
from canonical.probing.evaluate_probes import evaluate
from canonical.probing.plot_probes import plot_accuracy_per_layer
from canonical.probing.config import RunConfig

from typing import Callable, List


def main(
    hf_data_url: str,
    classifiers: List[Callable],
    remove_local: bool = False,
    language: str = "en",
):
    """Main function for running training, evaluation and visualisation of the probes."""
    # load the activations TODO: create this data - asked how to do it better.
    hf_activations = load_from_hf(hf_data_url)

    # init config
    run_cfg = RunConfig(
        language=language,
        n_layers=hf_activations["metadata"]["n_layers"],
        dataset_name=hf_activations["metadata"]["name"],
    )

    # create necessary folders
    results_path = create_results_path(run_cfg)

    # initialising splits
    train_X, train_y = check_XY(hf_activations["train"], hf_activations["train_labels"])
    validation_X, validation_y = check_XY(
        hf_activations["validation"], hf_activations["validation_labels"]
    )
    heldout_X, heldout_y = check_XY(
        hf_activations["heldout"], hf_activations["heldout_labels"]
    )

    # training
    probe_clfs = training(
        run_cfg,
        classifiers,
        train_X,
        train_y,
        remove_local=remove_local,
    )

    # evaluation
    validation_evals = evaluate(
        run_cfg,
        probe_clfs,
        validation_X,
        validation_y,
        save_path_prefix="Valid",
    )
    held_out_evals = evaluate(
        run_cfg,
        probe_clfs,
        heldout_X,
        heldout_y,
        save_path_prefix="Held",
    )
    # visualisation
    plot_accuracy_per_layer(run_cfg, validation_evals, save_path_prefix="Valid")
    plot_accuracy_per_layer(run_cfg, held_out_evals, save_path_prefix="Held")


if __name__ == "__main__":
    pass
