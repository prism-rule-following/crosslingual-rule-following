"""Main script for running the probes."""

from canonical.probing.utils import load_from_hf, check_XY
from canonical.probing.train_probes import training
from canonical.probing.evaluate_probes import evaluate
from typing import Callable


def main(
    hf_data_url: str,
    classifier: Callable,
    remove_local: bool = False,
    upload_probes_to_hf: str = True,
    eval_save_path: str = None,
):
    """Main function for running training, evaluation and visualisation of the probes."""
    # TODO: create this data
    # load the activations
    hf_activations = load_from_hf(hf_data_url)
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
        classifier,
        train_X,
        train_y,
        remove_local=remove_local,
        upload_to_hf_repo=upload_probes_to_hf,
    )
    # evaluation
    # TODO: upload it to hf as well
    validation_evals = evaluate(
        probe_clfs, validation_X, validation_y, save_path=eval_save_path
    )
    held_out_evals = evaluate(
        probe_clfs, heldout_X, heldout_y, save_path=eval_save_path
    )
    # TODO: visualisation


if __name__ == "__main__":
    pass
