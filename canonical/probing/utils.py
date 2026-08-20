"""Util function for training the probes."""

import os
import json
import numpy as np
from datasets import load_dataset
from typing import Dict

from canonical.probing.config import RunConfig


# Checking data
def check_X(layer_X: Dict[int, np.ndarray]):
    """Checks the input validity and converts it to numpy."""
    import torch

    for layer in layer_X:
        X = layer_X[layer]
        if isinstance(X, list):
            layer_X[layer] = np.array(X)
        elif isinstance(X, torch.Tensor):
            layer_X[layer] = X.detach().cpu().numpy()
        else:
            raise ValueError(f"X is not of the right format, got {type(X)}")

    return layer_X


def check_y(y):
    """Checks the input validity and converts it to numpy."""
    import torch

    if isinstance(y, list):
        y = np.array(y)
    elif isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    else:
        raise ValueError(f"y is not of the right format, got {type(y)}")
    return y


def check_XY(data, labels):
    """Checks the input validity and converts it to numpy."""
    # check that the shape of X and y matches
    pass


# Data loading and saving
def load_from_hf(url: str):
    try:
        return load_dataset(url)
    except Exception:
        print(f"An error occurred. Unable to load {url!r} from HuggingFace.")
        raise


# TODO: add various functions to create metadata files for each folder
# They should include the language too!


def upload_repo_to_hf(
    repo_path: str,
    cfg: RunConfig,
    remove_local: bool = False,
    repo_type: str = "model",
    save_path_prefix: str = "",
):
    """Uploads the data to HuggingFace."""
    from huggingface_hub import upload_folder

    try:
        upload_folder(
            folder_path=repo_path,
            repo_id=cfg.hf_repo_id,
            repo_type=repo_type,
        )
    except Exception as e:
        print(f"An error occurred while uploading probes to HuggingFace: {e}")
        raise
    # remove local if we dont want to keep it
    if remove_local:
        import shutil

        shutil.rmtree(repo_path)


def create_results_path(run_config):
    """Create the results folder that is going to keep trained probes,
    evaluation results and visualisations."""
    try:
        # one for trained probes
        os.makedirs(f"{run_config.trained_probes_path}", exist_ok=True)
        # one for evals
        os.makedirs(f"{run_config.eval_path}", exist_ok=True)
        # one for visualisations
        os.makedirs(f"{run_config.vis_folder}", exist_ok=True)
    except Exception as e:
        print(f"Error during creating results path: {e}")
        raise


# Metadata creation
def make_trained_probes_metadata(cfg: RunConfig, probe_models: Dict):
    """Creates and saves metadata file into the trained probes folder."""
    metadata = {
        "probe_models": [name for name in probe_models.keys()],
        **cfg.model_dump(),
    }
    try:
        with open(f"{cfg.trained_probes_path}/metadata.json", "w") as file:
            json.dump(metadata, file)
    except Exception as e:
        print(f"Exception during saving the metadata for trained probes: {e}")
        raise
