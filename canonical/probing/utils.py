"""Util function for training the probes."""

import os
import json
from huggingface_hub import upload_folder
import numpy as np
from datasets import load_dataset
from typing import Dict

from datetime import datetime as dt

from canonical.probing.config import PROBES_FOLDER, HF_REPO_ID


# Checking data
def check_X(layer_X: Dict[int, np.ndarray]):
    """Checks the input validity and converts it to numpy."""
    import torch

    for layer in layer_X:
        X = layer_X[layer]
        if isinstance(X, list):
            layer_X[layer] = np.array(X)
        elif isinstance(X, torch.tensor):
            layer_X[layer] = X.detach().cpu().numpy()
        else:
            raise ValueError(f"X is not of the right format, got {type(X)}")

    return layer_X


def check_y(y):
    """Checks the input validity and converts it to numpy."""
    import torch

    if isinstance(y, list):
        y = np.array(y)
    elif isinstance(y, torch.tensor):
        y = y.detach().cpu().numpy()
    else:
        raise ValueError(f"y is not of the right format, got {type(y)}")
    return y


def check_XY(data, labels):
    """Checks the input validity and converts it to numpy."""
    # shape
    if not len(X) == len(y):
        raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
    # type
    X = check_X(data)
    y = check_y(labels)
    return X, y


# Data loading and saving
def load_from_hf(url: str):
    try:
        return load_dataset(url)
    except Exception:
        print(f"An error occurred. Unable to load {url!r} from HuggingFace.")
        raise


def save_probes_to_disk(probe_dict: Dict[int, object], path: str):
    """Saves the data to disk."""

    # make folder name
    folder = f"{path}/{PROBES_FOLDER}/{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(folder, exist_ok=True)

    # create and save a metadata file
    metadata = {
        "probe_model": type(probe_dict[0]).__name__,
        "layers": list(probe_dict.keys()),
    }
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # save probes
    import skops.io as sio

    try:
        for layer, clf in probe_dict.items():
            sio.dump(clf, f"{folder}/layer_{layer}.skops")
    except Exception as e:
        print(f"An error occurred while saving probes to disk: {e}")
        raise
    return folder


def upload_to_hf(
    probe_data: Dict[int, object], repo_path: str, remove_local: bool = False
):
    """Uploads the data to HuggingFace."""
    from huggingface_hub import upload_folder

    folder_path = save_probes_to_disk(probe_data, repo_path)
    try:
        upload_folder(
            folder_path=folder_path,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
    except Exception as e:
        print(f"An error occurred while uploading probes to HuggingFace: {e}")
        raise
    if remove_local:
        import shutil

        shutil.rmtree(PROBES_FOLDER)
