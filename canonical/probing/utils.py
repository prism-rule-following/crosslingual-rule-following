"""Util function for training the probes."""

import os
import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from typing import Any, List, Dict, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from canonical.probing.config import RunConfig


def open_local_json(filepath):
    """Util function to open a local json file."""
    try:
        with open(filepath) as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Couldnt load {filepath} due to: {e}")
        raise


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


def check_length(*arrays: Any):
    """Checks the file length."""
    assert len({len(arr) for arr in arrays}) == 1


# Data loading and saving
def download_XY_from_hf(
    activations_path_in_repo: str,
    y_path_in_repo: str,
    repo_ix: str,
    repo_type: str = "dataset",
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads only certain file patterns from HF."""
    try:
        X_activations = hf_hub_download(
            repo_id=repo_ix,
            filename=activations_path_in_repo,
            repo_type=repo_type,
        )
        y_labels = hf_hub_download(
            repo_id=repo_ix,
            filename=y_path_in_repo,
            repo_type=repo_type,
        )
    except Exception as e:
        print(
            f"An error occurred. Unable to load {activations_path_in_repo} or {y_path_in_repo} from HuggingFace. {e}"
        )
        raise
    try:
        X_acts = np.load(X_activations)
        y = np.load(y_labels)
        return X_acts, y
    except Exception as e:
        print(f"Couldnt load downloaded X and y due to {e}")
        raise


def download_parquet_from_hf(
    text_index_path_in_repo: str,
    repo_ix: str,
    repo_type: str = "dataset",
):
    """Downloads the text data with indices corresponding to activations and labels.
    Function expects to download a parquet file from HF.
    """
    try:
        text_index_path = hf_hub_download(
            repo_id=repo_ix,
            filename=text_index_path_in_repo,
            repo_type=repo_type,
        )
        textdf = pd.read_parquet(text_index_path)
        return textdf
    except Exception as e:
        print(
            f"Error while processing text index. Path from HF {text_index_path}. Error: {e}"
        )
        raise


# TODO: add various functions to create metadata files for each folder
# They should include the language too!


def upload_repo_to_hf(
    repo_path: str,
    cfg: RunConfig,
    remove_local: bool = False,
    repo_type: str = "model",
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
        os.makedirs(f"{run_config.vis_path}", exist_ok=True)
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
        with open(f"{cfg.trained_probes_path}/metadata_{cfg.language}.json", "w") as file:
            json.dump(metadata, file)
    except Exception as e:
        print(f"Exception during saving the metadata for trained probes: {e}")
        raise


def extract_model_activations(
    model: Any,
    texts: List[str],
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
) -> np.ndarray:
    """Extract activations with TransformerLens HookedTransformer and returns a tensor."""
    try:
        _, cache = model.run_with_cache(
            texts,
            names_filter=lambda name: name.endswith(hook_name),
            pos_slice=pos_slice,
        )
        layers = sorted(int(name.split(".")[1]) for name in cache.keys())
        stacked = torch.stack(
            [cache[f"blocks.{l}.{hook_name}"].squeeze() for l in layers], dim=1
        )
        activations = stacked.cpu().detach().numpy()
        return activations
    except Exception as e:
        print(f"Error while extracting activations: {e}")
        raise


# TODO function for system chat settings for the model {"system": system_text, "user": query}
def make_chat_settings(
    model: Any, system_texts: List[str], queries: List[str]
) -> List[str]:
    chats = [
        [{"role": "system", "content": system_text}, {"role": "user", "content": query}]
        for system_text, query in zip(system_texts, queries)
    ]
    return [
        model.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        for chat in chats
    ]


CLASSIFIER_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "mlp": MLPClassifier,
    "knn": KNeighborsClassifier,
}


def build_classifiers(names: List[str]) -> List[Any]:
    return [CLASSIFIER_REGISTRY[name]() for name in names]
