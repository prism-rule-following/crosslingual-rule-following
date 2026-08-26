"""Util function for training the probes."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from canonical.probing.config import RunConfig
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def safe_roc_auc(y_true, y_score) -> float:
    """roc_auc_score needs both classes present; a permutation or a small eval split can produce only one by chance."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


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
        elif not isinstance(X, np.ndarray):
            raise ValueError(f"X is not of the right format, got {type(X)}")

    return layer_X


def check_y(y):
    """Checks the input validity and converts it to numpy."""
    import torch

    if isinstance(y, list):
        y = np.array(y)
    elif isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    elif not isinstance(y, np.ndarray):
        raise ValueError(f"y is not of the right format, got {type(y)}")
    return y


def check_length(*arrays: Any):
    """Checks the file length."""
    assert len({len(arr) for arr in arrays}) == 1


def split_by_layer(x: np.ndarray) -> Dict[int, np.ndarray]:
    return {layer: x[:, layer, :] for layer in range(x.shape[1])}


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


def download_jsonl_from_hf(
    data_path_in_repo: str,
    repo_ix: str,
    repo_type: str = "dataset",
) -> List[Dict]:
    """Reads and loads jsonl from HF."""
    try:
        data_path = hf_hub_download(
            repo_id=repo_ix,
            filename=data_path_in_repo,
            repo_type=repo_type,
        )
        with open(data_path) as f:
            datalines = [json.loads(line) for line in f]
        return datalines
    except Exception as e:
        print(
            f"Error while processing JSONL data. Path from HF {data_path_in_repo}. Error: {e}"
        )
        raise


# TODO: add various functions to create metadata files for each folder
# They should include the language too!


def upload_repo_to_hf(
    repo_path: str,
    cfg: Optional[RunConfig] = None,
    remove_local: bool = False,
    repo_type: str = "model",
    repo_id: Optional[str] = None,
    path_in_repo: Optional[str] = None,
):
    """Uploads the data to HuggingFace. Uploads to cfg.hf_repo_id unless repo_id is given."""
    from huggingface_hub import create_repo, upload_folder

    target_repo_id = repo_id or cfg.hf_repo_id
    try:
        create_repo(target_repo_id, repo_type=repo_type, exist_ok=True)
        upload_folder(
            folder_path=repo_path,
            repo_id=target_repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
        )
    except Exception as e:
        print(f"An error occurred while uploading probes to HuggingFace: {e}")
        raise
    # remove local if we dont want to keep it
    if remove_local:
        import shutil

        shutil.rmtree(repo_path)


def save_dataset_locally(
    local_dir: str,
    activations: np.ndarray,
    labels: np.ndarray,
    text,
    hook_name: str = "hook_resid_post",
) -> str:
    """Saves activations/labels/text locally as {hook_name}_activations.npy/{hook_name}_labels.npy/text.parquet(or .json)."""
    os.makedirs(local_dir, exist_ok=True)
    np.save(f"{local_dir}/{hook_name}_activations.npy", activations)
    np.save(f"{local_dir}/{hook_name}_labels.npy", labels)
    if isinstance(text, pd.DataFrame):
        text.to_parquet(f"{local_dir}/text.parquet")
    else:
        with open(f"{local_dir}/text.json", "w") as file:
            json.dump(text, file)
    return local_dir


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
        with open(
            f"{cfg.trained_probes_path}/metadata_{cfg.language}.json", "w"
        ) as file:
            json.dump(metadata, file)
    except Exception as e:
        print(f"Exception during saving the metadata for trained probes: {e}")
        raise


def save_clf_with_skops(
    run_config: RunConfig,
    trained_classifiers: Dict[str, Dict[int, object]],
    save_path_prefix: str = "",
) -> str:
    """Saves trained classifiers per layer into the run's trained-probes folder."""
    import skops.io as sio

    try:
        for name, layer_clfs in trained_classifiers.items():
            for layer, clf in layer_clfs.items():
                sio.dump(
                    clf,
                    f"{run_config.trained_probes_path}/{save_path_prefix}{name}_layer_{layer}_{run_config.language}.skops",
                )
    except Exception as e:
        raise ValueError(f"An error occurred while saving the classifiers: {e}")
    make_trained_probes_metadata(run_config, trained_classifiers)
    return run_config.trained_probes_path


def load_trained_clfs(
    trained_probes_path: str, language: str, save_path_prefix: str = ""
) -> Dict[str, Dict[int, object]]:
    """Loads previously saved skops classifiers from a trained-probes folder."""
    import glob

    import skops.io as sio

    valid_names = {cls.__name__ for cls in CLASSIFIER_REGISTRY.values()}
    trained_classifiers = {}
    suffix = f"_{language}.skops"
    pattern = os.path.join(trained_probes_path, f"{save_path_prefix}*{suffix}")
    for path in glob.glob(pattern):
        stem = os.path.basename(path)[len(save_path_prefix) : -len(suffix)]
        name, layer = stem.rsplit("_layer_", 1)
        if name not in valid_names:
            continue
        untrusted_types = sio.get_untrusted_types(file=path)
        trained_classifiers.setdefault(name, {})[int(layer)] = sio.load(
            path, trusted=untrusted_types
        )
    return trained_classifiers


def check_clf_match(loaded_names, classifiers) -> None:
    """Checks that a set of loaded classifier names matches the classifiers passed for this call."""
    expected = {type(clf).__name__ for clf in classifiers}
    actual = set(loaded_names)
    if expected != actual:
        raise ValueError(
            f"Loaded classifiers {sorted(actual)} do not match classifiers passed for this call {sorted(expected)}."
        )


def extract_model_activations(
    model: Any,
    texts: List[str],
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    batch_size: int = 4,
) -> np.ndarray:
    """Extract activations with TransformerLens HookedTransformer and returns a tensor.
    Runs in batches under no_grad, moving each batch to CPU and freeing GPU memory
    before the next, to avoid holding the whole input in GPU memory at once.
    """
    try:
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                _, cache = model.run_with_cache(
                    texts[start : start + batch_size],
                    names_filter=lambda name: name.endswith(hook_name),
                    pos_slice=pos_slice,
                )
                layers = sorted(int(name.split(".")[1]) for name in cache.keys())
                stacked = torch.stack(
                    [cache[f"blocks.{l}.{hook_name}"].squeeze(dim=1) for l in layers], dim=1
                )
                batches.append(stacked.cpu().detach().numpy())
                del cache, stacked
                torch.cuda.empty_cache()
        return np.concatenate(batches, axis=0)
    except Exception as e:
        print(f"Error while extracting activations: {e}")
        raise


# TODO function for system chat settings for the model {"system": system_text, "user": query}
def make_chat_settings(
    model: Any, system_texts: List[str], queries: List[str]
) -> List[str]:
    if len(system_texts) != len(queries):
        raise ValueError(
            f"system_texts and queries must be the same length, got {len(system_texts)} and {len(queries)}"
        )
    chats = [
        [{"role": "system", "content": system_text}, {"role": "user", "content": query}]
        for system_text, query in zip(system_texts, queries)
    ]
    return [
        model.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        for chat in chats
    ]


CLASSIFIER_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "mlp": MLPClassifier,
    "knn": KNeighborsClassifier,
}


# TODO: add arg per classifier. E.g. the names could be a dict {clf: {kwargs}, ...}
def build_classifiers(names: Dict[str, Dict]) -> List[Any]:
    return [CLASSIFIER_REGISTRY[name](**kwargs) for name, kwargs in names.items()]
