"""Donor vectors (dom) per language x layer, from cached activations joined
to Stage 0 verdicts. Pure numpy/pandas -- no GPU, no live model.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from canonical.causal.vector_patching.config import ACTIVATIONS_REPO, HOOK_NAME, MODEL_SLUGS
from canonical.causal.vector_patching.pair_selection import strip_id_suffix


def load_activations(
    model_id: str, language: str, hook_name: str = HOOK_NAME, repo_id: str = ACTIVATIONS_REPO
) -> pd.DataFrame:
    """One row per (canonical_id, rule_status), with an `activation` column
    holding that row's (n_layers, d_model) array. Active-condition (`_clean`)
    rows only -- matches judge-results-active-only's scope."""
    slug = MODEL_SLUGS[model_id]
    index_path = hf_hub_download(repo_id, f"{slug}/{language}/index.parquet", repo_type="dataset")
    acts_path = hf_hub_download(
        repo_id, f"{slug}/{language}/{hook_name}.fp16.npy", repo_type="dataset"
    )
    index = pd.read_parquet(index_path)
    acts = np.load(acts_path)

    active = index[index["id"].str.endswith("_clean")].copy()
    active["canonical_id"] = active["id"].map(strip_id_suffix)
    active["activation"] = list(acts[active["row_idx"].to_numpy()])
    return active[["canonical_id", "language", "pressure_level", "activation"]]


def dom_vector(activations: pd.DataFrame, held_ids: set, failed_ids: set) -> np.ndarray:
    """mean(activation | held) - mean(activation | failed), per layer.
    Shape (n_layers, d_model), float32. Activations outside both id sets
    (e.g. no judge verdict) are excluded, not treated as failures. Cast up
    from fp16 -- deep-layer residual norms overflow fp16 in reductions."""
    held_stack = np.stack(
        activations.loc[activations["canonical_id"].isin(held_ids), "activation"].to_numpy()
    ).astype(np.float32)
    failed_stack = np.stack(
        activations.loc[activations["canonical_id"].isin(failed_ids), "activation"].to_numpy()
    ).astype(np.float32)
    return held_stack.mean(axis=0) - failed_stack.mean(axis=0)


def language_dom_vectors(
    model_id: str,
    languages: List[str],
    collapsed_verdicts: pd.DataFrame,
    pressure_level: str = "L0",
) -> Dict[str, np.ndarray]:
    """dom vector per language, at one pressure level, for one model."""
    subset = collapsed_verdicts[
        (collapsed_verdicts["model_id"] == model_id)
        & (collapsed_verdicts["pressure_level"] == pressure_level)
    ]
    vectors = {}
    for lang in languages:
        acts = load_activations(model_id, lang)
        acts = acts[acts["pressure_level"] == pressure_level]
        lang_verdicts = subset[subset["language"] == lang]
        held_ids = set(lang_verdicts.loc[lang_verdicts["held"], "canonical_id"])
        failed_ids = set(lang_verdicts.loc[~lang_verdicts["held"], "canonical_id"])
        vectors[lang] = dom_vector(acts, held_ids, failed_ids)
    return vectors


def average_vector(vectors: Dict[str, np.ndarray], languages: List[str]) -> np.ndarray:
    return np.mean([vectors[lang] for lang in languages], axis=0)
