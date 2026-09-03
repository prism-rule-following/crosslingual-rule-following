"""w vectors: trained LogisticRegression probe coefficients. Same {lang: (n_layers, d_model)}
shape as vectors.language_dom_vectors, so it's a drop-in for feasibility
comparisons.
"""

from typing import Dict, List

import numpy as np
import skops.io as sio
from huggingface_hub import hf_hub_download

PROBE_REPO = "veerlosar/prism-model-activations"

MODEL_PROBE_DIRS = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
}


def _probe_path(model_id: str, language: str, layer: int) -> str:
    model_dir = MODEL_PROBE_DIRS[model_id]
    if model_id == "Qwen/Qwen3-8B":
        return f"{model_dir}/results_{language}/trained_probes/LogisticRegression_layer_{layer}_{language}.skops"
    return f"{model_dir}/adherence_gpt_results_{language}/trained_probes/LogisticRegression_layer_{layer}_{language}.skops"


def load_probe_direction(model_id: str, language: str, layer: int) -> np.ndarray:
    path = hf_hub_download(PROBE_REPO, _probe_path(model_id, language, layer), repo_type="dataset")
    untrusted = sio.get_untrusted_types(file=path)
    clf = sio.load(path, trusted=untrusted)
    return clf.coef_[0].astype(np.float32)


def language_probe_vectors(model_id: str, languages: List[str], n_layers: int) -> Dict[str, np.ndarray]:
    vectors = {}
    for lang in languages:
        layers = [load_probe_direction(model_id, lang, layer) for layer in range(n_layers)]
        vectors[lang] = np.stack(layers)
    return vectors
