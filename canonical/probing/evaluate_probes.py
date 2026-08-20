"""Script for evaluating the probes."""

import json
from typing import Dict
from datetime import datetime as dt
from sklearn.metrics import classification_report

from canonical.probing.config import RunConfig
from canonical.probing.utils import upload_repo_to_hf


def evaluate(
    cfg: RunConfig,
    probe_clfs,
    test_X,
    test_y,
    save_path_prefix: str = "",
) -> Dict[int, Dict[str, float]]:
    """Evaluate the probes.
    Computes classification report for each layer.
    """
    # computing evaluations
    evaluation_results = {}
    for model_name, layer_dict in probe_clfs.items():
        evaluation_results[model_name] = {}
        for layer, clf in layer_dict.items():
            predictions = clf.predict(test_X[layer])
            evaluation_results[model_name][layer] = classification_report(
                test_y, predictions, output_dict=True
            )
    # saving the results
    try:
        with open(f"{cfg.eval_path}/{save_path_prefix}Eval.json", "w") as f:
            json.dump(evaluation_results, f)
    except Exception as e:
        print(f"Error occurred while saving results: {e}")
        raise

    # uploading to hf
    upload_repo_to_hf(cfg.eval_path, cfg, repo_type="dataset")

    return evaluation_results


if __name__ == "__main__":
    pass
