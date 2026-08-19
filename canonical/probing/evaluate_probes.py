"""Script for evaluating the probes."""

import json
from datetime import datetime as dt
from sklearn.metrics import classification_report


def evaluate(probe_clfs, test_X, test_y, save_path: str = None):
    """Evaluate the probes.
    Computes classification report for each layer.
    """
    # computing evaluations
    evaluation_results = {}
    for layer, clf in probe_clfs.items():
        predictions = clf.predict(test_X[layer])
        evaluation_results[layer] = classification_report(
            test_y, predictions, output_dict=True
        )
    # saving the results
    if save_path is not None:
        folder = f"probe_evaluation_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        try:
            with open(f"{save_path}/{folder}.json", "w") as f:
                json.dump(evaluation_results, f)
        except Exception as e:
            print(f"Error occurred while saving results: {e}")
            raise

    return evaluation_results


if __name__ == "__main__":
    pass
