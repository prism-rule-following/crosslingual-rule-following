"""Probes training."""

from typing import Dict
import numpy as np

from canonical.probing.utils import upload_to_hf


def training(
    classifier,
    train_X: Dict[int, np.ndarray],
    train_y: np.ndarray,
    remove_local: bool = False,
    upload_to_hf_repo: str = None,
) -> Dict[int, object]:
    """Train the probes."""
    # match classifier
    if not hasattr(classifier, "fit"):
        raise ValueError(f"Classifier {classifier} does not have a fit() method.")
    # do the training
    trained_classifiers = {}
    try:
        # training per layer
        for layer in train_X:
            classifier.fit(train_X[layer], train_y)
            trained_classifiers[layer] = classifier
    except Exception as e:
        raise ValueError(f"An error occurred while training the classifier: {e}")
    # saving the probes
    if upload_to_hf_repo is not None:
        upload_to_hf(upload_to_hf_repo, remove_local=remove_local)

    return trained_classifiers
