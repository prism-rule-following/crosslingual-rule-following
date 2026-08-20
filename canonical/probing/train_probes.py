"""Probes training."""

from typing import Callable, Dict, List
from sklearn.base import clone
import numpy as np
from canonical.probing.config import RunConfig
from canonical.probing.utils import upload_repo_to_hf, make_trained_probes_metadata


def training(
    run_config: RunConfig,
    classifiers: List[Callable],
    train_X: Dict[int, np.ndarray],
    train_y: np.ndarray,
    remove_local: bool = False,
    save_path_prefix: str = "",
) -> Dict[int, object]:
    """Train the probes."""
    import skops.io as sio

    # check clfs
    valid_clfs = [clf for clf in classifiers if hasattr(clf, "fit")]
    if len(valid_clfs) == 0:
        raise ValueError(
            f"Classifiers do not have a fit() method. Pass valid classifiers."
        )

    # do the training
    trained_classifiers = {}
    for clf in valid_clfs:
        name = type(clf).__name__
        trained_classifiers[name] = {}
        try:
            # training per layer
            for layer in train_X:
                layer_clf = clone(clf)
                layer_clf.fit(train_X[layer], train_y)
                trained_classifiers[name][layer] = layer_clf
                sio.dump(
                    layer_clf,
                    f"{run_config.trained_probes_path}/{name}_layer_{layer}.skops",
                )

        except Exception as e:
            raise ValueError(f"An error occurred while training the classifier: {e}")
    # make and save metadata file
    make_trained_probes_metadata(run_config, trained_classifiers)

    # uploading the probes repo to hf
    upload_repo_to_hf(
        run_config.trained_probes_path,
        run_config,
        remove_local=remove_local,
        save_path_prefix=save_path_prefix,
    )

    return trained_classifiers
