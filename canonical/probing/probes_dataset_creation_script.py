import numpy as np
import pandas as pd
import random

from canonical.probing.utils import (
    download_text_index_from_hf,
    download_XY_from_hf,
    check_length,
)
from canonical.probing.pydantic_models import (
    SplitActivationDataset,
    IndexParquetColumns,
    words2labels,
    labels2words,
)


# TODO ALL
def test_heldout_split(
    x: np.ndarray, y: np.ndarray, textdf: pd.DataFrame
) -> SplitActivationDataset:
    """Split original activation dataset into train/test/heldout.
    Assumes some column names that could be accessed/changed in the pydantic_models.py.
    """
    # choose two pairs for heldout testing, take them out
    pos1, pos2 = random.sample(labels2words[1], k=2)
    neg1, neg2 = random.sample(labels2words[0], k=2)

    # split the rest into test and training, keep test and minimal at this point
    textdf["clean_id"] = textdf[IndexParquetColumns.str_id].apply(
        lambda ix: ix.rsplit("_", 1)[0]
    )
    held_split = textdf.loc[
        ~textdf[IndexParquetColumns.rule_status].isin([pos1, pos2, neg1, neg2])
    ].copy()
    held_ids = held_split[IndexParquetColumns.row_idx].tolist()
    held_x = x[held_ids].copy()
    held_y = y[held_ids].copy()

    # dropping held ids and splitting the rest
    rest_x = x[~held_ids].copy()
    rest_y = y[~held_ids].copy()
    rest_text = textdf.drop(held_ids, axis=0).copy()

    # splitting 80/20
    # TODO make sure the split is correct by the unique IDs, not rows (incorrect now!)
    sample_size = int(len(rest_text[IndexParquetColumns["clean_id"]]) * 0.2 / 100)
    test_ids = random.sample(
        rest_text[IndexParquetColumns.row_idx].tolist(), k=sample_size
    )
    test_x = rest_x[test_ids].copy()
    test_y = rest_y[test_ids].copy()
    test_text = rest_text.drop(test_ids, axis=0).copy()

    return SplitActivationDataset(
        held_x=held_x,
        held_y=held_y,
        test_x=test_x,
        test_y=test_y,
        # ...
    )


def create_from_xy_text(
    activations_in_hf: str,
    y_in_hf: str,
    text_id_hf: str,
    hf_repo_ix: str,
    hf_repo_type: str = "dataset",
):
    """Load necessary files and split the data into train, test and held out subsets."""
    # pull matching x and y
    X, y = download_XY_from_hf(
        activations_in_hf,
        y_in_hf,
        hf_repo_ix,
        repo_type=hf_repo_type,
    )
    # download the rules themselves to construct labels
    textid_df = download_text_index_from_hf(text_id_hf, hf_repo_ix, hf_repo_type)
    check_length(X, y, textid_df)
    # split into train, test and heldout
    test_heldout_split()


### Confound sets ###
# Neutral filler for the Rule:
def neutral_filler_data():
    """Replaces the rule with neutral text.
    Confirms whether the probe actually reads the rule itself and doesn't just
    encode the polarity of the status words.
    """
    pass


# Shopping status: (distractor word)
def distractor_word_data():
    """Replaces the 'Rule status:' with 'Shopping status:'
    or some other rule unrelated word.
    Should check how much attention a probe actually pays to the rule itself.
    By default, with the 'Shopping status:', a model should still recognise
    the rule as active with a slight confusion. If it breaks, it might have
    encoded a mechanism, like 'check the Rule status:, your label is there'.
    """
    pass


# 2 rules, opposite statuses + separate queries for each rule
def opposite_statuses_rules():
    """Checks how the probe survives two rules with different statuses.
    Example:
        System:
            Rule A: Do this. Rule status: active.
            Rule B: Do that. Rule status: cancelled.
        Query:
            (...) Apply rule A. -> positive
            (...) Apply rule B. -> negative
    This one rids the probe of any confounds above:
        1) overfitting to specific status words (held out test should check for this),
        2) learning polarity axis of status words, not actually encoding a rule enforcement itself
        3) finding the label in the 'Rule status:' without paying attention to the actual rule
        4) the internal rule enforcement is actually reflected in the behaviour (this one will need
        to be checked with adherence)
    Apart from that, it checks a deep understanding of what requires of the model.
    The problem is, a small model like 8B might not even perform well under this task.
    But in this case, the problem is the capacity.
    In case a probe fails training on the original dataset, we could train it on this one.
    """
    pass
