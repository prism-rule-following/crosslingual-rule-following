import random
from typing import Any

import numpy as np
import pandas as pd
from canonical.probing.pydantic_models import (
    CanonicalDatasetColumns,
    DistractorDataset,
    DoubleRuleDataset,
    IndexParquetColumns,
    NeutralFillerDataset,
    SplitActivationDataset,
    distractor_words,
    labels2words,
)
from canonical.probing.utils import (
    check_length,
    download_parquet_from_hf,
    download_XY_from_hf,
    extract_model_activations,
    make_chat_settings,
    open_local_json,
)


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
    textdf[IndexParquetColumns.clean_id] = textdf[IndexParquetColumns.str_id].apply(
        lambda ix: ix.rsplit("_", 1)[0]
    )
    held_split = textdf.loc[
        ~textdf[IndexParquetColumns.rule_status].isin([pos1, pos2, neg1, neg2])
    ].copy()
    held_ids = held_split[IndexParquetColumns.row_idx].tolist()
    held_x = x[held_ids].copy()
    held_y = y[held_ids].copy()

    # dropping held ids and splitting the rest
    held_mask = np.ones(len(x), dtype=bool)
    held_mask[held_ids] = False
    rest_x = x[held_mask].copy()
    rest_y = y[held_mask].copy()
    rest_text = textdf.drop(held_ids, axis=0).copy()

    # splitting 80/20
    unique_clean_ids = rest_text[IndexParquetColumns.clean_id].unique().tolist()
    sample_size = int(len(unique_clean_ids) * 0.2)  # test sample ids
    unique_ids = random.sample(unique_clean_ids, k=sample_size)
    test_ids = rest_text[rest_text[IndexParquetColumns.clean_id].isin(unique_ids)][
        IndexParquetColumns.row_idx
    ].tolist()

    # test split
    test_x = x[test_ids].copy()
    test_y = y[test_ids].copy()
    test_text = rest_text.loc[test_ids, :].copy()

    # train split
    train_text = rest_text.drop(test_ids, axis=0).copy()
    train_ids = train_text[IndexParquetColumns.row_idx].tolist()
    train_x = x[train_ids].copy()
    train_y = y[train_ids].copy()

    # check they all turned out in "shape"
    check_length(train_x, train_y, train_text)
    check_length(test_x, test_y, test_text)
    check_length(held_x, held_y, held_split)

    return SplitActivationDataset(
        held_x=held_x,
        held_y=held_y,
        test_x=test_x,
        test_y=test_y,
        train_x=train_x,
        train_y=train_y,
        held_text=held_split,
        test_text=test_text,
        train_text=train_text,
    )


def create_from_xy_text(
    activations_in_hf: str,
    y_in_hf: str,
    text_id_hf: str,
    hf_repo_ix: str,
    hf_repo_type: str = "dataset",
) -> SplitActivationDataset:
    """Load necessary files and split the data into train, test and held out subsets."""
    # pull matching x and y
    X, y = download_XY_from_hf(
        activations_in_hf,
        y_in_hf,
        hf_repo_ix,
        repo_type=hf_repo_type,
    )

    # download the rules themselves to construct labels
    textid_df = download_parquet_from_hf(text_id_hf, hf_repo_ix, hf_repo_type)
    check_length(X, y, textid_df)

    # split into train, test and heldout
    split_act_dataset = test_heldout_split(X, y, textid_df)
    return split_act_dataset


### Confound sets ###
# Neutral filler for the Rule:
def neutral_filler_data(
    neutral_fillers_path: str,
    model: Any,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
):
    """Replaces the rule with neutral text.
    Confirms whether the probe actually reads the rule itself and doesn't just
    encode the polarity of the status words.
    The function creates a new dataset from the existing json file.
    """
    # loading the file
    fillers = open_local_json(neutral_fillers_path)
    texts = [item["text"] for item in fillers]
    text_labels = [item["label"] for item in fillers]

    # extract activations
    activations = extract_model_activations(
        model, texts, hook_name=hook_name, pos_slice=pos_slice
    )
    check_length(activations, text_labels, texts)
    return NeutralFillerDataset(
        neutral_x=activations, neutral_y=text_labels, neutral_text=texts
    )


def load_neutral_filler_data(
    x_path: str, y_path: str, neutral_data_path: str
) -> NeutralFillerDataset:
    """Loads the already existing neutral filler dataset.
    Expects neutral text data to be a .json.
    """
    # load activations and text
    try:
        neutral_acts = np.load(x_path)
        neutral_labels = np.load(y_path)
        neutral_text = open_local_json(neutral_data_path)
    except Exception as e:
        print(f"Error while loading existing neutral filler data: {e}")
        raise
    return NeutralFillerDataset(
        neutral_x=neutral_acts, neutral_y=neutral_labels, neutral_text=neutral_text
    )


# Shopping status: (distractor word)
def distractor_word_data(
    original_text_hf: str,
    hf_repo_ix: str,
    model: Any,
    hf_repo_type: str = "dataset",
) -> DistractorDataset:
    """Replaces the 'Rule status:' with 'Shopping status:'
    or some other rule unrelated word.
    Should check how much attention a probe actually pays to the rule itself.
    By default, with the 'Shopping status:', a model should still recognise
    the rule as active with a slight confusion. If it breaks, it might have
    encoded a mechanism, like 'check the Rule status:, your label is there'.
    """
    # download the rules themselves to construct labels
    ogdf = download_parquet_from_hf(original_text_hf, hf_repo_ix, hf_repo_type)
    sample_df = ogdf.sample(n=500)

    # replacing the 'Rule status:'
    sample_df["distractors"] = sample_df[CanonicalDatasetColumns.system_rule].apply(
        lambda x: x.replace(
            "Rule status:", f"{random.choice(distractor_words)} status:"
        )
    )
    sample_texts = sample_df["distractors"].tolist()
    activations = extract_model_activations(
        model, sample_texts, hook_name="hook_resid_post", pos_slice=-1
    )
    labels = [1] * len(activations)
    check_length(activations, labels, sample_df)
    return DistractorDataset(
        distractor_x=activations, distractor_y=labels, distractor_text=sample_df
    )


# 2 rules, opposite statuses + separate queries for each rule
def opposite_statuses_rules(
    filepath: str, model: Any, hook_name: str = "hook_resid_post", pos_slice: int = -1
) -> DoubleRuleDataset:
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
    # load the generated file
    double_rule_data = open_local_json(filepath)
    texts = [item["double_rule"] for item in double_rule_data]
    labels = [item["double_rule_label"] for item in double_rule_data]
    queries = [item["double_rule_query"] for item in double_rule_data]

    # make chat settings
    chat = make_chat_settings(model, texts, queries)

    # extract activations
    activations = extract_model_activations(
        model, chat, hook_name=hook_name, pos_slice=pos_slice
    )
    check_length(texts, labels, queries, activations)
    return DoubleRuleDataset(
        doublerule_text=chat, doublerule_x=activations, doublerule_y=labels
    )


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Dataset creation utilities for probing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_xy_text = subparsers.add_parser("create-xy-text")
    create_xy_text.add_argument("--activations-in-hf", required=True)
    create_xy_text.add_argument("--y-in-hf", required=True)
    create_xy_text.add_argument("--text-id-hf", required=True)
    create_xy_text.add_argument("--hf-repo-ix", required=True)
    create_xy_text.add_argument("--hf-repo-type", default="dataset")

    neutral_filler = subparsers.add_parser("neutral-filler")
    neutral_filler.add_argument("--neutral-fillers-path", required=True)
    neutral_filler.add_argument("--model-name", required=True)
    neutral_filler.add_argument("--hook-name", default="hook_resid_post")
    neutral_filler.add_argument("--pos-slice", type=int, default=-1)

    load_neutral_filler = subparsers.add_parser("load-neutral-filler")
    load_neutral_filler.add_argument("--x-path", required=True)
    load_neutral_filler.add_argument("--y-path", required=True)
    load_neutral_filler.add_argument("--neutral-data-path", required=True)

    distractor = subparsers.add_parser("distractor")
    distractor.add_argument("--original-text-hf", required=True)
    distractor.add_argument("--hf-repo-ix", required=True)
    distractor.add_argument("--model-name", required=True)
    distractor.add_argument("--hf-repo-type", default="dataset")

    opposite_rules = subparsers.add_parser("opposite-rules")
    opposite_rules.add_argument("--filepath", required=True)
    opposite_rules.add_argument("--model-name", required=True)
    opposite_rules.add_argument("--hook-name", default="hook_resid_post")
    opposite_rules.add_argument("--pos-slice", type=int, default=-1)

    return parser


def main():
    from transformer_lens import HookedTransformer

    args = _build_arg_parser().parse_args()

    if args.command == "create-xy-text":
        dataset = create_from_xy_text(
            args.activations_in_hf,
            args.y_in_hf,
            args.text_id_hf,
            args.hf_repo_ix,
            hf_repo_type=args.hf_repo_type,
        )
        print(dataset)
    elif args.command == "neutral-filler":
        model = HookedTransformer.from_pretrained(args.model_name)
        dataset = neutral_filler_data(
            args.neutral_fillers_path,
            model,
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
        )
        print(dataset)
    elif args.command == "load-neutral-filler":
        dataset = load_neutral_filler_data(
            args.x_path, args.y_path, args.neutral_data_path
        )
        print(dataset)
    elif args.command == "distractor":
        model = HookedTransformer.from_pretrained(args.model_name)
        dataset = distractor_word_data(
            args.original_text_hf,
            args.hf_repo_ix,
            model,
            hf_repo_type=args.hf_repo_type,
        )
        print(dataset)
    elif args.command == "opposite-rules":
        model = HookedTransformer.from_pretrained(args.model_name)
        dataset = opposite_statuses_rules(
            args.filepath,
            model,
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
        )
        print(dataset)


if __name__ == "__main__":
    main()
