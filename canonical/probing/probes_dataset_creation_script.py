import random
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from canonical.probing.config import RunConfig
from canonical.probing.pydantic_models import (
    CanonicalDatasetColumns,
    DistractorDataset,
    DoubleRuleDataset,
    IndexParquetColumns,
    NeutralFillerDataset,
    NoKeywordRuleDataset,
    SplitActivationDataset,
    distractor_words,
    labels2words,
    words2labels,
)
from canonical.probing.utils import (
    check_length,
    download_jsonl_from_hf,
    download_parquet_from_hf,
    download_XY_from_hf,
    extract_model_activations,
    make_chat_settings,
    open_local_json,
    save_dataset_locally,
    upload_repo_to_hf,
)
from sklearn.model_selection import GroupShuffleSplit


def test_heldout_split(
    x: np.ndarray, y: np.ndarray, textdf: pd.DataFrame
) -> SplitActivationDataset:
    """Split original activation dataset into train/test/heldout.
    Assumes some column names that could be accessed/changed in the pydantic_models.py.
    """
    # choose two pairs for heldout testing, take them out
    seeded_random = random.Random(42)
    pos1, pos2 = seeded_random.sample(labels2words[1], k=2)
    neg1, neg2 = seeded_random.sample(labels2words[0], k=2)

    # split the rest into test and training, keep test and minimal at this point
    textdf[IndexParquetColumns.clean_id] = textdf[IndexParquetColumns.str_id].apply(
        lambda ix: ix.rsplit("_", 1)[0]
    )
    held_split = textdf.loc[
        textdf[IndexParquetColumns.rule_status].isin([pos1, pos2, neg1, neg2])
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

    # splitting 80/20 by clean_id groups, so variant rows never split across train/test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_pos, test_pos = next(
        splitter.split(rest_x, rest_y, groups=rest_text[IndexParquetColumns.clean_id])
    )

    # test split
    test_x = rest_x[test_pos].copy()
    test_y = rest_y[test_pos].copy()
    test_text = rest_text.iloc[test_pos].copy()

    # train split
    train_x = rest_x[train_pos].copy()
    train_y = rest_y[train_pos].copy()
    train_text = rest_text.iloc[train_pos].copy()

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


def get_xy(
    build_texts_and_labels: Callable[[], Tuple[List[str], np.ndarray]],
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    hf_dataset_repo: Optional[str] = None,
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    hf_repo_type: str = "dataset",
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared by every dataset-creation function below. If hf_dataset_repo is given,
    tries to download precomputed activations/labels from it first; on any failure
    (or if hf_dataset_repo isn't given), falls back to building texts/labels via
    build_texts_and_labels and extracting them on the fly via the model.
    build_texts_and_labels is only called in the on-the-fly case, so callers that need
    model.tokenizer to build their texts (chat formatting) don't need a model loaded
    at all when the cache hits.
    """
    if hf_dataset_repo:
        try:
            return download_XY_from_hf(
                activations_in_hf, y_in_hf, hf_dataset_repo, repo_type=hf_repo_type
            )
        except Exception as e:
            print(
                f"Couldn't download cached activations from {hf_dataset_repo}, extracting on the fly instead: {e}"
            )
    if model is None:
        raise ValueError(
            "No activations available: hf_dataset_repo wasn't given (or the download failed), "
            "and no model was provided for on-the-fly extraction."
        )
    texts, y = build_texts_and_labels()
    X = extract_model_activations(
        model, texts, hook_name=hook_name, pos_slice=pos_slice
    )
    return X, y


def create_canonical_dataset(
    jsonl_in_hf: str,
    hf_repo_ix: str,
    hf_repo_type: str = "dataset",
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    hf_dataset_repo: Optional[str] = None,
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    push_full_dataset_to_hf: bool = True,
    push_path_in_repo: Optional[str] = None,
    cfg: Optional[RunConfig] = None,
) -> SplitActivationDataset:
    """Load necessary files and split the data into train, test and held out subsets.
    Activations are downloaded from hf_dataset_repo if given (falling back to on-the-fly
    extraction if that fails), else extracted on the fly directly.
    Each jsonl line yields two rows: one for the active rule, one for the cancelled rule.
    """
    data = download_jsonl_from_hf(jsonl_in_hf, hf_repo_ix, hf_repo_type)

    rows = []
    for item in data:
        rows.append(
            {
                IndexParquetColumns.str_id: f"{item['id']}_0",
                IndexParquetColumns.rule_status: item["active_status"],
                CanonicalDatasetColumns.system_rule: item["system_rule"],
                "query": item["user_query"],
            }
        )
        rows.append(
            {
                IndexParquetColumns.str_id: f"{item['id']}_1",
                IndexParquetColumns.rule_status: item["revoked_status"],
                CanonicalDatasetColumns.system_rule: item["system_non_rule"],
                "query": item["user_query"],
            }
        )
    textid_df = pd.DataFrame(rows)
    textid_df[IndexParquetColumns.row_idx] = range(len(textid_df))
    y = np.array(
        [words2labels[status] for status in textid_df[IndexParquetColumns.rule_status]]
    )

    def build_chat_and_labels():
        systems = textid_df[CanonicalDatasetColumns.system_rule].tolist()
        queries = textid_df["query"].tolist()
        return make_chat_settings(model, systems, queries), y

    X, y = get_xy(
        build_chat_and_labels,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
        hf_dataset_repo=hf_dataset_repo,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        hf_repo_type=hf_repo_type,
    )

    check_length(X, y, textid_df)

    if push_full_dataset_to_hf:
        import tempfile
        from datetime import datetime

        model_name = getattr(getattr(model, "cfg", None), "model_name", None) or "model"
        language = cfg.language if cfg else "unk"
        path_in_repo = push_path_in_repo or (
            f"{model_name}/{datetime.now():%Y-%m-%d_%H-%M-%S}_{language}_full_dataset"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dataset_locally(tmp_dir, X, y, textid_df, hook_name=hook_name)
            upload_repo_to_hf(
                tmp_dir,
                repo_type=hf_repo_type,
                repo_id=hf_dataset_repo,
                path_in_repo=path_in_repo,
            )

    return test_heldout_split(X, y, textid_df)


### Confound sets ###
# Neutral filler for the Rule:
def neutral_filler_data(
    neutral_fillers_path: str,
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    hf_dataset_repo: Optional[str] = None,
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    hf_repo_type: str = "dataset",
):
    """Replaces the rule with neutral text.
    Confirms whether the probe actually reads the rule itself and doesn't just
    encode the polarity of the status words.
    The function creates a new dataset from the existing json file.
    """
    fillers = open_local_json(neutral_fillers_path)
    texts = [item["text"] for item in fillers]
    text_labels = np.array([int(item["label"]) for item in fillers])

    X, y = get_xy(
        lambda: (texts, text_labels),
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
        hf_dataset_repo=hf_dataset_repo,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        hf_repo_type=hf_repo_type,
    )
    check_length(X, y, fillers)
    return NeutralFillerDataset(neutral_x=X, neutral_y=y, neutral_text=fillers)


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
    model: Any = None,
    hf_repo_type: str = "dataset",
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    hf_dataset_repo: Optional[str] = None,
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
) -> DistractorDataset:
    """Replaces the 'Rule status:' with 'Shopping status:'
    or some other rule unrelated word.
    Should check how much attention a probe actually pays to the rule itself.
    By default, with the 'Shopping status:', a model should still recognise
    the rule as active with a slight confusion. If it breaks, it might have
    encoded a mechanism, like 'check the Rule status:, your label is there'.
    """
    # download the rules themselves to construct labels
    ogdf = download_jsonl_from_hf(original_text_hf, hf_repo_ix, hf_repo_type)
    sample_df = pd.DataFrame(ogdf).sample(n=min(500, len(ogdf)), random_state=42)

    # replacing the 'Rule status:'
    seeded_random = random.Random(42)
    sample_df["rule_distractors"] = sample_df[
        CanonicalDatasetColumns.system_rule
    ].apply(
        lambda x: x.replace(
            "Rule status:", f"{seeded_random.choice(distractor_words)} status:"
        )
    )
    sample_df["non_rule_distractors"] = sample_df[
        CanonicalDatasetColumns.system_non_rule
    ].apply(
        lambda x: x.replace(
            "Rule status:", f"{seeded_random.choice(distractor_words)} status:"
        )
    )
    sample_texts = (
        sample_df["rule_distractors"].tolist()
        + sample_df["non_rule_distractors"].tolist()
    )
    labels = np.array(([1] * len(sample_df)) + ([0] * len(sample_df)))

    X, y = get_xy(
        lambda: (sample_texts, labels),
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
        hf_dataset_repo=hf_dataset_repo,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        hf_repo_type=hf_repo_type,
    )
    check_length(X, y)
    return DistractorDataset(distractor_x=X, distractor_y=y, distractor_text=sample_df)


# 2 rules, opposite statuses + separate queries for each rule
def opposite_statuses_rules(
    filepath: str,
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    hf_dataset_repo: Optional[str] = None,
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    hf_repo_type: str = "dataset",
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
    texts = [item["double_rule_text"] for item in double_rule_data]
    labels = np.array([item["double_rule_label"] for item in double_rule_data])
    queries = [item["double_rule_query"] for item in double_rule_data]

    X, y = get_xy(
        lambda: (make_chat_settings(model, texts, queries), labels),
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
        hf_dataset_repo=hf_dataset_repo,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        hf_repo_type=hf_repo_type,
    )
    check_length(texts, y, queries, X)
    return DoubleRuleDataset(doublerule_text=texts, doublerule_x=X, doublerule_y=y)


def no_rule_keyword(
    model,
    jsonl_in_hf: str,
    repo_ix: str,
    repo_type: str = "dataset",
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    hf_dataset_repo: Optional[str] = None,
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
):
    """Tests whether a probe holds when there is no explicit 'Rule' keyword in the system."""
    # download original dataset
    data = download_jsonl_from_hf(
        data_path_in_repo=jsonl_in_hf, repo_ix=repo_ix, repo_type=repo_type
    )

    # construct new system clause
    # TODO: move the column names into smth structural, e.g. StrEnum, class
    positive_lines = [
        {
            "no_rule_label": 1,
            "no_rule_system_text": item["context"] + " " + item["rule_clause"] + ".",
            **item,
        }
        for item in data
    ]
    negative_lines = [
        {"no_rule_label": 0, "no_rule_system_text": item["context"], **item}
        for item in data
    ]
    new_lines = positive_lines + negative_lines
    labels = np.array([int(item["no_rule_label"]) for item in new_lines])
    systems = [item["no_rule_system_text"] for item in new_lines]
    user_queries = [item["user_query"] for item in new_lines]

    X, y = get_xy(
        lambda: (make_chat_settings(model, systems, user_queries), labels),
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
        hf_dataset_repo=hf_dataset_repo,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        hf_repo_type=repo_type,
    )
    check_length(y, systems, user_queries, X)

    # construct dataset
    return NoKeywordRuleDataset(nokrule_x=X, nokrule_y=y, nokrule_text=new_lines)


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset creation utilities for probing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_canonical = subparsers.add_parser("create-canonical-dataset")
    create_canonical.add_argument("--jsonl-in-hf", required=True)
    create_canonical.add_argument("--hf-repo-ix", required=True)
    create_canonical.add_argument("--hf-repo-type", default="dataset")
    create_canonical.add_argument("--activations-in-hf", default=None)
    create_canonical.add_argument("--y-in-hf", default=None)
    create_canonical.add_argument("--hf-dataset-repo", default=None)
    create_canonical.add_argument("--model-name", default=None)
    create_canonical.add_argument("--hook-name", default="hook_resid_post")
    create_canonical.add_argument("--pos-slice", type=int, default=-1)
    create_canonical.add_argument(
        "--no-push-full-dataset-to-hf",
        dest="push_full_dataset_to_hf",
        action="store_false",
        default=True,
    )
    create_canonical.add_argument("--push-path-in-repo", default=None)

    neutral_filler = subparsers.add_parser("neutral-filler")
    neutral_filler.add_argument("--neutral-fillers-path", required=True)
    neutral_filler.add_argument("--model-name", default=None)
    neutral_filler.add_argument("--hook-name", default="hook_resid_post")
    neutral_filler.add_argument("--pos-slice", type=int, default=-1)
    neutral_filler.add_argument("--hf-repo-type", default="dataset")
    neutral_filler.add_argument("--activations-in-hf", default=None)
    neutral_filler.add_argument("--y-in-hf", default=None)
    neutral_filler.add_argument("--hf-dataset-repo", default=None)

    load_neutral_filler = subparsers.add_parser("load-neutral-filler")
    load_neutral_filler.add_argument("--x-path", required=True)
    load_neutral_filler.add_argument("--y-path", required=True)
    load_neutral_filler.add_argument("--neutral-data-path", required=True)

    distractor = subparsers.add_parser("distractor")
    distractor.add_argument("--original-text-hf", required=True)
    distractor.add_argument("--hf-repo-ix", required=True)
    distractor.add_argument("--model-name", default=None)
    distractor.add_argument("--hf-repo-type", default="dataset")
    distractor.add_argument("--hook-name", default="hook_resid_post")
    distractor.add_argument("--pos-slice", type=int, default=-1)
    distractor.add_argument("--activations-in-hf", default=None)
    distractor.add_argument("--y-in-hf", default=None)
    distractor.add_argument("--hf-dataset-repo", default=None)

    opposite_rules = subparsers.add_parser("opposite-rules")
    opposite_rules.add_argument("--filepath", required=True)
    opposite_rules.add_argument("--model-name", default=None)
    opposite_rules.add_argument("--hook-name", default="hook_resid_post")
    opposite_rules.add_argument("--pos-slice", type=int, default=-1)
    opposite_rules.add_argument("--hf-repo-type", default="dataset")
    opposite_rules.add_argument("--activations-in-hf", default=None)
    opposite_rules.add_argument("--y-in-hf", default=None)
    opposite_rules.add_argument("--hf-dataset-repo", default=None)

    no_keyword = subparsers.add_parser("no-keyword")
    no_keyword.add_argument("--jsonl-in-hf", required=True)
    no_keyword.add_argument("--hf-repo-ix", required=True)
    no_keyword.add_argument("--model-name", default=None)
    no_keyword.add_argument("--hf-repo-type", default="dataset")
    no_keyword.add_argument("--hook-name", default="hook_resid_post")
    no_keyword.add_argument("--pos-slice", type=int, default=-1)
    no_keyword.add_argument("--activations-in-hf", default=None)
    no_keyword.add_argument("--y-in-hf", default=None)
    no_keyword.add_argument("--hf-dataset-repo", default=None)

    return parser


def main():
    args = _build_arg_parser().parse_args()

    def load_model_if_needed():
        if not args.model_name:
            return None
        from transformer_lens import HookedTransformer

        return HookedTransformer.from_pretrained(args.model_name)

    if args.command == "create-canonical-dataset":
        dataset = create_canonical_dataset(
            args.jsonl_in_hf,
            args.hf_repo_ix,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            hf_dataset_repo=args.hf_dataset_repo,
            model=load_model_if_needed(),
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
            push_full_dataset_to_hf=args.push_full_dataset_to_hf,
            push_path_in_repo=args.push_path_in_repo,
        )
        print(dataset)
    elif args.command == "neutral-filler":
        dataset = neutral_filler_data(
            args.neutral_fillers_path,
            load_model_if_needed(),
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            hf_dataset_repo=args.hf_dataset_repo,
        )
        print(dataset)
    elif args.command == "load-neutral-filler":
        dataset = load_neutral_filler_data(
            args.x_path, args.y_path, args.neutral_data_path
        )
        print(dataset)
    elif args.command == "distractor":
        dataset = distractor_word_data(
            args.original_text_hf,
            args.hf_repo_ix,
            load_model_if_needed(),
            hf_repo_type=args.hf_repo_type,
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            hf_dataset_repo=args.hf_dataset_repo,
        )
        print(dataset)
    elif args.command == "opposite-rules":
        dataset = opposite_statuses_rules(
            args.filepath,
            load_model_if_needed(),
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            hf_dataset_repo=args.hf_dataset_repo,
        )
        print(dataset)
    elif args.command == "no-keyword":
        dataset = no_rule_keyword(
            load_model_if_needed(),
            args.jsonl_in_hf,
            args.hf_repo_ix,
            repo_type=args.hf_repo_type,
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            hf_dataset_repo=args.hf_dataset_repo,
        )
        print(dataset)


if __name__ == "__main__":
    main()
