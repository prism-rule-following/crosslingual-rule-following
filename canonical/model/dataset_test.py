"""Unit tests for canonical/model/dataset.py.

No real network calls are made: HF loading is exercised via HFDataHelper /
load_dataset monkeypatches, and GitHub loading is exercised against local
tmp_path files plus a mocked urllib.request.urlopen for the remote-URL branch.
"""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

import canonical.model.dataset as ds
from canonical.model.dataset import (
    ACTIVE_STATUSES,
    PAIR_STATUS,
    Category,
    CrossLingualRuleFollowingDataset,
    DatasetConfig,
    DatasetLanguageCode,
    DatasetSource,
    GrammarType,
    HFDataHelper,
    PairType,
    PressureLevel,
    PressureName,
    RulePair,
    Topic,
    _apply_filters,
    _to_dataframe,
    _validate_rows,
    collate_behavioral,
    dataset_generator,
    load_from_github,
    split_constrast_pairs,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_checker(rule_status: str, binds: bool) -> dict:
    return {
        "rule_status": rule_status,
        "binds": binds,
        "violation_event": "violation_occurred",
        "checker_type": "llm_judge",
        "instruction": "LLM-judge instruction text.",
        "rubric": {
            "held_if": "Held condition.",
            "violated_if": "Violated condition.",
            "ignore": "Ignore condition.",
            "culture_invariant": True,
            "point_at": "the relevant span",
        },
    }


def make_row(**overrides) -> dict:
    """A fully valid RulePair dict. pair_type drives active_status/
    revoked_status (and their checkers) via PAIR_STATUS, so overriding just
    `pair_type` keeps the row internally consistent unless the caller also
    overrides active_status/revoked_status/active_checker/revoked_checker
    directly.
    """
    pair_type = overrides.get("pair_type", "active_cancelled")
    active_status, revoked_status = PAIR_STATUS[PairType(pair_type)]
    row = {
        "id": "en-1",
        "category": "ack_invert",
        "topic": "legal",
        "grammar_type": "imperative",
        "pressure_level": "L0",
        "pressure_name": "neutral",
        "checker_tier": "judgment",
        "language": "en",
        "context": "You are an assistant.",
        "rule_clause": "must comply",
        "pair_type": pair_type,
        "active_status": active_status.value,
        "revoked_status": revoked_status.value,
        "rule_text": "must comply. Rule status: active.",
        "non_rule_text": "must comply. Rule status: cancelled.",
        "system_rule": "SYSTEM: rule active",
        "system_non_rule": "SYSTEM: rule cancelled",
        "user_query": "What should I do?",
        "user_turns": None,
        "correct_answer": "A compliant answer.",
        "correct_keywords": ["comply"],
        "active_checker": make_checker(active_status.value, True),
        "revoked_checker": make_checker(revoked_status.value, False),
    }
    row.update(overrides)
    return row


@pytest.fixture
def sample_pairs() -> list:
    return [
        make_row(
            id="en-1",
            language="en",
            topic="legal",
            category="ack_invert",
            pair_type="active_cancelled",
        ),
        make_row(
            id="en-2",
            language="en",
            topic="mental_health",
            category="no_pii",
            grammar_type="polite_asking",
            pair_type="on_off",
        ),
        make_row(
            id="de-1",
            language="de",
            topic="finance",
            category="no_dosage",
            grammar_type="modal_obligation",
            pair_type="active_cancelled",
        ),
        make_row(
            id="en-3",
            language="en",
            topic="medical",
            category="scope_lock",
            grammar_type="polite_asking",
            pair_type="enabled_disabled",
        ),
    ]


@pytest.fixture
def sample_dataset_file(tmp_path, sample_pairs):
    path = tmp_path / "full_dataset.json"
    path.write_text(json.dumps({"pairs": sample_pairs}))
    return path


class FakeHFDataset:
    """Stands in for a HF `datasets.Dataset`, which exposes .to_pandas()."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


class FakeHfApi:
    """Records calls instead of touching the real Hugging Face Hub."""

    def __init__(self):
        self.create_repo_calls = []
        self.upload_file_calls = []
        self.get_paths_info_calls = []
        self.get_paths_info_return = []
        self.uploaded_df = None

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)

    def upload_file(self, **kwargs):
        self.upload_file_calls.append(kwargs)
        self.uploaded_df = pd.read_parquet(kwargs["path_or_fileobj"])

    def get_paths_info(self, **kwargs):
        self.get_paths_info_calls.append(kwargs)
        if isinstance(self.get_paths_info_return, Exception):
            raise self.get_paths_info_return
        return self.get_paths_info_return


# --------------------------------------------------------------------------- #
# load_from_github
# --------------------------------------------------------------------------- #
def test_load_from_github_local_file_with_pairs_wrapper(
    sample_dataset_file, sample_pairs
):
    df = load_from_github(str(sample_dataset_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_pairs)
    assert set(df["id"]) == {row["id"] for row in sample_pairs}


def test_load_from_github_local_file_bare_list(tmp_path, sample_pairs):
    path = tmp_path / "bare_list.json"
    path.write_text(json.dumps(sample_pairs))
    df = load_from_github(str(path))
    assert len(df) == len(sample_pairs)


def test_load_from_github_missing_local_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_from_github(str(missing))


def test_load_from_github_remote_url_mocked(monkeypatch, sample_pairs):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return json.dumps({"pairs": sample_pairs}).encode("utf-8")

    monkeypatch.setattr(ds.urllib.request, "urlopen", lambda url: FakeResponse())
    df = load_from_github(
        "https://raw.githubusercontent.com/example/repo/main/data.json"
    )
    assert len(df) == len(sample_pairs)


def test_load_from_github_invalid_json_falls_back_to_load_dataset(
    monkeypatch, tmp_path
):
    path = tmp_path / "not_json.json"
    path.write_text("this is not valid json")

    sentinel = object()
    monkeypatch.setattr(ds, "load_dataset", lambda fmt, data_files: sentinel)

    result = load_from_github(str(path))
    assert result is sentinel


def test_load_from_github_unsupported_raw_type_raises(monkeypatch, tmp_path):
    path = tmp_path / "weird.json"
    path.write_text(json.dumps({"pairs": {"not": "a list"}}))
    # raw.get("pairs", raw) -> {"not": "a list"} - a dict, not a list -> TypeError
    with pytest.raises(TypeError):
        load_from_github(str(path))


# --------------------------------------------------------------------------- #
# HFDataHelper.load_source_dataset (the HF loading path)
# --------------------------------------------------------------------------- #
def test_load_source_dataset_calls_load_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(repo_id):
        calls.append(repo_id)
        return {"train": FakeHFDataset(pd.DataFrame(sample_pairs_static()))}

    monkeypatch.setattr(ds, "load_dataset", fake_load_dataset)
    raw = HFDataHelper.load_source_dataset("some-org/some-repo")
    assert calls == ["some-org/some-repo"]
    assert "train" in raw


def test_load_source_dataset_reraises_on_failure(monkeypatch):
    def fake_load_dataset(repo_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(ds, "load_dataset", fake_load_dataset)
    with pytest.raises(RuntimeError, match="boom"):
        HFDataHelper.load_source_dataset("some-org/some-repo")


def sample_pairs_static() -> list:
    return [
        make_row(id="hf-1"),
        make_row(id="hf-2", pair_type="on_off"),
    ]


# --------------------------------------------------------------------------- #
# _to_dataframe
# --------------------------------------------------------------------------- #
def test_to_dataframe_picks_train_split_by_default():
    df = pd.DataFrame(sample_pairs_static())
    raw = {"train": FakeHFDataset(df), "test": FakeHFDataset(df.head(1))}
    result = _to_dataframe(raw, split=None)
    assert len(result) == len(df)


def test_to_dataframe_picks_first_key_if_no_train():
    df = pd.DataFrame(sample_pairs_static())
    raw = {"validation": FakeHFDataset(df)}
    result = _to_dataframe(raw, split=None)
    assert len(result) == len(df)


def test_to_dataframe_explicit_split():
    df_train = pd.DataFrame(sample_pairs_static())
    df_test = df_train.head(1)
    raw = {"train": FakeHFDataset(df_train), "test": FakeHFDataset(df_test)}
    result = _to_dataframe(raw, split="test")
    assert len(result) == 1


def test_to_dataframe_missing_split_raises_keyerror():
    raw = {"train": FakeHFDataset(pd.DataFrame(sample_pairs_static()))}
    with pytest.raises(KeyError):
        _to_dataframe(raw, split="validation")


def test_to_dataframe_passthrough_dataframe():
    df = pd.DataFrame(sample_pairs_static())
    result = _to_dataframe(df, split=None)
    assert result is df


def test_to_dataframe_unsupported_type_raises():
    with pytest.raises(TypeError):
        _to_dataframe(object(), split=None)


# --------------------------------------------------------------------------- #
# _validate_rows
# --------------------------------------------------------------------------- #
def test_validate_rows_keeps_valid_rows(sample_pairs):
    df = pd.DataFrame(sample_pairs)
    result = _validate_rows(df, strict=False)
    assert len(result) == len(sample_pairs)


def test_validate_rows_drops_invalid_row_when_not_strict(sample_pairs):
    bad_row = make_row(id="bad-1")
    del bad_row["system_rule"]  # required field missing
    df = pd.DataFrame(sample_pairs + [bad_row])
    result = _validate_rows(df, strict=False)
    assert len(result) == len(sample_pairs)
    assert "bad-1" not in set(result["id"])


def test_validate_rows_raises_when_strict(sample_pairs):
    bad_row = make_row(id="bad-1")
    del bad_row["system_rule"]
    df = pd.DataFrame(sample_pairs + [bad_row])
    with pytest.raises(ValidationError):
        _validate_rows(df, strict=True)


def test_validate_rows_handles_nan_optional_fields(sample_pairs):
    # pandas fills missing columns with NaN across rows of differing shape;
    # Optional[...] fields must accept that as None rather than failing.
    row_without_metadata = make_row(id="no-metadata")
    df = pd.DataFrame(
        [
            row_without_metadata,
            {
                **row_without_metadata,
                "id": "with-extra",
                "generation_metadata": None,
            },
        ]
    )
    result = _validate_rows(df, strict=False)
    assert len(result) == 2


def test_validate_rows_rejects_unknown_category(sample_pairs):
    bad_row = make_row(id="bad-category", category="banned_word")  # v1-only category
    df = pd.DataFrame(sample_pairs + [bad_row])
    result = _validate_rows(df, strict=False)
    assert "bad-category" not in set(result["id"])


# --------------------------------------------------------------------------- #
# _apply_filters
# --------------------------------------------------------------------------- #
def test_apply_filters_by_category(sample_pairs):
    df = pd.DataFrame(sample_pairs)
    config = DatasetConfig(url="x", source="gh", categories=[Category.ack_invert])
    result = _apply_filters(df, config)
    assert set(result["category"]) == {"ack_invert"}


def test_apply_filters_by_language(sample_pairs):
    df = pd.DataFrame(sample_pairs)
    config = DatasetConfig(url="x", source="gh", languages=[DatasetLanguageCode.de])
    result = _apply_filters(df, config)
    assert set(result["language"]) == {"de"}


def test_apply_filters_by_pressure_level(sample_pairs):
    df = pd.DataFrame(sample_pairs)
    config = DatasetConfig(url="x", source="gh", pressure_levels=["L0"])
    result = _apply_filters(df, config)
    assert set(result["pressure_level"]) == {"L0"}
    assert len(result) == len(sample_pairs)  # all sample rows are L0


def test_apply_filters_missing_column_is_noop(sample_pairs):
    df = pd.DataFrame(sample_pairs).drop(columns=["topic"])
    config = DatasetConfig(url="x", source="gh")
    result = _apply_filters(df, config)
    assert len(result) == len(sample_pairs)


def test_apply_filters_combined(sample_pairs):
    df = pd.DataFrame(sample_pairs)
    config = DatasetConfig(
        url="x",
        source="gh",
        categories=[Category.ack_invert],
        languages=[DatasetLanguageCode.en],
    )
    result = _apply_filters(df, config)
    assert len(result) == 1
    assert result.iloc[0]["id"] == "en-1"


# --------------------------------------------------------------------------- #
# split_constrast_pairs
# --------------------------------------------------------------------------- #
def test_split_constrast_pairs_produces_clean_and_revoked_rows():
    df = pd.DataFrame([make_row(id="r1", pair_type="active_cancelled")])
    result = split_constrast_pairs(df)
    assert len(result) == 2
    assert set(result["id"]) == {"r1_clean", "r1_revoked"}
    clean = result[result["id"] == "r1_clean"].iloc[0]
    revoked = result[result["id"] == "r1_revoked"].iloc[0]
    assert clean["rule_status"] == "active"
    assert revoked["rule_status"] == "cancelled"
    assert clean["system"] == df.iloc[0]["system_rule"]
    assert revoked["system"] == df.iloc[0]["system_non_rule"]


def test_split_constrast_pairs_assigns_the_matching_checker_only():
    df = pd.DataFrame([make_row(id="r1", pair_type="active_cancelled")])
    result = split_constrast_pairs(df)
    clean = result[result["id"] == "r1_clean"].iloc[0]
    revoked = result[result["id"] == "r1_revoked"].iloc[0]

    assert clean["checker"] == df.iloc[0]["active_checker"]
    assert revoked["checker"] == df.iloc[0]["revoked_checker"]
    # neither split row carries the other side's checker
    assert "active_checker" not in clean.index
    assert "revoked_checker" not in clean.index
    assert "active_checker" not in revoked.index
    assert "revoked_checker" not in revoked.index


def test_split_constrast_pairs_covers_all_pair_types():
    rows = [make_row(id=f"r-{pt.value}", pair_type=pt.value) for pt in PairType]
    result = split_constrast_pairs(pd.DataFrame(rows))
    assert len(result) == 2 * len(PairType)
    for pt in PairType:
        active, revoked = PAIR_STATUS[pt]
        clean = result[result["id"] == f"r-{pt.value}_clean"].iloc[0]
        assert clean["rule_status"] == active.value


def test_split_constrast_pairs_no_rows_dropped_for_mixed_pair_types():
    df = pd.DataFrame(
        [
            make_row(id="a", pair_type="active_cancelled"),
            make_row(id="b", pair_type="on_off"),
            make_row(id="c", pair_type="enabled_disabled"),
        ]
    )
    result = split_constrast_pairs(df)
    assert len(result) == 6  # every source row splits into 2, none dropped

    on_off = result[result["id"] == "b_clean"].iloc[0]
    assert on_off["rule_status"] == "on"
    enabled = result[result["id"] == "c_clean"].iloc[0]
    assert enabled["rule_status"] == "enabled"


def test_split_constrast_pairs_requires_active_and_revoked_status_columns():
    # No fallback here (unlike the old pair_type-keyed lookup): rows must
    # already carry active_status/revoked_status, i.e. have passed RulePair
    # validation, before reaching this function.
    df = pd.DataFrame([make_row(id="r1")]).drop(columns=["active_status"])
    with pytest.raises(KeyError):
        split_constrast_pairs(df)


# --------------------------------------------------------------------------- #
# DatasetConfig / RulePair validation
# --------------------------------------------------------------------------- #
def test_dataset_config_defaults_include_full_enums():
    config = DatasetConfig(url="x", source="gh")
    assert set(config.categories) == set(Category)
    assert set(config.languages) == set(DatasetLanguageCode)
    assert set(config.grammars) == set(GrammarType)
    assert set(config.topics) == set(Topic)
    assert set(config.contrastive_pairs) == set(PairType)
    assert set(config.pressure_levels) == set(PressureLevel)
    assert set(config.pressure_names) == set(PressureName)


def test_dataset_config_rejects_unsupported_language():
    with pytest.raises(ValidationError):
        DatasetConfig(url="x", source="gh", languages=["xx"])


def test_dataset_config_accepts_supported_language_subset():
    config = DatasetConfig(url="x", source="gh", languages=["en", "de"])
    assert config.languages == [DatasetLanguageCode.en, DatasetLanguageCode.de]


def test_rule_pair_ignores_extra_fields():
    row = RulePair(**make_row(extra_field="dropped"))
    assert "extra_field" not in row.model_dump()


def test_rule_pair_requires_mandatory_fields():
    incomplete = make_row()
    del incomplete["system_rule"]
    with pytest.raises(ValidationError):
        RulePair(**incomplete)


def test_rule_pair_rejects_mismatched_status_pair():
    bad = make_row(active_status="active", revoked_status="off")  # not a real pair
    with pytest.raises(ValidationError):
        RulePair(**bad)


# --------------------------------------------------------------------------- #
# dataset_generator / CrossLingualRuleFollowingDataset - GH source
# --------------------------------------------------------------------------- #
def test_dataset_generator_gh_source_end_to_end(sample_dataset_file):
    config = DatasetConfig(url=str(sample_dataset_file), source=DatasetSource.gh)
    df = dataset_generator(config)
    assert len(df) == 8
    assert set(df["rule_status"]) == {"active", "cancelled", "on", "off", "enabled", "disabled"}


def test_cross_lingual_dataset_gh_source(sample_dataset_file):
    config = DatasetConfig(url=str(sample_dataset_file), source=DatasetSource.gh)
    dataset = CrossLingualRuleFollowingDataset(config)
    assert len(dataset) == 8


def test_cross_lingual_dataset_gh_source_invalid_path_raises(tmp_path):
    config = DatasetConfig(url=str(tmp_path / "missing.json"), source=DatasetSource.gh)
    with pytest.raises(FileNotFoundError):
        CrossLingualRuleFollowingDataset(config)


# --------------------------------------------------------------------------- #
# dataset_generator / CrossLingualRuleFollowingDataset - HF source
# --------------------------------------------------------------------------- #
def test_dataset_generator_hf_source_mocked(monkeypatch, sample_pairs):
    calls = []

    def fake_load_source_dataset(repo_id):
        calls.append(repo_id)
        return {"train": FakeHFDataset(pd.DataFrame(sample_pairs))}

    monkeypatch.setattr(
        ds.HFDataHelper, "load_source_dataset", fake_load_source_dataset
    )

    config = DatasetConfig(url="some-org/rule-following-pairs", source=DatasetSource.hf)
    df = dataset_generator(config)

    assert calls == ["some-org/rule-following-pairs"]
    assert len(df) == 8  # same 4 source rows x2 as the GH fixture, none dropped


def test_cross_lingual_dataset_hf_source_mocked(monkeypatch, sample_pairs):
    monkeypatch.setattr(
        ds.HFDataHelper,
        "load_source_dataset",
        lambda repo_id: {"train": FakeHFDataset(pd.DataFrame(sample_pairs))},
    )
    config = DatasetConfig(url="some-org/rule-following-pairs", source=DatasetSource.hf)
    dataset = CrossLingualRuleFollowingDataset(config)
    assert len(dataset) == 8


def test_dataset_generator_hf_source_propagates_failure(monkeypatch):
    def fake_load_source_dataset(repo_id):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        ds.HFDataHelper, "load_source_dataset", fake_load_source_dataset
    )
    config = DatasetConfig(url="some-org/rule-following-pairs", source=DatasetSource.hf)
    with pytest.raises(RuntimeError, match="network down"):
        dataset_generator(config)


# --------------------------------------------------------------------------- #
# CrossLingualRuleFollowingDataset instance methods
# --------------------------------------------------------------------------- #
def test_from_dataframe_bypasses_fetch(sample_pairs):
    df = pd.DataFrame(sample_pairs)
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(df)
    assert dataset.config is None
    assert len(dataset) == len(sample_pairs)
    assert dataset[0]["id"] == sample_pairs[0]["id"]


def test_shuffle_is_deterministic_given_seed(sample_pairs):
    df = pd.DataFrame(sample_pairs)
    d1 = CrossLingualRuleFollowingDataset.from_dataframe(df).shuffle(seed=7)
    d2 = CrossLingualRuleFollowingDataset.from_dataframe(df).shuffle(seed=7)
    assert list(d1.df["id"]) == list(d2.df["id"])


def test_head_truncates(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )
    dataset.head(2)
    assert len(dataset) == 2


def test_head_beyond_length_returns_all_rows(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )
    dataset.head(1000)
    assert len(dataset) == len(sample_pairs)


def test_head_zero_returns_empty(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )
    dataset.head(0)
    assert len(dataset) == 0


def test_subset_single_value(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )
    subset = dataset.subset(language="de")
    assert len(subset) == 1
    assert subset.df.iloc[0]["language"] == "de"


def test_subset_list_value(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )
    subset = dataset.subset(topic=["legal", "finance"])
    assert set(subset.df["topic"]) == {"legal", "finance"}


def test_subset_does_not_mutate_original(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )
    original_len = len(dataset)
    dataset.subset(language="de")
    assert len(dataset) == original_len


def test_subset_missing_column_is_noop(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )
    subset = dataset.subset(nonexistent_column="anything")
    assert len(subset) == len(sample_pairs)


def test_build_indices_populates_and_nulls_correctly(sample_pairs):
    dataset = CrossLingualRuleFollowingDataset.from_dataframe(
        pd.DataFrame(sample_pairs)
    )

    def fn(row):
        if row["category"] == "no_pii":
            return None
        return (1, 2)

    dataset.build_indices(fn)
    no_pii = dataset.df[dataset.df["category"] == "no_pii"].iloc[0]
    other = dataset.df[dataset.df["category"] != "no_pii"].iloc[0]
    assert pd.isna(no_pii["correct_idx"])
    assert other["correct_idx"] == 1
    assert other["incorrect_idx"] == 2


def test_to_dataloader_batches_with_collate_behavioral(sample_dataset_file):
    # Post-split rows (system/rule_status columns), matching real usage.
    config = DatasetConfig(url=str(sample_dataset_file), source=DatasetSource.gh)
    dataset = CrossLingualRuleFollowingDataset(config)
    loader = dataset.to_dataloader(batch_size=8, collate_fn=collate_behavioral)
    batch = next(iter(loader))
    system, user_query, ids, checkers = batch
    assert len(system) == 4  # only the active-side half of each of the 4 pairs
    assert len(user_query) == 4
    assert len(ids) == 4
    assert len(checkers) == 4
    assert all(i.endswith("_clean") for i in ids)
    assert all(c["binds"] for c in checkers)  # active_checker always binds=True


# --------------------------------------------------------------------------- #
# collate_behavioral
# --------------------------------------------------------------------------- #
def make_split_row(**overrides) -> dict:
    """Shaped like dataset_generator's real (post-split) output."""
    row = {
        "id": "r1_clean",
        "system": "Follow the rule.",
        "user_query": "What do I do?",
        "rule_status": "active",
        "checker": make_checker("active", True),
    }
    row.update(overrides)
    return row


def test_collate_behavioral_keeps_only_active_side():
    batch = [
        make_split_row(id="r1_clean", rule_status="active"),
        make_split_row(
            id="r1_revoked", rule_status="cancelled", checker=make_checker("cancelled", False)
        ),
        make_split_row(id="r2_clean", rule_status="on", checker=make_checker("on", True)),
        make_split_row(
            id="r2_revoked", rule_status="off", checker=make_checker("off", False)
        ),
    ]
    system, user_query, ids, checkers = collate_behavioral(batch)
    assert ids == ["r1_clean", "r2_clean"]
    assert len(system) == 2
    assert len(user_query) == 2
    assert len(checkers) == 2
    assert all(c["binds"] for c in checkers)


def test_collate_behavioral_extracts_expected_fields():
    batch = [make_split_row(id="r1_clean"), make_split_row(id="r2_clean")]
    system, user_query, ids, checkers = collate_behavioral(batch)
    assert system == [row["system"] for row in batch]
    assert user_query == [row["user_query"] for row in batch]
    assert ids == [row["id"] for row in batch]
    assert checkers == [row["checker"] for row in batch]


def test_collate_behavioral_empty_batch_when_all_revoked():
    batch = [
        make_split_row(
            id="r1_revoked", rule_status="cancelled", checker=make_checker("cancelled", False)
        )
    ]
    system, user_query, ids, checkers = collate_behavioral(batch)
    assert system == user_query == ids == checkers == []


# --------------------------------------------------------------------------- #
# HFDataHelper (upload/exists/fetch)
# --------------------------------------------------------------------------- #
def test_hf_data_helper_init_does_not_eagerly_create_repo(monkeypatch):
    monkeypatch.setattr(ds, "HfApi", FakeHfApi)
    helper = HFDataHelper(repo_id="org/outputs")
    assert helper._api.create_repo_calls == []


def test_hf_data_helper_hf_path_format():
    monkeypatch_helper = HFDataHelper.__new__(HFDataHelper)
    path = monkeypatch_helper._hf_path("meta-llama/Llama-3", DatasetLanguageCode.en)
    assert path == "meta-llama__Llama-3/en.parquet"


def test_hf_data_helper_upload_calls_create_repo_and_upload_file(
    monkeypatch, sample_pairs
):
    monkeypatch.setattr(ds, "HfApi", FakeHfApi)
    helper = HFDataHelper(repo_id="org/outputs")
    df = pd.DataFrame(sample_pairs)

    helper.upload(df=df, model_id="gpt2", lang_code=DatasetLanguageCode.en)

    assert len(helper._api.create_repo_calls) == 1
    assert len(helper._api.upload_file_calls) == 1
    call = helper._api.upload_file_calls[0]
    assert call["path_in_repo"] == "gpt2/en.parquet"
    assert call["repo_id"] == "org/outputs"
    assert len(helper._api.uploaded_df) == len(sample_pairs)


def test_hf_data_helper_exists_true(monkeypatch):
    monkeypatch.setattr(ds, "HfApi", FakeHfApi)
    helper = HFDataHelper(repo_id="org/outputs")
    helper._api.get_paths_info_return = [{"path": "gpt2/en.parquet"}]
    assert helper.exists(model_id="gpt2", lang_code=DatasetLanguageCode.en) is True


def test_hf_data_helper_exists_false_when_not_found(monkeypatch):
    monkeypatch.setattr(ds, "HfApi", FakeHfApi)
    helper = HFDataHelper(repo_id="org/outputs")
    helper._api.get_paths_info_return = []
    assert helper.exists(model_id="gpt2", lang_code=DatasetLanguageCode.en) is False


def test_hf_data_helper_exists_false_on_api_error(monkeypatch):
    monkeypatch.setattr(ds, "HfApi", FakeHfApi)
    helper = HFDataHelper(repo_id="org/outputs")
    helper._api.get_paths_info_return = RuntimeError("not found")
    assert helper.exists(model_id="gpt2", lang_code=DatasetLanguageCode.en) is False


def test_hf_data_helper_fetch_reads_parquet(monkeypatch, tmp_path, sample_pairs):
    monkeypatch.setattr(ds, "HfApi", FakeHfApi)
    local_path = tmp_path / "gpt2_en.parquet"
    pd.DataFrame(sample_pairs).to_parquet(local_path, index=False)

    monkeypatch.setattr(ds, "hf_hub_download", lambda **kwargs: str(local_path))

    helper = HFDataHelper(repo_id="org/outputs")
    result = helper.fetch(model_id="gpt2", lang_code=DatasetLanguageCode.en)
    assert len(result) == len(sample_pairs)
