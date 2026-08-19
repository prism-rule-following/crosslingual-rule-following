"""Unit tests for canonical/evaluation/inference.py.

Uses a real tiny model (roneneldan/TinyStories-1M) for ModelRunner-level
behavior - tokenization, chat templating, generation, and activation
extraction all depend on real tokenizer/model mechanics that a mock would
just reassert rather than verify. run() orchestration is tested with fully
mocked ModelRunner/HFDataHelper/dataset collaborators, since that logic is
pure control flow and doesn't need real model weights.
"""

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
for _p in (str(_REPO_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import pytest
import torch

import inference as inf
from canonical.model.dataset import DatasetConfig, DatasetLanguageCode, DatasetSource

TINY_MODEL = "roneneldan/TinyStories-1M"


def make_checker(rule_status: str = "active", binds: bool = True) -> dict:
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
    """Shaped like dataset_generator's real output: the "system" column
    (not "system_rule") is what split_constrast_pairs actually produces,
    and "checker" is the single active_checker/revoked_checker matching
    rule_status (see split_constrast_pairs).
    """
    row = {
        "id": "r1",
        "category": "ack_invert",
        "topic": "legal",
        "grammar_type": "imperative",
        "language": "en",
        "system": "Follow the rule.",
        "user_query": "What do I do?",
        "pair_type": "active_cancelled",
        "rule_status": "active",
        "checker": make_checker("active", True),
        "pressure_level": "L0",
        "pressure_name": "neutral",
    }
    row.update(overrides)
    return row


@pytest.fixture
def tiny_dataset() -> pd.DataFrame:
    return pd.DataFrame([make_row(id="r1"), make_row(id="r2"), make_row(id="r3")])


@pytest.fixture
def base_config(tmp_path) -> inf.ModelGenerationConfig:
    return inf.ModelGenerationConfig(
        model_ids=[TINY_MODEL],
        dataset_config=DatasetConfig(url="x", source=DatasetSource.gh),
        language_codes=[DatasetLanguageCode.en],
        max_new_tokens=4,
        generation_batch_size=4,
        activation_batch_size=4,
        checkpoint_dir=str(tmp_path),
    )


_MINIMAL_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}{% if add_generation_prompt %}assistant:\n{% endif %}"
)


@pytest.fixture(scope="module")
def loaded_runner(tmp_path_factory) -> "inf.ModelRunner":
    config = inf.ModelGenerationConfig(
        model_ids=[TINY_MODEL],
        dataset_config=DatasetConfig(url="x", source=DatasetSource.gh),
        language_codes=[DatasetLanguageCode.en],
        max_new_tokens=4,
        generation_batch_size=4,
        activation_batch_size=4,
        checkpoint_dir=str(tmp_path_factory.mktemp("checkpoints")),
    )
    runner = inf.ModelRunner(config)
    runner.load(model_id=TINY_MODEL)
    runner.tokenizer.chat_template = _MINIMAL_CHAT_TEMPLATE
    runner.supports_system_role = runner._check_system_role_support()
    return runner


@pytest.fixture(autouse=True)
def _isolate_checkpoint_dir(request, tmp_path):
    # loaded_runner is module-scoped (real model load is expensive) but
    # checkpointing writes files keyed by model_id/language - give every test
    # a fresh checkpoint_dir so one test's checkpoint can't leak into another's.
    if "loaded_runner" in request.fixturenames:
        request.getfixturevalue("loaded_runner").config.checkpoint_dir = str(tmp_path)


# --------------------------------------------------------------------------- #
# ModelResponse / ModelGenerationConfig schema
# --------------------------------------------------------------------------- #
def test_model_response_valid_construction():
    resp = inf.ModelResponse(**make_row(id="a"), model_id=TINY_MODEL, response="r")
    assert resp.sample_idx == 0
    assert resp.model_id == TINY_MODEL
    assert resp.checker.binds is True
    assert resp.pair_type == "active_cancelled"
    assert resp.rule_status == "active"


def test_model_response_missing_required_field_raises():
    with pytest.raises(Exception):
        inf.ModelResponse(id="a", topic="legal")


def test_model_response_system_is_required():
    row = make_row(id="a")
    del row["system"]
    with pytest.raises(Exception):
        inf.ModelResponse(**row, model_id=TINY_MODEL, response="r")


def test_model_generation_config_defaults():
    config = inf.ModelGenerationConfig(
        model_ids=["m"],
        dataset_config=DatasetConfig(url="x", source=DatasetSource.gh),
        language_codes=[DatasetLanguageCode.en],
    )
    assert config.n_samples == 1
    assert config.run_inference_response is True
    assert config.run_inference_activations is False
    assert config.hf_result_repo is None
    assert config.hf_activations_repo is None


def test_model_generation_config_device_property(monkeypatch, base_config):
    monkeypatch.setattr(inf.utils, "get_device", lambda: "mps")
    assert base_config.device == "mps"


# --------------------------------------------------------------------------- #
# ModelRunner.load (real tiny model)
# --------------------------------------------------------------------------- #
def test_load_sets_padding_side_and_pad_token(loaded_runner):
    assert loaded_runner.tokenizer.padding_side == "left"
    assert loaded_runner.tokenizer.pad_token is not None


def test_load_enables_patching_cfg_flags(loaded_runner):
    assert loaded_runner.model.cfg.use_attn_result is True
    assert loaded_runner.model.cfg.use_split_qkv_input is True
    assert loaded_runner.model.cfg.use_hook_mlp_in is True


# --------------------------------------------------------------------------- #
# get_hook_filter - documents CURRENT hook names (not fixed in this pass)
# --------------------------------------------------------------------------- #
def test_get_hook_filter_current_hook_names(loaded_runner):
    n_layers = loaded_runner.model.cfg.n_layers
    expected = (
        ["hook_embed"]
        + [
            f"blocks.{i}.{hook}"
            for i in range(n_layers)
            for hook in inf.PROBING_HOOK_SUFFIXES
        ]
        + [
            f"blocks.{i}.{hook}"
            for i in range(n_layers)
            for hook in inf.PATCHING_HOOK_SUFFIXES
        ]
    )
    assert loaded_runner.get_hook_filter() == expected


def test_get_probing_and_patching_hooks_are_disjoint(loaded_runner):
    probing = loaded_runner.get_probing_hooks()
    patching = loaded_runner.get_patching_hooks()
    assert "hook_embed" in probing
    assert any(h.endswith("hook_resid_post") for h in probing)
    assert any(h.endswith("attn.qkv.hook_in") for h in patching)
    assert not set(probing) & set(patching)
    assert set(probing) | set(patching) == set(loaded_runner.get_hook_filter())


def test_extract_hidden_states_returns_arrays_and_index(loaded_runner, tiny_dataset):
    out = loaded_runner.extract_hidden_states(
        tiny_dataset, lang_code=DatasetLanguageCode.en
    )
    assert out.n_rows == len(tiny_dataset)
    assert set(out.index["id"]) == {"r1", "r2", "r3"}
    assert set(out.index.columns) == {
        "row_idx",
        "id",
        "rule_status",
        "grammar_type",
        "category",
        "topic",
        "pair_type",
        "pressure_level",
        "pressure_name",
        "language",
    }
    assert "hook_embed" in out.arrays
    assert any(g.endswith("hook_out") for g in out.arrays)
    assert not any("qkv" in g for g in out.arrays)
    assert out.arrays["hook_embed"].shape[0] == len(tiny_dataset)


# --------------------------------------------------------------------------- #
# format_chat_prompt / _check_system_role_support
# --------------------------------------------------------------------------- #
def test_format_chat_prompt_includes_user_text(loaded_runner):
    prompt = loaded_runner.format_chat_prompt("SYSTEM TEXT", "USER TEXT")
    assert isinstance(prompt, str)
    assert "USER TEXT" in prompt


def test_format_chat_prompt_folds_system_into_user_when_unsupported(loaded_runner):
    loaded_runner.supports_system_role = False
    try:
        prompt = loaded_runner.format_chat_prompt("SYS TEXT", "USER TEXT")
        assert "SYS TEXT" in prompt
        assert "USER TEXT" in prompt
    finally:
        loaded_runner.supports_system_role = True


def test_check_system_role_support_false_when_template_rejects_system(
    loaded_runner, monkeypatch
):
    def fake_apply(chat, **kwargs):
        if any(m["role"] == "system" for m in chat):
            raise ValueError("System role not supported")
        return "ok"

    monkeypatch.setattr(loaded_runner.tokenizer, "apply_chat_template", fake_apply)
    assert loaded_runner._check_system_role_support() is False


# --------------------------------------------------------------------------- #
# generate_response
# --------------------------------------------------------------------------- #
def test_generate_response_returns_one_row_per_example(loaded_runner, tiny_dataset):
    results = loaded_runner.generate_response(
        tiny_dataset, lang_code=DatasetLanguageCode.en
    )
    assert len(results) == len(tiny_dataset)
    for row in results:
        assert row["sample_idx"] == 0
        assert isinstance(row["response"], str)
        assert row["model_id"] == TINY_MODEL


def test_generate_response_n_samples_multiplies_output_rows(loaded_runner):
    dataset = pd.DataFrame([make_row(id="r1")])
    loaded_runner.config.n_samples = 2
    try:
        results = loaded_runner.generate_response(
            dataset, lang_code=DatasetLanguageCode.en
        )
        assert len(results) == 2
        assert sorted(r["sample_idx"] for r in results) == [0, 1]
    finally:
        loaded_runner.config.n_samples = 1


# --------------------------------------------------------------------------- #
# checkpointing (generate_response / extract_hidden_states)
# --------------------------------------------------------------------------- #
def test_generate_response_writes_checkpoint_file(loaded_runner, tiny_dataset):
    loaded_runner.generate_response(tiny_dataset, lang_code=DatasetLanguageCode.en)
    path = loaded_runner._checkpoint_path(DatasetLanguageCode.en, "responses")
    assert path.exists()
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    assert {row["id"] for row in lines} == {"r1", "r2", "r3"}


def test_generate_response_resumes_by_skipping_checkpointed_ids(loaded_runner):
    dataset = pd.DataFrame([make_row(id="r1"), make_row(id="r2")])

    already_done = inf.ModelResponse(
        **make_row(id="r1"), model_id=TINY_MODEL, response="PRE-EXISTING RESPONSE"
    ).model_dump()
    loaded_runner._append_checkpoint(
        DatasetLanguageCode.en, "responses", [already_done]
    )

    results = loaded_runner.generate_response(dataset, lang_code=DatasetLanguageCode.en)

    assert len(results) == 2
    by_id = {row["id"]: row for row in results}
    assert by_id["r1"]["response"] == "PRE-EXISTING RESPONSE"
    assert by_id["r2"]["response"] != "PRE-EXISTING RESPONSE"


def test_extract_hidden_states_writes_checkpoint_file(loaded_runner, tiny_dataset):
    loaded_runner.extract_hidden_states(tiny_dataset, lang_code=DatasetLanguageCode.en)
    done_path = loaded_runner._activation_done_path(DatasetLanguageCode.en)
    assert done_path.exists()
    with open(done_path) as f:
        ids = [line.strip() for line in f]
    assert set(ids) == {"r1", "r2", "r3"}
    shards = list(
        loaded_runner._activation_shards_dir(DatasetLanguageCode.en).glob("*.npy")
    )
    assert shards


def test_extract_hidden_states_resumes_without_duplicating(loaded_runner):
    dataset = pd.DataFrame([make_row(id="r1"), make_row(id="r2")])
    loaded_runner.extract_hidden_states(dataset, lang_code=DatasetLanguageCode.en)
    # second run: everything already checkpointed, must not redo or duplicate
    out = loaded_runner.extract_hidden_states(dataset, lang_code=DatasetLanguageCode.en)
    assert out.n_rows == 2
    assert out.arrays["hook_embed"].shape[0] == 2
    assert loaded_runner._activation_done_count(DatasetLanguageCode.en) == 2


# --------------------------------------------------------------------------- #
# run() orchestration - fully mocked, no real model
# --------------------------------------------------------------------------- #
class FakeLoadedDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def subset(self, **filters):
        return self  # single fake dataset; filtering is a no-op for these tests


class FakeRunner:
    """Stands in for ModelRunner: records what it's asked to do."""

    instances = []

    def __init__(self, config):
        self.config = config
        self.loaded_with = None
        self.generate_response_called = False
        self.extract_hidden_states_called = False
        self.upload_activations_called = False
        self.uploaded_activation_repo = None
        self.clear_activation_checkpoint_called = False
        FakeRunner.instances.append(self)

    def load(self, model_id):
        self.loaded_with = model_id

    def generate_response(self, dataset, lang_code):
        self.generate_response_called = True
        return [{"id": "r1", "response": "hi"}]

    def extract_hidden_states(self, dataset, lang_code):
        self.extract_hidden_states_called = True
        return inf.ActivationOutput(
            arrays={"hook_embed": [[0.0]]},
            index=pd.DataFrame([{"id": "r1", "row_idx": 0}]),
            n_rows=1,
        )

    def upload_activations(self, helper, act, lang_code):
        self.upload_activations_called = True
        self.uploaded_activation_repo = helper.repo_id

    def clear_activation_checkpoint(self, lang_code):
        self.clear_activation_checkpoint_called = True


class FakeHFDataHelper:
    """Stands in for HFDataHelper: records upload calls."""

    calls = []
    exists_return = False

    def __init__(self, repo_id):
        self.repo_id = repo_id

    def exists(self, model_id, lang_code):
        return FakeHFDataHelper.exists_return

    def exists_path(self, path_in_repo):
        return FakeHFDataHelper.exists_return

    def upload(self, df, model_id, lang_code):
        FakeHFDataHelper.calls.append(
            {
                "repo_id": self.repo_id,
                "model_id": model_id,
                "lang_code": lang_code,
                "n_rows": len(df),
            }
        )


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeRunner.instances.clear()
    FakeHFDataHelper.calls.clear()
    FakeHFDataHelper.exists_return = False
    yield
    FakeRunner.instances.clear()
    FakeHFDataHelper.calls.clear()
    FakeHFDataHelper.exists_return = False


def run_config(**overrides) -> inf.ModelGenerationConfig:
    defaults = dict(
        model_ids=["fake-model"],
        dataset_config=DatasetConfig(url="x", source=DatasetSource.gh),
        language_codes=[DatasetLanguageCode.en],
        push_to_hf=True,
        run_inference_response=True,
        run_inference_activations=False,
    )
    defaults.update(overrides)
    return inf.ModelGenerationConfig(**defaults)


def test_run_generates_and_uploads_responses_when_configured(monkeypatch):
    df = pd.DataFrame([make_row(id="r1")])
    monkeypatch.setattr(
        inf, "CrossLingualRuleFollowingDataset", lambda cfg: FakeLoadedDataset(df)
    )
    monkeypatch.setattr(inf, "ModelRunner", FakeRunner)
    monkeypatch.setattr(inf, "HFDataHelper", FakeHFDataHelper)

    config = run_config(hf_result_repo="org/results")
    inf.run(config)

    assert len(FakeRunner.instances) == 1
    assert FakeRunner.instances[0].loaded_with == "fake-model"
    assert FakeHFDataHelper.calls == [
        {
            "repo_id": "org/results",
            "model_id": "fake-model",
            "lang_code": DatasetLanguageCode.en,
            "n_rows": 1,
        }
    ]


def test_run_skips_model_already_uploaded(monkeypatch, capsys):
    df = pd.DataFrame([make_row(id="r1")])
    monkeypatch.setattr(
        inf, "CrossLingualRuleFollowingDataset", lambda cfg: FakeLoadedDataset(df)
    )
    monkeypatch.setattr(inf, "ModelRunner", FakeRunner)
    monkeypatch.setattr(inf, "HFDataHelper", FakeHFDataHelper)
    FakeHFDataHelper.exists_return = True

    config = run_config(hf_result_repo="org/results")
    inf.run(config)

    assert FakeRunner.instances[0].generate_response_called is False
    assert FakeHFDataHelper.calls == []
    assert "already uploaded" in capsys.readouterr().out


def test_run_skips_response_upload_when_repo_unset(monkeypatch, capsys):
    df = pd.DataFrame([make_row(id="r1")])
    monkeypatch.setattr(
        inf, "CrossLingualRuleFollowingDataset", lambda cfg: FakeLoadedDataset(df)
    )
    monkeypatch.setattr(inf, "ModelRunner", FakeRunner)
    monkeypatch.setattr(inf, "HFDataHelper", FakeHFDataHelper)

    config = run_config(hf_result_repo=None)
    inf.run(config)

    assert FakeHFDataHelper.calls == []
    assert "Skipping response upload" in capsys.readouterr().out


def test_run_runs_activation_pass_when_enabled(monkeypatch):
    df = pd.DataFrame([make_row(id="r1")])
    monkeypatch.setattr(
        inf, "CrossLingualRuleFollowingDataset", lambda cfg: FakeLoadedDataset(df)
    )
    monkeypatch.setattr(inf, "ModelRunner", FakeRunner)
    monkeypatch.setattr(inf, "HFDataHelper", FakeHFDataHelper)

    config = run_config(
        run_inference_response=False,
        run_inference_activations=True,
        hf_activations_repo="org/activations",
    )
    inf.run(config)

    runner = FakeRunner.instances[0]
    assert runner.extract_hidden_states_called is True
    assert runner.upload_activations_called is True
    assert runner.uploaded_activation_repo == "org/activations"
    assert runner.clear_activation_checkpoint_called is True


def test_run_iterates_all_model_ids(monkeypatch):
    df = pd.DataFrame([make_row(id="r1")])
    monkeypatch.setattr(
        inf, "CrossLingualRuleFollowingDataset", lambda cfg: FakeLoadedDataset(df)
    )
    monkeypatch.setattr(inf, "ModelRunner", FakeRunner)
    monkeypatch.setattr(inf, "HFDataHelper", FakeHFDataHelper)

    config = run_config(model_ids=["model-a", "model-b"], hf_result_repo="org/results")
    inf.run(config)

    assert [r.loaded_with for r in FakeRunner.instances] == ["model-a", "model-b"]
    assert len(FakeHFDataHelper.calls) == 2


def test_run_iterates_all_language_codes_per_model(monkeypatch):
    # The model should load once and be reused across every requested
    # language, uploading a separate file per language.
    df = pd.DataFrame([make_row(id="r1")])
    monkeypatch.setattr(
        inf, "CrossLingualRuleFollowingDataset", lambda cfg: FakeLoadedDataset(df)
    )
    monkeypatch.setattr(inf, "ModelRunner", FakeRunner)
    monkeypatch.setattr(inf, "HFDataHelper", FakeHFDataHelper)

    config = run_config(
        language_codes=[DatasetLanguageCode.en, DatasetLanguageCode.de],
        hf_result_repo="org/results",
    )
    inf.run(config)

    assert len(FakeRunner.instances) == 1  # model loaded once, not once per language
    assert [c["lang_code"] for c in FakeHFDataHelper.calls] == [
        DatasetLanguageCode.en,
        DatasetLanguageCode.de,
    ]


def test_run_reraises_on_failure(monkeypatch):
    def boom(cfg):
        raise RuntimeError("dataset load failed")

    monkeypatch.setattr(inf, "CrossLingualRuleFollowingDataset", boom)

    config = run_config()
    with pytest.raises(RuntimeError, match="dataset load failed"):
        inf.run(config)
