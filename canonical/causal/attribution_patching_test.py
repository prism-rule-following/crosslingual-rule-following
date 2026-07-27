from causal import attribution_patching
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from unittest.mock import patch
import contextlib
import torch
import pytest
import math

BRANCH = "dataset-design-gen-pipeline"
MODEL_ID = "roneneldan/TinyStories-1M"


@contextlib.contextmanager
def patch_cuda_only_zeros():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        yield
        return

    original_zeros = torch.zeros

    def patched_zeros(*args, **kwargs):
        requested = kwargs.get("device")
        if requested == "cuda" or (
            isinstance(requested, torch.device) and requested.type == "cuda"
        ):
            kwargs["device"] = device
        return original_zeros(*args, **kwargs)

    torch.zeros = patched_zeros
    try:
        yield
    finally:
        torch.zeros = original_zeros


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def to_dataloader(self, batch_size, collate_fn):
        return DataLoader(self._rows, batch_size=batch_size, collate_fn=collate_fn)


def _make_rows(n=4):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    correct_id = tokenizer(" a", add_special_tokens=False).input_ids[0]
    incorrect_id = tokenizer(" the", add_special_tokens=False).input_ids[0]
    assert correct_id != incorrect_id, "need two distinct token ids for the test"

    return [
        {
            "id": f"row{i}",
            "clean": "The rule is active. Answer:",
            "corrupted": "The rule is cancelled. Answer:",
            "correct_idx": correct_id,
            "incorrect_idx": incorrect_id,
        }
        for i in range(n)
    ]


@pytest.fixture
def config():
    return {
        "mode": "normal",
        "model_id": MODEL_ID,
        "dataset_config": {
            "url": "unused-because-mocked",
            "source": "gh",
            "language": ["en"],
            "category": ["bold_html"],
        },
        "batch_size": 2,
        "metrics": "logit_diff",
        "method": "EAP-IG-inputs",
        "steps": 2,
        "n_edges": 5,
    }


def test_run_valid_t0(config):
    fake_dataset = _FakeDataset(_make_rows())
    with (
        patch.object(
            attribution_patching,
            "CrossLingualRuleFollowingDataset",
            return_value=fake_dataset,
        ),
        patch_cuda_only_zeros(),
    ):

        result = attribution_patching.run(config)

    assert result is not None
    assert len(result.nodes) > 0
    assert len(result.edges) > 0
    assert len(result.edges) <= config["n_edges"]  # apply_greedy respects the budget
    assert not math.isnan(result.baseline)
    assert not math.isnan(result.corrupted_baseline)
    assert not math.isnan(result.circuit_faithfulness)
