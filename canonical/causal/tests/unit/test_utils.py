"""Unit tests for utils.py: intervention validation, metric resolution, JSON serialisation."""

import pytest
import torch

from canonical.causal.activation_patching.evaluation_metrics import logit_diff
from canonical.causal.activation_patching.utils import (
    resolve_metrics,
    to_serialisable,
    validate_intervention,
)


@pytest.mark.parametrize("intervention", ["mean", "patching", "zero"])
def test_validate_intervention_accepts_valid_values(intervention):
    assert validate_intervention(intervention) == intervention


def test_validate_intervention_rejects_unsupported_value():
    # 'mean-positional' is a real eap intervention type, but this codebase
    # deliberately restricts to mean/patching/zero.
    with pytest.raises(ValueError):
        validate_intervention("mean-positional")


def test_validate_intervention_rejects_garbage():
    with pytest.raises(ValueError):
        validate_intervention("not-a-real-intervention")


def test_resolve_metrics_returns_logit_diff_unwrapped():
    metrics = resolve_metrics(["logit_diff"])

    assert metrics == [logit_diff]


def test_resolve_metrics_raises_on_unknown_name():
    with pytest.raises(ValueError):
        resolve_metrics(["not_a_real_metric"])


def test_resolve_metrics_builds_adherence_from_checker_and_tokenizer():
    class FakeTokenizer:
        def decode(self, tokens):
            return "ok"

    metrics = resolve_metrics(["adherence"], checker=lambda text: 1.0, tokenizer=FakeTokenizer())

    assert len(metrics) == 1
    logits = torch.zeros(1, 1, 2)
    result = metrics[0](logits, None, torch.tensor([1]), torch.tensor([0]))
    assert result.item() == 1.0


def test_resolve_metrics_builds_multiple_metrics_in_requested_order():
    class FakeTokenizer:
        def decode(self, tokens):
            return "ok"

    metrics = resolve_metrics(
        ["logit_diff", "adherence"],
        checker=lambda text: 1.0,
        tokenizer=FakeTokenizer(),
    )

    assert len(metrics) == 2
    assert metrics[0] is logit_diff
    # metrics[1] must be the *built* adherence metric, not e.g. the raw
    # make_adherence_metric factory (which is also "callable" but takes
    # (checker, tokenizer, mean), not (logits, clean_logits, lengths, label)
    # -- so actually invoke it with the real 4-arg metric signature.
    assert metrics[1] is not logit_diff
    result = metrics[1](torch.zeros(1, 1, 2), None, torch.tensor([1]), torch.tensor([0]))
    assert result.item() == 1.0


def test_to_serialisable_converts_tensors_to_lists():
    assert to_serialisable(torch.tensor([1, 2, 3])) == [1, 2, 3]


def test_to_serialisable_converts_scalar_tensor_to_python_number():
    assert to_serialisable(torch.tensor(1.5)) == 1.5


def test_to_serialisable_stringifies_tuple_dict_keys():
    assert to_serialisable({(0, 1): "edge"}) == {"(0, 1)": "edge"}


def test_to_serialisable_recurses_into_nested_structures():
    nested = {"a": [torch.tensor(1.0), (torch.tensor(2), torch.tensor(3))]}

    assert to_serialisable(nested) == {"a": [1.0, [2, 3]]}


def test_to_serialisable_passes_through_plain_values():
    assert to_serialisable(5) == 5
    assert to_serialisable("hello") == "hello"
    assert to_serialisable(None) is None
