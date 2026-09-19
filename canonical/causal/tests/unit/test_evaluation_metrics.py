"""Unit tests for evaluation_metrics.py metric functions."""

import pytest
import torch

from canonical.causal.activation_patching.evaluation_metrics import (
    get_logit_positions,
    logit_difference,
    make_adherence_metric,
    make_internal_state_metric,
)


def test_get_logit_positions_picks_last_prompt_token():
    # batch of 2, seq len 3, vocab 2
    logits = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    input_length = torch.tensor([2, 3])  # -> positions 1 and 2 respectively

    result = get_logit_positions(logits, input_length)

    assert torch.equal(result, torch.tensor([[3.0, 4.0], [11.0, 12.0]]))


def test_logit_difference_matches_manual_computation():
    logits = torch.tensor([[[1.0, 5.0, 2.0]]])  # batch=1, seq=1, vocab=3
    input_length = torch.tensor([1])
    labels = torch.tensor([[1, 2]])  # compare vocab index 1 (5.0) vs index 2 (2.0)

    raw = logit_difference(logits, None, input_length, labels, mean=False, loss=False)
    assert torch.allclose(raw, torch.tensor([3.0]))

    as_loss = logit_difference(logits, None, input_length, labels, mean=False, loss=True)
    assert torch.allclose(as_loss, torch.tensor([-3.0]))

    meaned = logit_difference(logits, None, input_length, labels, mean=True, loss=False)
    assert torch.allclose(meaned, torch.tensor(3.0))


class _FakeTokenizer:
    def __init__(self, decode_fn):
        self._decode_fn = decode_fn

    def decode(self, tokens):
        return self._decode_fn(tokens)


def test_make_adherence_metric_scores_each_example_individually():
    tokenizer = _FakeTokenizer(lambda tokens: "BANNED" if tokens[-1].item() == 1 else "OK")
    checker = lambda text: 0.0 if text == "BANNED" else 1.0

    metric = make_adherence_metric(checker, tokenizer, mean=False)

    # batch of 2, seq=2, vocab=2: argmax predicts token 1 for row 0, token 0 for row 1
    logits = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )
    input_lengths = torch.tensor([2, 2])
    label = torch.tensor([0, 0])

    scores = metric(logits, None, input_lengths, label)

    assert torch.equal(scores, torch.tensor([0.0, 1.0]))


def test_make_adherence_metric_decodes_only_the_generated_span():
    """The metric must decode predictions starting exactly at input_lengths[batch] - 1
    (the position whose prediction is the first generated token) through to the end --
    not the whole sequence, and not off by one in either direction. A marker token
    planted only at that boundary position proves the exact slice used, rather than
    just checking that *some* prediction near the end got decoded.
    """
    decoded_calls = []

    class RecordingTokenizer:
        def decode(self, tokens):
            decoded_calls.append(tuple(tokens.tolist()))
            return "recorded"

    metric = make_adherence_metric(lambda text: 1.0, RecordingTokenizer(), mean=False)

    # batch=1, seq_len=5, vocab=10; craft logits so argmax at each position gives a
    # distinct, identifiable token id. Token 9 marks the boundary position (index 1,
    # since input_length=2 => boundary = input_length - 1 = 1).
    desired_preds = [0, 9, 1, 2, 3]
    logits = torch.zeros(1, 5, 10)
    for pos, token_id in enumerate(desired_preds):
        logits[0, pos, token_id] = 10.0

    input_lengths = torch.tensor([2])
    label = torch.tensor([0])

    metric(logits, None, input_lengths, label)

    assert decoded_calls == [(9, 1, 2, 3)]


def test_make_adherence_metric_mean_reduces_to_scalar():
    tokenizer = _FakeTokenizer(lambda tokens: "x")
    metric = make_adherence_metric(lambda text: 1.0, tokenizer, mean=True)

    logits = torch.zeros(3, 2, 2)
    input_lengths = torch.tensor([2, 2, 2])
    label = torch.tensor([0, 0, 0])

    result = metric(logits, None, input_lengths, label)

    assert result.item() == 1.0


@pytest.mark.xfail(
    reason="cosine_similarity body is a TODO stub (returns None), not implemented yet",
    strict=True,
)
def test_make_internal_state_metric_returns_a_tensor():
    metric = make_internal_state_metric(torch.zeros(3), torch.zeros(3))

    result = metric(torch.zeros(1, 1, 2), None, torch.tensor([1]), torch.tensor([0]))

    assert isinstance(result, torch.Tensor)
