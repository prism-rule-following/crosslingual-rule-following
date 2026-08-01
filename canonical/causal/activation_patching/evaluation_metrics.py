"""Metrics to pass to the EAP `evaluate_graph` function."""

from typing import Callable, Dict, Optional, Union

import torch

# The metric protocol evaluate_graph calls: metric(logits, clean_logits, input_lengths, label).
# clean_logits is None whenever evaluate_graph/evaluate_baseline run with skip_clean=True.
MetricFn = Callable[[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
    """Helper function to get logit positions."""
    batch_size = logits.size(0)
    index = torch.arange(batch_size, device=logits.device)

    logits = logits[index, input_length - 1]
    return logits


def logit_difference(
    logits: torch.Tensor,
    clean_logits: Optional[torch.Tensor],
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
    loss: bool = True,
) -> torch.Tensor:
    """Computes the logit difference between the clean and corrupted logits."""
    logits = get_logit_positions(logits=logits, input_length=input_length)
    last_token_logits = torch.gather(logits, -1, labels.to(logits.device))
    results = last_token_logits[:, 0] - last_token_logits[:, 1]
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


logit_diff = logit_difference


# Wrapper to adjust to the EAP `evaluate_graph` function signature
def make_adherence_metric(
    checker: Callable[[str], float], tokenizer, mean: bool = True
) -> MetricFn:
    """Computes the adherence metric.
    check: function that checks if the output adheres to the rule (returns 1.0 or 0.0)
    tokenizer: tokenizer to decode the output
    mean: whether to return the mean or the individual scores
    """

    def adherence(
        logits: torch.Tensor,
        clean_logits: Optional[torch.Tensor],
        input_lengths: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: adjust for the token span as well
        preds = logits.argmax(-1)
        # slicing the *generated* span (after the prompt)
        scores = [
            float(checker(tokenizer.decode(preds[batch, input_lengths[batch] - 1 :])))
            for batch in range(logits.shape[0])
        ]
        out = torch.tensor(scores, device=logits.device)
        return out.mean() if mean else out

    return adherence


# Or some other metric that compares internal states
def make_internal_state_metric(
    internal_cache: Union[Dict[str, torch.Tensor], torch.Tensor],
    target_internal_cache: Union[Dict[str, torch.Tensor], torch.Tensor],
    mean: bool = True,
) -> MetricFn:
    def cosine_similarity(
        logits: torch.Tensor,
        clean_logits: Optional[torch.Tensor],
        input_lengths: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: compare internal states of the model and the target internal states
        pass

    return cosine_similarity
