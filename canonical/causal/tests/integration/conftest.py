"""Fixtures for integration tests against a real toy TransformerLens model.

Model: "attn-only-1l" -- the smallest model TransformerLens supports (1 layer,
512 d_model, ~1M params). It's one of Neel Nanda's toy induction-head models,
hosted on HuggingFace under his account, and is explicitly referred to as a
"toy model" throughout TransformerLens's own docs/demos. See:
https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html

These fixtures follow the same Graph.from_model -> attribute -> apply_topn
recipe used in EAP-IG's own example notebook (greater_than.ipynb), just scaled
down to a tiny induction task instead of the greater-than dataset.
"""

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

transformer_lens = pytest.importorskip(
    "transformer_lens", reason="transformer_lens is not installed"
)
pytest.importorskip("eap.graph", reason="eap (EAP-IG) is not installed")
pytest.importorskip("eap.attribute", reason="eap (EAP-IG) is not installed")

if not torch.cuda.is_available():
    # eap hardcodes device='cuda' in several places (attribute.py's scoring
    # functions, utils.compute_mean_activations) with no way to override it,
    # so these fixtures/tests target a real CUDA device intentionally (e.g. a
    # Colab GPU runtime) rather than working around it locally.
    pytest.skip("integration tests require a CUDA device", allow_module_level=True)

from canonical.causal.activation_patching.evaluation_metrics import logit_difference

TOY_MODEL_NAME = "attn-only-1l"

# (clean, corrupted, correct_word, wrong_word) -- clean repeats an earlier
# bigram, which is exactly what induction heads (attn-only-1l's specialty)
# pick up on; corrupted swaps the subject so there's nothing to induct from.
_INDUCTION_EXAMPLES = [
    ("The cat sat. The cat", "The cat sat. The dog", " sat", " ran"),
    ("My dog runs fast. My dog", "My dog runs fast. Her cat", " runs", " sat"),
]


def _single_token_id(model, word: str) -> int:
    ids = model.to_tokens(word, prepend_bos=False)[0].tolist()
    assert len(ids) == 1, (
        f"{word!r} is not a single token for {TOY_MODEL_NAME}'s tokenizer; "
        "pick a different probe word for _INDUCTION_EXAMPLES"
    )
    return ids[0]


class ToyInductionDataset(Dataset):
    """Tiny (clean, corrupted, label) dataset for the induction task above."""

    def __init__(self, model):
        self.model = model

    def __len__(self):
        return len(_INDUCTION_EXAMPLES)

    def __getitem__(self, index):
        clean, corrupted, correct_word, wrong_word = _INDUCTION_EXAMPLES[index]
        label = (
            _single_token_id(self.model, correct_word),
            _single_token_id(self.model, wrong_word),
        )
        return clean, corrupted, label


def _collate(batch):
    clean, corrupted, labels = zip(*batch)
    return list(clean), list(corrupted), torch.tensor(labels)


@pytest.fixture(scope="session")
def tiny_model():
    """The smallest model TransformerLens supports."""
    model = transformer_lens.HookedTransformer.from_pretrained(
        TOY_MODEL_NAME, device="cuda"
    )
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    return model


@pytest.fixture(scope="session")
def tiny_dataloader(tiny_model):
    return DataLoader(ToyInductionDataset(tiny_model), batch_size=2, collate_fn=_collate)


@pytest.fixture(scope="session")
def neutral_dataloader():
    """intervention_dataloader for intervention='mean': eap.utils.compute_mean_activations
    computes the mean-ablation baseline over whatever dataloader you give it.
    """
    from canonical.causal.activation_patching.dataloaders import build_neutral_dataloader

    return build_neutral_dataloader(n_sentences=32, batch_size=16)


@pytest.fixture(scope="session")
def tiny_metric():
    return partial(logit_difference, mean=True, loss=True)


@pytest.fixture(scope="session")
def tiny_graph(tiny_model):
    """A tiny circuit on the toy model's real Graph structure.

    This deliberately skips eap.attribute(): as of eap==2.0.0 every scoring
    method (EAP, EAP-IG-inputs, clean-corrupted, ...) hardcodes
    device='cuda' for its internal scores tensor (see eap/attribute.py,
    get_scores_eap/get_scores_eap_ig/etc.), so it raises "Torch not compiled
    with CUDA enabled" on a CPU-only machine no matter which method is
    picked -- that's an upstream limitation, not something in our code.
    Graph.from_model itself has no such hardcoding (its tensors are plain
    torch.zeros(...)), so we build a real Graph and hand-pick edges forming
    the "circuit" instead of running real attribution. That's sufficient for
    exercising CircuitVerifier's own logic, which doesn't care how the
    circuit was selected.

    The picked edges must form a genuinely *connected* input->...->logits
    path: graph.prune() (called by every verify_* method) drops any edge
    that isn't part of one, silently shrinking an arbitrary/disconnected
    edge selection down to zero edges -- picking every edge touching a
    single attention head (its full q/k/v input plus its output straight to
    logits) guarantees a path that survives pruning.
    """
    from eap.graph import Graph

    graph = Graph.from_model(tiny_model)
    edge_names = ["input->a0.h0<q>", "input->a0.h0<k>", "input->a0.h0<v>", "a0.h0->logits"]

    graph.in_graph[:] = False
    for name in edge_names:
        graph.edges[name].in_graph = True
    graph.prune()

    assert graph.in_graph.sum().item() == len(edge_names), (
        "hand-picked circuit didn't survive graph.prune() -- edges must form a "
        "connected input->...->logits path"
    )
    return graph
