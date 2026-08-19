"""Tests for dataloaders.py.

test_build_neutral_dataloader_loads_real_sentences needs network access (it
downloads a small slice of GenericsKB from HuggingFace) but not a GPU, so it
doesn't belong under integration/'s CUDA-only skip guard. It's here to catch
schema drift/typos in NEUTRAL_TEXT_COLUMN -- exactly the bug that slipped
through originally (the column name differs across GenericsKB configs, and
guessing wrong only surfaces at dataloader-construction time).
"""

from canonical.causal.activation_patching.dataloaders import build_neutral_dataloader


def test_build_neutral_dataloader_loads_real_sentences():
    dataloader = build_neutral_dataloader(n_sentences=4, batch_size=4)

    batch = next(iter(dataloader))

    assert len(batch) == 4
    assert all(isinstance(sentence, str) and sentence.strip() for sentence in batch)
