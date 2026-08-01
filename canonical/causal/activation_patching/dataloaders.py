"""Dataloaders for activation-patching interventions.

Two things live here:
- CleanCorruptedCSVDataset / build_clean_corrupted_dataloader: the (clean,
  corrupted, label) evaluation dataloader CircuitVerifier is built around,
  loaded from a CSV, mirroring EAP-IG's own greater_than.ipynb EAPDataset.
- NeutralSentenceDataset / build_neutral_dataloader: a small, content-neutral
  sentence dataloader, meant to be handed to
  `evaluate_graph`/`CircuitVerifier.evaluate_with_edge_mask` as the
  `intervention_dataloader` whenever `intervention="mean"` -- eap's
  mean-ablation intervention computes mean activations over whatever
  dataloader you give it (see eap.utils.compute_mean_activations), so the
  choice of data determines the "neutral" baseline every mean-ablated edge
  gets replaced with.
"""

from typing import List

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class CleanCorruptedCSVDataset(Dataset):
    """(clean, corrupted, label) dataset loaded from a CSV with those column
    names, mirroring EAP-IG's own greater_than.ipynb EAPDataset."""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return row["clean"], row["corrupted"], row["label"]


def _collate_clean_corrupted(batch):
    clean, corrupted, labels = zip(*batch)
    return list(clean), list(corrupted), torch.tensor(labels)


def build_clean_corrupted_dataloader(csv_path: str, batch_size: int = 32) -> DataLoader:
    """A DataLoader over a CSV with clean/corrupted/label columns."""
    dataset = CleanCorruptedCSVDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_clean_corrupted)

# Smallest config of GenericsKB (~12.8k rows): plain, generic factual
# statements (e.g. "Dogs are mammals.") with no sentiment/opinion/narrative --
# about as content-neutral as naturally-occurring English sentences get.
NEUTRAL_DATASET_NAME = "community-datasets/generics_kb"
NEUTRAL_DATASET_CONFIG = "generics_kb_simplewiki"
# Confirmed against the real schema (this config's text column is 'sentence';
# other GenericsKB configs like generics_kb/generics_kb_best use
# 'generic_sentence' instead -- the two aren't interchangeable).
NEUTRAL_TEXT_COLUMN = "sentence"


class NeutralSentenceDataset(Dataset):
    """A small sample of neutral, generic sentences from GenericsKB."""

    def __init__(self, n_sentences: int = 64):
        rows = load_dataset(
            NEUTRAL_DATASET_NAME,
            NEUTRAL_DATASET_CONFIG,
            split=f"train[:{n_sentences}]",
        )
        self.sentences: List[str] = list(rows[NEUTRAL_TEXT_COLUMN])

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> str:
        return self.sentences[index]


def build_neutral_dataloader(n_sentences: int = 64, batch_size: int = 16) -> DataLoader:
    """A DataLoader of raw neutral-sentence strings.

    Batches are plain List[str], matching what
    eap.utils.compute_mean_activations expects when a dataloader doesn't yield
    (clean, corrupted, label) tuples: it uses `batch[0] if isinstance(batch,
    tuple) else batch` internally, so raw string batches work directly.
    """
    dataset = NeutralSentenceDataset(n_sentences=n_sentences)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=list)
