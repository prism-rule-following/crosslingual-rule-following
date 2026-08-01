"""Shared pytest fixtures for the causal-verification test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_torch():
    """Seed global torch RNG state before every test for reproducibility."""
    torch.manual_seed(0)


@pytest.fixture
def rng() -> torch.Generator:
    """A separately-seeded torch.Generator, for tests that need one explicitly
    (e.g. to compare two independent calls for reproducibility)."""
    generator = torch.Generator()
    generator.manual_seed(0)
    return generator
