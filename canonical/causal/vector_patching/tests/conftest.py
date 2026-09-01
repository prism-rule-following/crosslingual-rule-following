import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_torch():
    torch.manual_seed(0)
