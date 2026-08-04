# imports
# huggingface login
# load models
# set hyperparameters
# load data
# run forward pass for full_dataset_en
# save responses
# save activations layer and qkv
# upload to huggingface
#

import huggingface_hub
import torch
from transformers import AutoTokenizer
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities import utils
from pydantic import BaseModel, Field
from typing import List
from data.model import DatasetConfig
import pandas as pd
from data.model import CrossLingualRuleFollowingDataset


class ModelGenerationConfig(BaseModel):
    model_ids: List[str] = Field(..., description="")
    dataset_config: DatasetConfig = Field(..., description="")
    seeds: List[int] = Field(default_factory=List[0, 1, 2], description="")
    push_to_hf: bool = Field(default=False, description="")

    @property
    def device(self) -> str:
        return utils.get_device()

def collate_behavioral(
    batch: List[Dict[str, Any]],
) -> Tuple[List[str], List[Optional[str]], List[str]]:
    prompts = [r["system_rule"] + "\n" + r["user_query"] for r in batch]
    checkers = [r.get("checker") for r in batch]
    ids = [r["id"] for r in batch]
    return prompts, checkers, ids


class ModelRunner(BaseModel):
    def __init__(self, config: ModelGenerationConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self, model_id) -> None:
        print(f"Loading: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = TransformerBridge.boot_transformers(
            model_id, device=señf.config.device
        )
        self.model.enable_compatibility_mode(disable_warnings=True)

        n_layers = self.model.cfg.n_layers
        n_parameters = self.model.cfg.n_params
        n_heads = self.model.cfg.n_heads
        d_vocab = self.model.cfg.d_vocab
        architecture = self.model.cfg.architecture
        print(
            f"Model: {model_id} | {n_params:.1f}B params | {n_layers} layers | {n_heads} head | {d_vocab} vocabulary | {architecture} architecture"
        )

    def generate(self, pd: pd.DataFrame) -> None:
        assert self.model is not None, ValueError(
            "Initialize model by calling load() first"
        )


def run() -> None:
    try:
        dataset = CrossLingualRuleFollowingDataset(config.dataset_config)
        dataset_loader = dataset.to_dataloader(
            batch_size=config.batch_size, collate_fn=collate_EAP
        )