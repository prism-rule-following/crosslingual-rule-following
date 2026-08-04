import huggingface_hub
import torch
from data.model import DatasetConfig
import pandas as pd
from data.model import CrossLingualRuleFollowingDataset


def collate_behavioral(
    batch: List[Dict[str, Any]],
) -> Tuple[List[str], List[Optional[str]], List[str]]:
    prompts = [r["system_rule"] + "\n" + r["user_query"] for r in batch]
    checkers = [r.get("checker") for r in batch]
    ids = [r["id"] for r in batch]
    return prompts, checkers, ids


STATUS_LABELS = {
    "active_cancelled": ("active", "cancelled"),
    "on_off": ("on", "oof"),
    "true_false": ("true", "false"),
    "valid_invalid": ("valid", "invalid"),
    "enabled_disabled": ("enabled", "disabled"),
}


def separate_distinct_rows(input: pd.DataFrame) -> pd.DataFrame:
    input_dict = input.to_dict()


def main() -> None:
    dataset = CrossLingualRuleFollowingDataset(config.dataset_config)

    dataset_loader = dataset.to_dataloader(
        batch_size=config.batch_size, collate_fn=collate_EAP
    )
