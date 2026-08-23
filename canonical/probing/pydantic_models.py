from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# TODO ALL
words2labels = {  # pydantic?
    "active": 1,
    "cancelled": 0,
    "on": 1,
    "off": 0,
    "valid": 1,
    "invalid": 0,
    "true": 1,
    "false": 0,
    "enabled": 1,
    "disabled": 0,
}

labels2words = {  # pydantic?
    0: ["cancelled", "off", "invalid", "false", "disabled"],
    1: ["active", "on", "valid", "true", "enabled"],
}

distractor_words = [
    "Shopping",
    "Attendance",
    "Mood",
    "Art",
    "Politics",
    "Change",
    "Fund",
    "Fun",
    "Whatever",
    "Music",
    "Instrument",
    "Beauty",
    "Cosmetics",
    "Government",
    "Intelligence",
    "Education",
]


class SplitActivationDataset(
    BaseModel
):  # dataset for activations split into train, test and heldout
    model_config = ConfigDict(arbitrary_types_allowed=True)

    held_x: np.ndarray
    held_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    held_text: pd.DataFrame
    test_text: pd.DataFrame
    train_text: pd.DataFrame


class NeutralFillerDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    neutral_x: np.ndarray
    neutral_y: np.ndarray
    neutral_text: Dict


class DistractorDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    distractor_x: np.ndarray
    distractor_y: np.ndarray
    distractor_text: pd.DataFrame


class DoubleRuleDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    doublerule_x: np.ndarray
    doublerule_y: np.ndarray
    doublerule_text: pd.DataFrame


class IndexParquetColumns:
    str_id: str = "id"
    row_idx: str = "row_idx"
    rule_status: str = "rule_status"
    clean_id: str = "clean_id"


class CanonicalDatasetColumns:
    system_rule: str = "system_rule"
    # ...


class ShuffledLabelsResults(BaseModel):
    shuffled_eval_results: Dict[str, Dict[int, Dict]]
    shuffled_train_path: str
    shuffled_eval_path: str
