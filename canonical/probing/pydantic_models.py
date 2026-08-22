from pydantic import BaseModel, Field

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


class SplitActivationDataset(
    BaseModel
):  # dataset for activations split into train, test and heldout
    pass


class IndexParquetColumns(BaseModel):
    str_id: str = "id"
    row_idx: str = "row_idx"
    rule_status: str = "rule_status"
