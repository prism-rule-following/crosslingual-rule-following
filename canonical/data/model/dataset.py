# -*- coding: utf-8 -*-
"""Cross-lingual rule-following dataset utility."""

from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from pydantic import BaseModel, Field, ValidationError, field_validator
from torch.utils.data import DataLoader, Dataset

import json
import urllib.request


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class DatasetSource(StrEnum):
    hf = "hf"
    gh = "gh"


class DataCategories(StrEnum):
    ack_invert = "ack_invert"
    active_cancelled = "active_cancelled"
    banned_word = "banned_word"
    bold_html = "bold_html"
    directness = "directness"
    emotional_expressiveness = "emotional_expressiveness"
    humility = "humility"
    humor = "humor"
    include_word = "include_word"
    language = "language"
    second_word = "second_word"
    single_word = "single_word"
    start_with = "start_with"
    tone_provocation = "tone_provocation"
    word_count = "word_count"


class DatasetLanguageCode(StrEnum):
    en = "en"
    am = "am"
    de = "de"
    hi = "hi"
    ig = "ig"
    it = "it"
    ko = "ko"
    ru = "ru"
    sw = "sw"
    ta = "ta"
    tr = "tr"
    ur = "ur"
    yo = "yo"


class DatasetLanguageName(StrEnum):
    english = "English"
    amharic = "Amharic"
    german = "German"
    hindi = "Hindi"
    igbo = "Igbo"
    italian = "Italian"
    korean = "Korean"
    russian = "Russian"
    swahili = "Swahili"
    tamil = "Tamil"
    turkish = "Turkish"
    urdu = "Urdu"
    yoruba = "Yoruba"


LANGUAGE_NAMES: Dict[DatasetLanguageCode, DatasetLanguageName] = {
    DatasetLanguageCode.en: DatasetLanguageName.english,
    DatasetLanguageCode.am: DatasetLanguageName.amharic,
    DatasetLanguageCode.de: DatasetLanguageName.german,
    DatasetLanguageCode.hi: DatasetLanguageName.hindi,
    DatasetLanguageCode.ig: DatasetLanguageName.igbo,
    DatasetLanguageCode.it: DatasetLanguageName.italian,
    DatasetLanguageCode.ko: DatasetLanguageName.korean,
    DatasetLanguageCode.ru: DatasetLanguageName.russian,
    DatasetLanguageCode.sw: DatasetLanguageName.swahili,
    DatasetLanguageCode.ta: DatasetLanguageName.tamil,
    DatasetLanguageCode.tr: DatasetLanguageName.turkish,
    DatasetLanguageCode.ur: DatasetLanguageName.urdu,
    DatasetLanguageCode.yo: DatasetLanguageName.yoruba,
}


class DatasetTopic(StrEnum):
    general = "general"
    legal = "legal"
    finance = "finance"
    mental_health = "mental_health"
    medical = "medical"


class DatasetGrammar(StrEnum):
    imperative = "imperative"
    modal_obligation = "modal_obligation"
    polite_asking = "polite_asking"
    na = "n/a"  # banned_word and similar rows carry grammar_type "n/a"


class DatasetContrastivePair(StrEnum):
    active_cancelled = "active_cancelled"
    imperative_declarative = "imperative_declarative"  # present in data as pair_type
    on_off = "on_off"
    true_false = "true_false"
    valid_invalid = "valid_invalid"


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #
class RuleRow(BaseModel):
    """One dataset example. Unknown extra fields are allowed and preserved."""

    model_config = {"extra": "allow"}

    id: str
    category: DataCategories
    topic: str
    grammar_type: str
    language: DatasetLanguageCode
    context: str
    rule_text: str
    non_rule_text: str
    system_rule: str
    system_non_rule: str
    user_query: str
    checker: Optional[str] = None
    rule_clause: str
    correct_answer: Optional[str] = None
    pair_type: Optional[str] = None
    correct_keywords: List = []
    # Index fields are optional: only some categories can fill them.
    correct_idx: Optional[int] = None
    incorrect_idx: Optional[int] = None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class DatasetConfig(BaseModel):
    url: str = Field(..., description="HuggingFace dataset id or JSON file path/URL")
    source: DatasetSource = Field(..., description="dataset source")
    split: Optional[str] = Field(
        default=None,
        description="which split to use for HF DatasetDict; defaults to 'train' or first",
    )
    category: List[DataCategories] = Field(
        default_factory=lambda: list(DataCategories),
        description="keep only rows whose category is in this set",
    )
    languages: List[DatasetLanguageCode] = Field(
        default_factory=lambda: list(DatasetLanguageCode),
        description="keep only rows whose language is in this set",
    )
    grammar: List[DatasetGrammar] = Field(
        default_factory=lambda: list(DatasetGrammar),
        description="keep only rows whose grammar_type is in this set",
    )
    topic: List[DatasetTopic] = Field(
        default_factory=lambda: list(DatasetTopic),
        description="keep only rows whose topic is in this set",
    )
    contrastive_pair: List[DatasetContrastivePair] = Field(
        default_factory=lambda: list(DatasetContrastivePair),
        description="keep only rows whose pair_type is in this set",
    )
    validate_rows: bool = Field(
        default=True, description="run per-row pydantic validation on load"
    )
    strict: bool = Field(
        default=False,
        description="raise on the first invalid row instead of dropping it",
    )

    @field_validator("languages", mode="before")
    @classmethod
    def validate_language_code(cls, value: List[Any]) -> List[Any]:
        unsupported_languages = [x for x in value if not LANGUAGE_NAMES.get(x)]
        if unsupported_languages:
            raise ValueError(
                f"Unsupported languages included {unsupported_languages}. "
                "Update DatasetLanguageCode/LANGUAGE_NAMES to add support for this language."
            )
        return value


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_from_hf(url: str) -> Any:
    try:
        return load_dataset(url)
    except Exception:
        print(f"An error occurred. Unable to load {url!r} from HuggingFace.")
        raise


def load_from_github(url: str) -> Any:
    # Expects a raw content URL or local path to a .json / .jsonl file.
    try:
        if url.startswith("http://") or url.startswith("https://"):
            with urllib.request.urlopen(url) as resp:
                raw = json.load(resp)
        else:
            with open(url) as f:
                raw = json.load(f)
    except json.JSONDecodeError:
        try:
            return load_dataset("json", data_files=url)
        except Exception:
            print(f"An error occurred. Unable to load {url!r} as JSON.")
            raise
    except Exception:
        print(f"An error occurred. Unable to load {url!r} as JSON.")
        raise

    if isinstance(raw, dict):
        raw = raw.get("pairs", raw)
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    raise TypeError(f"Unsupported dataset object of type {type(raw)!r}")


def _to_dataframe(raw: Any, split: Optional[str]) -> pd.DataFrame:
    """Normalize whatever `load_dataset` returns into a pandas DataFrame."""
    # DatasetDict (has split keys) -> pick a split first.
    if isinstance(raw, dict):
        if split is not None:
            if split not in raw:
                raise KeyError(
                    f"split {split!r} not found; available: {list(raw.keys())}"
                )
            chosen = split
        else:
            chosen = "train" if "train" in raw else next(iter(raw.keys()))
        raw = raw[chosen]
    # HF Dataset -> DataFrame
    if hasattr(raw, "to_pandas"):
        return raw.to_pandas()
    # Already a DataFrame
    if isinstance(raw, pd.DataFrame):
        return raw

    raise TypeError(f"Unsupported dataset object of type {type(raw)!r}")


def _validate_rows(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    """Validate each row against RuleRow. Drop or raise on failure."""
    kept: List[Dict[str, Any]] = []
    errors: List[Tuple[str, ValidationError]] = []
    for i, row in enumerate(df.to_dict(orient="records")):
        # pandas fills missing keys with NaN (a float); convert to None so
        # Optional[...] fields validate instead of failing int/str coercion.
        row = {
            k: (None if (isinstance(v, float) and pd.isna(v)) else v)
            for k, v in row.items()
        }
        try:
            RuleRow(**row)
            kept.append(row)
        except ValidationError as e:
            errors.append((row.get("id", f"<index {i}>"), e))
            if strict:
                raise
    return pd.DataFrame(kept).reset_index(drop=True)


def _apply_filters(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Apply all subset filters. Missing columns are treated as 'no filter'."""

    def keep(col: str, allowed_values: List[str]) -> None:
        nonlocal df
        if col in df.columns:
            df = df[df[col].isin(allowed_values)]

    keep("category", [c.value for c in config.category])
    keep("language", [code.value for code in config.languages])
    keep("grammar_type", [g.value for g in config.grammar])
    keep("topic", [t.value for t in config.topic])
    keep("pair_type", [p.value for p in config.contrastive_pair])

    return df.reset_index(drop=True)


def dataset_generator(config: DatasetConfig) -> pd.DataFrame:
    """Load -> normalize -> (validate) -> filter. Returns a DataFrame."""
    raw = (
        load_from_hf(config.url)
        if config.source == DatasetSource.hf
        else load_from_github(config.url)
    )
    df = _to_dataframe(raw, config.split)
    if config.validate_rows:
        df = _validate_rows(df, config.strict)
    df = _apply_filters(df, config)
    return df


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CrossLingualRuleFollowingDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.df = dataset_generator(config)

    def __len__(self) -> int:
        return len(self.df)

    def shuffle(self, seed: Optional[int] = None) -> "CrossLingualRuleFollowingDataset":
        self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return self

    def head(self, n: int) -> "CrossLingualRuleFollowingDataset":
        self.df = self.df.head(n).reset_index(drop=True)
        return self

    def subset(self, **filters: Any) -> "CrossLingualRuleFollowingDataset":
        """Return a shallow copy filtered by column -> allowed value(s).

        Example: ds.subset(category=["start_with", "language"], topic="legal")
        """
        import copy

        df = self.df
        for col, val in filters.items():
            if col not in df.columns:
                continue
            allowed = val if isinstance(val, (list, set, tuple)) else [val]
            df = df[df[col].isin(list(allowed))]
        new = copy.copy(self)
        new.df = df.reset_index(drop=True)
        return new

    def build_indices(
        self,
        fn: Callable[[Dict[str, Any]], Optional[Tuple[int, int]]],
    ) -> "CrossLingualRuleFollowingDataset":
        """Populate correct_idx/incorrect_idx via a model-aware callback.

        `fn(row) -> (correct_idx, incorrect_idx) | None`. Rows where fn returns
        None (e.g. banned_word, which has no single-token contrast) are left
        with null indices. Keeps this util model-agnostic while giving a clean
        place to plug tokenization in.
        """
        correct, incorrect = [], []
        for row in self.df.to_dict(orient="records"):
            result = fn(row)
            if result is None:
                correct.append(None)
                incorrect.append(None)
            else:
                c, i = result
                correct.append(int(c))
                incorrect.append(int(i))
        self.df = self.df.assign(correct_idx=correct, incorrect_idx=incorrect)
        return self

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # neutral: hand back the whole row as a dict
        return self.df.iloc[index].to_dict()

    def to_dataloader(
        self, batch_size: int, collate_fn: Callable[..., Any], shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
        )
