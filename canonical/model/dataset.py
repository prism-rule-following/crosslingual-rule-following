"""
Schema for the judgment-tier rule-following dataset.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from collections import Counter
from enum import Enum
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset as TorchDataset


# --------------------------------------------------------------------------- #
# StrEnums (requirement 1)
# --------------------------------------------------------------------------- #
class StrEnum(str, Enum):
    """3.10-compatible StrEnum: serializes as its string value."""

    def __str__(self) -> str:  # pragma: no cover
        return self.value


class Category(StrEnum):
    ack_invert = "ack_invert"
    mandatory_referral = "mandatory_referral"
    no_dosage = "no_dosage"
    no_pii = "no_pii"
    no_verdict = "no_verdict"
    refuse_with_reason = "refuse_with_reason"
    scope_lock = "scope_lock"


class Topic(StrEnum):
    medical = "medical"
    mental_health = "mental_health"
    legal = "legal"
    finance = "finance"


class PairType(StrEnum):
    active_cancelled = "active_cancelled"
    on_off = "on_off"
    true_false = "true_false"
    valid_invalid = "valid_invalid"
    enabled_disabled = "enabled_disabled"


class PressureLevel(StrEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


class PressureName(StrEnum):
    neutral = "neutral"
    social_pressure = "social_pressure"
    authority_framing = "authority_framing"
    direct_override = "direct_override"
    emotional_appeal = "emotional_appeal"
    incremental_erosion = "incremental_erosion"


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


class CheckerTier(StrEnum):
    judgment = "judgment"


class GrammarType(StrEnum):
    """Governs the phrasing of the RULE clause in the system prompt.
    Orthogonal to PressureLevel, which governs the phrasing of the USER turn:
    the two axes modify different spans and never collide."""

    imperative = "imperative"  # "Never give a specific dosage."
    modal_obligation = "modal_obligation"  # "You must never give a specific dosage."
    polite_asking = (
        "polite_asking"  # "Please make sure you never give a specific dosage."
    )


class RuleStatus(StrEnum):
    """The token that appears after 'Rule status:' in each half of the pair."""

    # active-family (rule binds)
    active = "active"
    on = "on"
    true = "true"
    valid = "valid"
    enabled = "enabled"
    # revoked-family (rule does NOT bind)
    cancelled = "cancelled"
    off = "off"
    false = "false"
    invalid = "invalid"
    disabled = "disabled"


class CheckerType(StrEnum):
    llm_judge = "llm_judge"
    deterministic_function = "deterministic_function"


# Which status token is "active" vs "revoked" for each pair_type.
PAIR_STATUS = {
    PairType.active_cancelled: (RuleStatus.active, RuleStatus.cancelled),
    PairType.on_off: (RuleStatus.on, RuleStatus.off),
    PairType.true_false: (RuleStatus.true, RuleStatus.false),
    PairType.valid_invalid: (RuleStatus.valid, RuleStatus.invalid),
    PairType.enabled_disabled: (RuleStatus.enabled, RuleStatus.disabled),
}

ID_FORMAT = "rb_j_{category}_{topic}_{active_status}_{grammar_type}_{pressure_level}"


def build_id(
    category: str,
    topic: str,
    active_status: str,
    grammar_type: str,
    pressure_level: str,
) -> str:
    """Canonical row id, e.g. rb_j_ack_invert_medical_true_imperative_L5.

    A row's (category, topic, pressure_level) cell holds one row per
    pair_type; active_status is the single-token stand-in for pair_type
    that disambiguates them. grammar_type disambiguates the imperative /
    modal_obligation / polite_asking triple derived from the same source row.
    """
    return ID_FORMAT.format(
        category=category,
        topic=topic,
        active_status=active_status,
        grammar_type=grammar_type,
        pressure_level=pressure_level,
    )


# --------------------------------------------------------------------------- #
# Rubrics — one shape per checker_type (requirement 2)
# --------------------------------------------------------------------------- #
class LLMJudgeRubric(BaseModel):
    """Rubric handed to an LLM judge. Natural-language criteria."""

    held_if: str = Field(
        description="Condition under which the rule is judged to HOLD."
    )
    violated_if: str = Field(
        description="Condition under which the rule is judged VIOLATED."
    )
    ignore: str = Field(description="Behaviors the judge must NOT count as violations.")
    culture_invariant: bool = Field(
        default=True,
        description="If True, judge on a named event, not local-norm calibration.",
    )
    point_at: str = Field(
        description="What span the judge should point at as evidence."
    )

    @field_validator("held_if", "violated_if", "ignore", "point_at")
    @classmethod
    def _non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v.strip()


class DeterministicRubric(BaseModel):
    """Rubric for a pure-function checker. Machine-checkable spec."""

    function: str = Field(
        description="Name of the checker function to dispatch to, e.g. "
        "'check_yes_no_polarity'."
    )
    expected: str = Field(
        description="Expected canonical output for a PASS, e.g. 'no'. Must be "
        "a member of `labels`."
    )
    labels: List[str] = Field(
        description="Closed label set the answer is drawn from, e.g. " "['yes','no']."
    )
    truth_label: Optional[str] = Field(
        default=None, description="Ground-truth answer to the underlying question."
    )
    inverted_label: Optional[str] = Field(
        default=None,
        description="The inverted (rule-required) answer, opposite of truth.",
    )
    match: Literal["exact", "polarity", "substring"] = Field(
        default="polarity", description="How the reply is compared to `expected`."
    )

    @field_validator("function", "expected")
    @classmethod
    def _non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v.strip()

    @field_validator("labels")
    @classmethod
    def _labels_non_empty_unique(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("labels must contain at least two entries")
        if len(set(v)) != len(v):
            raise ValueError("labels must be unique")
        return v

    @model_validator(mode="after")
    def _labels_consistent(self) -> "DeterministicRubric":
        if self.expected not in self.labels:
            raise ValueError(f"expected '{self.expected}' not in labels {self.labels}")
        for name in ("truth_label", "inverted_label"):
            val = getattr(self, name)
            if val is not None and val not in self.labels:
                raise ValueError(f"{name} '{val}' not in labels {self.labels}")
        if (
            self.truth_label is not None
            and self.inverted_label is not None
            and self.truth_label == self.inverted_label
        ):
            raise ValueError("truth_label and inverted_label must differ")
        return self


# --------------------------------------------------------------------------- #
# Checker — tagged union on checker_type (requirement 2)
# --------------------------------------------------------------------------- #
ACTIVE_STATUSES = frozenset(
    {
        RuleStatus.active,
        RuleStatus.on,
        RuleStatus.true,
        RuleStatus.valid,
        RuleStatus.enabled,
    }
)
REVOKED_STATUSES = frozenset(
    {
        RuleStatus.cancelled,
        RuleStatus.off,
        RuleStatus.false,
        RuleStatus.invalid,
        RuleStatus.disabled,
    }
)


class _CheckerBase(BaseModel):
    rule_status: RuleStatus = Field(
        description="Which side (status token) this checker evaluates."
    )
    binds: bool = Field(
        description="True on the active side (rule must be obeyed), "
        "False on the revoked side (rule lifted)."
    )
    violation_event: str = Field(
        description="Discrete, nameable event that constitutes a violation."
    )

    @field_validator("violation_event")
    @classmethod
    def _event_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("violation_event must be a non-empty string")
        return v.strip()

    @model_validator(mode="after")
    def _binds_matches_status(self) -> "_CheckerBase":
        if self.binds and self.rule_status not in ACTIVE_STATUSES:
            raise ValueError(
                f"binds=True requires an active-family status, got "
                f"'{self.rule_status}'"
            )
        if not self.binds and self.rule_status not in REVOKED_STATUSES:
            raise ValueError(
                f"binds=False requires a revoked-family status, got "
                f"'{self.rule_status}'"
            )
        return self


class LLMJudgeChecker(_CheckerBase):
    checker_type: Literal[CheckerType.llm_judge] = Field(
        default=CheckerType.llm_judge, description="Discriminator: LLM-judged checker."
    )
    instruction: str = Field(
        description="Flat prompt string handed to the judge model."
    )
    rubric: LLMJudgeRubric = Field(
        description="Structured natural-language rubric for the judge."
    )

    @field_validator("instruction")
    @classmethod
    def _instruction_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("instruction must be a non-empty string")
        return v.strip()


class DeterministicChecker(_CheckerBase):
    checker_type: Literal[CheckerType.deterministic_function] = Field(
        default=CheckerType.deterministic_function,
        description="Discriminator: pure-function checker.",
    )
    rubric: DeterministicRubric = Field(
        description="Machine-checkable rubric (function + labels + expected)."
    )


Checker = Annotated[
    Union[LLMJudgeChecker, DeterministicChecker],
    Field(discriminator="checker_type"),
]


# --------------------------------------------------------------------------- #
# Per-row provenance
# --------------------------------------------------------------------------- #
class GenerationMethod(StrEnum):
    seed = "seed"  # original hand-authored / migrated row
    grammar_transform = "grammar_transform"  # re-registered from a seed row


class GenerationMetadata(BaseModel):
    """Provenance so 'which rows are synthetic?' is answerable per row."""

    method: GenerationMethod = Field(description="How the row was produced.")
    source_id: Optional[str] = Field(
        default=None,
        description="Seed example id this row was derived from (None for seeds).",
    )
    target_grammar: Optional[str] = Field(
        default=None,
        description="Grammar the source was re-registered into (transforms only).",
    )
    gen_model: Optional[str] = Field(
        default=None, description="Generator model slug (None for seeds)."
    )
    verify_model: Optional[str] = Field(
        default=None, description="Second-model verifier slug (None for seeds)."
    )
    prompt_version: Optional[str] = Field(
        default=None, description="Prompt template version tag."
    )
    verifier_verdict: Optional[dict] = Field(
        default=None, description="Second-model equivalence verdict payload."
    )
    gate_passed: bool = Field(
        default=True, description="Whether the row passed the quality gate."
    )
    ts: Optional[str] = Field(
        default=None, description="ISO-8601 UTC creation timestamp."
    )

    @model_validator(mode="after")
    def _transform_needs_source(self) -> "GenerationMetadata":
        if self.method == GenerationMethod.grammar_transform:
            if not self.source_id or not self.target_grammar:
                raise ValueError(
                    "grammar_transform rows require source_id and target_grammar"
                )
        return self


# --------------------------------------------------------------------------- #
# The dataset row
# --------------------------------------------------------------------------- #
class RulePair(BaseModel):
    id: str = Field(description="Stable unique id for the pair.")
    category: Category = Field(description="Rule category / checker family.")
    topic: Topic = Field(description="Domain the scenario is set in.")
    grammar_type: GrammarType = Field(
        default=GrammarType.imperative,
        description="Phrasing of the RULE clause (imperative / modal / polite).",
    )
    pressure_level: PressureLevel = Field(
        description="Adversarial pressure applied to the USER turn (L0–L5)."
    )
    pressure_name: PressureName = Field(
        description="Human-readable name of the pressure level."
    )
    checker_tier: CheckerTier = Field(
        default=CheckerTier.judgment,
        description="Evaluation tier; judgment = discrete nameable event.",
    )
    language: DatasetLanguageCode = Field(
        default=DatasetLanguageCode.en, description="ISO language code."
    )

    context: str = Field(description="Base system-prompt preamble.")
    rule_clause: str = Field(description="The bare rule clause, grammar-neutral core.")
    pair_type: PairType = Field(
        description="ACTIVE/REVOKED status-token pair used for the contrast."
    )

    active_status: RuleStatus = Field(
        description="Status token on the active side (rule binds)."
    )
    revoked_status: RuleStatus = Field(
        description="Status token on the revoked side (rule lifted)."
    )

    rule_text: str = Field(description="Rule clause + 'Rule status: <active>.'")
    non_rule_text: str = Field(description="Rule clause + 'Rule status: <revoked>.'")
    system_rule: str = Field(description="Full system prompt, active side.")
    system_non_rule: str = Field(description="Full system prompt, revoked side.")

    user_query: str = Field(description="Final user turn (the one that is scored).")
    user_turns: Optional[List[str]] = Field(
        default=None,
        description="Multi-turn build-up; required for L5, absent otherwise. "
        "Last element should equal user_query.",
    )

    correct_answer: str = Field(
        description="Reference compliant answer for the active side."
    )
    correct_keywords: List[str] = Field(
        description="Keywords expected in a compliant answer."
    )

    active_checker: Checker = Field(
        description="Checker for the active side (binds=True)."
    )
    revoked_checker: Checker = Field(
        description="Checker for the revoked side (binds=False)."
    )

    class Config:
        use_enum_values = True

    # -- cross-field validators ------------------------------------------- #
    @field_validator("correct_keywords")
    @classmethod
    def _keywords_non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("correct_keywords must not be empty")
        return v

    @model_validator(mode="after")
    def _status_pair_matches_pair_type(self) -> "RulePair":
        expected = PAIR_STATUS[PairType(self.pair_type)]
        got = (RuleStatus(self.active_status), RuleStatus(self.revoked_status))
        if got != expected:
            raise ValueError(
                f"(active_status, revoked_status)={got} does not match "
                f"pair_type '{self.pair_type}' (expected {expected})"
            )
        return self

    @model_validator(mode="after")
    def _l5_requires_turns(self) -> "RulePair":
        is_l5 = PressureLevel(self.pressure_level) == PressureLevel.L5
        if is_l5:
            if not self.user_turns or len(self.user_turns) < 2:
                raise ValueError("L5 requires user_turns with >= 2 turns")
            if self.user_turns[-1] != self.user_query:
                raise ValueError("last user_turn must equal user_query")
        elif self.user_turns is not None:
            raise ValueError("user_turns must be set only for L5")
        return self

    @model_validator(mode="after")
    def _checkers_paired(self) -> "RulePair":
        if not self.active_checker.binds:
            raise ValueError("active_checker must have binds=True")
        if self.revoked_checker.binds:
            raise ValueError("revoked_checker must have binds=False")
        if RuleStatus(self.active_checker.rule_status) != RuleStatus(
            self.active_status
        ):
            raise ValueError("active_checker.rule_status must equal active_status")
        if RuleStatus(self.revoked_checker.rule_status) != RuleStatus(
            self.revoked_status
        ):
            raise ValueError("revoked_checker.rule_status must equal revoked_status")
        # both checkers should be the same type within a category
        if self.active_checker.checker_type != self.revoked_checker.checker_type:
            raise ValueError("active and revoked checkers must share a checker_type")
        return self


class GenerationRecord(BaseModel):
    """Dataset-level provenance. Replaces per-row metadata: every row is
    synthetic, and transform variants are identifiable by the grammar_type
    segment embedded in their id (see ID_FORMAT), so origin need only be
    recorded once here."""

    all_synthetic: bool = Field(
        default=True,
        description="Whether every row in the dataset is synthetically produced.",
    )
    seed_method: Optional[str] = Field(
        default=None, description="How the imperative seed rows were produced."
    )
    transform_method: str = Field(
        default="grammar_transform",
        description="How non-imperative variants were derived from seeds.",
    )
    gen_model: Optional[str] = Field(
        default=None, description="Generator model slug for transforms."
    )
    prompt_version: Optional[str] = Field(
        default=None, description="Transform prompt template version."
    )
    id_format: str = Field(
        default=ID_FORMAT,
        description="Template used to build every row id; grammar_type is a "
        "plain segment in it, so a variant's grammar can be read off its id.",
    )
    ts: Optional[str] = Field(
        default=None, description="ISO-8601 UTC generation timestamp."
    )


class RowCounts(BaseModel):
    """Row-count breakdown by each single-field axis, so coverage/balance is
    readable straight from the file's own metadata, not just a side-report."""

    per_category: Dict[str, int] = Field(default_factory=dict)
    per_topic: Dict[str, int] = Field(default_factory=dict)
    per_grammar_type: Dict[str, int] = Field(default_factory=dict)
    per_pressure: Dict[str, int] = Field(default_factory=dict)
    per_rule_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts keyed by pair_type (active_cancelled, on_off, "
        "true_false, valid_invalid, enabled_disabled) — the axis that "
        "determines which 'Rule status: <token>' vocabulary a row uses.",
    )


def compute_row_counts(rows: List[dict]) -> RowCounts:
    """Build a RowCounts breakdown from a list of row dicts (pre- or
    post-validation — only plain field lookups, no RulePair required)."""

    def dist(key: str) -> Dict[str, int]:
        return dict(Counter(r[key] for r in rows))

    return RowCounts(
        per_category=dist("category"),
        per_topic=dist("topic"),
        per_grammar_type=dist("grammar_type"),
        per_pressure=dist("pressure_level"),
        per_rule_status=dist("pair_type"),
    )


class Metadata(BaseModel):
    draft: bool = Field(default=True, description="Draft flag.")
    total: int = Field(description="Number of pairs in the dataset.")
    checker_tier: CheckerTier = Field(
        default=CheckerTier.judgment, description="Evaluation tier."
    )
    note: str = Field(description="Free-form design note.")
    schema_version: str = Field(
        default="2.2-dataset-provenance", description="Schema version tag."
    )
    generation: Optional[GenerationRecord] = Field(
        default=None, description="Dataset-level provenance record (not per-row)."
    )
    counts: Optional[RowCounts] = Field(
        default=None,
        description="Row-count breakdown by category/topic/"
        "grammar_type/pressure/rule_status.",
    )

    @model_validator(mode="after")
    def _total_non_negative(self) -> "Metadata":
        if self.total < 0:
            raise ValueError("total must be non-negative")
        return self


def carry_metadata(source: dict, **overrides) -> "Metadata":
    """Build a new Metadata that carries over the SOURCE file's own metadata
    (draft, checker_tier, schema_version, note) instead of silently falling
    back to field defaults, then applies overrides (total, note, generation,
    counts, ...) on top. Every generate/verify output should build its
    Metadata through this, so metadata survives the whole pipeline instead
    of being reconstructed from scratch at each stage."""
    base = {
        "draft": source.get("draft", True),
        "checker_tier": source.get("checker_tier", CheckerTier.judgment),
        "note": source.get("note", ""),
        "schema_version": source.get("schema_version", "2.2-dataset-provenance"),
    }
    base.update(overrides)
    return Metadata(**base)


class Dataset(BaseModel):
    metadata: Metadata = Field(description="Dataset-level metadata.")
    pairs: List[RulePair] = Field(description="All rule pairs.")

    @model_validator(mode="after")
    def _total_matches(self) -> "Dataset":
        if self.metadata.total != len(self.pairs):
            raise ValueError(
                f"metadata.total ({self.metadata.total}) != "
                f"len(pairs) ({len(self.pairs)})"
            )
        return self


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
    categories: List[Category] = Field(
        default_factory=lambda: list(Category),
        description="keep only rows whose categories is in this set",
    )
    languages: List[DatasetLanguageCode] = Field(
        default_factory=lambda: list(DatasetLanguageCode),
        description="keep only rows whose languages is in this set",
    )
    grammars: List[GrammarType] = Field(
        default_factory=lambda: list(GrammarType),
        description="keep only rows whose grammar_types is in this set",
    )
    topics: List[Topic] = Field(
        default_factory=lambda: list(Topic),
        description="keep only rows whose topics is in this set",
    )
    pressure_levels: List[PressureLevel] = Field(
        default_factory=lambda: list(PressureLevel),
        description="keep only rows whose pressure_level is in this set",
    )
    pressure_names: List[PressureName] = Field(
        default_factory=lambda: list(PressureName),
        description="keep only rows whose pressure_name is in this set",
    )
    contrastive_pairs: List[PairType] = Field(
        default_factory=lambda: list(PairType),
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


def split_constrast_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        base = row.to_dict()
        clean_label, corrupt_label = row["active_status"], row["revoked_status"]
        constant_fields = {
            k: v
            for k, v in base.items()
            if k
            not in (
                "system_rule",
                "system_non_rule",
                "rule_text",
                "non_rule_text",
                "active_checker",
                "revoked_checker",
            )
        }

        rows.append(
            {
                **constant_fields,
                "id": f"{base['id']}_clean",
                "system": base["system_rule"],
                "rule_status": clean_label,
                "checker": base["active_checker"],
            }
        )

        rows.append(
            {
                **constant_fields,
                "id": f"{base['id']}_revoked",
                "system": base["system_non_rule"],
                "rule_status": corrupt_label,
                "checker": base["revoked_checker"],
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


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


def nan_to_none(row: Dict[str, Any]) -> Dict[str, Any]:
    """Pandas fills missing values with NaN (a float), which fails validation
    for Optional[...] fields expecting None. Normalize row dicts pulled from
    a DataFrame (e.g. via to_dict(orient="records")) before passing them to
    any pydantic model - RulePair here, but also e.g. ModelResponse elsewhere.
    """
    return {
        k: (None if (isinstance(v, float) and pd.isna(v)) else v)
        for k, v in row.items()
    }


def _validate_rows(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    """Validate each row against RulePair. Drop or raise on failure."""
    kept: List[Dict[str, Any]] = []
    errors: List[Tuple[str, ValidationError]] = []
    for i, row in enumerate(df.to_dict(orient="records")):
        row = nan_to_none(row)
        try:
            RulePair(**row)
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
            df = df[df[col].isin(allowed_values) | df[col].isna()]

    keep("category", [c.value for c in config.categories])
    keep("language", [code.value for code in config.languages])
    keep("grammar_type", [g.value for g in config.grammars])
    keep("topic", [t.value for t in config.topics])
    keep("pair_type", [p.value for p in config.contrastive_pairs])
    keep("pressure_level", [p.value for p in config.pressure_levels])
    keep("pressure_name", [p.value for p in config.pressure_names])

    return df.reset_index(drop=True)


def dataset_generator(config: DatasetConfig) -> pd.DataFrame:
    """Load -> normalize -> (validate) -> filter. Returns a DataFrame."""
    raw = (
        HFDataHelper.load_source_dataset(config.url)
        if config.source == DatasetSource.hf
        else load_from_github(config.url)
    )
    df = _to_dataframe(raw, config.split)
    if config.validate_rows:
        df = _validate_rows(df, config.strict)
    df = _apply_filters(df, config)
    return split_constrast_pairs(df)


def collate_behavioral(
    batch: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
    """Behavioral eval only scores the active-rule half of each pair - drop
    revoked-side rows (rule_status not in ACTIVE_STATUSES) before collating.
    Each surviving row's `checker` is its active_checker (see
    split_constrast_pairs), so callers get the right checker for free."""
    active_rows = [r for r in batch if r["rule_status"] in ACTIVE_STATUSES]
    system = [r["system"] for r in active_rows]
    user_query = [r["user_query"] for r in active_rows]
    ids = [r["id"] for r in active_rows]
    checkers = [r["checker"] for r in active_rows]
    return system, user_query, ids, checkers


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CrossLingualRuleFollowingDataset(TorchDataset):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.df = dataset_generator(config)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CrossLingualRuleFollowingDataset":
        """Wrap an already-loaded DataFrame directly, skipping fetch/validate/filter.

        For data that's already in the right shape - e.g. model outputs pulled
        back from HFDataHelper.fetch() - rather than the raw source dataset.
        """
        obj = cls.__new__(cls)
        obj.config = None
        obj.df = df.reset_index(drop=True)
        return obj

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


class HFDataHelper:
    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        self.token = os.environ.get("HF_TOKEN")
        self._api = HfApi()

    @staticmethod
    def load_source_dataset(repo_id: str) -> Any:
        try:
            return load_dataset(repo_id)
        except Exception:
            print(f"An error occurred. Unable to load {repo_id!r} from HuggingFace.")
            raise

    def _ensure_repo(self) -> None:
        self._api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token,
            exist_ok=True,
        )

    def _model_slug(self, model_id: str) -> str:
        return model_id.replace("/", "__")

    def _hf_path(self, model_id: str, lang_code: DatasetLanguageCode) -> str:
        return f"{self._model_slug(model_id)}/{lang_code.value}.parquet"

    def upload(
        self, df: pd.DataFrame, model_id: str, lang_code: DatasetLanguageCode
    ) -> None:
        self._ensure_repo()
        hf_path = self._hf_path(model_id, lang_code)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            self._api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=hf_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
            )
        os.unlink(tmp.name)

    def upload_file(self, path_or_fileobj: Any, path_in_repo: str) -> None:
        """Upload an arbitrary local file to a path inside this repo (used for
        activation .npy arrays and the index.parquet)."""
        self._ensure_repo()
        self._api.upload_file(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token,
        )

    def exists(self, model_id: str, lang_code: DatasetLanguageCode) -> bool:
        return self.exists_path(self._hf_path(model_id, lang_code))

    def exists_path(self, path_in_repo: str) -> bool:
        """Whether a specific path already exists in the repo (used to skip
        already-uploaded results)."""
        try:
            info = self._api.get_paths_info(
                repo_id=self.repo_id,
                repo_type="dataset",
                paths=[path_in_repo],
                token=self.token,
            )
            return len(info) > 0
        except Exception:
            return False

    def fetch(self, model_id: str, lang_code: DatasetLanguageCode) -> pd.DataFrame:
        """Download a previously-uploaded output file back into a DataFrame."""
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=self._hf_path(model_id, lang_code),
            token=self.token,
        )
        return pd.read_parquet(local_path)
