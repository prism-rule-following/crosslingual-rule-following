"""
Schema for the judgment-tier rule-following dataset.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Annotated, Dict, Literal, Optional, Union, List

from pydantic import BaseModel, Field, field_validator, model_validator


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
    rubric: LLMJudgeRubric = Field(
        description="Structured natural-language rubric for the judge."
    )


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
    language: str = Field(default="en", description="ISO language code.")

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
