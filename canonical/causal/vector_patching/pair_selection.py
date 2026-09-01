"""Stage 0: build the donor -> recipient minimal-pair table.

Loads the 3 judges' verdicts, collapses them into one per-(model, language,
id) verdict, computes per-language/category adherence rates, and emits the
donor->recipient pair table: same canonical `id`, donor held, recipient
didn't. Pure CPU/pandas.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

from canonical.causal.vector_patching.config import (
    HELD_VERDICT,
    HIGH_RESOURCE_HELD_RATE,
    JUDGE_RESULTS_REPO,
    JUDGES,
    LOW_RESOURCE_HELD_RATE,
    PRIMARY_PRESSURE_LEVEL,
)

# judge-results ids and model-inference-activations ids are both
# canonical_id + one of these suffixes
_ID_SUFFIXES = ("_clean", "_revoked")


def strip_id_suffix(row_id: str) -> str:
    for suffix in _ID_SUFFIXES:
        if row_id.endswith(suffix):
            return row_id[: -len(suffix)]
    return row_id


def load_judge_verdicts(
    judges: List[str] = JUDGES, repo_id: str = JUDGE_RESULTS_REPO
) -> pd.DataFrame:
    """Downloads and concatenates every judge's results.jsonl. Rows with a
    null verdict (judge failure / content-filter block) are dropped, not
    counted as failures."""
    frames = []
    for judge in judges:
        path = hf_hub_download(repo_id, f"{judge}/results.jsonl", repo_type="dataset")
        df = pd.read_json(path, lines=True)
        df["judge"] = judge
        frames.append(df)
    verdicts = pd.concat(frames, ignore_index=True)
    verdicts = verdicts[verdicts["verdict"].notna()].copy()
    verdicts["canonical_id"] = verdicts["id"].map(strip_id_suffix)
    verdicts["held"] = verdicts["verdict"] == HELD_VERDICT
    return verdicts


def log_distinct_verdicts(verdicts: pd.DataFrame) -> Dict[str, int]:
    return dict(Counter(verdicts["verdict"]))


def _majority(values: pd.Series) -> Tuple[bool, bool]:
    """Majority vote; returns (verdict, low_confidence). Ties break to
    False (not held) and set low_confidence."""
    counts = values.value_counts()
    if len(counts) == 1:
        return bool(counts.index[0]), False
    if counts.iloc[0] == counts.iloc[1]:
        return False, True
    return bool(counts.index[0]), False


def collapse_verdicts(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Majority of 3 samples per judge, then majority across judges present
    for that row. One row per (model_id, language, canonical_id, category,
    pressure_level): held, low_confidence, n_judges."""
    per_judge = (
        verdicts.groupby(
            ["model_id", "language", "canonical_id", "category", "pressure_level", "judge"]
        )["held"]
        .apply(lambda s: _majority(s)[0])
        .reset_index()
    )

    def _collapse_judges(group: pd.DataFrame) -> pd.Series:
        held, low_conf = _majority(group["held"])
        return pd.Series({"held": held, "low_confidence": low_conf, "n_judges": len(group)})

    collapsed = (
        per_judge.groupby(["model_id", "language", "canonical_id", "category", "pressure_level"])
        .apply(_collapse_judges, include_groups=False)
        .reset_index()
    )
    return collapsed


def compute_adherence_rates(
    collapsed: pd.DataFrame,
    model_id: str,
    pressure_level: str = PRIMARY_PRESSURE_LEVEL,
) -> pd.DataFrame:
    """Per-(language, category) and overall-per-language HELD rate."""
    subset = collapsed[
        (collapsed["model_id"] == model_id) & (collapsed["pressure_level"] == pressure_level)
    ]
    per_category = (
        subset.groupby(["language", "category"])["held"].mean().reset_index(name="held_rate")
    )
    overall = (
        subset.groupby("language")["held"].mean().reset_index(name="held_rate")
    )
    overall["category"] = "__overall__"
    return pd.concat([per_category, overall], ignore_index=True)


def classify_tiers(
    rates: pd.DataFrame,
    high_thresh: float = HIGH_RESOURCE_HELD_RATE,
    low_thresh: float = LOW_RESOURCE_HELD_RATE,
) -> Dict[str, str]:
    """Language -> "high" / "mid" / "low", from the overall held rate.
    """
    overall = rates[rates["category"] == "__overall__"].set_index("language")["held_rate"]
    tiers = {}
    for lang, rate in overall.items():
        if rate >= high_thresh:
            tiers[lang] = "high"
        elif rate <= low_thresh:
            tiers[lang] = "low"
        else:
            tiers[lang] = "mid"
    return tiers


def build_pair_table(
    collapsed: pd.DataFrame,
    model_id: str,
    pressure_level: str = PRIMARY_PRESSURE_LEVEL,
    donor_languages: Optional[List[str]] = None,
    recipient_languages: Optional[List[str]] = None,
) -> pd.DataFrame:
    """The donor -> recipient minimal-pair table Exp 2 patches against.
    """
    subset = collapsed[
        (collapsed["model_id"] == model_id) & (collapsed["pressure_level"] == pressure_level)
    ]
    donors = subset[subset["held"]]
    recipients = subset[~subset["held"]]
    if donor_languages is not None:
        donors = donors[donors["language"].isin(donor_languages)]
    if recipient_languages is not None:
        recipients = recipients[recipients["language"].isin(recipient_languages)]

    pairs = donors.merge(
        recipients,
        on="canonical_id",
        suffixes=("_donor", "_recipient"),
    )
    pairs = pairs[pairs["language_donor"] != pairs["language_recipient"]]
    return pairs[
        [
            "canonical_id",
            "language_donor",
            "language_recipient",
            "category_donor",
            "low_confidence_donor",
            "low_confidence_recipient",
        ]
    ].rename(columns={"category_donor": "category"})
