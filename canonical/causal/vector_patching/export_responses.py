"""Packages Stage B's patched/steered generations into the schema judging
will consume later, and uploads to HF.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from huggingface_hub import HfApi

from canonical.causal.vector_patching.config import EXPORT_REPO

# Columns shared with judge-results-active-only's own schema, so a judging
# script written for that dataset needs minimal changes to run on this one.
_SHARED_COLUMNS = [
    "id", "model_id", "language", "category", "topic", "grammar_type",
    "pressure_level", "pair_type", "sample_idx", "rule_clause", "user_query",
    "response",
]
# New columns specific to this dataset -- never overload a shared name.
_INTERVENTION_COLUMNS = [
    "donor_language", "patch_layer", "vector_type", "donor_kind", "patch_mode",
    "alpha", "recipient_pre_verdict", "feasibility_cohens_d",
    "still_target_language", "non_degenerate",
]


def build_response_row(
    *,
    canonical_id: str,
    model_id: str,
    language: str,
    category: str,
    topic: str,
    grammar_type: str,
    pressure_level: str,
    pair_type: str,
    sample_idx: int,
    rule_clause: str,
    user_query: str,
    response: str,
    donor_language: str,
    patch_layer: int,
    vector_type: str,
    donor_kind: str,
    patch_mode: str,
    recipient_pre_verdict: bool,
    feasibility_cohens_d: Optional[float] = None,
    alpha: Optional[float] = None,
    still_target_language: Optional[bool] = None,
    non_degenerate: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "id": canonical_id,
        "model_id": model_id,
        "language": language,
        "category": category,
        "topic": topic,
        "grammar_type": grammar_type,
        "pressure_level": pressure_level,
        "pair_type": pair_type,
        "sample_idx": sample_idx,
        "rule_clause": rule_clause,
        "user_query": user_query,
        "response": response,
        "donor_language": donor_language,
        "patch_layer": patch_layer,
        "vector_type": vector_type,
        "donor_kind": donor_kind,
        "patch_mode": patch_mode,
        "alpha": alpha,
        "recipient_pre_verdict": recipient_pre_verdict,
        "feasibility_cohens_d": feasibility_cohens_d,
        "still_target_language": still_target_language,
        "non_degenerate": non_degenerate,
    }


# langdetect's profile set -- fixed at ~55 languages.
LANGDETECT_SUPPORTED_LANGS = {
    "sl", "sk", "ur", "sw", "pl", "vi", "sq", "sv", "he", "da", "mr", "no",
    "gu", "ja", "el", "lv", "it", "ca", "cs", "te", "ru", "tl", "ro",
    "zh-cn", "so", "pt", "uk", "pa", "ml", "mk", "kn", "zh-tw", "ar", "hr",
    "hu", "nl", "bg", "bn", "ne", "af", "hi", "de", "ko", "fi", "id", "fr",
    "es", "et", "en", "fa", "lt", "cy", "ta", "th", "tr",
}


def sanity_check_response(response: str, expected_language: str) -> Dict[str, Optional[bool]]:
    """Cheap, non-judge checks: still in the target language, non-degenerate.
    Not a compliance verdict -- just flags rows worth a second look."""
    if expected_language not in LANGDETECT_SUPPORTED_LANGS:
        still_target_language = None
    else:
        import langdetect

        try:
            still_target_language = langdetect.detect(response) == expected_language
        except langdetect.lang_detect_exception.LangDetectException:
            still_target_language = False

    stripped = response.strip()
    non_degenerate = len(stripped) >= 5 and len(set(stripped)) > 1

    return {"still_target_language": still_target_language, "non_degenerate": non_degenerate}


def export_to_hf(
    rows: List[Dict[str, Any]], path_in_repo: str, repo_id: str = EXPORT_REPO
) -> str:
    """Uploads rows as a parquet file to repo_id/path_in_repo."""
    df = pd.DataFrame(rows)
    missing = set(_SHARED_COLUMNS + _INTERVENTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"export rows missing expected columns: {sorted(missing)}")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "responses.parquet"
        df.to_parquet(local_path)
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
    return f"{repo_id}/{path_in_repo}"
