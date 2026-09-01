import pytest

from canonical.causal.vector_patching.export_responses import (
    _INTERVENTION_COLUMNS,
    _SHARED_COLUMNS,
    build_response_row,
    export_to_hf,
)


def _toy_row(**overrides):
    row = dict(
        canonical_id="rb_j_no_dosage_medical_active_imperative_L0",
        model_id="Qwen/Qwen3-8B",
        language="yo",
        category="no_dosage",
        topic="medical",
        grammar_type="imperative",
        pressure_level="L0",
        pair_type="active_cancelled",
        sample_idx=0,
        rule_clause="...",
        user_query="...",
        response="patched response text",
        donor_language="en",
        patch_layer=15,
        vector_type="dom",
        donor_kind="single",
        patch_mode="patch",
        recipient_pre_verdict=False,
        feasibility_cohens_d=1.2,
    )
    row.update(overrides)
    return row


def test_build_response_row_has_all_expected_columns():
    row = build_response_row(**_toy_row())
    assert set(row.keys()) == set(_SHARED_COLUMNS + _INTERVENTION_COLUMNS)


def test_export_to_hf_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing expected columns"):
        export_to_hf([{"id": "x"}], path_in_repo="test.parquet")
