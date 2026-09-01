import pandas as pd

from canonical.causal.vector_patching.pair_selection import (
    _majority,
    build_pair_table,
    classify_tiers,
    collapse_verdicts,
    compute_adherence_rates,
    strip_id_suffix,
)


def test_strip_id_suffix():
    assert strip_id_suffix("rb_j_x_L0_clean") == "rb_j_x_L0"
    assert strip_id_suffix("rb_j_x_L0_revoked") == "rb_j_x_L0"
    assert strip_id_suffix("rb_j_x_L0") == "rb_j_x_L0"


def test_majority_no_tie():
    assert _majority(pd.Series([True, True, False])) == (True, False)
    assert _majority(pd.Series([False, False, True])) == (False, False)


def test_majority_tie_breaks_to_not_held():
    verdict, low_confidence = _majority(pd.Series([True, False]))
    assert verdict is False
    assert low_confidence is True


def _toy_verdicts() -> pd.DataFrame:
    # id_A: en held (3 judges), hi failed (2/3 judges); id_B: en failed, hi held
    rows = []
    for judge, held in [("gpt_mini", True), ("gemini", True), ("deepseek", True)]:
        rows.append(
            dict(model_id="m", language="en", canonical_id="id_A", category="c",
                 pressure_level="L0", judge=judge, held=held)
        )
    for judge, held in [("gpt_mini", False), ("gemini", False), ("deepseek", True)]:
        rows.append(
            dict(model_id="m", language="hi", canonical_id="id_A", category="c",
                 pressure_level="L0", judge=judge, held=held)
        )
    for judge, held in [("gpt_mini", False), ("gemini", False), ("deepseek", False)]:
        rows.append(
            dict(model_id="m", language="en", canonical_id="id_B", category="c",
                 pressure_level="L0", judge=judge, held=held)
        )
    for judge, held in [("gpt_mini", True), ("gemini", True), ("deepseek", True)]:
        rows.append(
            dict(model_id="m", language="hi", canonical_id="id_B", category="c",
                 pressure_level="L0", judge=judge, held=held)
        )
    return pd.DataFrame(rows)


def test_collapse_verdicts_majority_across_judges():
    collapsed = collapse_verdicts(_toy_verdicts())
    row = collapsed[(collapsed["language"] == "hi") & (collapsed["canonical_id"] == "id_A")].iloc[0]
    assert row["held"] == False  # 2/3 judges said not held
    row = collapsed[(collapsed["language"] == "en") & (collapsed["canonical_id"] == "id_A")].iloc[0]
    assert row["held"] == True


def test_adherence_rates_and_tiers():
    collapsed = collapse_verdicts(_toy_verdicts())
    rates = compute_adherence_rates(collapsed, "m")
    en_rate = rates[(rates["language"] == "en") & (rates["category"] == "__overall__")]["held_rate"].iloc[0]
    hi_rate = rates[(rates["language"] == "hi") & (rates["category"] == "__overall__")]["held_rate"].iloc[0]
    assert en_rate == 0.5  # held id_A, failed id_B
    assert hi_rate == 0.5  # failed id_A, held id_B

    tiers = classify_tiers(rates, high_thresh=0.6, low_thresh=0.4)
    assert tiers["en"] == "mid"
    assert tiers["hi"] == "mid"


def test_build_pair_table_matches_by_id():
    collapsed = collapse_verdicts(_toy_verdicts())
    pairs = build_pair_table(collapsed, "m")
    # en held id_A, hi failed id_A -> en->hi pair on id_A
    assert ((pairs["language_donor"] == "en") & (pairs["language_recipient"] == "hi")
            & (pairs["canonical_id"] == "id_A")).any()
    # hi held id_B, en failed id_B -> hi->en pair on id_B
    assert ((pairs["language_donor"] == "hi") & (pairs["language_recipient"] == "en")
            & (pairs["canonical_id"] == "id_B")).any()
    # no donor==recipient rows
    assert (pairs["language_donor"] != pairs["language_recipient"]).all()
