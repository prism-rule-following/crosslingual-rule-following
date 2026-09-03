import json

import pandas as pd
import pytest

from canonical.causal.vector_patching.finalize_export import audit
from canonical.causal.vector_patching.manifest import (
    IDS_PER_CATEGORY,
    RECIPIENTS,
    select_ids,
    verdict_meta,
)


def _toy_collapsed():
    rows = []
    for i in range(20):
        cid = f"id_{i:02d}"
        for judge in ("gpt_mini", "gemini", "deepseek"):
            rows.append(
                dict(model_id="Qwen/Qwen3-8B", language="en", canonical_id=cid,
                     category="ack_invert", pressure_level="L0", held=True,
                     low_confidence=False)
            )
            rows.append(
                dict(model_id="Qwen/Qwen3-8B", language="de", canonical_id=cid,
                     category="ack_invert", pressure_level="L0",
                     held=(i % 3 == 0), low_confidence=(i % 5 == 0))
            )
    return pd.DataFrame(rows)


def test_select_ids_deterministic_and_scoped():
    collapsed = _toy_collapsed()
    a = select_ids(collapsed, "ack_invert")
    b = select_ids(collapsed, "ack_invert")
    assert a == b
    assert len(a) == IDS_PER_CATEGORY
    assert len(set(a)) == IDS_PER_CATEGORY
    assert all(x.startswith("id_") for x in a)


def test_select_ids_insufficient_candidates_raises():
    collapsed = pd.DataFrame(
        [dict(model_id="Qwen/Qwen3-8B", language="en", canonical_id=f"id_{i}",
              category="scope_lock", pressure_level="L0", held=True)
         for i in range(3)]
    )
    with pytest.raises(ValueError):
        select_ids(collapsed, "scope_lock")


def test_verdict_meta():
    collapsed = _toy_collapsed()
    meta = verdict_meta(collapsed, "de", "id_00")
    assert meta["held"] is True
    meta = verdict_meta(collapsed, "de", "id_01")
    assert meta["held"] is False
    assert verdict_meta(collapsed, "en", "missing") is None


def _row(arm, cid, layer):
    return dict(
        arm=arm, id=cid, patch_layer=layer, category="ack_invert",
        model_id="Qwen/Qwen3-8B", donor_language="en", pressure_level="L0",
        response="ok", error=None, max_new_tokens=768,
        recipient_pre_verdict=False,
    )


def test_audit_clean_and_catches_duplicates():
    manifest = {
        "selected_ids": [f"id_{i:02d}" for i in range(25)],
        "id_category": {f"id_{i:02d}": "ack_invert" for i in range(25)},
        "model_id": "Qwen/Qwen3-8B",
        "donor_language": "en",
    }
    rows = [_row("baseline", f"id_{i:02d}", None) for i in range(25)]
    rows += [_row("dom", f"id_{i:02d}", layer) for i in range(25) for layer in [15, 24, 27, 29, 31]]
    rows += [_row("w", f"id_{i:02d}", layer) for i in range(25) for layer in [15, 24, 27, 29, 31]]
    problems, stats = audit(manifest, {"de": rows})
    assert problems == []
    assert stats["de"]["n"] == 275

    rows.append(_row("dom", "id_00", 15))
    problems, _ = audit(manifest, {"de": rows})
    assert any("duplicate" in p for p in problems)


def test_audit_catches_wrong_layer_set():
    manifest = {
        "selected_ids": [f"id_{i:02d}" for i in range(25)],
        "id_category": {f"id_{i:02d}": "ack_invert" for i in range(25)},
        "model_id": "Qwen/Qwen3-8B",
        "donor_language": "en",
    }
    rows = [_row("baseline", f"id_{i:02d}", None) for i in range(25)]
    rows += [_row("dom", f"id_{i:02d}", layer) for i in range(25) for layer in [12, 15, 24, 27, 29]]
    rows += [_row("w", f"id_{i:02d}", layer) for i in range(25) for layer in [15, 24, 27, 29, 31]]
    problems, _ = audit(manifest, {"de": rows})
    assert any("layer set" in p for p in problems)


def test_audit_catches_errors_and_empty():
    manifest = {
        "selected_ids": [f"id_{i:02d}" for i in range(25)],
        "id_category": {f"id_{i:02d}": "ack_invert" for i in range(25)},
        "model_id": "Qwen/Qwen3-8B",
        "donor_language": "en",
    }
    rows = [_row("baseline", f"id_{i:02d}", None) for i in range(25)]
    rows += [_row("dom", f"id_{i:02d}", layer) for i in range(25) for layer in [15, 24, 27, 29, 31]]
    rows += [_row("w", f"id_{i:02d}", layer) for i in range(25) for layer in [15, 24, 27, 29, 31]]
    rows[0]["error"] = "boom"
    rows[1]["response"] = ""
    problems, _ = audit(manifest, {"de": rows})
    assert any("generation errors" in p for p in problems)
    assert any("empty responses" in p for p in problems)