"""Assemble canonical run upload artifacts on the pod, per (model, language).

Reuses the exact production paths from ModelRunner (`_assemble_activations`,
`_build_activation_index`, `_activation_filename`) to build, from the
checkpointed numpy shards + responses jsonl:

  /workspace/inference_export/
    meta-llama__Llama-3.1-8B-Instruct/
      en/
        index.parquet
        hook_embed.fp16.npy
        hook_resid_post.fp16.npy
        hook_attn_out.fp16.npy
        hook_mlp_out.fp16.npy
        attn_q_input.fp16.npy
        attn_k_input.fp16.npy
        attn_v_input.fp16.npy
        hook_out.fp16.npy
    meta-llama__Llama-3.1-8B-Instruct_en_responses.parquet
    MANIFEST.json

Usage: assemble_export.py <model-id> [lang]   (lang defaults to en)

Verifies row counts, shapes, dtypes, and that every dataset row id appears
exactly once in the activation index.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/workspace/crosslingual-rule-following")

import numpy as np
import pandas as pd

from canonical.evaluation import inference as inf
from canonical.model.dataset import (
    ACTIVE_STATUSES,
    DatasetConfig,
    DatasetLanguageCode,
    DatasetSource,
    CrossLingualRuleFollowingDataset,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CKPT = Path("/workspace/inference")
EXPORT = Path("/workspace/inference_export")
DATA_DIR = Path("/workspace/data")
# n_layers per model (Llama-3.1-8B = 32, Qwen3-8B = 36). Must match the model
# whose shards are being assembled.
N_LAYERS = {"meta-llama/Llama-3.1-8B-Instruct": 32, "Qwen/Qwen3-8B": 36}
N_SAMPLES = 10


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID
    lang_code = (
        DatasetLanguageCode(sys.argv[2]) if len(sys.argv) > 2 else DatasetLanguageCode.en
    )
    lang = lang_code.value
    n_layers = N_LAYERS[model_id]
    data_json = DATA_DIR / lang / "test.jsonl"
    assert data_json.exists(), f"missing dataset file: {data_json}"
    ds = CrossLingualRuleFollowingDataset(
        DatasetConfig(
            url=str(data_json), source=DatasetSource.gh, validate_rows=True, strict=True
        )
    ).df
    expected_rows = len(ds)
    assert expected_rows % 2 == 0, "expected an even number of split rows"
    print(f"dataset rows: {expected_rows} (lang={lang})")

    config = inf.ModelGenerationConfig(
        model_ids=[model_id],
        dataset_config=DatasetConfig(
            url=str(data_json), source=DatasetSource.gh, validate_rows=True, strict=True
        ),
        language_codes=[lang_code],
        checkpoint_dir=str(CKPT),
    )
    runner = inf.ModelRunner(config)
    runner.model_id = model_id
    # Stub the model just enough for _group_hook_names (n_layers); no weights needed.
    runner.model = SimpleNamespace(cfg=SimpleNamespace(n_layers=n_layers))

    export_dir = EXPORT / model_id.replace("/", "__") / lang
    export_dir.mkdir(parents=True, exist_ok=True)

    # --- activations: assemble shards + index via production methods ---
    print("Assembling activations from shards...")
    arrays = runner._assemble_activations(lang_code)
    print(f"  groups: {sorted(arrays.keys())}")

    index = runner._build_activation_index(ds)
    print(f"  index rows: {len(index)} | columns: {list(index.columns)}")

    d_model = 4096
    expected = {
        "hook_embed": (expected_rows, d_model),
        "hook_resid_post": (expected_rows, n_layers, d_model),
        "hook_attn_out": (expected_rows, n_layers, d_model),
        "hook_mlp_out": (expected_rows, n_layers, d_model),
        "attn_q_input": (expected_rows, n_layers, d_model),
        "attn_k_input": (expected_rows, n_layers, d_model),
        "attn_v_input": (expected_rows, n_layers, d_model),
        "hook_out": (expected_rows, n_layers, d_model),
    }
    for group, shape in expected.items():
        assert group in arrays, f"missing group {group}"
        assert tuple(arrays[group].shape) == shape, (
            f"{group}: got {tuple(arrays[group].shape)} expected {shape}"
        )
        assert arrays[group].dtype == np.float16, f"{group}: dtype {arrays[group].dtype}"
        assert np.isfinite(arrays[group]).all(), f"{group}: non-finite values!"
    print("  all 8 groups: shape + fp16 + finite OK")

    # --- write activation files ---
    idx_path = export_dir / "index.parquet"
    index.to_parquet(idx_path, index=False)
    print(f"  wrote {idx_path} ({idx_path.stat().st_size/1e6:.1f} MB)")
    for group, arr in arrays.items():
        path = export_dir / inf._activation_filename(group, "float16")
        np.save(path, arr)
        print(f"  wrote {path.name} ({path.stat().st_size/1e6:.1f} MB)")

    # --- responses: retain only active-side rows and sample indices 0,1,2 ---
    resp_path = CKPT / f"{model_id.replace('/', '__')}_{lang}_responses.jsonl"
    raw_rows = [json.loads(line) for line in open(resp_path) if line.strip()]
    active_values = {status.value for status in ACTIVE_STATUSES}
    rows = [
        row
        for row in raw_rows
        if row["rule_status"] in active_values
        and int(row.get("sample_idx", 0)) in {0, 1, 2}
    ]
    expected_active_rows = sum(
        1 for value in index["rule_status"] if value in active_values
    )
    print(f"responses: {len(raw_rows)} raw rows -> {len(rows)} active/sample-3 rows")
    assert len(rows) == expected_active_rows * N_SAMPLES, (
        f"expected {expected_active_rows * N_SAMPLES} response rows, got {len(rows)}"
    )
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    assert all(len(v) == N_SAMPLES for v in by_id.values()), "not all active ids have 3 samples"
    assert len(by_id) == expected_active_rows, (
        f"expected {expected_active_rows} active ids, got {len(by_id)}"
    )
    assert all(isinstance(r["response"], str) and r["response"] for r in rows)
    df = pd.DataFrame(rows)
    resp_export = EXPORT / f"{model_id.replace('/', '__')}_{lang}_responses.parquet"
    df.to_parquet(resp_export, index=False)
    print(f"  wrote {resp_export} ({resp_export.stat().st_size/1e6:.1f} MB)")

    # --- cross-check: every index id has responses and vice versa ---
    idx_ids = set(index.loc[index["rule_status"].isin(active_values), "id"])
    resp_ids = set(by_id.keys())
    assert idx_ids == resp_ids, (
        f"index/responses id mismatch: {len(idx_ids ^ resp_ids)} ids differ"
    )
    print(f"  cross-check OK: {len(idx_ids)} ids identical between index and responses")

    manifest = {
        "model": model_id,
        "language": lang,
        "n_dataset_rows": expected_rows,
        "n_active_dataset_rows": expected_active_rows,
        "n_samples": N_SAMPLES,
        "n_response_rows": len(rows),
        "response_filter": {"active_statuses": sorted(active_values), "sample_idx": [0, 1, 2]},
        "activation_groups": {g: list(arrays[g].shape) for g in sorted(arrays)},
        "activation_dtype": "float16",
        "n_layers": n_layers,
        "d_model": d_model,
        "checkpoint_src": str(CKPT),
        "exported": str(EXPORT),
    }
    with open(EXPORT / f"MANIFEST_{model_id.replace('/', '__')}_{lang}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== ASSEMBLY + VERIFICATION: ALL PASS ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
