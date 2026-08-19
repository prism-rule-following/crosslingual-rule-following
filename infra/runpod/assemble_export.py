"""Assemble the canonical Llama run's final upload artifacts on the pod.

Reuses the exact production paths from ModelRunner (`_assemble_activations`,
`_build_activation_index`, `_activation_filename`) to build, from the
checkpointed numpy shards + responses jsonl:

  /workspace/inference_export/
    meta-llama__Llama-3.1-8B-Instruct/
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

Verifies row counts, shapes, dtypes, and that every dataset row id appears
exactly once in the activation index.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/crosslingual-rule-following")

import numpy as np
import pandas as pd

from canonical.evaluation import inference as inf
from canonical.model.dataset import (
    DatasetConfig,
    DatasetLanguageCode,
    DatasetSource,
    CrossLingualRuleFollowingDataset,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LANG = DatasetLanguageCode.en
CKPT = Path("/workspace/inference")
EXPORT = Path("/workspace/inference_export")
DATA_JSON = "/workspace/smoke_dataset.json"
EXPECTED_ROWS = 4680
N_SAMPLES = 10


def main():
    config = inf.ModelGenerationConfig(
        model_ids=[MODEL_ID],
        dataset_config=DatasetConfig(
            url=DATA_JSON, source=DatasetSource.gh, validate_rows=True, strict=True
        ),
        language_codes=[LANG],
        checkpoint_dir=str(CKPT),
    )
    runner = inf.ModelRunner(config)
    runner.model_id = MODEL_ID

    export_dir = EXPORT / MODEL_ID.replace("/", "__")
    export_dir.mkdir(parents=True, exist_ok=True)

    # --- activations: assemble shards + index via production methods ---
    print("Assembling activations from shards...")
    arrays = runner._assemble_activations(LANG)
    print(f"  groups: {sorted(arrays.keys())}")

    ds = CrossLingualRuleFollowingDataset(config.dataset_config).df
    print(f"  dataset rows: {len(ds)}")
    assert len(ds) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(ds)}"

    index = runner._build_activation_index(ds)
    print(f"  index rows: {len(index)} | columns: {list(index.columns)}")

    n_layers = 32
    d_model = 4096
    expected = {
        "hook_embed": (EXPECTED_ROWS, d_model),
        "hook_resid_post": (EXPECTED_ROWS, n_layers, d_model),
        "hook_attn_out": (EXPECTED_ROWS, n_layers, d_model),
        "hook_mlp_out": (EXPECTED_ROWS, n_layers, d_model),
        "attn_q_input": (EXPECTED_ROWS, n_layers, d_model),
        "attn_k_input": (EXPECTED_ROWS, n_layers, d_model),
        "attn_v_input": (EXPECTED_ROWS, n_layers, d_model),
        "hook_out": (EXPECTED_ROWS, n_layers, d_model),
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

    # --- responses: verify + export parquet ---
    resp_path = CKPT / f"{MODEL_ID.replace('/', '__')}_en_responses.jsonl"
    rows = [json.loads(line) for line in open(resp_path) if line.strip()]
    print(f"responses: {len(rows)} rows")
    assert len(rows) == EXPECTED_ROWS * N_SAMPLES, (
        f"expected {EXPECTED_ROWS * N_SAMPLES} response rows, got {len(rows)}"
    )
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    assert all(len(v) == N_SAMPLES for v in by_id.values()), "not all ids have 10 samples"
    assert len(by_id) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} unique ids, got {len(by_id)}"
    assert all(isinstance(r["response"], str) and r["response"] for r in rows)
    df = pd.DataFrame(rows)
    resp_export = EXPORT / f"{MODEL_ID.replace('/', '__')}_en_responses.parquet"
    df.to_parquet(resp_export, index=False)
    print(f"  wrote {resp_export} ({resp_export.stat().st_size/1e6:.1f} MB)")

    # --- cross-check: every index id has responses and vice versa ---
    idx_ids = set(index["id"])
    resp_ids = set(by_id.keys())
    assert idx_ids == resp_ids, (
        f"index/responses id mismatch: {len(idx_ids ^ resp_ids)} ids differ"
    )
    print(f"  cross-check OK: {len(idx_ids)} ids identical between index and responses")

    manifest = {
        "model": MODEL_ID,
        "language": "en",
        "n_dataset_rows": EXPECTED_ROWS,
        "n_samples": N_SAMPLES,
        "n_response_rows": len(rows),
        "activation_groups": {g: list(arrays[g].shape) for g in sorted(arrays)},
        "activation_dtype": "float16",
        "n_layers": n_layers,
        "d_model": d_model,
        "checkpoint_src": str(CKPT),
        "exported": str(EXPORT),
    }
    with open(EXPORT / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== ASSEMBLY + VERIFICATION: ALL PASS ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
