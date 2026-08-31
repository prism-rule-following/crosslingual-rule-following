"""Verify the REAL ModelRunner save path on a couple of real-model rows.

Exercises canonical/evaluation/inference.py's generate_response +
extract_hidden_states (the exact code used for canonical runs) on 2 dataset
rows per model, using the local verified JSON (no HF network). Verifies:

  - response checkpoint jsonl written with correct rows
  - activation done.jsonl + per-batch numpy shards written
  - assembled arrays shape == (n_rows, n_layers, ...) for per-layer hooks
  - index.parquet label columns + row_idx
  - resume behaviour: second call does not duplicate

push_to_hf=False everywhere. Checkpoints go to a throwaway dir.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/workspace/crosslingual-rule-following")

import pandas as pd
import torch

from canonical.evaluation import inference as inf
from canonical.model.dataset import (
    CrossLingualRuleFollowingDataset,
    DatasetConfig,
    DatasetLanguageCode,
    DatasetSource,
)

DATASET_PATH = "/workspace/smoke_dataset.json"
CKPT = "/workspace/smoke_ckpt_real"
MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"]
ROWS = 2


def check(cond, msg):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def main():
    config = inf.ModelGenerationConfig(
        model_ids=MODELS,
        dataset_config=DatasetConfig(
            url=DATASET_PATH, source=DatasetSource.gh, validate_rows=True, strict=True
        ),
        language_codes=[DatasetLanguageCode.en],
        max_new_tokens=16,
        generation_batch_size=4,
        activation_batch_size=4,
        activation_dtype="float16",
        checkpoint_dir=CKPT,
        push_to_hf=False,
        n_samples=1,
    )
    ds = CrossLingualRuleFollowingDataset(config.dataset_config)
    df = ds.df.head(ROWS).reset_index(drop=True)
    print(f"dataset rows: {len(df)} | columns: {list(df.columns)}")

    for model_id in MODELS:
        print(f"\n{'='*70}\nMODEL: {model_id}\n{'='*70}")
        runner = inf.ModelRunner(config)
        runner.load(model_id)
        runner.supports_system_role = runner._check_system_role_support()
        print(f"  system role: {runner.supports_system_role}")

        # ---- responses ----
        resp = runner.generate_response(df, lang_code=DatasetLanguageCode.en)
        check(len(resp) == ROWS, f"generate_response returned {len(resp)} rows")
        check(all(isinstance(r["response"], str) and r["response"] for r in resp),
              "all responses non-empty")
        print(f"  sample response [{resp[0]['rule_status']}]: {resp[0]['response'][:60]!r}")
        ckpt_path = runner._checkpoint_path(DatasetLanguageCode.en, "responses")
        check(ckpt_path.exists(), f"response checkpoint written: {ckpt_path.name}")
        with open(ckpt_path) as f:
            ckpt_rows = [line for line in f if line.strip()]
        check(len(ckpt_rows) == ROWS, f"checkpoint has {len(ckpt_rows)} rows")

        # ---- resume skips already-done ids ----
        resp2 = runner.generate_response(df, lang_code=DatasetLanguageCode.en)
        check(len(resp2) == ROWS, "resume: still returns full row set (completed ids reused)")
        mism = [(i, a["id"], b["id"], a["response"][:40], b["response"][:40])
                for i, (a, b) in enumerate(zip(resp2, resp))
                if a["response"] != b["response"]]
        if mism:
            print("  RESUME MISMATCH DETAIL:")
            for m in mism:
                print(f"    idx={m[0]} ids={m[1]}=={m[2]} | resp2={m[3]!r} | resp={m[4]!r}")
        check(not mism, "resume: responses identical (no regen)")

        # ---- activations ----
        act = runner.extract_hidden_states(df, lang_code=DatasetLanguageCode.en)
        check(act.n_rows == ROWS, f"activation n_rows={act.n_rows}")
        check(set(act.index["id"]) == set(df["id"]), "index ids match dataset")
        check("row_idx" in act.index.columns, "index has row_idx")
        print(f"  index columns: {list(act.index.columns)}")
        n_layers = runner.model.cfg.n_layers
        expected_shapes = {
            "hook_embed": (ROWS, 4096),
            "hook_resid_post": (ROWS, n_layers, 4096),
            "hook_attn_out": (ROWS, n_layers, 4096),
            "hook_mlp_out": (ROWS, n_layers, 4096),
        }
        for group, shp in expected_shapes.items():
            check(group in act.arrays, f"group present: {group}")
            check(tuple(act.arrays[group].shape) == shp,
                  f"{group} shape {tuple(act.arrays[group].shape)} == {shp}")
            check(act.arrays[group].dtype == "float16", f"{group} dtype fp16")
        # qkv / hook_out may be absent on some archs; warn but don't fail
        print(f"  groups present: {sorted(act.arrays.keys())}")
        for g in ("attn_qkv_input", "hook_out"):
            if g in act.arrays:
                print(f"    + extra group {g}: {tuple(act.arrays[g].shape)} {act.arrays[g].dtype}")

        # ---- done manifest + shards ----
        done_path = runner._activation_done_path(DatasetLanguageCode.en)
        check(done_path.exists(), "done.jsonl exists")
        shards = list(runner._activation_shards_dir(DatasetLanguageCode.en).glob("*.npy"))
        # Llama has no attn.qkv.hook_in under TransformerBridge -> 5 of 6
        # groups present (all probing groups + hook_out; attn_qkv_input absent).
        check(len(shards) >= 5, f"shard files written ({len(shards)})")

        # ---- resume: no duplicates ----
        act2 = runner.extract_hidden_states(df, lang_code=DatasetLanguageCode.en)
        check(act2.n_rows == ROWS, "activation resume n_rows unchanged")
        check(runner._activation_done_count(DatasetLanguageCode.en) == ROWS,
              "done manifest still ROWS (no duplicates)")

        # ---- cleanup this model's checkpoints ----
        runner.clear_checkpoint(DatasetLanguageCode.en, "responses")
        runner.clear_activation_checkpoint(DatasetLanguageCode.en)
        check(not ckpt_path.exists(), "response checkpoint cleared after upload-style cleanup")

        del runner
        torch.cuda.empty_cache()

    print(f"\n=== REAL-PATH SMOKE: ALL PASS ({MODELS}) ===")


if __name__ == "__main__":
    main()
