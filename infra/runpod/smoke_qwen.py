"""Qwen3-8B canonical smoke: enable_thinking=False end-to-end validation.

Exercises canonical/evaluation/inference.py's generate_response +
extract_hidden_states (the exact canonical code path) on 2 dataset rows for
Qwen/Qwen3-8B with enable_thinking=False. Verifies:

  - chat template: prompt contains the suppressed-thinking ' response\\n\\n'
    marker and NO reasoning/think markers
  - generated continuation contains no '<|im_start|>' (model did not open a
    thinking block on its own)
  - all 8 activation hook groups captured with correct shapes
    (hook_embed (R,4096); per-layer groups (R, n_layers, 4096)), fp16
  - index.parquet label columns + row_idx; ids match dataset
  - resume: second run does not duplicate
  - checkpoint cleanup

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

MODEL_ID = "Qwen/Qwen3-8B"
DATASET_PATH = "/workspace/smoke_dataset.json"
CKPT = "/workspace/smoke_ckpt_qwen"
ROWS = 2

ACTIVATION_GROUPS = [
    "hook_embed",
    "hook_resid_post",
    "hook_attn_out",
    "hook_mlp_out",
    "attn_q_input",
    "attn_k_input",
    "attn_v_input",
    "hook_out",
]


def check(cond, msg):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def main():
    config = inf.ModelGenerationConfig(
        model_ids=[MODEL_ID],
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
        enable_thinking=False,
    )
    ds = CrossLingualRuleFollowingDataset(config.dataset_config)
    df = ds.df.head(ROWS).reset_index(drop=True)
    print(f"dataset rows: {len(df)} | columns: {list(df.columns)}")
    check("system" in df.columns and "user_query" in df.columns,
          "dataset has system/user_query columns")

    print(f"\n{'='*70}\nMODEL: {MODEL_ID} (enable_thinking=False)\n{'='*70}")
    runner = inf.ModelRunner(config)
    runner.load(MODEL_ID)
    runner.supports_system_role = runner._check_system_role_support()
    print(f"  system role: {runner.supports_system_role}")
    n_layers = runner.model.cfg.n_layers
    print(f"  n_layers={n_layers} | d_model={runner.model.cfg.d_model}")

    # ---- chat template: thinking suppression ----
    prompt = runner.format_chat_prompt(df.iloc[0]["system"], df.iloc[0]["user_query"])
    # Ground truth (verified via pod byte dump): enable_thinking=False makes
    # Qwen3's template emit a CLOSED empty think block '<think>\n\n</think>'
    # and NO '<|im_start|>think' opening token, so the model responds directly.
    check("</think>" in prompt, "prompt closes the think block")
    check("<|im_start|>think" not in prompt, "prompt has no thinking-open marker")

    # ---- responses ----
    resp = runner.generate_response(df, lang_code=DatasetLanguageCode.en)
    check(len(resp) == ROWS, f"generate_response returned {len(resp)} rows")
    check(all(isinstance(r["response"], str) and r["response"] for r in resp),
          "all responses non-empty")
    for r in resp:
        print(f"  response [{r['rule_status']}]: {r['response'][:80]!r}")

    # raw continuation must not open a new turn (i.e. no thinking block)
    prompts = [runner.format_chat_prompt(r["system"], r["user_query"]) for r in resp]
    batch = runner.tokenizer(prompts, padding=True, return_tensors="pt").to(
        runner.config.device
    )
    outputs = runner.model.original_model.generate(
        **batch,
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        temperature=config.temperature,
    )
    input_len = batch["input_ids"].shape[1]
    raw_continuations = [
        runner.tokenizer.decode(o[input_len:], skip_special_tokens=False)
        for o in outputs
    ]
    for rc in raw_continuations:
        check("<|im_start|>" not in rc, f"raw continuation has no new turn start: {rc[:60]!r}")
    del outputs, batch
    torch.cuda.empty_cache()

    ckpt_path = runner._checkpoint_path(DatasetLanguageCode.en, "responses")
    check(ckpt_path.exists(), f"response checkpoint written: {ckpt_path.name}")
    with open(ckpt_path) as f:
        ckpt_rows = [line for line in f if line.strip()]
    check(len(ckpt_rows) == ROWS, f"checkpoint has {len(ckpt_rows)} rows")

    # ---- resume skips already-done ids ----
    resp2 = runner.generate_response(df, lang_code=DatasetLanguageCode.en)
    check(len(resp2) == ROWS, "resume: still returns full row set")
    mism = [(i, a["id"], b["id"], a["response"][:40], b["response"][:40])
            for i, (a, b) in enumerate(zip(resp2, resp))
            if a["response"] != b["response"]]
    for m in mism:
        print(f"  RESUME MISMATCH: idx={m[0]} ids={m[1]}=={m[2]} | {m[3]!r} | {m[4]!r}")
    check(not mism, "resume: responses identical (no regen)")

    # ---- activations ----
    act = runner.extract_hidden_states(df, lang_code=DatasetLanguageCode.en)
    check(act.n_rows == ROWS, f"activation n_rows={act.n_rows}")
    check(set(act.index["id"]) == set(df["id"]), "index ids match dataset")
    check("row_idx" in act.index.columns, "index has row_idx")
    print(f"  index columns: {list(act.index.columns)}")
    check(set(act.arrays.keys()) == set(ACTIVATION_GROUPS),
          f"all 8 activation groups present: {sorted(act.arrays.keys())}")
    for g in ACTIVATION_GROUPS:
        expected = (ROWS, 4096) if g == "hook_embed" else (ROWS, n_layers, 4096)
        check(tuple(act.arrays[g].shape) == expected,
              f"{g} shape {tuple(act.arrays[g].shape)} == {expected}")
        check(act.arrays[g].dtype == "float16", f"{g} dtype fp16")
        check(bool(act.arrays[g].reshape(-1)[:1000].isfinite().all()),
              f"{g} finite")

    # ---- done manifest + shards ----
    done_path = runner._activation_done_path(DatasetLanguageCode.en)
    check(done_path.exists(), "done.jsonl exists")
    shards = list(runner._activation_shards_dir(DatasetLanguageCode.en).glob("*.npy"))
    check(len(shards) == 8, f"shard files written ({len(shards)})")

    # ---- resume: no duplicates ----
    act2 = runner.extract_hidden_states(df, lang_code=DatasetLanguageCode.en)
    check(act2.n_rows == ROWS, "activation resume n_rows unchanged")
    check(runner._activation_done_count(DatasetLanguageCode.en) == ROWS,
          "done manifest still ROWS (no duplicates)")

    # ---- cleanup this model's checkpoints ----
    runner.clear_checkpoint(DatasetLanguageCode.en, "responses")
    runner.clear_activation_checkpoint(DatasetLanguageCode.en)
    check(not ckpt_path.exists(), "response checkpoint cleared")

    print(f"\n=== QWEN SMOKE (enable_thinking=False): ALL PASS ===")


if __name__ == "__main__":
    main()
