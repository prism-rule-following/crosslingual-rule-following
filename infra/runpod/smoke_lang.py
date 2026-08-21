"""Per-language canonical smoke: end-to-end validation for one (model, lang).

Exercises canonical/evaluation/inference.py's generate_response +
extract_hidden_states (the exact canonical code path) on a small number of
dataset rows for a given model + language. Verifies:

  - dataset loads + validates from the local jsonl for this language
  - all 8 activation hook groups captured with correct shapes
    (hook_embed (R,4096); per-layer groups (R, n_layers, 4096)), fp16
  - index.parquet label columns + row_idx; ids match dataset
  - resume: second run does not duplicate
  - checkpoint cleanup

Usage: smoke_lang.py <model-id> <lang> <dataset-jsonl> [rows]
Example: smoke_lang.py meta-llama/Llama-3.1-8B-Instruct de /workspace/data/de/test.jsonl 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/workspace/crosslingual-rule-following")

import numpy as np
import torch

from canonical.evaluation import inference as inf
from canonical.model.dataset import (
    CrossLingualRuleFollowingDataset,
    DatasetConfig,
    DatasetLanguageCode,
    DatasetSource,
)

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
    model_id = sys.argv[1]
    lang_code = DatasetLanguageCode(sys.argv[2])
    dataset_path = sys.argv[3]
    rows = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    ckpt = Path(f"/workspace/smoke_ckpt_{lang_code.value}")
    is_qwen = "Qwen" in model_id

    print(f"MODEL={model_id} LANG={lang_code.value} DATA={dataset_path} ROWS={rows}")

    config = inf.ModelGenerationConfig(
        model_ids=[model_id],
        dataset_config=DatasetConfig(
            url=dataset_path, source=DatasetSource.gh, validate_rows=True, strict=True
        ),
        language_codes=[lang_code],
        max_new_tokens=16,
        generation_batch_size=4,
        activation_batch_size=4,
        activation_dtype="float16",
        checkpoint_dir=str(ckpt),
        push_to_hf=False,
        n_samples=1,
        enable_thinking=False,
    )

    ds = CrossLingualRuleFollowingDataset(config.dataset_config)
    full = ds.df
    print(f"dataset rows: {len(full)} (lang={lang_code.value})")
    check(len(full) > 0, "dataset non-empty")
    check((full["language"] == lang_code.value).all(),
          "all rows match target language")
    check(full["id"].is_unique, "ids unique")
    df = full.head(rows).reset_index(drop=True)

    runner = inf.ModelRunner(config)
    runner.load(model_id)
    runner.supports_system_role = runner._check_system_role_support()
    print(f"  system role: {runner.supports_system_role}")
    n_layers = runner.model.cfg.n_layers
    print(f"  n_layers={n_layers} | d_model={runner.model.cfg.d_model}")

    if is_qwen:
        prompt = runner.format_chat_prompt(df.iloc[0]["system"], df.iloc[0]["user_query"])
        check(" response" in prompt, "qwen: prompt closes the think block")
        check("<|im_start|>think" not in prompt, "qwen: prompt has no thinking-open marker")

    resp = runner.generate_response(df, lang_code=lang_code)
    check(len(resp) == rows, f"generate_response returned {len(resp)} rows")
    check(all(isinstance(r["response"], str) and r["response"] for r in resp),
          "all responses non-empty")
    for r in resp:
        print(f"  response [{r['rule_status']}]: {r['response'][:80]!r}")

    ckpt_path = runner._checkpoint_path(lang_code, "responses")
    check(ckpt_path.exists(), f"response checkpoint written: {ckpt_path.name}")
    with open(ckpt_path) as f:
        ckpt_rows = [line for line in f if line.strip()]
    check(len(ckpt_rows) == rows, f"checkpoint has {len(ckpt_rows)} rows")

    resp2 = runner.generate_response(df, lang_code=lang_code)
    check(len(resp2) == rows, "resume: still returns full row set")
    mism = [a["response"] != b["response"] for a, b in zip(resp2, resp)]
    check(not any(mism), "resume: responses identical (no regen)")

    act = runner.extract_hidden_states(df, lang_code=lang_code)
    check(act.n_rows == rows, f"activation n_rows={act.n_rows}")
    check(set(act.index["id"]) == set(df["id"]), "index ids match dataset")
    check("row_idx" in act.index.columns, "index has row_idx")
    check(set(act.arrays.keys()) == set(ACTIVATION_GROUPS),
          f"all 8 activation groups present: {sorted(act.arrays.keys())}")
    for g in ACTIVATION_GROUPS:
        expected = (rows, 4096) if g == "hook_embed" else (rows, n_layers, 4096)
        check(tuple(act.arrays[g].shape) == expected,
              f"{g} shape {tuple(act.arrays[g].shape)} == {expected}")
        check(act.arrays[g].dtype == "float16", f"{g} dtype fp16")
        check(bool(np.isfinite(act.arrays[g].reshape(-1)[:1000]).all()),
              f"{g} finite")

    done_path = runner._activation_done_path(lang_code)
    check(done_path.exists(), "done.jsonl exists")

    act2 = runner.extract_hidden_states(df, lang_code=lang_code)
    check(act2.n_rows == rows, "activation resume n_rows unchanged")
    check(runner._activation_done_count(lang_code) == rows,
          "done manifest still ROWS (no duplicates)")

    runner.clear_checkpoint(lang_code, "responses")
    runner.clear_activation_checkpoint(lang_code)
    check(not ckpt_path.exists(), "response checkpoint cleared")

    print(f"\n=== SMOKE {model_id} / {lang_code.value}: ALL PASS ===")


if __name__ == "__main__":
    main()