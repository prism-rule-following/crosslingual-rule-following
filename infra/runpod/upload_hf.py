"""Upload the canonical Llama-3.1-8B run outputs to HF from the Mac.

Layout matches the production HFDataHelper conventions:
  model-inference-responses:
    meta-llama__Llama-3.1-8B-Instruct/en.parquet
  model-inference-activations:
    meta-llama__Llama-3.1-8B-Instruct/en/
      index.parquet
      hook_embed.fp16.npy
      hook_resid_post.fp16.npy
      hook_attn_out.fp16.npy
      hook_mlp_out.fp16.npy
      attn_q_input.fp16.npy
      attn_k_input.fp16.npy
      attn_v_input.fp16.npy
      hook_out.fp16.npy
"""

import json
import os
import sys
import time

from huggingface_hub import HfApi

EXPORT = "/Users/ayesha/Projects/crosslingual-rule-following/.local/llama_export"
MODEL_SLUG = "meta-llama__Llama-3.1-8B-Instruct"
EN_DIR = f"{MODEL_SLUG}/en"
RESP_REPO = "crosslingual-rule-following/model-inference-responses"
ACT_REPO = "crosslingual-rule-following/model-inference-activations"
TOKEN = os.environ["HF_TOKEN"]

RESP_PATHS = [
    (f"{EXPORT}/{MODEL_SLUG}_en_responses.parquet", f"{MODEL_SLUG}/en.parquet"),
]
ACT_GROUPS = [
    "hook_embed", "hook_resid_post", "hook_attn_out", "hook_mlp_out",
    "attn_q_input", "attn_k_input", "attn_v_input", "hook_out",
]
ACT_PATHS = [("index.parquet", f"{EN_DIR}/index.parquet")] + [
    (f"{g}.float16.npy", f"{EN_DIR}/{g}.fp16.npy") for g in ACT_GROUPS
]


def upload_all(api, repo, local_dir, paths, tag):
    api.create_repo(repo_id=repo, repo_type="dataset", token=TOKEN, exist_ok=True)
    for local_name, remote_path in paths:
        src = os.path.join(local_dir, local_name)
        size = os.path.getsize(src) / 1e6
        t0 = time.time()
        print(f"[{tag}] uploading {remote_path} ({size:.1f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=src,
            path_in_repo=remote_path,
            repo_id=repo,
            repo_type="dataset",
            token=TOKEN,
        )
        print(f"  done in {time.time()-t0:.1f}s", flush=True)


def main():
    api = HfApi(token=TOKEN)
    who = api.whoami()
    print("auth as:", who.get("name"))

    # 1. responses (small)
    print("\n=== RESPONSES ===")
    upload_all(api, RESP_REPO, EXPORT, RESP_PATHS, "resp")

    # 2. activations (large)
    print("\n=== ACTIVATIONS ===")
    act_dir = os.path.join(EXPORT, MODEL_SLUG)
    upload_all(api, ACT_REPO, act_dir, ACT_PATHS, "act")

    print("\n=== UPLOAD COMPLETE ===")


if __name__ == "__main__":
    main()
