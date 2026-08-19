"""Upload canonical run outputs to HF from the Mac.

Layout matches the production HFDataHelper conventions:
  model-inference-responses:
    {model__slug}/en.parquet
  model-inference-activations:
    {model__slug}/en/
      index.parquet
      hook_embed.fp16.npy
      hook_resid_post.fp16.npy
      hook_attn_out.fp16.npy
      hook_mlp_out.fp16.npy
      attn_q_input.fp16.npy
      attn_k_input.fp16.npy
      attn_v_input.fp16.npy
      hook_out.fp16.npy

Usage: upload_hf.py <model-slug> [export-dir]
  model-slug  e.g. meta-llama__Llama-3.1-8B-Instruct or Qwen__Qwen3-8B
  export-dir  local dir containing the export (defaults by model):
                .local/llama_export or .local/qwen_export
"""

import os
import sys
import time

from huggingface_hub import HfApi

ROOT = "/Users/ayesha/Projects/crosslingual-rule-following"
DEFAULT_EXPORTS = {
    "meta-llama__Llama-3.1-8B-Instruct": f"{ROOT}/.local/llama_export",
    "Qwen__Qwen3-8B": f"{ROOT}/.local/qwen_export",
}
RESP_REPO = "crosslingual-rule-following/model-inference-responses"
ACT_REPO = "crosslingual-rule-following/model-inference-activations"
TOKEN = os.environ["HF_TOKEN"]

ACT_GROUPS = [
    "hook_embed", "hook_resid_post", "hook_attn_out", "hook_mlp_out",
    "attn_q_input", "attn_k_input", "attn_v_input", "hook_out",
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
    model_slug = sys.argv[1]
    export = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_EXPORTS[model_slug]
    en_dir = f"{model_slug}/en"

    resp_paths = [
        (f"{export}/{model_slug}_en_responses.parquet", f"{model_slug}/en.parquet"),
    ]
    act_paths = [("index.parquet", f"{en_dir}/index.parquet")] + [
        (f"{g}.float16.npy", f"{en_dir}/{g}.fp16.npy") for g in ACT_GROUPS
    ]

    api = HfApi(token=TOKEN)
    who = api.whoami()
    print("auth as:", who.get("name"))
    print(f"model: {model_slug} | export: {export}")

    # 1. responses (small)
    print("\n=== RESPONSES ===")
    upload_all(api, RESP_REPO, export, resp_paths, "resp")

    # 2. activations (large)
    print("\n=== ACTIVATIONS ===")
    act_dir = os.path.join(export, model_slug)
    upload_all(api, ACT_REPO, act_dir, act_paths, "act")

    print("\n=== UPLOAD COMPLETE ===")


if __name__ == "__main__":
    main()
