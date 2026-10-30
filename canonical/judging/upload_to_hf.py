"""
Uploads a judge's final.jsonl to `hf_results_repo`, at
{hf_path_prefix}/{judge}/results.jsonl -- one folder per judge, same
pattern as the inference dataset's one folder per model.

SETUP: .env with
    HF_TOKEN=hf_xxx   (needs write access to the target org)

RUN:
    python -m canonical.judging.upload_to_hf --config canonical/judging/configs/judge_gpt_mini.json
"""

import os

from huggingface_hub import HfApi

from canonical.judging.common import get_secret
from canonical.judging.config import parse_config_arg

if __name__ == "__main__":
    config = parse_config_arg("Upload a judge's final.jsonl to HF.")
    if not config.hf_results_repo:
        raise SystemExit("hf_results_repo is not set in this config.")

    hf_token = get_secret("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN not set -- add it to .env (needs write access).")

    local_file = config.final_file
    if not os.path.exists(local_file):
        raise SystemExit(
            f"{local_file} not found -- run consolidate.py for this judge first."
        )

    api = HfApi(token=hf_token)
    # Creates the repo if it doesn't exist yet (harmless no-op if it does).
    api.create_repo(
        repo_id=config.hf_results_repo,
        repo_type="dataset",
        private=config.hf_private,
        exist_ok=True,
    )

    remote_path = config.hf_remote_path()
    print(f"Uploading {local_file} -> {config.hf_results_repo}/{remote_path} ...")
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=remote_path,
        repo_id=config.hf_results_repo,
        repo_type="dataset",
    )
    print(
        f"Done. https://huggingface.co/datasets/{config.hf_results_repo}/tree/main/{os.path.dirname(remote_path)}"
    )
