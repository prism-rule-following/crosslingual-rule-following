"""Upload one response parquet under a versioned path without overwriting old data."""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "crosslingual-rule-following/model-inference-responses"
PREFIX = "active_only_768_n3"


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: upload_responses_hf.py <model-slug> <lang> <parquet>")
    model_slug, lang, parquet = sys.argv[1:]
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo_id=REPO, repo_type="dataset", token=token, exist_ok=True)
    remote_dir = f"{PREFIX}/{model_slug}"
    for local_path, remote_name in [
        (Path(parquet), f"{remote_dir}/{lang}.parquet"),
        (Path(parquet).with_suffix(".manifest.json"), f"{remote_dir}/{lang}.manifest.json"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=REPO,
            repo_type="dataset",
            token=token,
        )
        print(f"uploaded {remote_name}")


if __name__ == "__main__":
    main()
