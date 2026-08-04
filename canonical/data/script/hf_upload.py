import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from canonical.model.dataset import DatasetLanguageCode
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-file",
        required=True,
        type=Path,
        help="Path to a single full_dataset*.json file to upload",
    )
    ap.add_argument("--repo-id", required=True, help="Hugging Face repo id")
    ap.add_argument("--language-code", required=True, help="Language code")
    ap.add_argument(
        "--path-in-repo",
        required=True,
        help="Base destination path inside the repo (the language code and "
        "filename are appended automatically, e.g. 'data' -> 'data/de/full_dataset.json')",
    )

    args = ap.parse_args()

    if not args.data_file.is_file():
        raise ValueError(f"Path {args.data_file} does not exist or is not a file")

    try:
        DatasetLanguageCode(args.language_code)
    except ValueError:
        raise ValueError(
            f"Language code {args.language_code!r} not supported. "
            "Update the DatasetLanguageCode enum."
        )

    with open(args.data_file, encoding="utf-8") as f:
        data = json.load(f)

    if "pairs" not in data:
        raise ValueError(
            f"{args.data_file} has no top-level 'pairs' key - nothing to upload"
        )

    # Upload only the rows themselves, not the surrounding {"metadata": ...,
    # "pairs": [...]} wrapper - HF dataset loaders expect a bare list of
    # records, not an arbitrarily-nested structure.
    pairs_bytes = json.dumps(data["pairs"], ensure_ascii=False, indent=2).encode(
        "utf-8"
    )

    api = HfApi()

    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    # Upload a single file, named uniformly inside its own language folder -
    # the language is already encoded in the folder, so the filename itself
    # doesn't need a repeated language suffix.
    api.upload_file(
        path_or_fileobj=pairs_bytes,
        path_in_repo=f"{args.path_in_repo}/{args.language_code}/full_dataset.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
