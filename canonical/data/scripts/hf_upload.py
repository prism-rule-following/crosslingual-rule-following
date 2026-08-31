import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from canonical.model.dataset import DatasetLanguageCode, RulePair
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
        "filename are appended automatically, e.g. 'data' -> 'data')",
    )
    ap.add_argument(
        "--split",
        default="test",
        help="Dataset split name (e.g. train, test, validation).",
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
    if "metadata" not in data:
        raise ValueError(
            f"{args.data_file} has no top-level 'metadata' key - nothing to upload"
        )

    pairs = [RulePair.model_validate(d) for d in data["pairs"]]
    pairs_bytes = "\n".join(
        json.dumps(p.model_dump(mode="json"), ensure_ascii=False) for p in pairs
    ).encode("utf-8")

    api = HfApi()

    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    base_path = f"{args.path_in_repo}/{args.language_code}"
    api.upload_file(
        path_or_fileobj=pairs_bytes,
        path_in_repo=f"{base_path}/{args.split}.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
