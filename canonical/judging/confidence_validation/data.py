"""Pin public HF snapshots and validate response files without model loading."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .core import content_hash, normalize_row, response_key

EXPECTED_MODELS = ("Qwen__Qwen3-8B", "meta-llama__Llama-3.1-8B-Instruct")
EXPECTED_LANGUAGES = ("en", "de", "hi", "ig", "it", "ko", "ru", "tr", "ur", "yo")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid JSON in {path}:{n}") from exc
    return rows


def write_json(path, data):
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def validate_rows(rows):
    seen, normalized = set(), []
    for row in rows:
        row = normalize_row(row)
        key = response_key(row)
        if key in seen:
            raise ValueError(f"duplicate response identity {key}")
        seen.add(key)
        # Validate rubric coverage before ever sending rows to a provider.
        from .core import build_prompt
        try:
            build_prompt(row)
            row["confidence_judge_eligible"] = True
            row["confidence_exclusion_reason"] = None
        except ValueError:
            if row["checker"].get("checker_type") != "deterministic_function":
                raise
            row["confidence_judge_eligible"] = False
            row["confidence_exclusion_reason"] = "deterministic checker has no HELD/VIOLATED prose rubric"
        row["content_sha256"] = content_hash(row)
        normalized.append(row)
    if not normalized:
        raise ValueError("empty response dataset")
    return normalized


def prepare_snapshot(repo, revision, out, languages=None, models=None, subset=""):
    """Read-only remote access; local output is deliberately exclusive-create."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    out = Path(out)
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    api = HfApi()
    info = api.dataset_info(repo, revision=revision)
    sha = info.sha
    requested = [f"{subset + '/' if subset else ''}{m}/{lang}.parquet"
                 for m in models or EXPECTED_MODELS for lang in languages or EXPECTED_LANGUAGES]
    files = set(api.list_repo_files(repo, repo_type="dataset", revision=sha))
    found = sorted(set(requested) & files)
    missing = sorted(set(requested) - files)
    if not found:
        raise ValueError("no requested response files available at this snapshot")
    out.mkdir(parents=True)
    manifest = {"repo": repo, "requested_revision": revision, "revision": sha,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "files": [], "missing_files": missing, "status": "in_progress"}
    # Preserve source pointers even if an integrity gate fails later.
    write_json(out / "source.json", manifest)
    rows = []
    for filename in found:
        local = hf_hub_download(repo, filename, repo_type="dataset", revision=sha)
        table = pq.read_table(local)
        batch = table.to_pylist()
        expected_model, expected_language = Path(filename).parent.name, Path(filename).stem
        for row in batch:
            if str(row.get("model_id", "")).replace("/", "__") != expected_model or row.get("language") != expected_language:
                raise ValueError(f"row model/language mismatches source file {filename}")
            row["source_file"] = filename
            row["source_repo"] = repo
            row["source_revision"] = sha
        manifest["files"].append({"path": filename, "sha256": file_hash(local),
                                  "bytes": Path(local).stat().st_size, "rows": len(batch),
                                  "columns": table.column_names})
        rows.extend(batch)
    rows = validate_rows(rows)
    write_jsonl(out / "responses.jsonl", rows)
    manifest.update({"status": "validated", "n_rows": len(rows),
                     "responses_sha256": file_hash(out / "responses.jsonl"),
                     "sample_indices": dict(Counter(str(r["sample_idx"]) for r in rows)),
                     "counts_model_language": dict(Counter(f'{r["model_id"]}|{r["language"]}' for r in rows)),
                     "categories": dict(Counter(r.get("category", "unknown") for r in rows)),
                     "confidence_eligible_n": sum(r["confidence_judge_eligible"] for r in rows),
                     "confidence_excluded_categories": dict(Counter(r.get("category") for r in rows if not r["confidence_judge_eligible"])),
                     "empty_responses": sum(not r["response"].strip() for r in rows)})
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="crosslingual-rule-following/crv-model-inference-responses")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subset", default="")
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = prepare_snapshot(args.repo, args.revision, args.output_dir, args.languages, args.models, args.subset)
    print(json.dumps({k: result[k] for k in ("revision", "n_rows", "missing_files", "sample_indices")}, indent=2))


if __name__ == "__main__":
    main()
