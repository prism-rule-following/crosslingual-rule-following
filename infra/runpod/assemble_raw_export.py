"""Build an unfiltered raw export (all samples, active+revoked) from a
response JSONL checkpoint. Used to persist the full originals to HF before
workspace cleanup."""

import json
import sys
from pathlib import Path

import pandas as pd

EXPORT_ROOT = Path("/workspace/inference_export_raw")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: assemble_raw_export.py <model-slug> <lang>")
    model_slug, lang = sys.argv[1:]
    model_id = model_slug.replace("__", "/")

    candidates = [
        Path("/workspace/inference_768") / f"{model_slug}_{lang}_responses.jsonl",
        Path("/workspace/inference_en768") / f"{model_slug}_{lang}_responses.jsonl",
        Path("/workspace/inference") / f"{model_slug}_{lang}_responses.jsonl",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"no checkpoint for {model_slug}/{lang}")

    rows = [json.loads(line) for line in src.open() if line.strip()]
    if not rows:
        raise SystemExit(f"empty checkpoint: {src}")

    export_dir = EXPORT_ROOT / model_slug
    export_dir.mkdir(parents=True, exist_ok=True)
    output = export_dir / f"{lang}.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False)

    sample_counts = {}
    for r in rows:
        key = int(r.get("sample_idx", 0))
        sample_counts[key] = sample_counts.get(key, 0) + 1
    status_counts = {}
    for r in rows:
        s = str(r.get("rule_status"))
        status_counts[s] = status_counts.get(s, 0) + 1

    manifest = {
        "model": model_id,
        "language": lang,
        "source_checkpoint": str(src),
        "n_rows": len(rows),
        "n_unique_ids": len({r["id"] for r in rows}),
        "samples_per_idx": sample_counts,
        "status_counts": status_counts,
        "unfiltered": True,
    }
    (export_dir / f"{lang}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()