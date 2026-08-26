"""Build an active-only, three-sample response export from raw JSONL checkpoints."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/workspace/crosslingual-rule-following")

from canonical.model.dataset import (  # noqa: E402
    ACTIVE_STATUSES,
    CrossLingualRuleFollowingDataset,
    DatasetConfig,
    DatasetLanguageCode,
    DatasetSource,
)

CHECKPOINT_DIRS = [Path("/workspace/inference_768"), Path("/workspace/inference_en768")]
EXPORT_ROOT = Path("/workspace/inference_export_active3")
N_SAMPLES = 3
SAMPLE_INDICES = {0, 1, 2}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: assemble_responses_export.py <model-slug> <lang>")

    model_slug, lang = sys.argv[1:]
    model_id = model_slug.replace("__", "/")
    lang_code = DatasetLanguageCode(lang)
    dataset_path = Path("/workspace/data") / lang / "test.jsonl"
    dataset = CrossLingualRuleFollowingDataset(
        DatasetConfig(
            url=str(dataset_path),
            source=DatasetSource.gh,
            validate_rows=True,
            strict=True,
        )
    ).df
    active_values = {status.value for status in ACTIVE_STATUSES}
    # The loader returns already-split rows, so use the split row's status and
    # id directly rather than the repeated source-level active_status field.
    active_ids = set(
        dataset.loc[dataset["rule_status"].isin(active_values), "id"].astype(str)
    )

    response_path = next(
        (
            directory / f"{model_slug}_{lang}_responses.jsonl"
            for directory in CHECKPOINT_DIRS
            if (directory / f"{model_slug}_{lang}_responses.jsonl").exists()
        ),
        None,
    )
    if response_path is None:
        raise FileNotFoundError(f"response checkpoint not found for {model_slug}/{lang}")

    raw_rows = [json.loads(line) for line in response_path.open() if line.strip()]
    rows = [
        row
        for row in raw_rows
        if row["rule_status"] in active_values
        and row["id"] in active_ids
        and int(row.get("sample_idx", 0)) in SAMPLE_INDICES
    ]
    by_id = {}
    for row in rows:
        by_id.setdefault(row["id"], []).append(row)
    if set(by_id) != active_ids:
        raise AssertionError(
            f"incomplete active export: expected {len(active_ids)} ids, "
            f"got {len(by_id)} ids / {len(rows)} rows"
        )
    # Some IDs may have more than 3 samples (e.g. a duplicate-run overlap). Keep
    # the first 3 rows per ID, deterministically in checkpoint order.
    rows = [
        row
        for group in by_id.values()
        for row in group[:N_SAMPLES]
    ]

    export_dir = EXPORT_ROOT / model_slug
    export_dir.mkdir(parents=True, exist_ok=True)
    output = export_dir / f"{lang}.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False)
    manifest = {
        "model": model_id,
        "language": lang,
        "source_checkpoint": str(response_path),
        "active_statuses": sorted(active_values),
        "sample_indices": sorted(SAMPLE_INDICES),
        "n_active_ids": len(active_ids),
        "n_response_rows": len(rows),
    }
    (export_dir / f"{lang}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
