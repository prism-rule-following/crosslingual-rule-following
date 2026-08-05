# data

Rule generators, ACTIVE/REVOKED pair construction (STATUS-field contrastive pairs), per-language prompt sets, and the translation/HITL pipeline (IndicTrans2/NLLB, GEMBA-MQM, native-speaker spot-checks). See project doc §2.

## Files

- `full_dataset.json` — English base, the canonical source. `full_dataset_<lang>.json` (am, de, hi, ig, it, ko, ru, sw, ta, tr, ur, yo) — per-language translations, same row `id`s as the English base.
- `../model/dataset.py` (`canonical/model/dataset.py`) — `RuleRow`/`DatasetConfig` schema and `CrossLingualRuleFollowingDataset` loader.
- `script/` — the scripts below.

## Scripts

### `script/validate_datasets.py` — consistency validator

Runs the full schema/consistency check across all 13 `full_dataset*.json` files (identical contrastive pairs, missing `pair_type`, translation-introduced drift vs. the English base, invalid category values, missing/LLM-judge checkers, duplicate rows/ids, missing required fields). Produces a grouped, color-coded `.xlsx` workbook (one sheet per language + a SUMMARY sheet) for triage, plus an optional JSON report.

Run as a plain script (no special invocation needed):

```bash
python canonical/data/script/validate_datasets.py --data-dir canonical/data --out validation_report.xlsx
```

Optional flags:
- `--json-out validation_report.json` — also write a machine-readable JSON report.
- `--ci` — exit non-zero if any *blocking* issue is found (for CI gating).

Requires `openpyxl`. English base (`full_dataset.json`) must be present — it's the reference for the drift check.

### `script/hf_upload.py` — push a dataset file to the Hugging Face Hub

Uploads a single `full_dataset*.json` file to `<repo_id>/<path-in-repo>/<language_code>/full_dataset.json` on the Hub. Run once per language.

Must be invoked as a module from the **repo root** (not as a bare script path), since it imports via the full `canonical.*` path:

```bash
uv run python3 -m canonical.data.script.hf_upload \
  --data-file canonical/data/full_dataset.json \
  --repo-id crosslingual-rule-following/rule-following-pairs \
  --language-code en \
  --path-in-repo data
```

Requires an `HF_TOKEN` with write access to the target repo (loaded via `.env`/`load_dotenv()`), and the target dataset repo must already exist on the Hub — create it once via the website, or add `api.create_repo(repo_id=..., repo_type="dataset", exist_ok=True)` before the upload call if you want the script to create it automatically.
