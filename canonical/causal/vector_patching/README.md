# vector_patching

Cross-lingual causal repair (Experiments 2-4): does copying a language that
followed a rule into a language that didn't, at one layer, flip adherence?

## Pipeline

| Stage | Module | Needs GPU? | Status |
|---|---|---|---|
| 0. Donor->recipient pairs | `pair_selection.py` | No | done, run on real data |
| 1. Donor vectors (dom) | `vectors.py` | No | done, run on real data |
| A. Feasibility pre-check | `feasibility.py` | No | done, run on real data |
| B. Patch/steer + generate | `intervene.py`, `run_sweep.py`, `run_stage_b.py` | **Yes** | done for `patch` mode, yo/ig recipients (2x RTX 5090, real Qwen3-8B) -- `steer` mode untested |
| Export | `export_responses.py`, `finalize_export.py` | No (runs right after B) | done, uploaded + verified (see "Where the data is" below) |

**Judging is out of scope here.** Stage B produces generations and uploads
them, labeled; scoring them is a separate, later pass (same as the rest of
the project's judge pipeline) run by whoever owns that.

## Where the data is

Dataset repo (private, HF org): **[crosslingual-rule-following/vector-patching-responses](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses)**

One parquet file per Stage B run, under `{model_slug}/`. Current run:

- **[qwen3-8b/exp2_yo_ig_20260902_021144.parquet](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses/blob/main/qwen3-8b/exp2_yo_ig_20260902_021144.parquet)**
  -- Qwen3-8B, recipients yo + ig, real GPU generations, 0 errors. See
  "Upload log" at the bottom of this file for the exact row/layer/donor
  breakdown.

You need an HF token with read access to the org (same one used everywhere
else in this project) -- it's a private repo, not public.

### Scope of the current upload (read before assuming coverage)

- **Recipients**: only `yo` and `ig` -- the two languages that genuinely
  collapse on Qwen3-8B (see `pair_selection.classify_tiers`). Not a full
  sweep of all 10 languages as recipients.
- **Donors**: the other 8 languages individually, plus a pooled
  `all_avg` direction (mean dom vector across all 8). **No `hr_avg`
  row exists** -- every one of those 8 donors is tier="high" for both
  recipients, so `hr_avg` would be numerically identical to `all_avg`;
  computing both was pure waste, so only `all_avg` was run.
- **Layers**: 24-31 (the feasibility-grid anchor band, peak at 28) plus
  layer 15 for both recipients (convergence point across three unrelated
  methods/models -- see progress log), plus one cheap extra probe layer
  per recipient (L12 for ig, L19-20 for yo) that Stage A ranked as
  likely-too-weak-to-work, kept anyway to see *how* it fails, not just
  whether.
- **Pairs**: subsampled to 25 (donor, recipient) id-pairs per donor per
  recipient, not the full pair table (thousands of rows) -- a first
  statistical sweep, not exhaustive.
- **Not in this run**: the full 10x9 donor/recipient matrix, `w`
  (trained-probe) vectors, `steer` mode (Exp 3), `same_lang_control`,
  Llama-3.1-8B. All supported by the schema/pipeline already, just not
  run yet.

### Loading it

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "crosslingual-rule-following/vector-patching-responses",
    "qwen3-8b/exp2_yo_ig_20260902_021144.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)
```

Or list everything that's been uploaded so far instead of hardcoding a
filename:

```python
from huggingface_hub import HfApi

files = HfApi().list_repo_files(
    "crosslingual-rule-following/vector-patching-responses", repo_type="dataset"
)
```

### Common filters, once loaded

```python
# Just the anchor-layer patches for one recipient
df[(df.language == "yo") & (df.patch_layer == 28)]

# One donor -> recipient pair across all its layers
df[(df.language == "ig") & (df.donor_language == "it")]

# Only the pooled all-language-average donor (donor_language is null here)
df[df.donor_kind == "all_avg"]

# Rows worth a second look before reading by hand: response drifted out
# of the recipient's language, or looks degenerate
df[(df.still_target_language == False) | (df.non_degenerate == False)]
```

**This is unjudged data.** `response` is the raw patched generation --
there's no compliance verdict in this file yet. `still_target_language`/
`non_degenerate` are cheap sanity flags only (see data dictionary below),
not a judge score. Scoring is a separate pass, same pipeline as the rest
of the project's judge-results data (same shared columns, see below).

## Data dictionary — exported responses

Shared columns match `judge-results-active-only`'s own
schema so a judging script written for that dataset needs minimal changes:

`id`, `model_id`, `language` (recipient), `category`, `topic`,
`grammar_type`, `pressure_level`, `pair_type`, `sample_idx`, `rule_clause`,
`user_query`, `response` (the patched/steered generation).

New columns, specific to this dataset:

| Column | Meaning |
|---|---|
| `donor_language` | Language whose activation supplied the patch/steer direction |
| `patch_layer` | Layer the hook was applied at (`blocks.{layer}.hook_resid_post`) |
| `vector_type` | `dom` (diff-of-means) or `w` (trained probe coefficient) |
| `donor_kind` | `single` / `hr_avg` / `all_avg` / `same_lang_control` |
| `patch_mode` | `patch` (swap, Exp 2) or `steer` (additive nudge, Exp 3) |
| `alpha` | Steer strength (steer mode only, else null) |
| `recipient_pre_verdict` | Did the recipient hold the rule *before* patching (always False by construction for Exp 2 pairs) |
| `feasibility_cohens_d` | Stage A's separation score for this (donor, recipient, layer, direction) |
| `still_target_language` | Cheap langdetect check, not a compliance verdict |
| `non_degenerate` | Cheap non-empty/non-repeating check, not a compliance verdict |

## Key design notes

- Patch/steer position: **last prompt token only**, matching how every
  cached activation and dom/probe vector in this project was computed.
  Patching a different position invalidates the donor vectors.
- Pairs are built at `pressure_level == "L0"` (clean/neutral) by default --
  matches the rest of the project's causal work. The full L0-L4 ladder is
  in the same judge data for a stretch pass.
- `judge-results-active-only` verdicts are `HELD`/`VIOLATED`/null; null
  rows (judge failure / content-filter block) are dropped, not counted as
  failures.
- Cached activations are fp16; residual-stream norms at deep layers
  overflow fp16 in dot products/norms, so every numeric op upcasts to
  float32 first (`vectors.py`, `feasibility.py`).
- Stage B's model loading (`run_sweep.load_model`) mirrors
  `canonical/evaluation/inference.py`'s exact recipe
  (`TransformerBridge.boot_transformers` + `enable_compatibility_mode` +
  the qkv/mlp cfg flags) so hook names match the rest of the project.
- Known perf caveat: Stage B generates via TransformerLens-compatible
  `model.hooks(...)` + `model.generate()`, not the raw
  `model.original_model.generate()` path `inference.py` uses for its
  (much larger, unpatched) response runs. If that's too slow at scale on
  the pod, swap in a raw PyTorch forward hook + `original_model.generate()`
  -- not done here since it can't be benchmarked without a GPU.

## Upload log

- **20260902_021144** (4725 total rows, 0 generation errors across both recipients, verified by read-back after upload)
  - HF: [qwen3-8b/exp2_yo_ig_20260902_021144.parquet](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses/blob/main/qwen3-8b/exp2_yo_ig_20260902_021144.parquet)
  - `ig`: 2250 rows (0 errors), layers [12, 15, 24, 25, 26, 27, 28, 29, 30, 31], donor_kind counts {'single': 2000, 'all_avg': 250}
  - `yo`: 2475 rows (0 errors), layers [15, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31], donor_kind counts {'single': 2200, 'all_avg': 275}
