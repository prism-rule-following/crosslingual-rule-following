# vector_patching

Cross-lingual causal repair (Experiments 2-4): does copying a language that
followed a rule into a language that didn't, at one layer, flip adherence?

## Pipeline

| Stage | Module | Needs GPU? | Status |
|---|---|---|---|
| 0. Donor->recipient pairs | `pair_selection.py` | No | done, run on real data |
| 1. Donor vectors (dom) | `vectors.py` | No | done, run on real data |
| A. Feasibility pre-check | `feasibility.py` | No | done, run on real data |
| B. Patch/steer + generate | `intervene.py`, `run_sweep.py` | **Yes** | written, CPU-smoke-tested only |
| Export | `export_responses.py` | No (runs right after B) | written, unit-tested |

**Judging is out of scope here.** Stage B produces generations and uploads
them, labeled; scoring them is a separate, later pass (same as the rest of
the project's judge pipeline) run by whoever owns that.

## Data dictionary — exported responses

Uploaded to `crosslingual-rule-following/vector-patching-responses` (one
parquet per run). Shared columns match `judge-results-active-only`'s own
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
