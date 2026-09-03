# Vector patching responses

Cross-lingual causal repair experiments (Exp 2–4): does transplanting one
layer of a rule-following language's residual stream into a language that
didn't follow the rule flip adherence? Raw generations are published here;
semantic judging is a separate pass.

## The all-languages dataset (primary, 2026-09-03)

**HF file:**
[`qwen3-8b/exp2_all_langs_20260903_191445.parquet`](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses/blob/main/qwen3-8b/exp2_all_langs_20260903_191445.parquet)
in the private dataset `crosslingual-rule-following/vector-patching-responses`.

One clean all-language run: English donor activations patched into nine
recipient languages at five shared layers, with matched baseline, `dom`,
and `w` generations per prompt. **2,475 rows, 0 generation errors.**

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-8B` (non-quantized) |
| Donor | English (`en`) |
| Recipients | `de, hi, ig, it, ko, ru, tr, ur, yo` |
| Pressure | `L0` |
| Patch layers (shared, all recipients) | `15, 24, 27, 29, 31` |
| Vectors | `dom` = English difference-of-means; `w` = English LogisticRegression probe coefficients |
| Prompts | 25 frozen IDs, 5 per category × 5 categories, seed 0 |
| Generation | Greedy (`temperature=0`, `do_sample=false`), thinking disabled, `max_new_tokens=768`, `stop_at_eos=true` |
| Position | Final prompt token, `blocks.{layer}.hook_resid_post`, fired once |
| Baseline | Same-run unpatched generation (deterministic, comparable) |

Rows per language: `25 baseline + 125 dom (25 × 5 layers) + 125 w (25 × 5
layers) = 275`. The 25 prompt IDs are identical across all nine languages
(English-held, per the frozen manifest) so cross-language and cross-arm
comparisons are matched at the prompt level.

### The intervention

At the final prompt token of the recipient prompt, one residual-stream layer:

```text
x' = x - (x · u)u + c_donor u
```

where `u` is the unit `dom` or `w` direction and `c_donor` is the English
activation at the same layer projected onto `u` (the English model's real
coordinate for that prompt). This is activation patching, not weight
editing: the recipient's component along `u` is replaced by the donor's.

### Field dictionary (22 columns)

Prompt and content fields (shared schema with `judge-results-active-only`):

| Column | Type | Meaning |
|---|---|---|
| `id` | str | Canonical prompt ID; the matching key across conditions and languages (25 unique) |
| `model_id` | str | `Qwen/Qwen3-8B` |
| `language` | str | Recipient language: one of `de hi ig it ko ru tr ur yo` |
| `category` | str | Rule category: `ack_invert`, `mandatory_referral`, `no_dosage`, `refuse_with_reason`, `scope_lock` (5 prompts each) |
| `topic` | str | Scenario domain: `finance`, `legal`, `medical`, `mental_health` |
| `grammar_type` | str | Rule grammar variant: `imperative`, `modal_obligation`, `polite_asking` |
| `pressure_level` | str | `L0` (neutral pressure) |
| `pair_type` | str | Binary pair style: `active_cancelled`, `enabled_disabled`, `on_off`, `true_false`, `valid_invalid` |
| `sample_idx` | int | Prompt sample index; always `0` in this run |
| `rule_clause` | str | Rule text in the recipient language (what the model was told) |
| `user_query` | str | User request in the recipient language |
| `response` | str | Raw generated response (prompt stripped, special tokens removed); lossless |

Intervention fields:

| Column | Type | Meaning |
|---|---|---|
| `vector_type` | str | `none` (baseline), `dom` (diff-of-means), `w` (probe direction) |
| `donor_language` | str / null | `en` for patched rows; `null` for baseline |
| `patch_layer` | float / null | Patched layer (`15, 24, 27, 29, 31`); `null` for baseline |
| `donor_kind` | str | `baseline` or `single` (one English donor) |
| `patch_mode` | str | `none` (baseline) or `patch` (activation swap) |
| `alpha` | null | Always null — this run is patching, not steering |
| `recipient_pre_verdict` | bool | The recipient's own pre-run judge verdict for this prompt (collapsed 3-judge majority); **real metadata, not a post-hoc score** — `True` means the recipient already followed the rule before any intervention |
| `feasibility_cohens_d` | null | Always null — no Stage A feasibility score in this run |

Audit fields (cheap screens, **not** compliance verdicts):

| Column | Type | Meaning |
|---|---|---|
| `still_target_language` | bool / null | `langdetect` says the response is in the recipient language; `null` for `ig`/`yo` (unsupported by the detector) — do not treat null as "wrong language" |
| `non_degenerate` | bool | Non-empty and not a single repeated character; not a quality label |

### Usage guide

Load:

```python
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/crosslingual-rule-following/vector-patching-responses/"
    "qwen3-8b/exp2_all_langs_20260903_191445.parquet"
)
# or: hf_hub_download(repo_id, "qwen3-8b/exp2_all_langs_20260903_191445.parquet",
#                     repo_type="dataset") then pd.read_parquet(path)
```

Canonical filters:

```python
w_rows   = df[df.vector_type == "w"]
baseline = df[df.vector_type == "none"]
dom_rows = df[df.vector_type == "dom"]
de_w_l15 = df[(df.language == "de") & (df.vector_type == "w") & (df.patch_layer == 15)]
```

Compare conditions (matched at prompt level):

```python
piv = df.pivot_table(
    index=["id", "language"], columns="vector_type",
    values="response", aggfunc="first",
)
# piv["w"] vs piv["none"] and piv["dom"] vs piv["none"], per language/category
```

Do's and don'ts:

- **Do** compare `w`/`dom` against the same-run `baseline` — all three are
  deterministic greedy, so differences are intervention effects.
- **Do** analyze per prompt (paired by `id`), per language, per category,
  per layer, before pooling anything. Use bootstrap/permutation intervals
  clustered by prompt; do not treat the 5 layer rows as independent
  observations.
- **Do** check `recipient_pre_verdict` before claiming an intervention
  changed behavior — e.g. `ru` was already holding these rules (25/25),
  while `ig` (4/25) and `yo` (9/25) were mostly failing pre-run.
- **Don't** compare against the archived original response files: those
  used stochastic decoding (`n_samples=3`, `temperature=1.0`), so any
  difference from them is confounded by decoding, not just the patch.
- **Don't** treat `still_target_language` / `non_degenerate` as adherence.
  Known failure modes to screen for: `w` rows repeating/echoing the prompt,
  `dom` rows drifting to Chinese script or collapsing to short refusals.
  Rule adherence needs the separate judging pipeline.

### Known limitations

- No judge verdicts in this file — generations only. Run the judge pipeline
  with the same rubric across all nine languages and all three arms.
- Igbo/Yoruba responses have no `still_target_language` value (detector
  limitation) and `ig`/`yo` generations are long (often near the 768-token
  cap), so per-language timing differs — not a bug.
- The plan's audit section said "4,725 rows"; that figure is a stale copy
  from the earlier 10-layer Igbo/Yoruba run. The internally consistent
  design (5 shared layers) is 275 rows/language = 2,475 total, which is
  what this file contains and what the structural audit enforces.
- Frozen manifest (25 IDs, categories, verdict metadata):
  `canonical/causal/vector_patching/manifests/exp2_all_langs_20260903_163124.json`.

## Historical runs (do not merge with the above)

Earlier multi-donor `dom` sweeps on Igbo/Yoruba with recipient-specific
layers and a different selection policy:

| File | Rows | Contents |
|---|---:|---|
| [`qwen3-8b/exp2_yo_ig_20260902_021144.parquet`](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses/blob/main/qwen3-8b/exp2_yo_ig_20260902_021144.parquet) | 4,725 | 8 donor languages + `all_avg`, 10/11 layers, 200-token cap (~44% truncated), old selection (en-held ∩ recipient-failed) |
| [`qwen3-8b/exp2_w_yo_ig_20260903_011725.parquet`](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses/blob/main/qwen3-8b/exp2_w_yo_ig_20260903_011725.parquet) | 1,100 | English donor, matched baseline/`dom`/`w`, ig/yo only, 768 tokens, recipient-specific layers |

These were the runs that motivated the all-language design. The
all-languages file is the fair cross-language comparison: shared layers,
shared 25 prompts, same-run deterministic baselines.

## Upload log

- **20260903_191445** (2,475 rows, 0 generation errors, verified by
  read-back after upload): manifest `exp2_all_langs_20260903_163124`,
  9 languages × 275 rows (25 baseline + 125 `dom` + 125 `w`), layers
  [15, 24, 27, 29, 31], donor `en`.
- **20260903_011725** (1,100 rows, verified): `w` vs `dom`, ig/yo.
- **20260902_021144** (4,725 rows, verified): multi-donor `dom` sweep,
  ig/yo, 200-token cap.

## Pipeline (code)

| Stage | Module | Needs GPU? |
|---|---|---|
| 0. Donor→recipient pairs | `pair_selection.py` | No |
| 1. Donor vectors (dom) | `vectors.py` | No |
| A. Feasibility pre-check | `feasibility.py` | No |
| B. Patch/steer + generate | `intervene.py`, `run_exp2_all_langs.py` | Yes |
| Export | `finalize_export.py` (structural audit + upload) | No |

Key design notes: patch position is the last prompt token only (matching
how every cached activation and vector was computed); all numeric ops
upcast fp16→fp32 (deep-layer residual norms overflow fp16); model loading
mirrors `canonical/evaluation/inference.py`
(`TransformerBridge.boot_transformers` + `enable_compatibility_mode` + the
qkv/mlp cfg flags).