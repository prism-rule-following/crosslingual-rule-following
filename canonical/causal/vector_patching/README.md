# Vector patching

Causal activation-patching experiments for cross-lingual rule following. Stage B produces raw model generations; semantic compliance is judged separately.

## Published data

The files are in the private Hugging Face repository [`crosslingual-rule-following/vector-patching-responses`](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses).

| File | Run | Rows | Contents |
|---|---|---:|---|
| [`qwen3-8b/exp2_w_yo_ig_20260903_011725.parquet`](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses/blob/main/qwen3-8b/exp2_w_yo_ig_20260903_011725.parquet) | Qwen `w` vs `dom` | 1,100 | Matched English donor, Igbo/Yoruba recipients, baseline + `dom` + `w` |
| [`qwen3-8b/exp2_yo_ig_20260902_021144.parquet`](https://huggingface.co/datasets/crosslingual-rule-following/vector-patching-responses/blob/main/qwen3-8b/exp2_yo_ig_20260902_021144.parquet) | Earlier Qwen `dom` run | 4,725 | Multi-donor `dom` generations; broader and not a matched control for the latest file |

The latest file is the primary dataset for the English-donor `w` versus `dom` comparison. It was verified after upload: 525 Igbo rows, 575 Yoruba rows, and no recorded generation errors.

## Latest run: Qwen `w` vs `dom`

The experiment asks whether two English-derived directions cause similar recipient-language behavior when patched into Qwen activations:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Donor | English (`en`) |
| Recipients | Igbo (`ig`), Yoruba (`yo`) |
| Pressure | `L0` |
| Seed | `0` |
| Layers | Igbo: `12, 15, 24–31`; Yoruba: `15, 19, 20, 24–31` |
| Generation | Greedy, Qwen chat template, thinking disabled, `max_new_tokens=768` |
| GPUs | Two GPUs in parallel: Igbo on `cuda:0`, Yoruba on `cuda:1` |

Each recipient has 25 matched prompt IDs:

| Recipient | Baseline | `dom` | `w` | Total |
|---|---:|---:|---:|---:|
| Igbo | 25 | 250 (25 × 10 layers) | 250 (25 × 10 layers) | 525 |
| Yoruba | 25 | 275 (25 × 11 layers) | 275 (25 × 11 layers) | 575 |

### Prompt selection

The 25 IDs per recipient are a reproducible pilot sample, not the full canonical dataset:

1. Collapse the three judge-result streams to one `HELD`/not-held verdict per model, language, canonical ID, category, and pressure level; null verdicts are dropped and ties count as not held.
2. Keep IDs where English is held and the recipient language is not held for the same canonical rule/query pair.
3. Sample 25 eligible IDs without replacement with `numpy.random.default_rng(0)`.
4. Reuse the same IDs for baseline, `dom`, `w`, and every tested layer. There is no category balancing or extra low-confidence exclusion.

The resulting category mix is not balanced: Igbo = 3 `scope_lock`, 4 `no_dosage`, 5 `ack_invert`, 6 `mandatory_referral`, 7 `refuse_with_reason`; Yoruba = 1, 1, 6, 9, and 8 respectively.

### Intervention labels

| `vector_type` | `donor_kind` | `patch_mode` | Meaning |
|---|---|---|---|
| `none` | `baseline` | `none` | Unpatched generation |
| `dom` | `single` | `patch` | English difference-of-means direction |
| `w` | `single` | `patch` | English LogisticRegression probe-coefficient direction |

For patched rows, the hook operates at `blocks.{layer}.hook_resid_post` on the final prompt token, once during generation. The recipient activation is rotated onto the selected direction and given the English donor coordinate for the same canonical ID and layer:

```text
x' = x - (x · u)u + c_donor u
```

Here `u` is the `dom` or `w` direction and `c_donor` is the English donor activation projected onto `u`. This is activation patching, not model-weight editing.

## Dataset schema

The latest Qwen `w` vs `dom` file has 25 columns.

### Prompt and generation fields

| Field | Values / type | Meaning |
|---|---|---|
| `id` | string | Canonical prompt ID; the matching key across conditions |
| `model_id` | string | `Qwen/Qwen3-8B` |
| `language` | `ig`, `yo` | Recipient language |
| `category` | `scope_lock`, `no_dosage`, `ack_invert`, `mandatory_referral`, `refuse_with_reason` | Rule category |
| `topic` | string | Scenario topic |
| `grammar_type` | string | Rule/query grammar variant |
| `pressure_level` | `L0` | User-pressure condition |
| `pair_type` | `valid_invalid`, `true_false`, `active_cancelled`, `enabled_disabled`, `on_off` | Binary option/pair used by the prompt |
| `sample_idx` | integer | Prompt sample index; `0` in this run |
| `rule_clause` | string | Translated rule clause shown to the model |
| `user_query` | string | Translated user request |
| `response` | string | Raw generated response |

### Intervention and audit fields

| Field | Values / type | Meaning |
|---|---|---|
| `donor_language` | `en` or null | English donor for patched rows; null for baseline |
| `patch_layer` | integer or null | Patched transformer layer; null for baseline |
| `vector_type` | `none`, `dom`, `w` | Experimental condition |
| `donor_kind` | `baseline`, `single` | No donor for baseline; one English donor for patched rows |
| `patch_mode` | `none`, `patch` | Whether activation patching was applied |
| `alpha` | nullable numeric | Steering strength; null in this file because this is patching |
| `recipient_pre_verdict` | boolean | Recipient’s pre-run judge verdict; false by construction for selected IDs, not an output score |
| `feasibility_cohens_d` | nullable numeric | Optional Stage A geometry score; null in this file |
| `still_target_language` | nullable boolean | Language-detection audit field; null here because the detector does not support Igbo/Yoruba reliably |
| `non_degenerate` | boolean | Cheap non-empty/repetition-insensitive sanity check; not a quality or compliance label |
| `error` | nullable string | Generation error, if any |
| `max_new_tokens` | integer | Generation cap: `768` |
| `gen_time_s` | numeric | Generation wall-clock time in seconds |

### Row interpretation

To select the `w` rows for Yoruba at layer 28:

```python
rows[(rows.language == "yo") &
     (rows.vector_type == "w") &
     (rows.patch_layer == 28)]
```

To compare all conditions for one matched prompt:

```python
rows[rows.id == prompt_id].sort_values(["language", "vector_type", "patch_layer"])
```

The same prompt ID can therefore have one baseline row and one row per tested layer for each patched condition.

## Scope and caveats

- This is a 25-prompt-per-recipient pilot, not an exhaustive run over the roughly 2,250 canonical prompts available per language.
- The latest file contains raw generations, not final compliance judgments. `recipient_pre_verdict` describes prompt selection only.
- `w` is a probe direction trained on English activations; `dom` is an English difference-of-means direction. Neither label means the response is correct.
- The earlier `dom` file uses a different multi-donor, broader run design. Use the latest file for the clean matched English-donor comparison.
- `still_target_language` and `non_degenerate` should not be treated as reliable semantic-quality labels for this low-resource-language run.

## Code map

| Purpose | File |
|---|---|
| Select matched donor/recipient prompts | [`pair_selection.py`](pair_selection.py) |
| Build English probe (`w`) vectors | [`probe_vectors.py`](probe_vectors.py) |
| Run Qwen `w` Stage B generation | [`run_qwen_w_stage_b.py`](run_qwen_w_stage_b.py) |
| Run activation intervention | [`intervene.py`](intervene.py) |
| Export and validate Parquet | [`export_responses.py`](export_responses.py), [`finalize_export.py`](finalize_export.py) |
