# Obligation direction extraction (difference-in-means)

Extracts obligation directions from Qwen3-8B and Llama-3.1-8B by contrasting
minimal-pair system rules that differ by a single modal token.

- **must_text (clean) - may_text (corrupt)**   -> obligation vs permission
- **must_text (clean) - neutral_text (corrupt)** -> obligation vs descriptive-norm

Method follows Arditi et al. 2024 (diff-in-means over the residual stream, per
layer), extended with three token-position anchors and causal selection.

## Positions
| anchor | what it reads | aggregation |
|---|---|---|
| `contrast_token`   | the swapped token (`mandatory`/`optional`/`customary` ...) | **per-frame DIM, then averaged** (avoids cross-frame index misalignment) |
| `rule_clause_end`  | last token of the system rule clause | pooled |
| `post_instruction` | last prompt token before generation (Arditi's anchor) | pooled |

Positions are located by **token-id sub-span matching**, so indexing is correct
for both tokenizers despite different word-boundary handling.

## Files
- `hyperparameters.json` - all model/data/extraction/selection config
- `extract_dim.py` - caches resid_post, builds per-layer DIMs for both contrasts
  at all three positions, computes per-layer Cohen's d separation, runs the
  adj->modal transfer check. Outputs `dim_report_<model>.json` +
  `dim_candidates_<model>.pt`.
- `intervene_select.py` - ablate / add / KL causal selection of (l*, i*) over the
  candidate directions. Outputs `intervention_report_<model>.json`.

## Run
```bash
pip install -r requirements.txt
# put obligation_full.json next to the scripts (or pass --data)

python extract_dim.py      --model qwen3-8b    --config hyperparameters.json --data obligation_full.json
python extract_dim.py      --model llama3.1-8b --config hyperparameters.json --data obligation_full.json

python intervene_select.py --model qwen3-8b    --config hyperparameters.json --data obligation_full.json
python intervene_select.py --model llama3.1-8b --config hyperparameters.json --data obligation_full.json
```
Add `--limit 8` for a quick smoke test.

## Notes / knobs
- **Prompt construction**: system = `"<context> Rule: <must/may/neutral_text>"`,
  user = the constant `user_query` (L0). The obligation manipulation lives in the
  system rule; the query is held fixed, so any activation difference is
  attributable to the rule.
- **Qwen3 thinking**: `enable_thinking:false` in config keeps the template
  non-reasoning so the post-instruction anchor is well-defined.
- **normalize_directions**: unit-norm the saved DIMs (default true). Raw norms
  are always saved per layer in the report.
- **Selection scoring** (`selection.intervention.comply_words` / `permit_words`):
  the target-token lists are a starting point. For a rigorous behaviour score,
  replace with a held-out judged generation eval; the hooks (ablate/add) are the
  reusable part.
- **Transfer check**: compares the DIM built from adjective frames
  (`adj_mandatory`,`adj_obligatory`) against modal frames
  (`modal_core`,`modal_have`) at `contrast_token`. High cosine => the direction
  encodes obligation, not the surface string "must".

## HF + checkpointing + optimizations (added)

### Repos (auto-created, private by default)
- dataset in:  `nunaa/canonical_obligation_dataset`
- activations:  `nunaa/crosslingual_rf-activations`  (row cache; off by default)
- directions:   `nunaa/crosslingual_rf-directions`   (`dim_candidates_<model>.pt`)
- results:      `nunaa/crosslingual_rf-results`       (`dim_report_<model>.json`)

Set `export HF_TOKEN=hf_...` first. All repo names live in `hyperparameters.json -> hf`.

### Run (pulls dataset from hub, pushes artifacts)
```bash
python extract_dim.py --model qwen3-8b    --config hyperparameters.json
python extract_dim.py --model llama3.1-8b --config hyperparameters.json
```
`--no-push` to skip upload, `--data file.json` to use a local dataset, `--limit N` to smoke-test.

Manual HF ops:
```bash
python hf_io.py pull-dataset --out obligation_full.json
python hf_io.py push --kind directions --model qwen3-8b --path dim_out/dim_candidates_qwen3-8b.pt
```

### Checkpointing (`hyperparameters.json -> checkpoint`)
Row-level, Drive-backed. Each row's activations (all members x positions) are written
atomically to `<drive_dir>/<model>/row_activations/<row_id>.pt` as soon as it's computed.
On restart the run skips any row already on disk, so a crash resumes mid-extraction with
no recompute. Point `drive_dir` at your mounted Drive (Colab:
`/content/drive/MyDrive/...`). The DIM math reloads from the row cache, so extraction and
analysis are decoupled.

### Optimizations (`hyperparameters.json -> optim`)
- batched forward (`batch_size`), bf16, `use_cache=false`
- `attn_implementation`: `sdpa` (default) or `flash_attention_2` (falls back to sdpa if unavailable)
- `torch_compile` with `compile_mode`
- right-padding + per-prompt unpadding so batching never corrupts position indices
- `empty_cache_every_n_batches` clears CUDA cache + gc; model is freed before the CPU DIM math
- hooks are only used in `intervene_select.py`, and only on needed layers

## Two experiments, one pipeline (stimulus_mode + presets)

The concept experiment and the rule-following experiment differ only in how the
stimulus reaches the model. Select with `--preset`:

| preset | stimulus_mode | prompt | positions | answers |
|---|---|---|---|---|
| `rule_following` | `system_user` | system=`<context> Rule: <rule>`, user=`<query>` | contrast_token, rule_clause_end, post_instruction | does the model *follow* the rule? |
| `concept` | `user_only` | single user turn = the rule sentence | contrast_token, sentence_end | does the model *represent* obligation? |
| `concept_raw` | `raw_sentence` | raw sentence, **no chat template** | contrast_token, sentence_end | concept, stripped of all chat framing |

```bash
python extract_dim.py --model qwen3-8b    --preset rule_following
python extract_dim.py --model qwen3-8b    --preset concept
python extract_dim.py --model qwen3-8b    --preset concept_raw
```

Why three: `concept_raw` -> `concept` -> `rule_following` is a ladder of added
framing. If the obligation direction survives all three, it's a concept, not an
artifact of one prompt format. `must_may` is the headline contrast
(obligation vs permission); `must_neutral`, and any `should`/`can` you add later,
are a *different deontic axis* -- report them separately, don't average them in.

**Guardrails (fail loud):**
- `post_instruction` is rejected unless `stimulus_mode=system_user` (it only reads
  chat scaffolding otherwise).
- unknown `stimulus_mode` is rejected.
- the transfer check auto-skips when the dataset has <2 frames (expected for the
  concept/cross-lingual stimuli, which have no adj/modal frame structure).

**Checkpoint / output namespacing:** artifacts and the Drive row cache are keyed by
`<model>__<preset>`, so the three experiments never overwrite each other and each
resumes independently.

### Notes for the cross-lingual (Igbo/Yoruba) concept run
- Use `concept` or `concept_raw`. Provide a dataset whose rows carry the deontic
  sentence per language with `must_token`/`may_token` set to the language's modal
  marker so `contrast_token` locates correctly (token-id matching is language-agnostic).
- The transfer check is frame-based and will skip; the cross-lingual patch
  (English obligation dir -> Yoruba/Igbo permission run, with within-language
  positive controls) lives in the patching script, not here.

## Per-language, per-concept runs (required flags)

Every run is one homogeneous dataset for one `(concept, language)`, passed as flags.
`--language` and `--concept` are **required** on both scripts. This keeps runs from
ever overwriting each other as you extend to new languages and concept types.

```bash
python extract_dim.py --model qwen3-8b --preset concept --concept obligation --language en
python extract_dim.py --model qwen3-8b --preset concept --concept obligation --language yoruba
python extract_dim.py --model qwen3-8b --preset concept --concept obligation --language igbo
python extract_dim.py --model qwen3-8b --preset concept --concept negation   --language yoruba
```

### Naming (identical everywhere)
`run_tag = <model>__<preset>` and `group = <concept>/<language>/<run_tag>`.

| artifact | path |
|---|---|
| local directions | `dim_out/<concept>/<language>/<model>__<preset>/dim_candidates.pt` |
| local report | `dim_out/<concept>/<language>/<model>__<preset>/dim_report.json` |
| intervention report | `dim_out/<concept>/<language>/<model>__<preset>/intervention_report.json` |
| Drive row cache | `<drive_dir>/<concept>__<language>__<model>__<preset>/row_activations/` |
| HF (each of the 3 repos) | `<concept>/<language>/<model>__<preset>/...` |

So `nunaa/crosslingual_rf-directions` grows as:
```
obligation/en/qwen3-8b__concept/dim_candidates.pt
obligation/yoruba/qwen3-8b__concept/dim_candidates.pt
obligation/igbo/qwen3-8b__concept/dim_candidates.pt
negation/yoruba/qwen3-8b__concept/dim_candidates.pt
...
```

### Dataset resolution (`hf.dataset_files`)
When pulling from the hub, the file inside `nunaa/canonical_obligation_dataset` is
resolved by, in order: an explicit `hf.dataset_files["<concept>/<language>"]` entry,
then `"<concept>_<language>.json"` at repo root, then `"<concept>/<language>.json"`,
then a warned fallback to the first json. Add new languages/concepts to
`hf.dataset_files` as you create them, or just name the files `<concept>_<language>.json`.

Local runs skip the hub: pass `--data <concept>_<language>.json` (or set
`hf.dataset_load: local`).

### Intervention (same flags)
```bash
python intervene_select.py --model qwen3-8b --preset rule_following \
    --concept obligation --language en
```
It loads `dim_candidates.pt` from the matching group dir and writes
`intervention_report.json` beside it. Push it with:
```bash
python hf_io.py push --kind results \
    --group-path obligation/en/qwen3-8b__rule_following \
    --path dim_out/obligation/en/qwen3-8b__rule_following/intervention_report.json
```
