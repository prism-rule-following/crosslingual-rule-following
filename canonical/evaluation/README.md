# Evaluation

Checker functions (regex/deterministic) and calibrated LLM judge for behavioural adherence scoring — Stage One §3.1.

## Adherence evalution

* `adherence_scoring.py` - deterministic checker functions and adherence evaluation script with either checker function, or LLM judge
* `llm_judge.py` - code for LLM as a judge. Logits if the model is available, sampling for API.

## Inference output — HF dataset format

Running `inference.py` generates responses (and optionally activations) and
uploads them to HF (when `push_to_hf: true`). Everything is organized per model
and language:

* Model dir: `{model_id}` with `/` replaced by `__` (e.g. `meta-llama__Llama-3.1-8B-Instruct`).
* Language: ISO code (`en`, `de`, `hi`, `ig`, `it`, `ko`, `ru`, `tr`, `ur`, `yo`).

### 1. Responses repo — `hf_result_repo` (e.g. `model-inference-responses`)

The repo holds three top-level folders plus the legacy root layout:

| Folder | Contents |
|---|---|
| `active_only_768_n3/{model}/{lang}.parquet` | **Analysis-ready.** Active-side rows only, `sample_idx ∈ {0,1,2}`. 2,340 active IDs × 3 samples = 7,020 rows per (model, lang), + `{lang}.manifest.json`. |
| `raw_768_full/{model}/{lang}.parquet` | **Full 768-token era originals.** Unfiltered: all 10 samples, active + revoked statuses, + manifest (row counts, sample/status distribution). |
| `raw_50token_original/{model}/{lang}.parquet` | Original 50-token-era responses (superseded, preserved for reference). |
| `{model}/{lang}.parquet` (root) | Legacy original uploads. Do not overwrite. |

Each response row contains the source row fields (`id`, `system`,
`user_query`, `category`, `topic`, `grammar_type`, `pair_type`, `rule_status`,
`checker`, `pressure_*`, ...) plus `model_id`, `response` (generated text) and
`sample_idx` (0..n_samples-1). Rows are fan-out: `n_dataset_rows × n_samples`.

### 2. Activations repo — `hf_activations_repo` (e.g. `model-inference-activations`)

One directory per (model, language): `{model_dir}/{lang}/` containing:

* `index.parquet` — labels + join key, one row per dataset row (no sample
  fan-out; activations are cached once per row, deterministically, in dataset
  order). Columns: `row_idx`, `id`, `rule_status`, `grammar_type`, `category`,
  `topic`, `pair_type`, `pressure_level`, `pressure_name`, `language`.
* `*.fp16.npy` — one fp16 array per activation hook group. Row `i` of every
  array corresponds to `index.parquet` row `row_idx == i`.
  * `hook_embed` — (n_rows, d_model), embedding at the last (decision) token.
  * `hook_resid_post`, `hook_attn_out`, `hook_mlp_out`, `attn_q_input`,
    `attn_k_input`, `attn_v_input`, `hook_out` — (n_rows, n_layers, d_model),
    per-layer activations at the last token.

### Joining activations to rules / prompts

* `rule_status` is the probe label: active-family (`active`, `on`, `true`,
  `valid`, `enabled`) = rule binds; revoked-family (`cancelled`, `off`,
  `false`, `invalid`, `disabled`) = rule lifted.
* `id` joins to the responses parquet (same id) for the full `system` /
  `user_query` / `response` / `checker` text. Each contrastive pair is stored
  as adjacent `{id}_clean` (rule) / `{id}_revoked` (non-rule) rows.

## Config / CLI (768-token era)

Hyperparameter JSONs live in `canonical/evaluation/`:

* `hyperparameter_768_{llama,qwen}_{lang}.json` — canonical per-language configs (batch 30).
* `hyperparameter_en768_{llama,qwen}.json` — English runs (checkpoints under `inference_en768`).
* `oldpod_768/` — A100 variants (generation batch 90). `l4pod_768/` — L4 variants (batch 45).

Key settings: `max_new_tokens: 768` (was 50 — truncated outputs confound the
LLM-judge coherence gate), `temperature: 1.0`, `n_samples: 10` (analysis uses
samples 0–2), `run_inference_activations: false` (activations are
prompt-only and unaffected by generation length).

Runtime overrides (CLI flags on `inference.py`):

* `--active-only` — generate only rows whose `rule_status` is in `ACTIVE_STATUSES`
  (`active`, `on`, `true`, `valid`, `enabled`).
* `--n-samples N` — override the sample count (3 used for the active-only redo).
* `--generation-batch-size N` — override the config batch size.

`assemble_responses_export.py` builds the `active_only_768_n3` parquets from raw
checkpoints; `assemble_raw_export.py` builds the unfiltered `raw_768_full` /
`raw_50token_original` parquets; `upload_responses_hf.py` uploads them under
versioned folder prefixes without overwriting existing files. `auto_lane.sh`
is the per-GPU serial queue supervisor used for batch runs on pods.