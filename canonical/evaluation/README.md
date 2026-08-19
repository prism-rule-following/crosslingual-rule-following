# Evaluation

Checker functions (regex/deterministic) and calibrated LLM judge for behavioural adherence scoring — Stage One §3.1.

## Adherence evalution

* `adherence_scoring.py` - deterministic checker functions and adherence evaluation script with either checker function, or LLM judge
* `llm_judge.py` - code for LLM as a judge. Logits if the model is available, sampling for API.

## Inference output — HF dataset format

Running `inference.py` generates two kinds of outputs and uploads them to HF
(when `push_to_hf: true`). Everything is organized per model and language:

* Model dir: `{model_id}` with `/` replaced by `__` (e.g. `meta-llama__Llama-3.1-8B-Instruct`).
* Language dir: the ISO code under `language_codes` (currently `en`).

### 1. Responses repo — `hf_result_repo` (e.g. `model-inference-responses`)

One parquet per (model, language): `{model_dir}/{lang}.parquet`.

Contains one row per generated sample: the source row fields (`id`, `system`,
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

