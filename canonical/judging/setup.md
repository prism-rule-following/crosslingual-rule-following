# Crosslingual Rule-Following Judge Pipeline

Judges model completions (Qwen3-8B, Llama-3.1-8B-Instruct) across 10 languages for
whether they followed a given system rule, using 3 independent judges (GPT-5.4-mini,
Gemini 3.7 Flash, DeepSeek-V4-Pro), then analyzes the results.

## Repo structure

```
canonical/judging/
  judge_gpt_mini.py           # Azure GPT-5.4-mini judge
  judge_deepseek.py           # Azure DeepSeek-V4-Pro judge
  judge_gemini.py             # Gemini via OpenRouter
  judge_gemini_vertex.py      # Gemini via direct Vertex AI (faster path)
  config.py                   # JudgeConfig + --config loader
  common.py                   # shared prompt, rubric, cross-checks, judging loop
  configs/                    # one JSON config per judge
  retry_failed_rows.py        # retries transient failures from a judge run
  consolidate.py              # merges an original run + retry pass
  merge_gemini_sources.py     # merges OpenRouter + Vertex Gemini sources
  dedupe_judge.py             # removes duplicate rows from a judge's output
  upload_to_hf.py             # pushes a judge's final.jsonl to HF

analysis/
  analysis_report.py          # single-judge HELD/language/model breakdown
  two_judge_agreement.py      # GPT vs Gemini agreement analysis
  three_judge_agreement.py    # full 3-judge consensus (majority + 2-judge fallback)
  metadata_adherence_analysis.py  # adherence by category/topic/grammar/pressure
  build_report.py             # generates the final PDF report

.env.example                  # copy to .env, fill in real keys, never commit .env
```

## Setup

### 1. Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install openai anthropic google-cloud-aiplatform huggingface_hub matplotlib reportlab pypdf
```

### 2. Credentials

```bash
cp .env.example .env
```

Fill in:
- **Azure** (`AZURE_GPT_*`, `AZURE_DEEPSEEK_*`): needed for the GPT-mini and
  DeepSeek judges.
- **OpenRouter** (`OPENROUTER_*`): needed for the Gemini judge (OpenRouter path).
  Two keys supported -- the scripts auto-switch on a 402 (credit exhaustion).
- **Vertex AI** (`VERTEX_*`): needed for the faster direct-Gemini path
  (`judge_gemini_vertex.py`), ~4x throughput vs. OpenRouter.
- **HF_TOKEN**: needed for `upload_to_hf.py`. Must be issued *after* you've been
  granted org membership on the target HF org, or uploads 403.

### 3. Config

Every judging script takes `--config <file>`, a JSON file validated by
`JudgeConfig` in `config.py` (same pattern as `canonical/evaluation/inference.py`'s
`--hyperparameter-file`). One file per judge lives in `configs/`:

| Field | Meaning |
|---|---|
| `judge` | `gpt_mini`, `deepseek`, `gemini` (OpenRouter) or `gemini_vertex` |
| `judge_model` | deployment / model name; overrides the provider's env var |
| `dataset_repo` | HF repo with inference responses -- the inference config's `hf_result_repo` |
| `dataset_subset` | folder inside `dataset_repo`; `""` = root `{model}/{lang}.parquet` written by `inference.py`, `"active_only_768_n3"` = the assembled export |
| `model_ids`, `language_codes` | which (model, language) parquets to judge -- same ids as the inference configs |
| `sample_indices` | judge only these `sample_idx` values; `null` = all |
| `results_dir` | local output root; each judge writes to `{results_dir}/{judge}/` |
| `max_workers`, `retry_max_workers`, `max_retries` | concurrency and rate-limit backoff |
| `skip_succeeded_from` | other results files whose successful rows are skipped (Vertex skipping OpenRouter rows) |
| `merge_sources` | files merged by `merge_gemini_sources.py` |
| `hf_results_repo`, `hf_path_prefix`, `hf_private` | upload target for `upload_to_hf.py` |

Secrets (API keys, endpoints, `HF_TOKEN`) stay in `.env`.

## Running the judging pipeline

Run from the repo root. Each judge is independent:

```bash
python -m canonical.judging.judge_gpt_mini      --config canonical/judging/configs/judge_gpt_mini.json
python -m canonical.judging.judge_deepseek      --config canonical/judging/configs/judge_deepseek.json
# Gemini: pick ONE path, or both then merge
python -m canonical.judging.judge_gemini        --config canonical/judging/configs/judge_gemini.json
python -m canonical.judging.judge_gemini_vertex --config canonical/judging/configs/judge_gemini_vertex.json
```

Retry failures (works for any judge), then merge original + retry into `final.jsonl`:

```bash
python -m canonical.judging.retry_failed_rows --config canonical/judging/configs/judge_<judge>.json
python -m canonical.judging.consolidate       --config canonical/judging/configs/judge_<judge>.json
```

If you ran Gemini via both paths, merge instead of consolidating (writes the gemini `final.jsonl`):

```bash
python -m canonical.judging.merge_gemini_sources --config canonical/judging/configs/judge_gemini.json
```

Remove duplicates if a run was interrupted/resumed:

```bash
python -m canonical.judging.dedupe_judge --config canonical/judging/configs/judge_<judge>.json
```

Push each judge's clean `final.jsonl` to HF:

```bash
python -m canonical.judging.upload_to_hf --config canonical/judging/configs/judge_<judge>.json
```

## Running the analysis

Each script takes one or more `final.jsonl` paths as arguments:

```bash
# Single judge
python analysis/analysis_report.py results/gpt_mini/final.jsonl

# Two-judge agreement
python analysis/two_judge_agreement.py results/gpt_mini/final.jsonl results/gemini/final.jsonl

# Full 3-judge consensus (the primary analysis)
python analysis/three_judge_agreement.py results/gpt_mini/final.jsonl results/gemini/final.jsonl results/deepseek/final.jsonl

# Adherence broken down by category/topic/grammar/pressure -- aggregate and per-model
python analysis/metadata_adherence_analysis.py results/gpt_mini/final.jsonl results/gemini/final.jsonl results/deepseek/final.jsonl
```

Then build the PDF report (reads the chart PNGs generated alongside the analysis
scripts -- run the analysis scripts first):

```bash
python analysis/build_report.py
```

## Known data caveats (carry these into any downstream analysis)

- DeepSeek has a ~3.4% permanent failure rate from Azure's own content-safety
  filter, concentrated on `L2`/`L3` adversarial-pressure prompts -- not a bug,
  handled via the 2-judge fallback in `three_judge_agreement.py`.
- 904 Gemini rows (0.64%) were judged with `gemini-2.5-flash` instead of
  `gemini-3.7-flash` due to an early config default -- labeled via `judge_model`,
  not silently mixed in.
- 412 rows (0.29%) have no resolvable consensus across all 3 judges -- excluded
  from consensus HELD/VIOLATED percentages, not forced into a guess.