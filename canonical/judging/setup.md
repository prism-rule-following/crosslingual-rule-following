# Crosslingual Rule-Following Judge Pipeline

Judges model completions (Qwen3-8B, Llama-3.1-8B-Instruct) across 10 languages for
whether they followed a given system rule, using 3 independent judges (GPT-5.4-mini,
Gemini 3.7 Flash, DeepSeek-V4-Pro), then analyzes the results.

## Repo structure

```
judging/
  judge_gpt_mini.py           # Azure GPT-5.4-mini judge
  judge_deepseek.py           # Azure DeepSeek-V4-Pro judge
  judge_gemini.py             # Gemini via OpenRouter
  judge_gemini_vertex.py      # Gemini via direct Vertex AI (faster path)
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

### 3. Dataset

Judging scripts pull from the HF dataset `crosslingual-rule-following/model-inference-responses`,
subset `active_only_768_n3`. No local download step needed -- each judge script
fetches it directly.

## Running the judging pipeline

Run each judge independently (they don't depend on each other):

```bash
python judging/judge_gpt_mini.py
python judging/judge_deepseek.py
# Gemini: pick ONE path, or both then merge
python judging/judge_gemini.py            # OpenRouter -- slower, works everywhere
python judging/judge_gemini_vertex.py     # Vertex AI -- faster, needs GCP access
```

If any judge run has failures, retry them:

```bash
python judging/retry_failed_rows.py
python judging/consolidate.py             # merges original + retry -> final.jsonl
```

If you ran Gemini via both paths:

```bash
python judging/merge_gemini_sources.py    # -> results/gemini/final.jsonl
```

Check for and remove duplicates if a run was interrupted/resumed:

```bash
python judging/dedupe_judge.py <gpt_mini|deepseek|gemini>
```

Push results to HF once each judge's `final.jsonl` is clean:

```bash
python judging/upload_to_hf.py <gpt_mini|deepseek|gemini>
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
