# Log-probability judging and human validation

CPU-only workflow for Ayesha's camera-ready validation task. This package is
independent of the historical judge scripts, which initialize credentials and
download datasets at import time on current `main`. Importing this package or
running offline tests does not initialize clients or make judge calls.

## What is ready / what still needs keys

### Live pilot results — 22 October 2026

The first live capability pilot has now run. Azure GPT returned usable scoring
probabilities with **top_k=5** (20 is rejected by this deployment). Azure
DeepSeek returned usable probabilities with top_k=20; its terminal EOS token
is now explicitly handled and covered by a regression test. The same Azure
resource/key serves both deployments. `config_verified.json` records this
tested two-judge setup, with a 512-token output budget and per-judge top-k.
These are single-example capability checks, not measured judge accuracy.

The configured **Gemini 3.7 Flash currently cannot supply logprobs on either
tested route**: Vertex returns `Logprobs is not supported for this model`;
OpenRouter rejects parameter-support routing, and none of its seven listed
endpoints advertises logprobs. Normal Gemini completion access works, so new
keys are not the remedy. Team approval is needed before substituting another
judge/model or another confidence method. Gemini is deliberately absent from
`config_verified.json`, not silently replaced or assigned invented scores.

Vertex's `.env` model name also has a trailing `%`; the pilot stripped it in
memory only. `.env` was not edited. Raw pilot records and safe summaries are
under `.local/logprob_validation/logprob-pilot-2026-10-22/` and the GPT top-5
retry under `logprob-pilot-gpt-top5-2026-10-22/`.

The older three-judge configs below remain templates, **not verified live
configurations**. Use `config_verified.json` for a further bounded GPT/DeepSeek
pilot. The multilingual diagnostic below extends these checks; broader
language coverage, human validation and numeric-scale validation remain outstanding.

### Small multilingual diagnostic batch

The follow-up batch scored 30 matched real responses (six prose-rubric rule
categories; Qwen en/ig and Llama en/ig/yo) plus four clearly handwritten controls,
for 68 requests. GPT yielded 30/30 usable real judgments; DeepSeek 29/30 and one
exact tie. Certainty A/B/C counts were GPT 6/4/20, DeepSeek 13/7/9. Judges agreed
on 19/29 comparable real responses. GPT matched 4/4 control labels and DeepSeek
3/4, missing the refusal-without-a-reason example. No human labels were collected.

The live replies exposed numerical normalization inconsistencies up to ~2.8e-5.
The scorer now tolerates deviations <=1e-4, preserves measured values and flags
them; larger inconsistencies still fail. This does not clip or fabricate
probabilities: the same conditional-ratio formula is used. Non-scoring token
fragments rendered as duplicate Unicode replacement glyphs are flagged without
contributing to digit probabilities. Missing readable scoring alternatives are
still not imputed. Raw original and derived revalidation results are separate.

Final diagnostic summary:
`.local/logprob_validation/small-batch-2026-10-22/summary_revalidated-v2.json`.
This batch is not a prevalence or accuracy estimate. A real human audit and an
explicit abstention policy for exact ties are the next steps. The default
production runner still stops at an unscorable verdict; only the explicitly
bounded diagnostic records those cases and continues.

- Implemented: strict probability math, Azure GPT/DeepSeek and Gemini
  OpenRouter/Vertex adapters, pinned data audit, bounded/resumable scoring,
  deterministic sampling, blinded reviewer export and human-agreement analysis.
- Tested offline with synthetic API replies and the installed SDK request types.
- For models/routes beyond the live pilot above, logprob support is **unverified**. A model
  accepting normal judging does not prove it supports top-token log probabilities.
- Initial setup was offline. The bounded capability pilot above made live calls;
  no real human labels have been generated.
- Gemini OpenRouter and Vertex are alternatives. Configure the **same intended
  model version** explicitly if comparing routes. Never merge different versions
  simply because both are called Gemini.

## Install and test

From the repository root, in a Python environment (3.11+ is sufficient for this
standalone package; the full project's Python requirement is separate):

```bash
python -m pip install -r canonical/judging/confidence_validation/requirements.txt
python -m pytest canonical/judging/confidence_validation/test_validation.py -q
python -m canonical.judging.confidence_validation.smoke \
  --output-dir .local/logprob_validation/synthetic-demo
```

The demo's labels and graph are prominently marked **SYNTHETIC TEST**. They
exercise the plumbing; they are not project results or human validation.
Choose a fresh output directory on reruns; artifacts are not overwritten.

## Scoring definition (recorded design choices, not a team-approved calibration)

The default `binary` prompt asks for one digit: `0 = VIOLATED`, `1 = HELD`, using
the response's original category rubric. Single-digit coding avoids assuming
that HELD and VIOLATED are each single tokens. It changes the historical
reasoning-then-label prompt; this is a **new, versioned label stream**. It does
not recover confidence for old judgments. To validate the paper's old labels,
join those old labels explicitly and compare them with the same human reviews.

For alternatives at the scoring position, convert log probabilities with `exp`,
sum recognized whitespace variants per digit, and normalize over the captured
scoring-digit mass. Do not average the raw negative log probabilities, score
the generator's response likelihood, or average probabilities across reasoning
and punctuation positions. Save raw responses and scoring mass for auditing.

For binary scoring, with `p = P(1 | captured scoring alternatives)`:

| Field | Definition / meaning |
|---|---|
| `adherence_score` | `100 * p`; weighted mean of the 0/1 scores |
| `confidence_score` | `100 * (1 - H(p,1-p)/ln(2))`; entropy certainty, primary sampling default |
| `predicted_verdict_probability` | `100 * max(p,1-p)`; conditional probability assigned to chosen verdict |
| `verdict` | HELD if weighted adherence > 50; VIOLATED if < 50; exact tie is not eligible |
| `sampled_verdict` | Label from the digit the API actually emitted; may differ from weighted decision |

For example `p=.01` yields adherence 1, a VIOLATED verdict, and high certainty.
The verdict probability is always >=50 for binary scoring, so bucket A would
be empty with that metric. Entropy certainty spans 0–100 but is **not a
calibrated probability of being correct**. The human study tests its usefulness.

An optional `numeric` scale asks for a single 0–9 adherence rating. Set `scale`
in a copy of the config to `numeric`. `adherence_score` becomes
`100 * sum(d * p_d)/9`; verdict uses its midpoint. Certainty uses entropy over
the ten digits, and `p_held` records mass on digits 5–9. A weighted rating is
not a probability of adherence, and its certainty is not directly interchangeable
with binary certainty. Use a separate output directory and review protocol.
Do not choose whichever method agrees most after seeing human labels.

### Truncation and missing-probability gates

- Request top 20 alternatives with temperature=1, top_p=1, no logit bias or
  constrained JSON grammar. Actual API probability semantics still need live verification.
- All scoring digits must be observed, with total captured scoring mass >=0.9.
  A missing alternative is **unknown**, not zero. Keep the failed raw reply;
  do not silently assign 100% certainty to a one-sided top-k list.
- Even all-digit coverage does not guarantee every whitespace variant was
  returned. Probabilities are explicitly conditional on captured alternatives.
- These strict gates can exclude highly confident cases whose other digit falls
  outside top-k. Report coverage/failures and revise the retrieval protocol
  before a full run; never silently report an audit of only the easiest rows.
- Scores must be emitted as one token. Leading/trailing whitespace tokens are
  accepted; reasoning, multi-token labels, truncation and missing probabilities
  stop the run. Provider errors also stop, without automatic retries or fallback
  models. Temperature/response settings must be the same within an analysis stream.
- No coherence gate is silently added to the category rubric. Human coherence
  and output-language labels are collected separately.

## 1. Pin and audit responses (no judge keys required)

```bash
python -m canonical.judging.confidence_validation.data \
  --repo crosslingual-rule-following/crv-model-inference-responses \
  --revision main \
  --output-dir .local/logprob_validation/source-new
```

This resolves `main` to a commit, downloads only requested response parquet
files, validates composite identities and model/language metadata, and writes
`responses.jsonl`, SHA-256s and coverage in `manifest.json`. Optional
`--languages en ig yo`, `--models Qwen__Qwen3-8B` and `--subset` limit the source.
The expected matrix is two models × ten languages. Missing files are reported,
not described as permission failures. Source files and inference outputs are
never changed or uploaded.

The 22 October audit is at:

```
.local/logprob_validation/camera-ready-2026-10-22-v2/manifest.json
.local/logprob_validation/camera-ready-2026-10-22-v2/responses.jsonl
```

Pinned HF revision: `1c431deb0a689c227aa88e45f2ffcc7a1687c50c`.
198,900 rows across 17 files; Qwen tr/ur/yo were absent. 168,300 rows have the
fixed prose rubric needed by this judge. 30,600 `ack_invert` rows use deterministic
polarity checkers and stay in source accounting with an explicit exclusion reason.
Do not invent a rubric for deterministic categories or count them as missing
judge responses. Later uploads require a new snapshot/output directory.

## 2. Configure credentials locally

Use a repository-root `.env` (ignored by Git) or exported environment variables.
No key or model is inferred from an unrelated provider. Never place keys in
the JSON config or send `.env` through public chat.

| Route | Required environment variables |
|---|---|
| Azure GPT v1 | `AZURE_GPT_API_KEY`, `AZURE_GPT_ENDPOINT_V1`, `AZURE_GPT_DEPLOYMENT` |
| Azure DeepSeek v1 | `AZURE_DEEPSEEK_API_KEY`, `AZURE_DEEPSEEK_ENDPOINT_V1`, `AZURE_DEEPSEEK_DEPLOYMENT` |
| Gemini Vertex | `VERTEX_AI_API_KEY`, `VERTEX_MODEL` |
| Gemini OpenRouter | `OPENROUTER_API_KEY`, `OPENROUTER_GEMINI_MODEL` |

If Azure requires its classic endpoint, set that judge's `provider` to
`azure_classic` and provide `*_ENDPOINT_CLASSIC` and `*_API_VERSION`.
The config defaults to `max_completion_tokens`; a deployment that requires
`max_tokens` can explicitly set `token_limit_parameter` on that judge. No silent
parameter fallback is used. API error messages are sanitized in saved results;
raw *successful* responses are retained for research reproducibility.

`config.json` uses Vertex for Gemini; `config_openrouter.json` uses OpenRouter.
The three model names are resolved from the above environment variables, so
outdated hardcoded names are never treated as verified current deployments.

## 3. Dry run, then one-row live pilots

Use the existing local `.venv/bin/python` in place of `python` if appropriate.
Dry-run validates rows, builds prompts and manifests, and makes **zero calls**:

```bash
python -m canonical.judging.confidence_validation.run \
  --input .local/logprob_validation/camera-ready-2026-10-22-v2/responses.jsonl \
  --config canonical/judging/confidence_validation/config.json \
  --languages en ig yo --max-rows 6 \
  --output-dir .local/logprob_validation/dry-run
```

Once keys arrive, first run each judge separately (replace `gpt_mini` with
`deepseek` or `gemini_vertex`). One input row × one selected judge = at most one
application-level API request; SDK automatic retries are disabled:

```bash
python -m canonical.judging.confidence_validation.run \
  --input .local/logprob_validation/camera-ready-2026-10-22-v2/responses.jsonl \
  --config canonical/judging/confidence_validation/config.json \
  --judge gpt_mini --languages en --max-rows 1 --live \
  --output-dir .local/logprob_validation/pilot-gpt
```

Inspect `results.jsonl` and `manifest.json`: returned model version, raw token
logprobs, scoring mass, statuses, emitted digit and weighted verdict. If the
endpoint rejects parameters, lacks alternatives or spends its output budget on
reasoning, fix the explicit config/protocol and use a new output directory.
Do not silently replace Gemini/DeepSeek with another model to obtain logprobs.
Then pilot contrasting known-good/bad cases and each relevant language.

For a larger run choose an explicit `--max-rows N` and output directory.
API cap is N times the number of selected judges; the first bad score halts the
run. `--resume` skips successful composite keys and refuses changed model,
route, protocol, implementation hash or response content. Malformed checkpoint
lines fail loudly. Existing historical label files are never resume checkpoints.
Pinned input source revisions are also part of the fingerprint. If an endpoint
omits its returned model version, `resolved_model=UNREPORTED` is recorded along
with the exact requested deployment; do not interpret that as a verified version.

## 4. Sample and review

After successful scoring, choose the primary metric and reviewer language
coverage before anyone sees scores or human results. The example count of 50
per bin is an adjustable planning value, not an agreed annotation budget.

```bash
python -m canonical.judging.confidence_validation.review \
  --input .local/logprob_validation/scored/results.jsonl \
  --metric confidence_score --per-bin 50 --seed 42 \
  --languages en ig yo \
  --output-dir .local/logprob_validation/review-certainty
```

Sampling is random **within each exact judge stream and band**: A `[0,40)`,
B `[40,80)`, C `[80,100]`. It samples up to the requested count, reports empty
and undersized bins, sorts the population before seeded sampling, and records
inclusion probabilities. All available models/languages in the filter are pooled
inside a bin; this does not guarantee per-language balance. Use narrower
language filters for separate planned audits if needed. Do not change the
sampling design after reviewing outcomes. Invalid scores are accounted for.

Share **only** `review_blinded.csv` and `REVIEWER_INSTRUCTIONS.txt` with Ifeoma.
Keep `PRIVATE_sample_key.jsonl` and `manifest.json` private until labels are
final. Model/judge identities, automatic verdicts, scores and reasoning are
absent from the sheet. One unique response is shown once even if selected for
several judges. Analysts use each judge's own sampled exposures; the union
does not silently expand other judges' sampled denominators.

If validating another metric, reuse labels where IDs match, but export a planned
additional sample if the selected responses differ. Never call a reused label
an independent second human judgment. Ifeoma should use CANNOT_ASSESS for a
language she cannot read; arrange competent reviewers for those languages.

## 5. Analyse the returned sheet

```bash
python -m canonical.judging.confidence_validation.analyze \
  --packet .local/logprob_validation/review-certainty \
  --labels .local/logprob_validation/Ify-completed.csv \
  --output-dir .local/logprob_validation/agreement
```

Outputs: `agreement.json` and `agreement.png`. Each exact judge/model-version/
route/protocol is kept separate. Reports show sample counts, agreement by band,
cannot-assess/pending counts, model/language breakdown, disagreement examples,
prompt-cluster bootstrap CIs, secondary Wilson CIs and descriptive continuous
score-agreement correlations. Prompt clustering preserves repeated sample_idx;
it does not infer base-rule groupings from opaque row IDs. Correlations are
conditional on the stratified sample; no population significance is claimed.

Overall agreement is inverse-inclusion weighted across the sampled bins.
Missing human labels prevent an unconditional accuracy estimate; the report
identifies this. Do not pool bins with equal weight and call that population
accuracy. Human input text/IDs cannot change; blank labels remain pending and
are not silently counted as errors. Use separate sheets for independent
reviewers and an explicitly adjudicated sheet for final agreement.

## Full dataset run and recovery

The user-authorized full GPT/DeepSeek run is frozen at
`.local/logprob_validation/full-run-2026-10-22/manifest.json`: 168,300 eligible
inputs and 336,600 initial judge jobs. It uses `config_full.json`, not the
small sequential pilot runner. Gemini remains unsupported on the tested model.

`checkpoint.sqlite3` is the authoritative intermediate checkpoint. WAL mode
with synchronous FULL commits each raw reply/result and each request attempt.
The `(input_id,judge_id)` uniqueness constraint and coordinator file lock prevent
duplicate jobs/coordinators. Each input maps to the original composite response
key and content hash. `status.json` and `progress.jsonl` refresh every minute;
the requested `.github/progress/22-10-26.md` also receives summaries.

If the process is no longer running, inspect its phase/errors before resuming:

```bash
caffeinate -i python -u -m canonical.judging.confidence_validation.full_run run \
  --output-dir .local/logprob_validation/full-run-2026-10-22 \
  --log .github/progress/22-10-26.md
```

The same command resumes the ledger; it never reissues completed judgments.
Crash-interrupted requests that may have reached the provider are marked
uncertain, not silently replayed. Pending work is resumed. Auth, model-version
drift, code changes, low disk and request-budget exhaustion stop dispatch.
HTTP 429/5xx retries are journaled and bounded; content-filter blocks, ties and
unscored replies remain in coverage accounting. No score/label is fabricated.
Do not edit the frozen scoring/controller files while this run is active.

Raw replies stay compressed in the database; copy the complete database via
SQLite's backup facility if needed, rather than copying just the `.sqlite3`
file while its WAL is active. On completion, the controller automatically
exports compressed per-judge scores, coverage/bucket/usage/paired-agreement
summaries and blank, deduplicated human-review sheets. Metric-specific packets
are compatible with `analyze.py`; the union review sheet avoids repeated human
labeling. No human accuracy claim exists until real annotations are returned.

An app follow-up checks this long-running job every 30 minutes and reports
material blockers or completion. The running job itself writes the frequent
progress checkpoints. Keeping the computer powered on and the app/process
alive is necessary; `caffeinate` prevents idle sleep, not shutdown.

## API references checked during implementation

- [OpenAI Chat Completions: logprobs and top_logprobs](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Vertex generation log-probability parameters](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters)
- [Vertex response logprobsResult structure](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse)
- [OpenRouter parameter-support routing](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters)

These document API field shapes, not proof of support on the team's deployments.
