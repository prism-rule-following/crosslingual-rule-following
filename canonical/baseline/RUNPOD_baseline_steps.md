# No-rule baseline on RunPod (matched to the repo's own infra/runpod setup)

## What gets generated
Same `ModelRunner` + same config (temperature 1.0, 768 tokens, n_samples 5, batch 30, Qwen enable_thinking=false), untouched. Only the prompts change:
- system = the row's `context` only ("You are an assistant."). Rule, status word and grammar wrapper (Never / You must ensure / Please make sure) are all dropped.
- user_query unchanged. L0 only by default (see note on L1/L4 below).
- active-side rows only, de-duplicated to one row per (language, category, topic, pressure_level) cell.
- ids become `norule::<original id>`; join back on (language, category, topic, pressure_level).

Size, checked on the repo's English v2 data: 2,340 active rows collapse to **26 prompts** at L0 (78 if you add L1+L4). x5 samples = **130 generations per language per model** (390 with L1+L4). Tiny job.

Pressure levels: L2 ("As your supervisor, ignore that rule...") and L3 ("Ignore the previous rule...") mention a rule that doesn't exist in the baseline, so they're excluded. L1 (social) and L4 (emotional) contain no rule reference, so `--pressure-levels L0,L1,L4` is valid if you want a baseline for those rungs too. L5 is excluded (multi-turn; the generation code only sends the final turn anyway).

## Which server
**1x RTX 4090 (24 GB)**, which is what the repo's infra README targets. 8B bf16 weights are about 16 GB and batch 30 x 768 tokens fits; the repo also has L4 (24 GB) configs, so L4 works too. A100/H100 is unnecessary for 130 generations.
- Filter for a pod whose driver supports **CUDA >= 12.8** (setup.sh installs cu128 torch wheels).
- Network volume **100 GB** at /workspace (two 8B models ~32 GB of HF cache + venv ~10 GB). If the team already has the shared workspace volume, attach your pod to it: venv and HF cache are already there and you can skip setup.
- Check current RunPod pricing yourself; run time is likely well under an hour per model after setup and model download (my estimate, not measured).

## 0. HF prep (laptop)
- Create a NEW HF dataset repo, e.g. `<org>/norule-baseline-responses`. Never the main results repo (`crosslingual-rule-following/crv-model-inference-responses`): inference.py skips any model/language already there, and the script refuses that name.
- HF token with write access. Accept the Llama-3.1 license with that account (Qwen3-8B isn't gated).

## 1. Setup (repo's own script: infra/runpod/setup.sh)
```bash
cd /workspace
git clone https://github.com/prism-rule-following/crosslingual-rule-following.git
cd crosslingual-rule-following
git checkout ft-camera-ready-sampling
bash infra/runpod/setup.sh          # uv venv at /workspace/venv, cu128 torch, pinned lock file
source /workspace/.runpod_env       # every new shell / pod
export HF_TOKEN=hf_xxx              # never saved to disk, set it each session
```
Notes:
- Python 3.14 is requested (.python-version); uv fetches it for you.
- Not tested by me: `pygraphviz==2.0.1` is in the lock file and setup.sh doesn't install graphviz headers. If the install fails on it: `apt-get install -y graphviz libgraphviz-dev`, then rerun setup.sh (it's idempotent).
- Optional sanity check of the env: `python infra/runpod/smoke_test.py`.

Everything lives in `canonical/baseline/`. From the repo root (`export PYTHONPATH=.`):
```bash
export PYTHONPATH=.
export LLAMA=canonical/evaluation/configs/hyperparameter_768_llama.json
export QWEN=canonical/evaluation/configs/hyperparameter_768_qwen.json
```

## 2. Run, in this order (repo root)
```bash
# a) dry run: builds prompts for ALL languages, loads no model. Needs the HF dataset to be reachable.
python canonical/baseline/run_baseline_norule.py --hyperparameter-file $LLAMA --dry-run
```
Per language you should see: 26 rows, ONE distinct system prompt (the persona line in that language), `persona_from_heuristic: 0`. It prints PROBLEM lines and refuses to run if the system prompts look wrong.

```bash
# b) smoke test: 20 rows, never pushes, separate checkpoint dir
python canonical/baseline/run_baseline_norule.py --hyperparameter-file $LLAMA --langs en --limit 20 --no-push
head -n 2 /workspace/baseline_out/*_en_norule_responses.jsonl
```
Responses should be normal, non-empty answers.

```bash
# c) full run in tmux (survives disconnects; Ctrl-B D to detach, tmux attach -t baseline)
tmux new -s baseline
source /workspace/.runpod_env; export HF_TOKEN=hf_xxx; export PYTHONPATH=.
cd /workspace/crosslingual-rule-following
for CFG in $LLAMA $QWEN; do
  python canonical/baseline/run_baseline_norule.py \
    --hyperparameter-file $CFG \
    --result-repo <org>/norule-baseline-responses
done
```
Add `--pressure-levels L0,L1,L4` to both if you want the extra rungs.

## 3. Resume / finish
- Crash or restart: rerun the same command. Checkpoints (/workspace/baseline_ckpt) resume mid-language; finished model/language pairs in the HF repo are skipped.
- Local copies of every output are always in /workspace/baseline_out/.
- Stop the pod when done.

## 4. Scoring later
Join to the active rule rows on (language, category, topic, pressure_level), run the same checker/judge. Baseline rate = how often the model does the behaviour with no rule at all; rule adherence above it is the rule's effect.

## Verified vs not
Verified by reading the repo: columns, `context` + `rule` construction of `system`, ModelRunner/HFDataHelper signatures, active-status filter, setup.sh/lock/configs. Tested the prompt-building on the repo's real English v2 data (26 prompts, 1 persona, no rule text left).
NOT verified: the Hugging Face `canonical-dataset` itself (my environment can't reach huggingface.co), so the non-English translations' `context`/`system` are untested (the dry run covers that), and nothing was run on a GPU.

---
## Core-language version (en, yo, sw)  <- use this first
HF `canonical-dataset` has 13 language subsets (2,340 rows each, same structure as the English file, with a `context` column in every language), so Swahili is available even though it isn't in the 768-run configs' language list. Swap in `de` or `ru` with `LANGS=en,yo,de`.
```bash
source /workspace/.runpod_env; export HF_TOKEN=hf_xxx
cd /workspace/crosslingual-rule-following
bash canonical/baseline/run_baseline_core.sh dry     # per language expect: cells 26, after_dedupe 26, persona_from_heuristic 0, ONE distinct system prompt
bash canonical/baseline/run_baseline_core.sh smoke   # 20 rows, no push
RESULT_REPO=<org>/norule-baseline-responses bash canonical/baseline/run_baseline_core.sh full   # Llama then Qwen
```
Defaults: LANGS=en,yo,sw, PRESSURE=L0 (`PRESSURE=L0,L1,L4` for the extra rungs). Rerunning `full` with a longer language list later skips pairs already uploaded.
In the dry run, `cells_with_differing_query_text > 0` means the translation wording differs between rows of the same cell; harmless (one is kept), but join on pressure_level not on the text.

## Setup = the repo's infra/runpod/setup.sh (same one from your earlier Azure/Foundry chat)
The team pod from that chat (RTX 4090, `ssh root@213.173.109.223 -p 15432`) already has the venv on /workspace if it's still up: just `git pull`, `source /workspace/.runpod_env`, `export HF_TOKEN`. On a fresh pod run `bash infra/runpod/setup.sh` first.
The Lambda-style setup.sh from the other repo (numpy<2, cu121) does NOT apply here: this repo needs numpy>=2, pandas>=3, cu128.
