#!/usr/bin/env bash
# Core-language no-rule baseline. Run from the repo root on the pod, after:
#   source /workspace/.runpod_env ; export HF_TOKEN=hf_xxx
#
#   bash canonical/baseline/run_baseline_core.sh dry     # build prompts only, no model (check the output!)
#   bash canonical/baseline/run_baseline_core.sh smoke   # 20 rows, 1 model, no push
#   bash canonical/baseline/run_baseline_core.sh full    # both models, pushes to $RESULT_REPO
#
# Override with env vars, e.g.  LANGS=en,yo,de PRESSURE=L0 N_SAMPLES=20 bash canonical/baseline/run_baseline_core.sh full
set -euo pipefail

LANGS="${LANGS:-en,yo,sw}"
PRESSURE="${PRESSURE:-L0}"            # L0 = clean no-rule baseline (the run reported in REPORT.md)
N_SAMPLES="${N_SAMPLES:-5}"        # responses per prompt; reported run used 5 (26 prompts x 5 = 130 per language per model)
GEN_BATCH="${GEN_BATCH:-30}"      # lower if out-of-memory
RESULT_REPO="${RESULT_REPO:-}"            # NEW HF dataset repo, never the main results repo
SCRIPT=canonical/baseline/run_baseline_norule.py
CFG_DIR=canonical/evaluation/configs
LLAMA="$CFG_DIR/hyperparameter_768_llama.json"
QWEN="$CFG_DIR/hyperparameter_768_qwen.json"
export PYTHONPATH=.

[ -f "$SCRIPT" ] || { echo "Run from the repo root: bash canonical/baseline/run_baseline_core.sh dry|smoke|full" >&2; exit 1; }
stage="${1:-}"

case "$stage" in
  dry)
    python "$SCRIPT" --hyperparameter-file "$LLAMA" --langs "$LANGS" --pressure-levels "$PRESSURE" --n-samples "$N_SAMPLES" --dry-run
    ;;
  smoke)
    python "$SCRIPT" --hyperparameter-file "$LLAMA" --langs "${LANGS%%,*}" --pressure-levels "$PRESSURE" \
      --n-samples "$N_SAMPLES" --generation-batch-size "$GEN_BATCH" --limit 20 --no-push
    echo; head -n 2 /workspace/baseline_out/*_norule_responses.jsonl
    ;;
  full)
    [ -n "$RESULT_REPO" ] || { echo "Set RESULT_REPO=<org>/<new-baseline-repo> first." >&2; exit 1; }
    for CFG in "$LLAMA" "$QWEN"; do
      python "$SCRIPT" --hyperparameter-file "$CFG" --langs "$LANGS" --pressure-levels "$PRESSURE" \
        --n-samples "$N_SAMPLES" --generation-batch-size "$GEN_BATCH" \
        --result-repo "$RESULT_REPO" --checkpoint-dir /workspace/baseline_ckpt
    done
    ;;
  *)
    echo "usage: bash canonical/baseline/run_baseline_core.sh dry|smoke|full" >&2; exit 2 ;;
esac
