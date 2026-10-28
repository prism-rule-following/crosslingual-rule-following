#!/usr/bin/env bash
# End-to-end no-rule baseline: generate -> check -> prep -> judge (3 judges) -> retry -> consolidate -> consensus -> analysis.
# Run from the repo root inside tmux:   source /workspace/.runpod_env && bash canonical/baseline/run_all.sh <stage>
#
# stages:  gen | check | judge | retry | finish | all     (all = gen,check,judge,retry,finish; resumable, re-run is safe)
#
# Settings (env vars):
#   MODELS   hf model ids, comma separated   default: both repo models (Llama-3.1-8B-Instruct, Qwen3-8B)
#   LANGS    default = all 13 dataset languages: en,am,de,hi,ig,it,ko,ru,sw,ta,tr,ur,yo   (or e.g. LANGS=en,yo,sw)
#   N_SAMPLES default 5    PRESSURE default L0    GEN_BATCH default 30
#   RESULT_REPO   HF dataset repo for generations (new repo, NOT crv-model-inference-responses)  [required for gen]
#   JUDGE_REPO    HF repo for judge results (optional)
set -euo pipefail
export PYTHONPATH=.
B=canonical/baseline
MODELS="${MODELS:-meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen3-8B}"
ALL13=en,am,de,hi,ig,it,ko,ru,sw,ta,tr,ur,yo
LANGS="${LANGS:-$ALL13}"; [ "$LANGS" = "all" ] && LANGS=$ALL13
N_SAMPLES="${N_SAMPLES:-5}"; PRESSURE="${PRESSURE:-L0}"; GEN_BATCH="${GEN_BATCH:-30}"
OUT=/workspace/baseline_out; JIN=/workspace/baseline_judge_in; CFGD=/workspace/baseline_judge_cfg; RES=results/baseline
IFS=, read -ra _P <<< "$PRESSURE"; EXPECT_PROMPTS=$(( 26 * ${#_P[@]} ))
stage="${1:-all}"

cfg_for() {   # model id -> hyperparameter file (one file per model family)
  case "$1" in
    meta-llama/*) echo canonical/evaluation/configs/hyperparameter_768_llama.json ;;
    Qwen/*)       echo canonical/evaluation/configs/hyperparameter_768_qwen.json ;;
    *) echo "no config for $1: copy a hyperparameter_768_*.json, set model_ids, add a case here" >&2; exit 1 ;;
  esac
}

do_gen() {
  [ -n "${RESULT_REPO:-}" ] || { echo "Set RESULT_REPO=<org>/<new-baseline-repo>" >&2; exit 1; }
  IFS=, read -ra MS <<< "$MODELS"
  for M in "${MS[@]}"; do
    echo "=== generating: $M | langs=$LANGS | n=$N_SAMPLES"
    python $B/run_baseline_norule.py --hyperparameter-file "$(cfg_for "$M")" --models "$M" --langs "$LANGS" \
      --pressure-levels "$PRESSURE" --n-samples "$N_SAMPLES" --generation-batch-size "$GEN_BATCH" \
      --result-repo "$RESULT_REPO" --checkpoint-dir /workspace/baseline_ckpt
  done
}
do_check() { python $B/check_baseline.py $OUT "$N_SAMPLES" $EXPECT_PROMPTS; }

do_judge() {
  python $B/prep_baseline_for_judging.py --in $OUT --out $JIN
  python $B/make_judge_configs.py --models "$MODELS" --langs "$LANGS" --out-dir $CFGD --data $JIN --results-dir $RES ${JUDGE_REPO:+--hf-results-repo $JUDGE_REPO}
  mkdir -p logs
  # 3 judges in parallel, each resumable (re-running skips finished rows)
  for J in gpt_mini deepseek gemini; do
    nohup python -m canonical.judging.judge_$J --config $CFGD/judge_$J.json > logs/judge_$J.log 2>&1 &
  done
  echo "judges running in background; follow with: tail -f logs/judge_*.log   (wait for all three: 'wait' or: pgrep -f canonical.judging.judge_)"
  wait
}

do_retry() {   # only failed/missing rows; DeepSeek content-filter rows are permanent failures and are skipped by design
  for J in gpt_mini deepseek gemini; do
    python -m canonical.judging.retry_failed_rows --config $CFGD/judge_$J.json || true
    python -m canonical.judging.consolidate --config $CFGD/judge_$J.json
  done
}

do_finish() {
  python $B/build_consensus.py --out $B/data/rule_scored.jsonl $B/data/rule_verdicts_gemini_L0.jsonl $B/data/rule_verdicts_deepseek_L0.jsonl $B/data/rule_verdicts_gpt_mini_L0.jsonl
  python $B/build_consensus.py --out $B/data/baseline_scored.jsonl $RES/gemini/final.jsonl $RES/deepseek/final.jsonl $RES/gpt_mini/final.jsonl
  python $B/analyze.py \
    --baseline gemini=$RES/gemini/final.jsonl gpt_mini=$RES/gpt_mini/final.jsonl deepseek=$RES/deepseek/final.jsonl \
    --rule gemini=$B/data/rule_verdicts_gemini_L0.jsonl gpt_mini=$B/data/rule_verdicts_gpt_mini_L0.jsonl deepseek=$B/data/rule_verdicts_deepseek_L0.jsonl \
    --out $B/data
  python $B/make_figures.py && python $B/make_html.py
  echo "done. Review $B/REPORT.html, then commit canonical/baseline (NOT results/ or logs/)."
}

case "$stage" in
  gen) do_gen ;; check) do_check ;; judge) do_judge ;; retry) do_retry ;; finish) do_finish ;;
  all) do_gen; do_check; do_judge; do_retry; do_finish ;;
  *) echo "usage: bash $B/run_all.sh gen|check|judge|retry|finish|all" >&2; exit 2 ;;
esac
