#!/usr/bin/env bash
# Capability baseline: ask directly (no rule) for a referral / reasoned refusal, then judge whether the act was performed.
# Run from repo root inside tmux:  source /workspace/.runpod_env && bash canonical/baseline/run_capability.sh <stage>
# stages: dry | gen | judge | retry | finish | all      (resumable)
# env: LANGS (default all 13)  N_SAMPLES (default 10)  MODELS  GEN_BATCH (default 30)
set -euo pipefail
export PYTHONPATH=.
B=canonical/baseline
MODELS="${MODELS:-meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen3-8B}"
LANGS="${LANGS:-en,am,de,hi,ig,it,ko,ru,sw,ta,tr,ur,yo}"; N_SAMPLES="${N_SAMPLES:-10}"; GEN_BATCH="${GEN_BATCH:-30}"
OUT=/workspace/capability_out
stage="${1:-all}"
cfg_for() { case "$1" in meta-llama/*) echo canonical/evaluation/configs/hyperparameter_768_llama.json ;; Qwen/*) echo canonical/evaluation/configs/hyperparameter_768_qwen.json ;; *) echo "no config for $1" >&2; exit 1 ;; esac; }

do_gen() {  # $1 = extra flag
  IFS=, read -ra MS <<< "$MODELS"
  for M in "${MS[@]}"; do
    echo "=== capability generation: $M | langs=$LANGS | n=$N_SAMPLES"
    python $B/run_capability_act.py --hyperparameter-file "$(cfg_for "$M")" --models "$M" --langs "$LANGS" \
      --variants A,B --n-samples "$N_SAMPLES" --generation-batch-size "$GEN_BATCH" --out-dir $OUT ${1:-}
  done
}
do_judge() {
  mkdir -p logs
  for V in A B; do
    python $B/prep_baseline_for_judging.py --in $OUT --out /workspace/capability_judge_in_$V --pattern "*_cap${V}_responses.jsonl" --restore-user-query
    python $B/make_judge_configs.py --models "$MODELS" --langs "$LANGS" --out-dir /workspace/capability_judge_cfg_$V \
      --data /workspace/capability_judge_in_$V --results-dir results/capability_$V
  done
  # 6 judge jobs (3 judges x 2 variants) -> run 3 at a time to stay inside API rate limits
  for V in A B; do
    for J in gpt_mini deepseek gemini; do
      nohup python -m canonical.judging.judge_$J --config /workspace/capability_judge_cfg_$V/judge_$J.json > logs/cap${V}_judge_$J.log 2>&1 &
    done
    wait
  done
}
do_retry() {
  for V in A B; do for J in gpt_mini deepseek gemini; do
    python -m canonical.judging.retry_failed_rows --config /workspace/capability_judge_cfg_$V/judge_$J.json || true
    python -m canonical.judging.consolidate --config /workspace/capability_judge_cfg_$V/judge_$J.json
  done; done
}
do_finish() {
  python $B/analyze_capability.py \
    --cap-a gemini=results/capability_A/gemini/final.jsonl gpt_mini=results/capability_A/gpt_mini/final.jsonl deepseek=results/capability_A/deepseek/final.jsonl \
    --cap-b gemini=results/capability_B/gemini/final.jsonl gpt_mini=results/capability_B/gpt_mini/final.jsonl deepseek=results/capability_B/deepseek/final.jsonl \
    --out $B/data/capability
  mkdir -p $B/data/capability/responses && cp $OUT/*_responses.jsonl $B/data/capability/responses/
  echo "done: $B/data/capability/ (csv + png + raw responses). Commit canonical/baseline only."
}
case "$stage" in
  dry) do_gen --dry-run ;; gen) do_gen ;; judge) do_judge ;; retry) do_retry ;; finish) do_finish ;;
  all) do_gen; do_judge; do_retry; do_finish ;;
  *) echo "usage: bash $B/run_capability.sh dry|gen|judge|retry|finish|all" >&2; exit 2 ;;
esac
