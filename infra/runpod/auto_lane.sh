#!/bin/bash
# One supervisor owns one GPU and runs its queue serially.
# Usage: auto_lane.sh <gpu> <batch-size> <name|config|log> [...]
set -u

GPU="$1"
BATCH="$2"
shift 2
cd /workspace/crosslingual-rule-following
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/root/hf_cache_local/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for spec in "$@"; do
  IFS='|' read -r name config log <<< "$spec"
  pattern="canonical.evaluation.inference.*${name}"

  # If a job was already started before this supervisor, wait instead of
  # creating a second model process on the same GPU.
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep 30
  done

  while true; do
    printf '[%s] START %s GPU%s active-only n_samples=3 batch=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$name" "$GPU" "$BATCH" >> "$log"
    CUDA_VISIBLE_DEVICES="$GPU" /workspace/venv/bin/python \
      -m canonical.evaluation.inference \
      --hyperparameter-file "$config" \
      --active-only --n-samples 3 \
      --generation-batch-size "$BATCH" >> "$log" 2>&1
    rc=$?
    printf '[%s] EXIT %s rc=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$name" "$rc" >> "$log"
    [ "$rc" -eq 0 ] && break
    sleep 30
  done
done
