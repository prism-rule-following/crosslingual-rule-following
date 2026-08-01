#!/usr/bin/env bash
# Idempotent environment setup for the crosslingual-rule-following RunPod workspace.
#
# Design goal: the venv, uv's package cache, and the HF model cache all live on
# the PERSISTENT network volume (mounted at $WORKSPACE_DIR, default /workspace),
# not on the pod's ephemeral container disk. That means whoever attaches a new
# pod to this workspace just activates the existing venv instead of
# reinstalling ~10GB of torch/CUDA deps and re-downloading multi-GB model
# weights every time. Uses uv (not pip) for venv + installs, and keeps the
# venv isolated from the base image's preinstalled torch to avoid the version
# conflicts RunPod's own images warn about.
#
# Usage (from anywhere, after cloning the repo into the workspace):
#   bash infra/runpod/setup.sh
#
# Safe to re-run: it skips steps that are already done.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/venv}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$WORKSPACE_DIR/hf_cache}"
LOCK_FILE="$REPO_DIR/infra/runpod/requirements.lock.txt"
REQ_FILE="$REPO_DIR/infra/runpod/requirements.txt"

echo "== crosslingual-rule-following RunPod setup =="
echo "Repo:      $REPO_DIR"
echo "Workspace: $WORKSPACE_DIR"
echo "Venv:      $VENV_DIR"
echo "HF cache:  $HF_CACHE_DIR"
echo

if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "ERROR: $WORKSPACE_DIR does not exist. Set WORKSPACE_DIR to your mounted persistent volume." >&2
    exit 1
fi

# ---- 1. GPU sanity check -----------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Is this pod actually GPU-backed?" >&2
    exit 1
fi
echo "-- GPU --"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

# ---- 2. System packages (best-effort, only if apt + sudo/root available) --
# tmux is included here so every pod attach gets it via this one script
# instead of a separate manual `apt-get install tmux` each time.
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    MISSING_PKGS=""
    command -v git >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS git"
    command -v tmux >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS tmux"
    command -v gcc >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS build-essential"
    if [ -n "$MISSING_PKGS" ]; then
        echo "-- Installing system packages:$MISSING_PKGS --"
        apt-get update -qq
        # shellcheck disable=SC2086
        apt-get install -y -qq $MISSING_PKGS >/dev/null
    fi
fi

# ---- 3. uv ----------------------------------------------------------------
# uv manages the venv and installs; its download/build cache is redirected
# onto the persistent volume so a fresh pod attach doesn't re-fetch wheels
# uv already fetched on a previous pod.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE_DIR/uv_cache}"
mkdir -p "$UV_CACHE_DIR"
if ! command -v uv >/dev/null 2>&1; then
    echo "-- Installing uv --"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- 4. Persistent, isolated venv -----------------------------------------
# Deliberately NOT --system-site-packages: RunPod's own pytorch images warn
# that mixing venv installs with the image's preinstalled torch causes
# version conflicts. We keep the venv fully isolated and pin torch ourselves
# below, so the env is identical whether the base image ships torch or not.
mkdir -p "$WORKSPACE_DIR"
if [ ! -d "$VENV_DIR" ]; then
    echo "-- Creating venv at $VENV_DIR --"
    uv venv "$VENV_DIR"
else
    echo "-- Reusing existing venv at $VENV_DIR --"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---- 5. torch (CUDA-matched wheel, always pinned explicitly) --------------
# Uninstall + reinstall unconditionally (per RunPod's guidance) to avoid
# conflicts with whatever torch build the base image or a previous venv
# state left behind. Cheap after the first run: uv's persistent cache means
# this is a local install, not a re-download.
echo "-- Installing torch/torchvision (cu128 wheels) --"
uv pip uninstall torch torchvision torchaudio >/dev/null 2>&1 || true
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ---- 6. Everything else: lock file if present, else floor-pin + freeze --
if [ -f "$LOCK_FILE" ]; then
    echo "-- Installing from requirements.lock.txt (reproducible, team-shared) --"
    uv pip install -r "$LOCK_FILE"
else
    echo "-- No lock file yet. Installing floor-pinned requirements.txt --"
    uv pip install -r "$REQ_FILE"
    echo "-- Freezing to requirements.lock.txt --"
    uv pip freeze | grep -v -E '^(torch|torchvision|torchaudio)==' > "$LOCK_FILE"
    echo
    echo "  >>> First install on this workspace. requirements.lock.txt was generated at:"
    echo "      $LOCK_FILE"
    echo "  >>> Commit and push this file so the rest of the team installs the same versions:"
    echo "      cd $REPO_DIR && git add infra/runpod/requirements.lock.txt && git commit -m 'lock runpod env' && git push"
    echo
fi

# ---- 7. Persistent HF cache ----------------------------------------------
mkdir -p "$HF_CACHE_DIR"
ENV_FILE="$WORKSPACE_DIR/.runpod_env"
cat > "$ENV_FILE" <<EOF
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WORKSPACE_DIR="$WORKSPACE_DIR"
export VENV_DIR="$VENV_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
source "$VENV_DIR/bin/activate"
EOF
echo "-- Wrote activation dotfile: $ENV_FILE --"

echo
echo "== Setup complete =="
echo "For this and every future shell/pod attach, run:"
echo "    source $ENV_FILE"
echo "    export HF_TOKEN=hf_xxx   # not persisted anywhere, set it fresh each pod"
echo
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
