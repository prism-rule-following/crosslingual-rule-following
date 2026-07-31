# RunPod environment

Shared setup for running experiments on RunPod GPU pods against the team's
persistent workspace volume. The goal: everyone gets the exact same package
versions, and nobody re-installs torch/transformer_lens or re-downloads
model weights every time a new pod attaches to the workspace.

## How it works

- The **venv**, **uv's package cache**, and the **HF model cache** all live
  on the persistent network volume (default mount: `/workspace`), not on the
  pod's ephemeral container disk. Attach a new pod to the same workspace →
  the venv is already there.
- Package management is via [**uv**](https://docs.astral.sh/uv/), not raw
  pip — faster, and its cache means "reinstalling" on a new pod is a local
  operation, not a re-download.
- The venv is **fully isolated** (no `--system-site-packages`). RunPod's
  official pytorch images ship a preinstalled torch, but mixing that with
  venv installs causes version conflicts — so `setup.sh` explicitly
  uninstalls any stray torch/torchvision/torchaudio in the venv and installs
  the pinned `cu128` wheels itself every run. This is deliberately
  unconditional (not skipped when torch looks present) to guarantee everyone
  ends up on the same build regardless of what a given pod image ships.
- The **first person** to run `setup.sh` on a fresh workspace installs
  floor-pinned versions from `requirements.txt` and freezes them to
  `requirements.lock.txt`, which gets committed. **Everyone after that**
  installs from the lock file, so the whole team runs identical versions.
- `setup.sh` is idempotent — re-running it on a pod that already has the venv
  reuses it; only the torch uninstall/reinstall step always runs (cheap,
  thanks to uv's cache).

## First-time setup (one person, once per workspace)

1. Attach an RTX 4090 pod, connect (SSH / VS Code Remote), confirm you're
   inside the persistent workspace (usually mounted at `/workspace`).

2. Clone the repo into the workspace and check out this branch:
   ```bash
   cd /workspace
   git clone https://github.com/prism-rule-following/crosslingual-rule-following.git
   cd crosslingual-rule-following
   git checkout runpod-setup   # or main, once merged
   ```

3. Run setup:
   ```bash
   bash infra/runpod/setup.sh
   ```
   This installs `uv` if missing, creates `/workspace/venv`, installs torch
   (cu128) + everything in `requirements.txt` via uv, and writes
   `infra/runpod/requirements.lock.txt`.

4. Activate and set your HF token (needed for gated models like Llama-3.1;
   not persisted anywhere — export it fresh every pod/session):
   ```bash
   source /workspace/.runpod_env
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   Make sure your HF account has accepted the license for
   `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen3-8B` on huggingface.co.

5. Smoke-test:
   ```bash
   python infra/runpod/smoke_test.py
   ```
   Should print a generation from both models and `[PASS]` for each.

6. Commit and push the generated lock file so everyone else gets the exact
   same versions:
   ```bash
   git add infra/runpod/requirements.lock.txt
   git commit -m "Lock RunPod env dependencies"
   git push
   ```

## Every subsequent pod attach (anyone, every time)

The venv and HF cache already exist on the volume — no reinstall needed.
`HF_TOKEN` is never written to disk, so it has to be exported again in every
new pod/shell:

```bash
cd /workspace/crosslingual-rule-following
git pull
source /workspace/.runpod_env
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

That's it. `setup.sh` is safe to re-run too (e.g. if `requirements.lock.txt`
changed) — it detects the existing venv and only installs what's missing.

## Long-running jobs (tmux)

`setup.sh` installs `tmux` as part of its system-package step, so it's there
without a separate manual install. Use it for anything that should survive
a disconnect:

```bash
tmux new -s mysession        # start a session
source /workspace/.runpod_env
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python your_script.py
# Ctrl-B D to detach, walk away — job keeps running
tmux attach -t mysession     # reattach later to check progress
```

Useful shortcuts while attached: `Ctrl-B D` detach, `Ctrl-B [` scroll (`q` to
exit), `Ctrl-C` interrupt (safe — current batch finishes writing first).

## Updating dependencies

Don't hand-edit `requirements.lock.txt`. Instead:

1. Edit `requirements.txt` (bump/add the floor-pinned package).
2. `rm infra/runpod/requirements.lock.txt`
3. Re-run `bash infra/runpod/setup.sh` on one pod, run `smoke_test.py` to confirm nothing broke.
4. Commit the regenerated `requirements.lock.txt` and let the team `git pull && bash infra/runpod/setup.sh`.

## Files

| File | Purpose |
|---|---|
| `setup.sh` | Idempotent venv + dependency + HF cache setup |
| `requirements.txt` | Floor-pinned deps (source of truth for versions) |
| `requirements.lock.txt` | Exact frozen versions (generated, committed, what everyone actually installs) |
| `smoke_test.py` | Loads Llama-3.1-8B-Instruct + Qwen3-8B and generates, to confirm the env works end-to-end |

## Troubleshooting

- **`nvidia-smi not found`**: the pod isn't GPU-backed or drivers aren't up — check the RunPod pod config, not this script.
- **`torch.cuda.is_available()` is False after setup**: check the pod's driver
  CUDA version (`nvidia-smi`, top right) and make sure it's ≥ 12.8; if the
  pod is on an older driver, change the `--index-url` in `setup.sh` step 5
  (currently `cu128`) to match.
- **Gated repo 403 on model load**: accept the license on the model's
  huggingface.co page with the account matching your `HF_TOKEN`.
- **Different people getting different behavior**: check
  `uv pip freeze | diff - infra/runpod/requirements.lock.txt` — if it's not
  empty, re-run `setup.sh` to reconcile against the lock file.
