#!/usr/bin/env python3
"""
Hugging Face I/O for the obligation-direction pipeline.

- pull the canonical dataset (nunaa/canonical_obligation_dataset)
- create (if missing) and push to three repos:
    activations -> nunaa/crosslingual_rf-activations
    directions  -> nunaa/crosslingual_rf-directions
    results     -> nunaa/crosslingual_rf-results

Auth: set HF_TOKEN in the environment (or the env name in config.hf.token_env).

CLI:
    python hf_io.py pull-dataset --config hyperparameters.json --out obligation_full.json
    python hf_io.py push --config hyperparameters.json --kind activations --model qwen3-8b --path dim_out/acts_qwen3-8b
    python hf_io.py push --config hyperparameters.json --kind directions  --model qwen3-8b --path dim_out/dim_candidates_qwen3-8b.pt
    python hf_io.py push --config hyperparameters.json --kind results     --model qwen3-8b --path dim_out/dim_report_qwen3-8b.json
"""
import os, json, argparse, glob

def _cfg(p): return json.load(open(p))

def _token(cfg):
    env = cfg["hf"].get("token_env", "HF_TOKEN")
    tok = os.environ.get(env)
    if not tok:
        raise SystemExit(f"[hf] no token in ${env}. `export {env}=hf_...` first.")
    return tok

def _repo_id(cfg, kind):
    return {"activations": cfg["hf"]["activations_repo"],
            "directions":  cfg["hf"]["directions_repo"],
            "results":     cfg["hf"]["results_repo"]}[kind]

def ensure_repo(api, repo_id, private, token):
    from huggingface_hub import create_repo
    create_repo(repo_id, repo_type="dataset", private=private,
                exist_ok=True, token=token)

# --------------------------------------------------------------------------- #
def pull_dataset(cfg, out_path, concept=None, language=None):
    """Pull the canonical dataset for a given (concept, language).

    Resolution order for the file inside the dataset repo:
      1. config.hf.dataset_files["<concept>/<language>"] if present (explicit map)
      2. "<concept>_<language>.json" at repo root
      3. first .json in the repo (last-resort fallback)
    If datasets.load_dataset works with a matching config/split, that is tried first.
    """
    hf = cfg["hf"]; token = _token(cfg)
    if hf.get("dataset_load", "hub") == "local":
        print("[hf] dataset_load=local; nothing to pull"); return
    repo = hf["dataset_repo"]

    # explicit filename map wins
    fmap = hf.get("dataset_files", {})
    key = f"{concept}/{language}" if (concept and language) else None
    target_name = fmap.get(key) if key else None
    if target_name is None and concept and language:
        target_name = f"{concept}_{language}.json"

    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(repo, repo_type="dataset", token=token)
    pick = None
    if target_name and target_name in files:
        pick = target_name
    elif target_name:
        # allow nested placement like <concept>/<language>.json
        nested = f"{concept}/{language}.json"
        pick = nested if nested in files else None
    if pick is None:
        cand = [f for f in files if f.endswith(".json")]
        if not cand:
            raise SystemExit(f"[hf] no matching json in {repo} for {key}; files={files}")
        pick = cand[0]
        print(f"[hf][warn] exact file for {key} not found; falling back to {pick}")
    fp = hf_hub_download(repo, pick, repo_type="dataset", token=token)
    rows = json.load(open(fp))
    json.dump(rows, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[hf] pulled {len(rows)} rows from {repo}/{pick} -> {out_path}")

def push(cfg, kind, group_path, path):
    """group_path is the repo-relative prefix, e.g. '<concept>/<lang>/<model>__<preset>'."""
    from huggingface_hub import HfApi, upload_file, upload_folder
    token = _token(cfg); api = HfApi(token=token)
    repo_id = _repo_id(cfg, kind)
    if cfg["hf"].get("create_if_missing", True):
        ensure_repo(api, repo_id, cfg["hf"].get("repo_private", True), token)
    if os.path.isdir(path):
        upload_folder(folder_path=path, path_in_repo=group_path,
                      repo_id=repo_id, repo_type="dataset", token=token,
                      commit_message=f"{kind}: {group_path}")
        print(f"[hf] pushed folder {path} -> {repo_id}/{group_path}")
    else:
        dest = f"{group_path}/{os.path.basename(path)}"
        upload_file(path_or_fileobj=path, path_in_repo=dest,
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message=f"{kind}: {dest}")
        print(f"[hf] pushed file {path} -> {repo_id}/{dest}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pull-dataset")
    p1.add_argument("--config", default="hyperparameters.json")
    p1.add_argument("--concept", required=True)
    p1.add_argument("--language", required=True)
    p1.add_argument("--out", default=None)
    p2 = sub.add_parser("push")
    p2.add_argument("--config", default="hyperparameters.json")
    p2.add_argument("--kind", required=True, choices=["activations","directions","results"])
    p2.add_argument("--group-path", required=True,
                    help="repo-relative prefix, e.g. obligation/yoruba/qwen3-8b__concept")
    p2.add_argument("--path", required=True)
    a = ap.parse_args(); cfg = _cfg(a.config)
    if a.cmd == "pull-dataset":
        out = a.out or f"{a.concept}_{a.language}.json"
        pull_dataset(cfg, out, concept=a.concept, language=a.language)
    elif a.cmd == "push":
        push(cfg, a.kind, a.group_path, a.path)

if __name__ == "__main__":
    main()
