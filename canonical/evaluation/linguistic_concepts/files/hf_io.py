#!/usr/bin/env python3
"""
Hugging Face I/O for the obligation-direction pipeline.

- pull the canonical dataset (nunaa/canonical_obligation_dataset), either the
  frame-generalization split ({lang}_{split}.json) or the scenario-generalization
  split ({lang}_{split}_scenario.json), selected via `dataset_variant`.
- create (if missing) and push to three repos:
    activations -> nunaa/crosslingual_rf-activations
    directions  -> nunaa/canonical_crosslingual_rf-directions
    results     -> nunaa/canonical_crosslingual_rf-results (AUC/ prefix; see extract_dim.py)

Auth: set HF_TOKEN in the environment (or the env name in config.hf.token_env).

CLI:
    python hf_io.py pull-dataset --config hyperparameters.json --concept obligation --language ig --split test --variant scenario --out obligation_ig_test_scenario.json
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
def _rows_from_parquet(fp):
    import pandas as pd
    return pd.read_parquet(fp).to_dict(orient="records")

def _rows_from_jsonl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]

def pull_dataset(cfg, out_path, concept=None, language=None, split=None, dataset_variant="frame"):
    """Pull one split of the canonical dataset for a given (concept, language).

    `split` (e.g. 'train' / 'test') overrides config.hf.dataset_split when given.
    `dataset_variant` selects which generalization split the file belongs to:
      "frame"    -> data/{language}_{split}.json           (disjoint frame_ids)
      "scenario" -> data/{language}_{split}_scenario.json  (disjoint scenarios)
    `want_split` (train/test) still drives the row-level "split" column filter,
    since row['split'] is the same base train/test tag under either variant.
    Handles the common HF layouts:
      A. a `datasets`-format repo (Parquet/Arrow under data/, possibly many configs)
      B. explicit files anywhere in the repo (root OR data/), matched by name,
         with split-aware candidates (<concept>_<language>_<split>.* etc.).
    """
    hf = cfg["hf"]; token = _token(cfg)
    if hf.get("dataset_load", "hub") == "local":
        print("[hf] dataset_load=local; nothing to pull"); return
    if dataset_variant not in ("frame", "scenario"):
        raise SystemExit(f"[hf] dataset_variant must be 'frame' or 'scenario', got {dataset_variant!r}")
    repo = hf["dataset_repo"]
    want_split = split or hf.get("dataset_split", "train")
    file_split = want_split if dataset_variant == "frame" else f"{want_split}_scenario"

    # ---------- A) try datasets library (understands the repo's own format) ----------
    def _write(rows, how):
        json.dump(rows, open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"[hf] pulled {len(rows)} rows ({dataset_variant}) from {repo} via {how} -> {out_path}")

    try:
        from datasets import load_dataset, get_dataset_config_names
        split = want_split
        # if the dataset has named configs, prefer one matching concept/language
        cfg_name = None
        try:
            names = get_dataset_config_names(repo, token=token)
            if names:
                want = {f"{concept}_{language}", f"{concept}-{language}",
                        str(language), str(concept)}
                cfg_name = next((n for n in names if n in want), None)
                if cfg_name is None and len(names) == 1:
                    cfg_name = names[0]
        except Exception:
            pass
        ds = load_dataset(repo, cfg_name, split=split, token=token) if cfg_name \
            else load_dataset(repo, split=split, token=token)
        rows = [dict(r) for r in ds]
        # if concept/language/split are columns, filter to the requested slice
        if rows and concept and "concept" in rows[0]:
            rows = [r for r in rows if str(r.get("concept")) == str(concept)] or rows
        if rows and language and "language" in rows[0]:
            rows = [r for r in rows if str(r.get("language")) == str(language)] or rows
        if rows and "split" in rows[0]:
            rows = [r for r in rows if str(r.get("split")) == str(want_split)] or rows
        _write(rows, f"load_dataset(config={cfg_name}, split={split})")
        return
    except Exception as e:
        print(f"[hf] load_dataset path failed ({e}); trying direct file download")

    # ---------- B) recursive file listing (descend into data/ etc.) ----------
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        files = [f.rfilename for f in api.list_repo_tree(repo, repo_type="dataset",
                                                         recursive=True, token=token)]
    except Exception:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo, repo_type="dataset", token=token)

    data_files = [f for f in files if f.lower().endswith((".json", ".jsonl", ".parquet"))]
    if not data_files:
        raise SystemExit(f"[hf] no data files (.json/.jsonl/.parquet) in {repo}; saw: {files}")

    fmap = hf.get("dataset_files", {})
    key = f"{concept}/{language}/{file_split}"
    key2 = f"{concept}/{language}"
    candidates = []
    if fmap.get(key):
        candidates.append(fmap[key])
    if fmap.get(key2):
        candidates.append(fmap[key2])
    if language:
        # LANGUAGE-ONLY naming (no concept in the filename), e.g. data/yo_test.json
        # or data/yo_test_scenario.json for the scenario-generalization variant.
        # This is the actual layout for the cross-lingual repo, so it's checked FIRST.
        for ext in (".json", ".jsonl", ".parquet"):
            candidates += [
                f"{language}_{file_split}{ext}", f"data/{language}_{file_split}{ext}",
                f"{file_split}/{language}{ext}", f"data/{file_split}/{language}{ext}",
            ]
    if concept and language:
        stem = f"{concept}_{language}"
        for ext in (".json", ".jsonl", ".parquet"):
            # split-aware names first, then split subdir, then plain
            candidates += [
                f"{stem}_{file_split}{ext}", f"data/{stem}_{file_split}{ext}",
                f"{concept}/{language}/{file_split}{ext}", f"data/{concept}/{language}/{file_split}{ext}",
                f"{file_split}/{stem}{ext}", f"data/{file_split}/{stem}{ext}",
                f"{stem}{ext}", f"data/{stem}{ext}",
                f"{concept}/{language}{ext}", f"data/{concept}/{language}{ext}",
            ]

    pick = next((c for c in candidates if c in data_files), None)
    if pick is None:
        # prefer a file whose name contains the split, then parquet, then first
        split_hits = [f for f in data_files if file_split in f.lower()]
        pref = split_hits or [f for f in data_files if f.endswith(".parquet")] or data_files
        pick = pref[0]
        print(f"[hf][warn] no exact match for {key}; falling back to {pick}. "
              f"available: {data_files}")

    fp = hf_hub_download(repo, pick, repo_type="dataset", token=token)
    if pick.endswith(".parquet"):
        rows = _rows_from_parquet(fp)
    elif pick.endswith(".jsonl"):
        rows = _rows_from_jsonl(fp)
    else:
        rows = json.load(open(fp))
    # if a split column exists in a combined file, filter
    if rows and isinstance(rows[0], dict) and "split" in rows[0]:
        rows = [r for r in rows if str(r.get("split")) == str(want_split)] or rows
    _write(rows, f"file {pick} (split={want_split})")

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
    p1.add_argument("--split", default=None, help="train | test (defaults to config.hf.dataset_split)")
    p1.add_argument("--variant", choices=["frame", "scenario"], default="frame",
                    help="frame: {split}.json (disjoint frame_ids). "
                         "scenario: {split}_scenario.json (disjoint scenarios).")
    p1.add_argument("--out", default=None)
    p2 = sub.add_parser("push")
    p2.add_argument("--config", default="hyperparameters.json")
    p2.add_argument("--kind", required=True, choices=["activations","directions","results"])
    p2.add_argument("--group-path", required=True,
                    help="repo-relative prefix, e.g. obligation/yoruba/scenario/qwen3-8b__concept")
    p2.add_argument("--path", required=True)
    a = ap.parse_args(); cfg = _cfg(a.config)
    if a.cmd == "pull-dataset":
        out = a.out or f"{a.concept}_{a.language}_{a.variant}.json"
        pull_dataset(cfg, out, concept=a.concept, language=a.language,
                     split=a.split, dataset_variant=a.variant)
    elif a.cmd == "push":
        push(cfg, a.kind, a.group_path, a.path)

if __name__ == "__main__":
    main()
