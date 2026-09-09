#!/usr/bin/env python3
"""
Per-layer logit-lens for the full sweep: every (dataset_variant, contrast,
position, language, model) combo extract_dim.py produced, and EVERY layer of
each one's candidate tensor -> raw top/bottom tokens per layer. No vocabulary
scoring, no verdict -- you read/translate the tokens.

Directions live on HF one-per-combo (extract_dim.py no longer writes a combined
multi-direction file): concept/variant/contrast/position/language/model__preset/
dim_candidates.pt, holding {"{contrast}__{position}": tensor[n_layers+1, d]}.
This script pulls each one directly rather than requesting a list of names, so
"run for everything" doesn't require enumerating combos by hand.

Labeling is explicit to avoid the classic off-by-one:
  candidate tensors are indexed 0..n_layers where index 0 is the EMBEDDING output
  and index i>=1 is the residual stream AFTER decoder block (i-1). Each row prints
  tensor_index and the resid interpretation.

An optional random-direction column (same layer) is included as a noise baseline:
tokens that look like the random column are not meaningful.

Outputs (local + optional HF push), one dir per combo:
  <out>/<concept>/<variant>/<contrast>/<position>/<language>/<model>__<preset>/
    per_layer.json   tensor_index, resid_after_block, top_plus, top_minus, top_random
    per_layer.csv    tensor_index, resid_after_block, top_plus(joined), top_minus(joined)
No plots (kept clean); the tokens are the artifact.

Like extract_dim.py, model loading is the expensive part and is per-MODEL, not
per-combo: for a given model_key the tokenizer and unembedding matrix are the
same regardless of language/variant/contrast/position, so the model loads once
and every combo under it reuses it. The model's weights are evicted from the
local HF cache once its combos are done (see hf_io.evict_model_cache) for the
same reason extract_dim.py does this -- two 8B+ models cached at once can blow
a constrained disk quota.

All of one model's combo outputs are pushed to the results repo (under
logit_lens/) as ONE commit, not one commit per combo -- HF caps repo commits at
128/hour, and a 200+-combo sweep would blow through that at one commit each.

A missing direction file (that combo's extract_dim.py run never completed, or
hasn't been pushed yet) is skipped with a log line regardless of
--continue-on-error, since it's a data-availability gap, not a bug; an actual
error computing/writing the lens for a combo that WAS found still respects
--continue-on-error the normal way.

Loads .env for HF_TOKEN.

Usage:
  python logit_lens_all_layers.py --models qwen3-8b llama3.1-8b \
    --languages en de hi ig it ko ru tr ur yo \
    --dataset-variants frame scenario \
    --concept obligation --preset rule_following --push --continue-on-error
"""
import os, json, argparse, csv, gc
from pathlib import Path
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import hf_io
from extract_dim import load_cfg  # reuses preset application + guardrails, no duplication

load_dotenv()
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@torch.no_grad()
def project_tokens(W, tok, vec_unit, topk):
    u = vec_unit.to(W.dtype).to(W.device)
    s = W @ u
    dec = lambda ids: [tok.decode([i]).strip() for i in ids]
    top = dec(torch.topk(s, topk).indices.tolist())
    bot = dec(torch.topk(-s, topk).indices.tolist())
    return top, bot


@torch.no_grad()
def per_layer_lens(model, tok, cand_tensor, topk, seed=0, with_random=True):
    """cand_tensor: [n_layers+1, d]. index 0 = embedding output; index i = resid
    after decoder block (i-1). Returns list of per-index dicts with raw tokens."""
    W = model.get_output_embeddings().weight  # [vocab, d]
    d = cand_tensor.shape[1]
    rows = []
    g = torch.Generator().manual_seed(seed)
    for idx in range(cand_tensor.shape[0]):
        v = cand_tensor[idx].float()
        resid = "embedding" if idx == 0 else f"after_block_{idx-1}"
        if v.norm() < 1e-6:
            rows.append({"tensor_index": idx, "resid_after_block": resid, "empty": True})
            continue
        top, bot = project_tokens(W, tok, v / v.norm(), topk)
        entry = {"tensor_index": idx, "resid_after_block": resid, "empty": False,
                 "top_plus": top, "top_minus": bot}
        if with_random:
            rnd = torch.randn(d, generator=g)
            rtop, _ = project_tokens(W, tok, (rnd / rnd.norm()), topk)
            entry["top_random_baseline"] = rtop
        rows.append(entry)
    return rows


def write_combo(out_dir, name, rows, with_random):
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_dir / "per_layer.json", "w"), indent=2, ensure_ascii=False)
    with open(out_dir / "per_layer.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["tensor_index", "resid_after_block", "top_plus", "top_minus"]
        if with_random: header.append("top_random_baseline")
        w.writerow(header)
        for r in rows:
            if r.get("empty"):
                w.writerow([r["tensor_index"], r["resid_after_block"], "EMPTY", "", ""]); continue
            row = [r["tensor_index"], r["resid_after_block"],
                   " ".join(r["top_plus"]), " ".join(r["top_minus"])]
            if with_random: row.append(" ".join(r.get("top_random_baseline", [])))
            w.writerow(row)
    valid = [r for r in rows if not r.get("empty")]
    mid = valid[len(valid) // 2] if valid else None
    preview = f"mid-layer +[{' '.join(mid['top_plus'][:6])}]" if mid else "(all-empty)"
    print(f"  [{name}] {len(valid)} layers -> {out_dir}/per_layer.csv  {preview}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="one or more model keys from config.models, e.g. --models qwen3-8b llama3.1-8b")
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--preset", default="rule_following")
    ap.add_argument("--concept", default="obligation")
    ap.add_argument("--languages", nargs="+", required=True,
                    help="one or more language codes, e.g. --languages en de ig")
    ap.add_argument("--dataset-variants", nargs="+", choices=["frame", "scenario"], default=["frame"])
    ap.add_argument("--dim-candidates", default=None,
                    help="local root mirroring the directions repo layout "
                         "(<concept>/<variant>/<contrast>/<position>/<lang>/<model>__<preset>/dim_candidates.pt); "
                         "if omitted, each combo is hub-downloaded individually")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--no-random", action="store_true", help="skip the random-direction baseline column")
    ap.add_argument("--out", default="logitlens_out")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="log and skip a combo that errors while computing/writing its lens, "
                         "instead of aborting the whole sweep. A combo whose direction file "
                         "simply isn't on the hub yet is always skipped, with or without this.")
    ap.add_argument("--keep-model-cache", action="store_true",
                    help="don't evict a model's downloaded weights from the local HF cache "
                         "once its combos are done (see extract_dim.py's flag of the same name)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_root = Path(args.out)
    token = os.environ.get("HF_TOKEN")
    cfg0 = json.load(open(args.config))
    acfg = cfg0["ablation"]

    combos = [(m, l, v) for m in args.models for l in args.languages for v in args.dataset_variants]
    print(f"[sweep] {len(args.models)} model(s) x {len(args.languages)} language(s) x "
          f"{len(args.dataset_variants)} variant(s) = {len(combos)} (model, language, variant) groups "
          f"(each expands to contrasts x positions directions)")

    completed, skipped, failures = [], [], []
    for model_key in args.models:
        cfg, mcfg = load_cfg(args.config, model_key, preset=args.preset)
        ecfg = cfg["extraction"]; positions = ecfg["positions"]; contrasts = ecfg["contrasts"]
        preset_tag = ecfg.get("_active_preset", ecfg["stimulus_mode"])
        run_tag = f"{model_key}__{preset_tag}"

        print(f"\n{'=' * 70}\n[load model] {model_key}\n{'=' * 70}")
        tok = AutoTokenizer.from_pretrained(mcfg["hf_name"], trust_remote_code=mcfg.get("trust_remote_code", False))
        model = AutoModelForCausalLM.from_pretrained(
            mcfg["hf_name"], torch_dtype=DTYPES[args.dtype], device_map=args.device,
            trust_remote_code=mcfg.get("trust_remote_code", False)).eval()

        model_patterns = []
        try:
            for lang in args.languages:
                for variant in args.dataset_variants:
                    for contrast in contrasts:
                        for position in positions:
                            name = f"{contrast}__{position}"
                            tag = f"{args.concept}/{variant}/{contrast}/{position}/{lang}/{run_tag}"

                            # ---- fetch this combo's direction (skip, don't fail, if absent) ----
                            try:
                                if args.dim_candidates:
                                    cand_path = os.path.join(args.dim_candidates, tag, "dim_candidates.pt")
                                    if not os.path.exists(cand_path):
                                        raise FileNotFoundError(cand_path)
                                else:
                                    from huggingface_hub import hf_hub_download
                                    cand_path = hf_hub_download(acfg["directions_repo"], f"{tag}/dim_candidates.pt",
                                                                repo_type="dataset", token=token)
                                cands = torch.load(cand_path)
                            except Exception as e:
                                print(f"[skip] {tag}: direction not found ({e!r})")
                                skipped.append((tag, repr(e)))
                                continue
                            if name not in cands:
                                print(f"[skip] {tag}: file present but missing key {name!r}")
                                skipped.append((tag, f"missing key {name}"))
                                continue

                            # ---- compute + write (respects --continue-on-error) ----
                            try:
                                rows = per_layer_lens(model, tok, cands[name], args.topk,
                                                      with_random=not args.no_random)
                                combo_dir = out_root / args.concept / variant / contrast / position / lang / run_tag
                                write_combo(combo_dir, tag, rows, with_random=not args.no_random)
                                model_patterns.append(f"{args.concept}/{variant}/{contrast}/{position}/{lang}/{run_tag}/*")
                                completed.append(tag)
                            except Exception as e:
                                if not args.continue_on_error:
                                    raise
                                print(f"[error] {tag} failed: {e!r} -- continuing (--continue-on-error)")
                                failures.append((tag, repr(e)))
        finally:
            print(f"[unload model] {model_key}")
            del tok, model
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if not args.keep_model_cache:
                hf_io.evict_model_cache(mcfg["hf_name"])

        if args.push and model_patterns:
            hf_io.push_batch(cfg, "results", str(out_root), model_patterns, path_in_repo="logit_lens",
                             commit_message=f"logit_lens: {len(model_patterns)} combos for {model_key}")

    print(f"\n[sweep] {len(completed)} completed, {len(skipped)} skipped (no direction on hub), "
          f"{len(failures)} failed")
    for tag, info in skipped:
        print(f"  [skip]  {tag}: {info}")
    for tag, info in failures:
        print(f"  [error] {tag}: {info}")
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
