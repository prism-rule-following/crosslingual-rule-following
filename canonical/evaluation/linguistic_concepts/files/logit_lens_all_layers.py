#!/usr/bin/env python3
"""
Per-layer logit-lens for a set of directions -> raw top/bottom tokens per layer.

For each direction and EACH layer index of the candidate tensor, project that
layer's slice through the unembedding and record the top-k and bottom-k tokens on
BOTH poles. No vocabulary scoring, no verdict -- you read/translate the tokens.

Labeling is explicit to avoid the classic off-by-one:
  candidate tensors are indexed 0..n_layers where index 0 is the EMBEDDING output
  and index i>=1 is the residual stream AFTER decoder block (i-1). Each row prints
  tensor_index and the resid interpretation.

An optional random-direction column (same layer) is included as a noise baseline:
tokens that look like the random column are not meaningful.

Outputs (local + optional HF push):
  <out>/<dir>/per_layer.json   tensor_index, resid_after_block, top_plus, top_minus, top_random
  <out>/<dir>/per_layer.csv    tensor_index, resid_after_block, top_plus(joined), top_minus(joined)
No plots (kept clean); the tokens are the artifact.

Loads .env for HF_TOKEN.

Usage:
  python logit_lens_all_layers.py --model Qwen/Qwen3-8B --model-key qwen3-8b \
    --concept obligation --language en --preset rule_following \
    --names must_neutral__contrast_token,must_neutral__rule_clause_end,must_may__rule_clause_end,must_may__contrast_token \
    --push
"""
import os, json, argparse, csv
from pathlib import Path
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--concept", default="obligation")
    ap.add_argument("--language", default="en")
    ap.add_argument("--preset", default="rule_following")
    ap.add_argument("--names", required=True, help="comma-separated direction names")
    ap.add_argument("--dim-candidates", default=None)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--no-random", action="store_true", help="skip the random-direction baseline column")
    ap.add_argument("--out", default="logitlens_out")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--results-repo", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(args.config)); acfg = cfg["ablation"]
    token = os.environ.get("HF_TOKEN")
    from huggingface_hub import hf_hub_download
    group = f"{args.concept}/{args.language}/{args.model_key}__{args.preset}"
    cand_path = args.dim_candidates or hf_hub_download(
        acfg["directions_repo"], f"{group}/dim_candidates.pt", repo_type="dataset", token=token)
    cands = torch.load(cand_path)
    names = [n.strip() for n in args.names.split(",") if n.strip()]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype],
                                                 device_map=args.device).eval()

    for name in names:
        if name not in cands:
            print(f"[skip] {name} not in candidates"); continue
        rows = per_layer_lens(model, tok, cands[name], args.topk,
                              with_random=not args.no_random)
        d_out = out / name; d_out.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(d_out / "per_layer.json", "w"), indent=2, ensure_ascii=False)
        with open(d_out / "per_layer.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["tensor_index", "resid_after_block", "top_plus", "top_minus"]
            if not args.no_random: header.append("top_random_baseline")
            w.writerow(header)
            for r in rows:
                if r.get("empty"):
                    w.writerow([r["tensor_index"], r["resid_after_block"], "EMPTY", "", ""]); continue
                row = [r["tensor_index"], r["resid_after_block"],
                       " ".join(r["top_plus"]), " ".join(r["top_minus"])]
                if not args.no_random: row.append(" ".join(r.get("top_random_baseline", [])))
                w.writerow(row)
        print(f"[{name}] wrote {len([r for r in rows if not r.get('empty')])} layers -> {d_out}/per_layer.csv")
        # console preview: a few representative layers
        valid = [r for r in rows if not r.get("empty")]
        for r in (valid[len(valid)//4], valid[len(valid)//2], valid[-1]):
            print(f"  idx {r['tensor_index']:>2} ({r['resid_after_block']}): +[{' '.join(r['top_plus'][:8])}]")

    if args.push:
        from huggingface_hub import create_repo, upload_folder
        repo = args.results_repo or acfg.get("directions_results_repo")
        create_repo(repo, repo_type="dataset", private=cfg["hf"].get("repo_private", True),
                    exist_ok=True, token=token)
        dest = f"logit_lens/{group}"
        upload_folder(folder_path=str(out), path_in_repo=dest, repo_id=repo,
                      repo_type="dataset", token=token, commit_message=f"per-layer logit-lens {group}")
        print(f"[hf] pushed {out} -> {repo}/{dest}")


if __name__ == "__main__":
    main()
