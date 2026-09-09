#!/usr/bin/env python3
"""
Causal selection of (l*, i*) for obligation directions, following Arditi et al. 2024.

For each candidate direction (contrast x position, per layer) this measures:
  - ablation effect : project the direction OUT of resid at all layers/positions,
                      measure change in the target-behaviour score
  - addition effect : add c * r at the extraction layer, all positions,
                      measure induced change
  - KL penalty      : KL(base || intervened) on a control set, to penalise
                      directions whose interventions break general behaviour

Behaviour score here is task-appropriate for obligation rules:
we score the log-prob mass the model puts on COMPLYING with the rule's action
vs a permissive continuation, at the first generated token(s). Because the
"right" target tokens depend on the model, this is wired via a small scoring
spec you can edit in hyperparameters.json -> selection.intervention.

Run AFTER extract_dim.py (needs dim_candidates_<model>.pt).
    python intervene_select.py --model qwen3-8b --config hyperparameters.json
"""
import os, json, argparse, math
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

from extract_dim import load_cfg  # shared: applies presets + guardrails

# ------- resid hooks ------------------------------------------------------- #
def get_layers(model):
    # works for Llama/Qwen HF CausalLM
    return model.model.layers

def add_hook_ablate(model, unit_dir):
    """Ablate unit_dir from the output resid of every decoder layer (all positions)."""
    handles = []
    u = unit_dir  # [d] on device
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        proj = (h * u).sum(-1, keepdim=True) * u
        h2 = h - proj
        if isinstance(out, tuple):
            return (h2,) + tuple(out[1:])
        return h2
    for lyr in get_layers(model):
        handles.append(lyr.register_forward_hook(hook))
    return handles

def add_hook_addition(model, vec, layer_idx):
    """Add vec to the output resid of a single layer (all positions)."""
    handles = []
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h2 = h + vec
        if isinstance(out, tuple):
            return (h2,) + tuple(out[1:])
        return h2
    handles.append(get_layers(model)[layer_idx].register_forward_hook(hook))
    return handles

def clear(handles):
    for h in handles: h.remove()

# ------- scoring ----------------------------------------------------------- #
@torch.no_grad()
def next_token_logits(model, tok, prompt_ids, device):
    """prompt_ids: a list of token ids (from build_prompt)."""
    input_ids = torch.tensor([prompt_ids], device=device)
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    return out.logits[0, -1].float().cpu()  # [vocab]

def target_score(logits, tok, comply_words, permit_words):
    """log P(comply) - log P(permit) using first-subtoken ids of each word list."""
    def mass(words):
        ids = set()
        for w in words:
            for variant in (" "+w, w):
                t = tok(variant, add_special_tokens=False).input_ids
                if t: ids.add(t[0])
        if not ids: return torch.tensor(-1e9)
        lp = torch.log_softmax(logits, -1)
        return torch.logsumexp(lp[list(ids)], 0)
    return float(mass(comply_words) - mass(permit_words))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--preset", default="rule_following",
                    help="which extraction preset these candidates came from")
    ap.add_argument("--language", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg, mcfg = load_cfg(args.config, args.model, preset=args.preset)
    icfg = cfg["selection"]["intervention"]
    dcfg = cfg["data"]
    preset_tag = cfg["extraction"].get("_active_preset", args.preset)
    lang, concept = args.language, args.concept
    run_tag = f"{args.model}__{preset_tag}"
    group_path = f"{concept}/{lang}/{run_tag}"
    local_dir = os.path.join(cfg["output"]["dir"], concept, lang, run_tag)
    data_path = args.data or f"{concept}_{lang}.json"
    rows = json.load(open(data_path))
    if args.limit: rows = rows[:args.limit]

    device = mcfg.get("device", "cuda")
    tok = AutoTokenizer.from_pretrained(mcfg["hf_name"], trust_remote_code=mcfg.get("trust_remote_code", False))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_name"], torch_dtype=DTYPES[mcfg["dtype"]],
        device_map=device, trust_remote_code=mcfg.get("trust_remote_code", False)).eval()

    cand_path = args.candidates or os.path.join(local_dir, "dim_candidates.pt")
    cands = torch.load(cand_path)   # name -> [nl+1, d]

    # scoring word lists: comply = do the mandated action; permit = leave it optional.
    # Edit these in the config if you want model-specific target tokens.
    comply_words = icfg.get("comply_words", ["Yes", "You", "I", "Please", "Consult", "Refuse"])
    permit_words = icfg.get("permit_words", ["It", "Maybe", "Optionally", "You", "Sure"])

    # build eval prompts (constant query + the CLEAN rule, i.e. obligation present).
    # Intervention/steering is only well-defined when there is behaviour to move,
    # i.e. the rule-following framing. We honour the active stimulus_mode but warn
    # if it is a concept mode (no downstream action to bypass/induce).
    from extract_dim import build_prompt
    mode = cfg["extraction"].get("stimulus_mode", "system_user")
    if mode != "system_user":
        print(f"[intervene][warn] stimulus_mode='{mode}': ablate/add scores measure "
              f"next-token shifts on a bare stimulus, not rule adherence. "
              f"Behavioural selection is intended for 'system_user'.")
    eval_rows = rows[: icfg.get("n_eval_prompts", 40)]
    prompts = []  # list of id-lists
    for r in eval_rows:
        ids, _, _ = build_prompt(
            tok, mcfg, mode,
            rule_text=r[dcfg["rule_field_map"]["clean"]],
            context=r.get(dcfg["context_field"]) if mode == "system_user" else None,
            query=r.get(dcfg["query_field"]) if mode == "system_user" else None,
            raw_add_special_tokens=cfg["extraction"].get("raw_add_special_tokens", True),
        )
        prompts.append(ids)

    # baseline scores
    base_logits = [next_token_logits(model, tok, p, device) for p in prompts]
    base_score = sum(target_score(l, tok, comply_words, permit_words) for l in base_logits) / len(base_logits)

    report = {"model": args.model, "preset": preset_tag, "concept": concept, "language": lang,
              "stimulus_mode": mode, "baseline_target_score": base_score, "candidates": {}}

    for name, dvec in cands.items():
        dvec = dvec.to(device)                              # [nl+1, d]
        nl = dvec.shape[0]
        # evaluate a few candidate layers around the separation-best to save compute
        # (here: sweep all non-embedding layers; comment to restrict)
        layer_range = range(1, nl)   # skip embedding layer 0
        best = None
        per_layer = {}
        for l in layer_range:
            r_l = dvec[l]
            if r_l.norm() < 1e-6: 
                continue
            u = r_l / r_l.norm()

            # ablation: remove u everywhere, expect obligation signal to DROP
            h = add_hook_ablate(model, u)
            abl = [next_token_logits(model, tok, p, device) for p in prompts]
            clear(h)
            abl_score = sum(target_score(x, tok, comply_words, permit_words) for x in abl)/len(abl)

            # addition: add r at layer l everywhere, expect obligation signal to RISE
            add_vec = icfg.get("add_coeff", 1.0) * r_l
            h = add_hook_addition(model, add_vec, l)
            add = [next_token_logits(model, tok, p, device) for p in prompts]
            clear(h)
            add_score = sum(target_score(x, tok, comply_words, permit_words) for x in add)/len(add)

            # KL on control: base vs ablated distribution (proxy for collateral damage)
            kl = 0.0
            for lb, la in zip(base_logits, abl):
                pb = torch.log_softmax(lb, -1); pa = torch.softmax(la, -1)
                kl += torch.sum(pa * (torch.log(pa+1e-9) - pb)).item()
            kl /= len(base_logits)

            bypass = base_score - abl_score      # want > 0 (ablation removes obligation)
            induce = add_score - base_score      # want > 0 (addition adds obligation)
            score  = bypass + induce - icfg.get("kl_weight", 1.0)*max(kl,0) if icfg.get("kl_penalty", True) else bypass+induce
            per_layer[l] = {"bypass": bypass, "induce": induce, "kl": kl, "score": score}
            if best is None or score > best[1]:
                best = (l, score)

        report["candidates"][name] = {"best_layer": best[0] if best else None,
                                      "best_score": best[1] if best else None,
                                      "per_layer": per_layer}
        if best:
            print(f"  {name:38s} -> l*={best[0]:>2}  score={best[1]:+.3f}  "
                  f"(bypass={per_layer[best[0]]['bypass']:+.3f} induce={per_layer[best[0]]['induce']:+.3f} kl={per_layer[best[0]]['kl']:.3f})")

    os.makedirs(local_dir, exist_ok=True)
    out = os.path.join(local_dir, "intervention_report.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")

if __name__ == "__main__":
    main()
