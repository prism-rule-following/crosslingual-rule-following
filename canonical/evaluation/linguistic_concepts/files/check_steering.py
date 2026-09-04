#!/usr/bin/env python3
"""
Steering (addition) check: the INVERSE of ablation, and the strongest single
causal-validity test.

ADD the direction to the residual stream at all layers >= its layer, on NEUTRAL
prompts, and measure whether the model's output shifts toward obligation language,
compared to adding a random direction of the same norm at the same layer.

Two read-outs:
  1. next-token log-prob mass on obligation vs permission vocabulary (quantitative)
  2. a base + steered generation sample AT EVERY COEFFICIENT (qualitative eyeball,
     not just at the largest one -- lets you see the dose-response in the actual
     text, not just the logprob numbers)

  obligation_specific_shift = (obligation-mass gain from the direction)
                            - (obligation-mass gain from a random direction)
  > 0 and growing with coefficient, WITHOUT the output collapsing, means the
  direction causally induces obligation -> it encodes the concept, not just
  correlates with it. ~0 means it is inert as a causal handle (correlate only).

DIRECTION LANGUAGE vs PROMPT LANGUAGE are independent:
  --dir-language : which extracted direction to load (default "en"). Set this to
                   the SAME value as --language to use a direction extracted
                   natively in that language, instead of the default cross-lingual
                   transfer test (English-extracted direction steered on
                   --language prompts).
  --language     : which language's prompts to steer on and generate/compare in.

THINKING is a binary switch (default OFF, --enable-thinking to turn on). If you
enable it, raise --max-new-tokens well above your typical reasoning-trace length
or generations will be truncated mid-thought.

The coefficient is auto-scaled to the residual norm at the target layer (a fixed
small coeff is meaningless when norms are ~500-1500), and a sweep is run so you can
see the dose-response and where (if anywhere) the model breaks.

Loads .env (HF_TOKEN etc).

Usage:
  python check_steering.py --model Qwen/Qwen3-8B --model-key qwen3-8b \
      --concept obligation --language yo --dir-language en --preset rule_following \
      --name must_may__post_instruction --coeff-fracs 0.05,0.1,0.2,0.4 \
      --max-new-tokens 768
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

OBLIGATION_WORDS = ["must", "shall", "required", "require", "obligated", "obligation",
                    "mandatory", "obliged", "duty", "necessary", "need"]
PERMISSION_WORDS = ["may", "can", "optional", "allowed", "allow", "permitted", "permit",
                    "could", "free", "welcome", "able"]


def get_layers(model):
    return model.model.layers


class AddDir:
    """add coeff * unit to resid at all layers >= from_layer."""
    def __init__(self, model, unit, from_layer, coeff):
        self.h = []; self.u = unit; self.c = coeff
        for i, blk in enumerate(get_layers(model)):
            if i >= from_layer:
                self.h.append(blk.register_forward_hook(self._hook()))
    def _hook(self):
        u, c = self.u, self.c
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h + c * u.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook
    def remove(self):
        for x in self.h: x.remove()
    def __enter__(self): return self
    def __exit__(self, *a): self.remove()


def load_rows(p):
    if p.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(p).to_dict(orient="records")
    return [json.loads(l) for l in open(p) if l.strip()]


def strip_think(t):
    """Same convention as steering_poc.py: thinking is a binary switch, this is a
    safety net for a misconfigured run, not a designed-for state. Returns
    (answer_text, truncated). Unterminated <think> -> ("", True): no real answer."""
    if "<think>" in t:
        if "</think>" in t:
            return t.split("</think>", 1)[1].strip(), False
        return "", True
    return t, False


def fmt_prompt(tok, r, enable_thinking=False):
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": r.get("system", "")},
             {"role": "user", "content": r.get("user_query", "")}],
            tokenize=False, add_generation_prompt=True, **kw)
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{r.get('system','')}\n\n{r.get('user_query','')}"}],
            tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def resid_norm_at(model, tok, prompts, from_layer, device):
    """mean ||h|| at the target layer, to scale the steering coefficient sensibly."""
    got = {}
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        got["n"] = h.norm(dim=-1).float().mean().item()
    hd = get_layers(model)[from_layer].register_forward_hook(hook)
    enc = tok(prompts, padding=True, return_tensors="pt").to(device)
    model(**enc); hd.remove()
    return got["n"]


@torch.no_grad()
def obligation_mass(model, tok, prompts, device, add=None, batch=8):
    def mass(logits, words):
        ids = set()
        for w in words:
            for var in (" " + w, w):
                t = tok(var, add_special_tokens=False).input_ids
                if t: ids.add(t[0])
        lp = torch.log_softmax(logits, -1)
        return torch.logsumexp(lp[:, list(ids)], -1)     # [B]
    tok.padding_side = "left"
    ob, pe = [], []
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s+batch]
        enc = tok(chunk, padding=True, return_tensors="pt").to(device)
        if add is None:
            logits = model(**enc).logits[:, -1, :].float()
        else:
            ctx = add()
            try:
                logits = model(**enc).logits[:, -1, :].float()
            finally:
                ctx.remove()
        ob += mass(logits, OBLIGATION_WORDS).cpu().tolist()
        pe += mass(logits, PERMISSION_WORDS).cpu().tolist()
    return float(np.mean(ob)), float(np.mean(pe))


@torch.no_grad()
def sample_generations(model, tok, prompts, device, max_new, add=None, k=1):
    """Generate k samples. Returns list of (raw_text, answer_text, truncated)."""
    tok.padding_side = "left"
    chunk = prompts[:k]
    enc = tok(chunk, padding=True, return_tensors="pt").to(device)
    gen = lambda: model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
    if add is None:
        out = gen()
    else:
        ctx = add()
        try: out = gen()
        finally: ctx.remove()
    L = enc["input_ids"].shape[1]
    raws = [tok.decode(o[L:], skip_special_tokens=True) for o in out]
    results = []
    for raw in raws:
        answer, truncated = strip_think(raw)
        results.append({"raw": raw, "answer": answer, "think_truncated": truncated})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--concept", default="obligation")
    ap.add_argument("--language", default="en", help="language of the PROMPTS steered on")
    ap.add_argument("--dir-language", default="en",
                    help="language the DIRECTION was extracted from (default 'en', the "
                        "cross-lingual transfer test). Set equal to --language to use a "
                        "direction extracted natively in that language instead.")
    ap.add_argument("--preset", default="rule_following")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dim-report", default=None)
    ap.add_argument("--dim-candidates", default=None)
    ap.add_argument("--neutral-data", default=None, help="neutral prompts (local override)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--coeff-fracs", default="0.05,0.1,0.2,0.4",
                    help="steering coefficients as FRACTIONS of the layer's mean resid norm")
    ap.add_argument("--coeffs", default=None,
                    help="ABSOLUTE steering coefficients (comma list). If given, overrides "
                         "--coeff-fracs; each is added as coeff*unit at all layers >= L. "
                         "Interpret relative to the layer's resid norm (printed at runtime).")
    ap.add_argument("--max-new-tokens", type=int, default=768,
                    help="raised from the old 250 default: short generations get "
                        "truncated before the dose-response is visible in the text, "
                        "and truncation is especially costly if thinking is enabled.")
    ap.add_argument("--enable-thinking", type=lambda s: s.lower() != "false", default=False,
                    help="Qwen3 thinking mode. Binary switch: OFF (default, no <think> "
                        "tokens ever appear) or ON with --max-new-tokens large enough "
                        "that the block always closes. Not designed to run truncated.")
    ap.add_argument("--gen-samples", type=int, default=1,
                    help="how many base+steered generation samples to show PER COEFFICIENT")
    ap.add_argument("--out", default="validate_out")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(args.config)); acfg = cfg["ablation"]
    token = os.environ.get("HF_TOKEN")
    from huggingface_hub import hf_hub_download
    # DIRECTION comes from dir_language's group path; PROMPTS come from language.
    dir_group = f"{args.concept}/{args.dir_language}/{args.model_key}__{args.preset}"
    hf = lambda repo, path: hf_hub_download(repo, path, repo_type="dataset", token=token)

    if args.enable_thinking and args.max_new_tokens < 512:
        print(f"[WARNING] enable_thinking=True with max_new_tokens={args.max_new_tokens}, "
              f"below the recommended floor of 512. Thinking traces will likely be cut "
              f"off mid-thought, producing samples with NO usable answer. Raise "
              f"--max-new-tokens before trusting these results.")

    rep_path = args.dim_report or hf(acfg["directions_results_repo"], f"{dir_group}/dim_report.json")
    cand_path = args.dim_candidates or hf(acfg["directions_repo"], f"{dir_group}/dim_candidates.pt")
    report = json.load(open(rep_path)); cands = torch.load(cand_path)
    d = report["directions"][args.name]
    ho = d.get("held_out")
    layer_dim = ho["at_train_best_layer"]["layer"] if ho else d["in_sample"]["best_layer"]
    from_layer = max(layer_dim - 1, 0)
    vec = cands[args.name][layer_dim].float()
    vec_unit = vec / (vec.norm() + 1e-8)
    print(f"[dir] {args.name} (extracted from '{args.dir_language}') "
          f"dim_idx={layer_dim} resid_layer={from_layer}")
    print(f"[prompts] language='{args.language}'  enable_thinking={args.enable_thinking}  "
          f"max_new_tokens={args.max_new_tokens}")

    slug = acfg["model_slug_map"].get(args.model_key, args.model.replace("/", "__"))
    sweep_path = args.neutral_data or hf(acfg["sweep_repo"], f"{acfg['sweep_subset']}/{slug}/{args.language}.parquet")
    rows = [r for r in load_rows(sweep_path) if str(r.get("pressure_level", "L0")) == "L0"]
    oblig = set(acfg.get("obligation_categories", []))
    # NEUTRAL prompts = non-obligation categories (so any obligation shift is induced, not primed)
    neu_rows = [r for r in rows if r.get("category") not in oblig][:args.n]
    if args.neutral_data:
        neu_rows = load_rows(args.neutral_data)[:args.n]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype],
                                                 device_map=args.device).eval()
    prompts = [fmt_prompt(tok, r, args.enable_thinking) for r in neu_rows]
    dev = args.device
    u = vec_unit.to(dev)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(vec.numel(), generator=g); rnd = (rnd / rnd.norm()).to(dev)

    norm = resid_norm_at(model, tok, prompts, from_layer, dev)
    print(f"[scale] mean resid norm at layer {from_layer}: {norm:.1f}")

    base_ob, base_pe = obligation_mass(model, tok, prompts, dev, add=None)
    print(f"[base] obligation logprob={base_ob:.3f} permission logprob={base_pe:.3f}")

    # generate the BASELINE sample(s) ONCE (unsteered, deterministic) -- reused
    # alongside every coefficient's steered sample below, rather than regenerated.
    base_gens = sample_generations(model, tok, prompts, dev, args.max_new_tokens,
                                   add=None, k=args.gen_samples)

    sweep = {}
    if args.coeffs:
        coeff_list = [(None, float(x)) for x in args.coeffs.split(",") if x.strip()]  # signed OK
    else:
        fr = [float(x) for x in args.coeff_fracs.split(",") if x.strip()]
        coeff_list = [(f, f * norm) for f in fr]
    for frac, coeff in coeff_list:
        key = frac if frac is not None else f"c{coeff:+g}"
        dir_ob, dir_pe = obligation_mass(model, tok, prompts, dev,
                                         add=lambda c=coeff: AddDir(model, u, from_layer, c))
        rnd_ob, rnd_pe = obligation_mass(model, tok, prompts, dev,
                                         add=lambda c=coeff: AddDir(model, rnd, from_layer, c))
        d_ob = dir_ob - base_ob; d_pe = dir_pe - base_pe
        r_ob = rnd_ob - base_ob; r_pe = rnd_pe - base_pe
        shift_ob = d_ob - r_ob            # obligation-specific (vs random)
        shift_pe = d_pe - r_pe            # permission-specific (vs random)

        # SAMPLE AT EVERY COEFFICIENT (not just the largest) -- steered generation
        # under this exact coefficient, shown next to the (reused) baseline sample.
        steer_gens = sample_generations(model, tok, prompts, dev, args.max_new_tokens,
                                        add=lambda c=coeff: AddDir(model, u, from_layer, c),
                                        k=args.gen_samples)

        sweep[key] = {"coeff": coeff, "frac_of_norm": coeff / norm,
                      "dir_delta_ob": d_ob, "dir_delta_pe": d_pe,
                      "rand_delta_ob": r_ob, "rand_delta_pe": r_pe,
                      "obligation_specific_shift": shift_ob,
                      "permission_specific_shift": shift_pe,
                      "sample_generations": {"base": base_gens, "steered": steer_gens}}
        print(f"[{key}] coeff={coeff:+8.1f} ({coeff/norm:+.2f}x)  "
              f"dOb={d_ob:+.2f} dPe={d_pe:+.2f}  "
              f"ob_shift={shift_ob:+.2f} pe_shift={shift_pe:+.2f}")
        for i in range(len(steer_gens)):
            b, s = base_gens[i], steer_gens[i]
            print(f"    base[{i}]  (truncated={b['think_truncated']}): {b['answer'][:300]!r}")
            print(f"    steer[{i}] (truncated={s['think_truncated']}): {s['answer'][:300]!r}")

    def is_degenerate(t):
        toks = t.split()
        if len(toks) < 5: return len(set(t)) < 6
        return (len(set(toks)) / len(toks) < 0.2) or (len(set(t)) < 8)
    # "broke" now checks EVERY coefficient's steered samples, not just the largest
    broke_by_coeff = {k: any(is_degenerate(g["answer"]) or g["think_truncated"]
                             for g in v["sample_generations"]["steered"])
                      for k, v in sweep.items()}
    broke = any(broke_by_coeff.values())

    pos = {k: v for k, v in sweep.items() if v["coeff"] > 0}
    neg = {k: v for k, v in sweep.items() if v["coeff"] < 0}
    # +coeff should push OBLIGATION vocab up; -coeff should push PERMISSION vocab up.
    pos_ok = pos and all(v["obligation_specific_shift"] > 0.3 for v in pos.values())
    neg_ok = neg and all(v["permission_specific_shift"] > 0.3 for v in neg.values())
    best_ob = max((v["obligation_specific_shift"] for v in pos.values()), default=float("nan"))
    best_pe = max((v["permission_specific_shift"] for v in neg.values()), default=float("nan"))

    if broke:
        verdict = ("NOT A CLEAN CAUSAL HANDLE: doses large enough to affect the output collapse "
                   "the model into degenerate/truncated text at one or more coefficients. In the "
                   "coherent band the pole-specific shift is ~0 or matched by a random direction "
                   "-> correlate, not a usable lever.")
    elif (pos and pos_ok) or (neg and neg_ok):
        verdict = ("DIRECTIONAL CAUSAL SIGNAL (coherent): +coeff pushes obligation and/or -coeff "
                   "pushes permission above a random direction, without breaking the model at "
                   "any tested coefficient. Worth confirming with judged adherence.")
    else:
        verdict = ("INERT (in coherent band): neither +coeff (obligation) nor -coeff (permission) "
                   "shifts vocabulary above a random direction -> correlate, not a causal handle.")
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"  +coeff best obligation_specific_shift: {best_ob:+.3f}" if pos else "  (no +coeffs)")
    print(f"  -coeff best permission_specific_shift: {best_pe:+.3f}" if neg else "  (no -coeffs)")
    print(f"  broke by coefficient: {broke_by_coeff}")

    result = {"direction": args.name, "dir_language": args.dir_language, "language": args.language,
              "resid_layer": from_layer, "resid_norm_at_layer": norm,
              "enable_thinking": args.enable_thinking, "max_new_tokens": args.max_new_tokens,
              "base_obligation_logprob": base_ob, "base_permission_logprob": base_pe,
              "sweep": sweep, "broke_by_coeff": broke_by_coeff,
              "best_obligation_specific_shift": best_ob,
              "best_permission_specific_shift": best_pe,
              "steered_degenerate_any": broke, "verdict": verdict}
    out_name = f"steering_{args.name}_{args.dir_language}to{args.language}.json"
    json.dump(result, open(out / out_name, "w"), indent=2, ensure_ascii=False)
    print(f"[done] {out}/{out_name}")


if __name__ == "__main__":
    main()
