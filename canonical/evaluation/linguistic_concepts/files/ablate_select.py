#!/usr/bin/env python3
"""
Ablation-based causal selection of an obligation direction (Arditi-style).

Pipeline:
  1. Shortlist candidates from dim_report.json by held-out AUC:
     top-2 at contrast_token + top-2 at sentence_end  (4 candidates).
     Each candidate = (contrast, position, layer, unit-vector from dim_candidates.pt).
     + 1 random-direction control per candidate layer (specificity check).
  2. Baseline HELD looked up by the EXACT sweep ids from the pushed judge results.
  3. For each candidate: directional ablation (project the candidate's own-layer
     unit vector out of resid at ALL layers >= its layer) during generation on
     100 English sweep prompts. Plain-PyTorch forward hooks.
  4. KL guard: mean first-token KL vs baseline on neutral reference prompts,
     pre-registered cutoff 0.2 (also records median/p95/max).
  5. Coherence guard: local heuristic per response (4-gram repeat, gzip ratio,
     length), pre-registered coherence_rate >= 0.90. Per-response metrics saved.
  6. Judge the 100 ablated responses inline with GPT-mini (imported rubric).
  7. Winner = max HELD-drop among candidates passing BOTH gates. Selection-only
     (the causal claim lives in the cross-lingual patch, not here).

Outputs:
  <out>/selected_direction.pt        {vector, layer, contrast, position, ...}
  <out>/selection_report.json        per-candidate metrics + pre-registered gates

Judge functions are imported from judge_gpt_mini.py (single source of truth).

NOTE: this is designed to run on the pod (needs torch+CUDA, HF access, and the
Azure GPT env vars judge_gpt_mini.py expects). It is not runnable in the
sandbox; syntax is validated here.
"""

import os, json, gzip, argparse, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- reuse the judge exactly as the pipeline uses it -------------------------
# judge_gpt_mini.py builds the prompt, calls Azure GPT-mini, returns HELD/VIOLATED.
import importlib.util


def _import_judge(path):
    spec = importlib.util.spec_from_file_location("judge_gpt_mini", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# pre-registered gates (fixed BEFORE looking at results)
KL_CUTOFF = 0.2  # mean first-token KL vs baseline on neutral refs
COHERENCE_MIN = 0.90  # fraction of responses that must be coherent


# --------------------------------------------------------------------------- #
# candidate shortlist from dim_report.json
# --------------------------------------------------------------------------- #
def shortlist_candidates(
    report, n_per_pos=2, positions=("contrast_token", "sentence_end")
):
    """Rank directions by held-out AUC (at train-selected layer) per position,
    take top-n each. Returns list of dicts with contrast/position/layer/dim_index."""
    rows = []
    for name, d in report["directions"].items():
        pos = d["position"]
        if pos not in positions:
            continue
        ho = d.get("held_out")
        # held-out AUC at the train-selected layer (the honest number)
        if ho and "at_train_best_layer" in ho:
            layer = ho["at_train_best_layer"]["layer"]
            auc = ho["at_train_best_layer"]["auc"]
            auc_source = "held_out@train_best_layer"
        else:
            # fall back to in-sample if no test split was run (flagged in report)
            ins = d["in_sample"]
            layer = ins["best_layer"]
            auc = ins["best_auc"]
            auc_source = "in_sample@best_layer"
        rows.append(
            {
                "name": name,
                "contrast": d["contrast"],
                "position": pos,
                "layer_dim_index": layer,
                "auc": auc,
                "auc_source": auc_source,
            }
        )
    # top n per position by AUC
    picked = []
    for pos in positions:
        pos_rows = sorted(
            [r for r in rows if r["position"] == pos],
            key=lambda r: r["auc"],
            reverse=True,
        )[:n_per_pos]
        picked += pos_rows
    return picked


def dim_index_to_resid_layer(dim_index):
    """dim_candidates.pt is indexed 0..n_layers with index 0 = embedding.
    resid_post of decoder layer L corresponds to dim index L+1, so the
    decoder layer to hook from is dim_index - 1."""
    return max(dim_index - 1, 0)


# --------------------------------------------------------------------------- #
# ablation hooks (plain PyTorch): project unit dir out of resid at layers >= L
# --------------------------------------------------------------------------- #
def get_decoder_layers(model):
    # Llama / Qwen: model.model.layers
    return model.model.layers


class DirectionalAblation:
    """Register forward hooks that project `unit` out of each decoder block's
    residual output, for every block index >= from_layer. Arditi directional
    ablation: same direction removed at all layers >= L."""

    def __init__(self, model, unit_vec, from_layer):
        self.handles = []
        u = unit_vec / (unit_vec.norm() + 1e-8)
        self.u = u
        layers = get_decoder_layers(model)
        for i, blk in enumerate(layers):
            if i >= from_layer:
                self.handles.append(blk.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        u = self.u

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # project out: h <- h - (h . u) u   (u on same device/dtype)
            uu = u.to(dtype=h.dtype, device=h.device)
            coef = (h * uu).sum(-1, keepdim=True)
            h2 = h - coef * uu
            if isinstance(out, tuple):
                return (h2,) + tuple(out[1:])
            return h2

        return hook

    def remove(self):
        for hdl in self.handles:
            hdl.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


# --------------------------------------------------------------------------- #
# prompt formatting (faithful to inference.py)
# --------------------------------------------------------------------------- #
def format_chat_prompt(tok, system, user, supports_system, enable_thinking=None):
    if supports_system:
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    else:
        chat = [{"role": "user", "content": f"{system}\n\n{user}"}]
    kw = {}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    return tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, **kw
    )


def supports_system_role(tok):
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


@torch.no_grad()
def generate_batch(model, tok, prompts, device, max_new_tokens, temperature, do_sample):
    tok.padding_side = "left"
    enc = tok(prompts, padding=True, return_tensors="pt").to(device)
    gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if do_sample:
        gen_kw.update(do_sample=True, temperature=max(temperature, 1e-5))
    else:
        gen_kw.update(do_sample=False)
    out = model.generate(**enc, **gen_kw)
    in_len = enc["input_ids"].shape[1]
    return [tok.decode(o[in_len:], skip_special_tokens=True) for o in out]


# --------------------------------------------------------------------------- #
# KL guard: mean first-token KL vs baseline on neutral reference prompts
# --------------------------------------------------------------------------- #
@torch.no_grad()
def first_token_kl(model, tok, ref_prompts, unit_vec, from_layer, device, batch=8):
    """KL(baseline || ablated) on the first-token (last-position) distribution,
    averaged over ref_prompts. Returns dict of mean/median/p95/max."""
    tok.padding_side = "left"
    kls = []
    for s in range(0, len(ref_prompts), batch):
        chunk = ref_prompts[s : s + batch]
        enc = tok(chunk, padding=True, return_tensors="pt").to(device)
        base_logits = model(**enc).logits[:, -1, :].float()
        with DirectionalAblation(model, unit_vec, from_layer):
            abl_logits = model(**enc).logits[:, -1, :].float()
        p = torch.log_softmax(base_logits, -1)
        q = torch.log_softmax(abl_logits, -1)
        # KL(P||Q) = sum P (logP - logQ)
        kl = (p.exp() * (p - q)).sum(-1)  # [B]
        kls += kl.cpu().tolist()
    a = np.array(kls)
    return {
        "mean_kl": float(a.mean()),
        "median_kl": float(np.median(a)),
        "p95_kl": float(np.percentile(a, 95)),
        "max_kl": float(a.max()),
        "n_ref": len(kls),
    }


# --------------------------------------------------------------------------- #
# coherence heuristic (local, no API)
# --------------------------------------------------------------------------- #
def ngram_repeat_ratio(text, n=4):
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return 1.0 - uniq / len(grams)  # high = repetitive


def compression_ratio(text):
    b = text.encode("utf-8")
    if len(b) == 0:
        return 1.0
    return len(gzip.compress(b)) / len(b)  # low = repetitive/degenerate


def is_coherent(text, min_len=3, max_repeat=0.5, min_compress=0.30):
    toks = text.split()
    if len(toks) < min_len:
        return False, {"reason": "too_short", "len": len(toks)}
    rep = ngram_repeat_ratio(text)
    comp = compression_ratio(text)
    if rep > max_repeat:
        return False, {
            "reason": "high_ngram_repeat",
            "ngram_repeat": rep,
            "compression": comp,
            "len": len(toks),
        }
    if comp < min_compress:
        return False, {
            "reason": "low_compression",
            "ngram_repeat": rep,
            "compression": comp,
            "len": len(toks),
        }
    return True, {
        "reason": None,
        "ngram_repeat": rep,
        "compression": comp,
        "len": len(toks),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument(
        "--model-key",
        required=True,
        help="config key for slug map + directions path, e.g. qwen3-8b",
    )
    ap.add_argument("--concept", default="obligation")
    ap.add_argument("--language", default="en")
    ap.add_argument(
        "--preset",
        default="concept_raw",
        help="which DIM preset's directions to use (matches the extract run)",
    )
    # optional LOCAL overrides (if given, skip the HF pull for that input)
    ap.add_argument(
        "--dim-report", default=None, help="local override for dim_report.json"
    )
    ap.add_argument(
        "--dim-candidates", default=None, help="local override for dim_candidates.pt"
    )
    ap.add_argument(
        "--sweep-data", default=None, help="local override for the sweep parquet/jsonl"
    )
    ap.add_argument(
        "--kl-ref-data", default=None, help="local override for neutral KL refs"
    )
    ap.add_argument(
        "--baseline-results",
        default=None,
        help="local override for gpt_mini results.jsonl",
    )
    ap.add_argument("--judge-script", default="judge_gpt_mini.py")
    ap.add_argument("--out", default="ablate_out")
    ap.add_argument("--n-sweep", type=int, default=100)
    ap.add_argument(
        "--n-ref", type=int, default=40, help="neutral KL reference prompts"
    )
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="MATCH the baseline inference run's setting",
    )
    ap.add_argument(
        "--do-sample",
        type=lambda s: s.lower() == "true",
        default=True,
        help="set false for deterministic (greedy) generation; match baseline",
    )
    ap.add_argument(
        "--pressure-level",
        default="L0",
        help="baseline + sweep pressure level to evaluate (DIM work is L0)",
    )
    ap.add_argument(
        "--enable-thinking", type=lambda s: s.lower() == "true", default=None
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(args.config))
    acfg = cfg["ablation"]
    global KL_CUTOFF, COHERENCE_MIN
    KL_CUTOFF = acfg.get("kl_cutoff", KL_CUTOFF)
    COHERENCE_MIN = acfg.get("coherence_min", COHERENCE_MIN)
    judge_valid_min = acfg.get("judge_valid_min", 0.95)

    judge = _import_judge(args.judge_script)

    from huggingface_hub import hf_hub_download

    tok_env = os.environ.get("HF_TOKEN")
    slug = acfg["model_slug_map"].get(args.model_key, args.model.replace("/", "__"))
    group = f"{args.concept}/{args.language}/{args.model_key}__{args.preset}"

    def hf_file(repo, path):
        return hf_hub_download(repo, path, repo_type="dataset", token=tok_env)

    def load_rows(path):
        if path.endswith(".parquet"):
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
        return [json.loads(l) for l in open(path) if l.strip()]

    # ---- DIM report + candidates (directions repo, or local override) ----
    if args.dim_report and args.dim_candidates:
        rep_path, cand_path = args.dim_report, args.dim_candidates
    else:
        rep_path = hf_file(acfg["results_repo"], f"{group}/dim_report.json")
        cand_path = hf_file(acfg["directions_repo"], f"{group}/dim_candidates.pt")
    print(f"[dim] report={rep_path}")
    report = json.load(open(rep_path))
    cand_tensors = torch.load(cand_path)
    candidates = shortlist_candidates(report)
    print(f"[shortlist] {len(candidates)} candidates:")
    for c in candidates:
        c["resid_layer"] = dim_index_to_resid_layer(c["layer_dim_index"])
        print(
            f"   {c['name']:32s} pos={c['position']:14s} dim_idx={c['layer_dim_index']} "
            f"resid_layer={c['resid_layer']} auc={c['auc']:.3f} ({c['auc_source']})"
        )

    # ---- sweep rows: model-inference-responses / <subset> / <slug> / <lang>.parquet ----
    if args.sweep_data:
        sweep_all = load_rows(args.sweep_data)
    else:
        sweep_file = f"{acfg['sweep_subset']}/{slug}/{args.language}.parquet"
        sweep_all = load_rows(hf_file(acfg["sweep_repo"], sweep_file))
    sweep_all = [
        r for r in sweep_all if str(r.get("language", args.language)) == args.language
    ]
    # restrict to the pressure level the DIM was built for (L0)
    sweep_all = [
        r
        for r in sweep_all
        if str(r.get("pressure_level", "L0")) == args.pressure_level
    ]

    # ---- KL refs: derive from NON-obligation categories of the same inference set ----
    oblig = set(acfg.get("obligation_categories", []))
    if args.kl_ref_data:
        kl_ref_all = load_rows(args.kl_ref_data)
    elif acfg.get("kl_ref_source") == "non_obligation_categories":
        kl_ref_all = [r for r in sweep_all if r.get("category") not in oblig]
        print(
            f"[kl-ref] derived {len(kl_ref_all)} non-obligation rows for the KL guard"
        )
    else:
        raise SystemExit(
            "no --kl-ref-data and kl_ref_source is not 'non_obligation_categories'"
        )
    kl_ref_all = [r for r in kl_ref_all if r.get("user_query")]

    # ---- baseline verdicts keyed by (id, sample_idx), filtered to this model+lang+pressure ----
    if args.baseline_results:
        base_path = args.baseline_results
    else:
        base_path = hf_file(acfg["baseline_repo"], acfg["baseline_file"])
    baseline = {}  # (id, sample_idx) -> verdict
    for line in open(base_path):
        if not line.strip():
            continue
        j = json.loads(line)
        if str(j.get("model_id")) != args.model:
            continue
        if str(j.get("language")) != args.language:
            continue
        if str(j.get("pressure_level", "L0")) != args.pressure_level:
            continue
        baseline[(j["id"], int(j.get("sample_idx", 0)))] = j.get("verdict")
    if not baseline:
        raise SystemExit(
            f"[baseline] no rows after filter model={args.model} lang={args.language} "
            f"pressure={args.pressure_level}; check baseline repo/file"
        )
    n_samples = 1 + max(si for (_id, si) in baseline.keys())
    print(
        f"[baseline] {len(baseline)} (id,sample) verdicts; inferred n_samples={n_samples}"
    )

    # sweep prompts = obligation-category rows that HAVE a baseline verdict; sample n_sweep prompts
    prompt_ids = sorted(
        {
            r["id"]
            for r in sweep_all
            if ((not oblig) or r.get("category") in oblig)
            and any((r["id"], s) in baseline for s in range(n_samples))
        }
    )
    random.Random(args.seed).shuffle(prompt_ids)
    prompt_ids = prompt_ids[: args.n_sweep]
    # one representative row per prompt id (for system/user_query/checker), keep all fields
    row_by_id = {}
    for r in sweep_all:
        if r["id"] in prompt_ids and r["id"] not in row_by_id:
            row_by_id[r["id"]] = r

    # ---- HELD-ONLY instance eval set: (id, sample_idx) whose baseline verdict == HELD ----
    eval_instances = [
        (pid, s)
        for pid in prompt_ids
        for s in range(n_samples)
        if baseline.get((pid, s)) == "HELD"
    ]
    n_held = len(eval_instances)
    n_total_inst = sum(
        1 for pid in prompt_ids for s in range(n_samples) if (pid, s) in baseline
    )
    if n_held == 0:
        raise SystemExit("[eval] no baseline-HELD instances in the sampled prompts")
    print(
        f"[eval] {n_held} baseline-HELD instances (of {n_total_inst} total) across "
        f"{len(prompt_ids)} prompts x {n_samples} samples -- ablation measures flip rate on these"
    )

    # KL refs disjoint from the sweep prompt ids
    ref_rows = [r for r in kl_ref_all if r["id"] not in set(prompt_ids)][: args.n_ref]
    print(f"[kl-ref] {len(ref_rows)} neutral refs")
    if len(ref_rows) < args.n_ref:
        print(f"[kl-ref][warn] only {len(ref_rows)} neutral refs (< {args.n_ref})")

    # ---- load model ----
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPES[args.dtype], device_map=args.device
    ).eval()
    sys_ok = supports_system_role(tok)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # verify hook assumption: decoder block returns hidden state as output[0].
    # Probe one block on a dummy forward and check shape [.., .., d_model].
    _probe = {}

    def _probe_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _probe["shape"] = tuple(h.shape)
        _probe["is_tuple"] = isinstance(out, tuple)

    _h = get_decoder_layers(model)[0].register_forward_hook(_probe_hook)
    with torch.no_grad():
        model(**tok("probe", return_tensors="pt").to(args.device))
    _h.remove()
    assert _probe.get("shape", (None,))[-1] == d_model, (
        f"decoder block output[0] last dim {_probe.get('shape')} != hidden_size {d_model}; "
        f"hook target is not resid_post for this model -- inspect the block signature"
    )
    print(
        f"[hook] verified block output[0] last-dim == hidden_size {d_model} "
        f"(tuple_output={_probe['is_tuple']})"
    )

    # eval prompts: one per baseline-HELD instance (id, sample_idx), each carrying its
    # source row for the judge. Same prompt text across samples of a prompt; the sample_idx
    # only matches it to the right baseline verdict (all baseline verdicts here are HELD).
    eval_rows = []
    for pid, s in eval_instances:
        r = dict(row_by_id[pid])
        r["_sample_idx"] = s
        eval_rows.append(r)
    eval_prompts = [
        format_chat_prompt(
            tok,
            r.get("system", ""),
            r.get("user_query", ""),
            sys_ok,
            args.enable_thinking,
        )
        for r in eval_rows
    ]
    ref_prompts = [
        format_chat_prompt(
            tok,
            r.get("system", ""),
            r.get("user_query", ""),
            sys_ok,
            args.enable_thinking,
        )
        for r in ref_rows
    ]

    # ---- build candidate + control direction list ----
    def unit_for(cand):
        vec = cand_tensors[cand["name"]][cand["layer_dim_index"]].float()
        assert (
            vec.numel() == d_model
        ), f"direction dim {vec.numel()} != model hidden size {d_model} for {cand['name']}"
        return vec / (vec.norm() + 1e-8)

    # one fixed random direction PER LAYER (shared across candidates at that layer)
    random_units = {}

    def random_unit_for_layer(layer_dim_index):
        if layer_dim_index not in random_units:
            g = torch.Generator().manual_seed(args.seed + layer_dim_index)
            rnd = torch.randn(d_model, generator=g)
            random_units[layer_dim_index] = rnd / (rnd.norm() + 1e-8)
        return random_units[layer_dim_index]

    runs = []
    seen_control_layers = set()
    for c in candidates:
        runs.append({**c, "is_control": False, "unit": unit_for(c)})
        if (
            c["layer_dim_index"] not in seen_control_layers
        ):  # one control per layer, not per candidate
            seen_control_layers.add(c["layer_dim_index"])
            runs.append(
                {
                    "name": f"random@resid{c['resid_layer']}",
                    "contrast": "random",
                    "position": "-",
                    "layer_dim_index": c["layer_dim_index"],
                    "resid_layer": c["resid_layer"],
                    "auc": None,
                    "auc_source": "control",
                    "is_control": True,
                    "unit": random_unit_for_layer(c["layer_dim_index"]),
                }
            )

    # ---- evaluate each run ----
    def judge_responses(rows, responses):
        held = 0
        n = 0
        invalid = 0
        verdicts = []
        for r, resp in zip(rows, responses):
            jr = dict(r)
            jr["response"] = resp
            prompt = judge.build_judge_prompt(jr)
            v = judge.judge_gpt_mini(prompt)
            verdicts.append(v.get("verdict"))
            if v.get("verdict") in ("HELD", "VIOLATED"):
                n += 1
                if v["verdict"] == "HELD":
                    held += 1
            else:
                invalid += 1
        return (held / max(n, 1)), n, invalid, verdicts

    results = []
    for run in runs:
        tag = run["name"] + ("" if not run["is_control"] else " [control]")
        print(f"\n[run] {tag}  resid_layer>={run['resid_layer']}")
        unit = run["unit"].to(args.device)

        # KL guard
        kl = first_token_kl(
            model, tok, ref_prompts, unit, run["resid_layer"], args.device
        )
        print(
            f"   KL mean={kl['mean_kl']:.3f} median={kl['median_kl']:.3f} p95={kl['p95_kl']:.3f}"
        )

        # ablated generation on the CANONICAL eval set
        with DirectionalAblation(model, unit, run["resid_layer"]):
            responses = []
            bs = 16
            for s in range(0, len(eval_prompts), bs):
                responses += generate_batch(
                    model,
                    tok,
                    eval_prompts[s : s + bs],
                    args.device,
                    args.max_new_tokens,
                    args.temperature,
                    args.do_sample,
                )

        # coherence (record failure reasons)
        coh_flags, coh_metrics = [], []
        fail_reasons = {"too_short": 0, "high_ngram_repeat": 0, "low_compression": 0}
        for resp in responses:
            ok, m = is_coherent(resp)
            coh_flags.append(ok)
            coh_metrics.append(m)
            if not ok and m.get("reason") in fail_reasons:
                fail_reasons[m["reason"]] += 1
        coherence_rate = sum(coh_flags) / max(len(coh_flags), 1)

        # judge on the HELD-only eval instances. Baseline HELD rate here is 1.0 by
        # construction, so the causal effect is the FLIP RATE = 1 - ablated_HELD_rate.
        abl_held_rate, n_judged, n_invalid, verdicts = judge_responses(
            eval_rows, responses
        )
        n_flipped = sum(1 for v in verdicts if v == "VIOLATED")
        flip_rate = 1.0 - abl_held_rate  # fraction of obeying responses that broke
        judge_valid_rate = n_judged / max(len(eval_rows), 1)

        kl_pass = kl["mean_kl"] <= KL_CUTOFF
        coh_pass = coherence_rate >= COHERENCE_MIN
        judge_ok = judge_valid_rate >= 0.95  # data-quality gate: enough valid verdicts
        print(
            f"   ablated_held={abl_held_rate:.3f} flip_rate={flip_rate:+.3f} "
            f"({n_flipped}/{n_judged} flipped) coh={coherence_rate:.2f} "
            f"valid_judge={judge_valid_rate:.2f} kl_pass={kl_pass} coh_pass={coh_pass} judge_ok={judge_ok}"
        )

        results.append(
            {
                "name": run["name"],
                "contrast": run["contrast"],
                "position": run["position"],
                "layer_dim_index": run["layer_dim_index"],
                "resid_layer": run["resid_layer"],
                "auc": run["auc"],
                "auc_source": run["auc_source"],
                "is_control": run["is_control"],
                "baseline_held": 1.0,
                "ablated_held": abl_held_rate,
                "flip_rate": flip_rate,
                "n_flipped": n_flipped,
                "n_eval_held_instances": len(eval_rows),
                "n_judged": n_judged,
                "n_invalid_judgments": n_invalid,
                "judge_valid_rate": judge_valid_rate,
                **kl,
                "coherence_rate": coherence_rate,
                "coherence_fail_reasons": fail_reasons,
                "per_response_coherence": coh_metrics,
                "kl_pass": kl_pass,
                "coherence_pass": coh_pass,
                "judge_data_ok": judge_ok,
            }
        )

    # ---- select winner: max FLIP RATE among NON-control passing ALL gates ----
    eligible = [
        r
        for r in results
        if (not r["is_control"])
        and r["kl_pass"]
        and r["coherence_pass"]
        and r["judge_data_ok"]
    ]
    winner = max(eligible, key=lambda r: r["flip_rate"]) if eligible else None
    for r in results:
        r["selected"] = bool(
            winner
            and r["name"] == winner["name"]
            and r["resid_layer"] == winner["resid_layer"]
            and not r["is_control"]
        )

    # numeric specificity: winner flip rate vs the random control at the winner's layer
    specificity = None
    if winner is not None:
        ctrl = next(
            (
                r
                for r in results
                if r["is_control"] and r["resid_layer"] == winner["resid_layer"]
            ),
            None,
        )
        if ctrl is not None:
            specificity = {
                "winner_flip_rate": winner["flip_rate"],
                "control_flip_rate": ctrl["flip_rate"],
                "specificity_gap": winner["flip_rate"] - ctrl["flip_rate"],
                "control_name": ctrl["name"],
            }

    report_out = {
        "model": args.model,
        "language": args.language,
        "pressure_level": args.pressure_level,
        "n_prompts": len(prompt_ids),
        "n_samples": n_samples,
        "n_held_instances": len(eval_rows),
        "n_total_instances": n_total_inst,
        "n_kl_ref": len(ref_rows),
        "generation": {
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "note": "must match the baseline inference run's generation config",
        },
        "metric": "flip_rate = 1 - ablated_HELD over baseline-HELD instances (baseline HELD = 1.0 by construction)",
        "set_role": "SELECTION set (not the final causal effect; causal claim = cross-lingual patch)",
        "pre_registered_gates": {
            "kl_cutoff_mean_first_token": KL_CUTOFF,
            "coherence_rate_min": COHERENCE_MIN,
            "judge_valid_rate_min": 0.95,
            "no_candidate_passes_behavior": "fail_loud_select_none",
        },
        "specificity_vs_random": specificity,
        "candidates": results,
        "winner": winner["name"] if winner else None,
    }
    json.dump(report_out, open(out / "selection_report.json", "w"), indent=2)

    if winner is None:
        print(
            "\n[select] NO candidate passed all gates. Selecting none. See selection_report.json"
        )
    else:
        vec = cand_tensors[winner["name"]][winner["layer_dim_index"]].float()
        torch.save(
            {
                "vector": vec,
                "unit": (vec / (vec.norm() + 1e-8)),
                "resid_layer": winner["resid_layer"],
                "dim_index": winner["layer_dim_index"],
                "contrast": winner["contrast"],
                "position": winner["position"],
                "model": args.model,
                "flip_rate": winner["flip_rate"],
                "mean_kl": winner["mean_kl"],
                "coherence_rate": winner["coherence_rate"],
                "specificity_vs_random": specificity,
                "note": "selection-only; causal claim via cross-lingual patch",
            },
            out / "selected_direction.pt",
        )
        print(
            f"\n[select] winner = {winner['name']} @resid_layer {winner['resid_layer']} "
            f"flip_rate={winner['flip_rate']:+.3f}"
        )
        if specificity:
            print(
                f"[specificity] winner flip {specificity['winner_flip_rate']:+.3f} "
                f"vs random {specificity['control_flip_rate']:+.3f} "
                f"= gap {specificity['specificity_gap']:+.3f}"
            )
    print(f"[done] {out/'selection_report.json'}")


if __name__ == "__main__":
    main()
