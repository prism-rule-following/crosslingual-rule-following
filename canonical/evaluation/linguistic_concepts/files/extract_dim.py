#!/usr/bin/env python3
"""
Difference-in-means extraction for obligation directions (optimized + checkpointed).

Two experiment families, one pipeline, selected by --preset (or extraction.stimulus_mode):

  rule_following : system=<context + rule>, user=<query>   (rule adherence framing)
                   positions: contrast_token, rule_clause_end, post_instruction
  concept        : user=<rule sentence only>               (linguistic concept framing)
                   positions: contrast_token, sentence_end
  concept_raw    : raw sentence, NO chat template          (purest concept stimulus)
                   positions: contrast_token, sentence_end

Contrasts:  must_text (clean) - may_text (corrupt)      -> obligation vs permission
            must_text (clean) - neutral_text (corrupt)  -> obligation vs descriptive-norm

Optimizations: batched fwd, bf16, use_cache=False, SDPA/flash-attn, torch.compile,
periodic torch.cuda.empty_cache(). Checkpointing: Drive-backed row-level, resumable.
HF: pulls dataset from hub, pushes directions/results (+ optional activations).

Usage:
    export HF_TOKEN=hf_...
    python extract_dim.py --model qwen3-8b    --preset rule_following
    python extract_dim.py --model qwen3-8b    --preset concept
    python extract_dim.py --model llama3.1-8b --preset concept_raw --no-push
"""

import os, json, argparse, math, gc
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from checkpoint import RowCheckpoint
import hf_io

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

VALID_MODES = {"system_user", "user_only", "raw_sentence"}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_cfg(path, model_key, preset=None):
    cfg = json.load(open(path))
    if model_key not in cfg["models"]:
        raise SystemExit(f"model '{model_key}' not in config: {list(cfg['models'])}")

    # Apply a preset (overrides extraction fields) if requested.
    if preset:
        presets = cfg.get("presets", {})
        if preset not in presets:
            raise SystemExit(f"preset '{preset}' not in config: {list(presets)}")
        for k, v in presets[preset].items():
            cfg["extraction"][k] = v
        cfg["extraction"]["_active_preset"] = preset

    ec = cfg["extraction"]
    mode = ec.get("stimulus_mode", "system_user")
    if mode not in VALID_MODES:
        raise SystemExit(f"stimulus_mode '{mode}' invalid; choose {VALID_MODES}")

    # --- guardrails ---------------------------------------------------------
    positions = ec["positions"]
    # post_instruction only meaningful when there is an instruction to follow
    if "post_instruction" in positions and mode != "system_user":
        raise SystemExit(
            f"[config] position 'post_instruction' is only valid for stimulus_mode="
            f"'system_user'; current mode is '{mode}'. Use 'sentence_end' instead."
        )
    # sentence_end is the concept-mode anchor; rule_clause_end is the rule-mode anchor.
    if "sentence_end" in positions and mode == "system_user":
        print("[config][warn] 'sentence_end' with system_user is unusual; "
              "'rule_clause_end' is the intended rule-mode anchor.")
    if "rule_clause_end" in positions and mode == "raw_sentence":
        print("[config][warn] 'rule_clause_end' in raw_sentence mode behaves like "
              "'sentence_end' (whole sequence is the clause).")

    return cfg, cfg["models"][model_key]


# --------------------------------------------------------------------------- #
# prompt construction (mode-aware)
# --------------------------------------------------------------------------- #
def build_prompt(tok, mcfg, mode, *, rule_text, context=None, query=None,
                 raw_add_special_tokens=True):
    """
    Returns (encoded_ids_list, rule_text, has_special_prefix).

    - system_user : chat template, system=<context + rule>, user=<query>
    - user_only   : chat template, single user turn = rule_text
    - raw_sentence: NO chat template; tokenize rule_text directly

    We return token ids (not a string) so raw mode can control add_special_tokens
    and every mode shares one downstream code path for position finding.
    """
    if mode == "raw_sentence":
        ids = tok(rule_text, add_special_tokens=raw_add_special_tokens).input_ids
        return ids, rule_text, raw_add_special_tokens

    if mode == "user_only":
        msgs = [{"role": "user", "content": rule_text}]
    elif mode == "system_user":
        system = f"{context} Rule: {rule_text}".strip() if context else rule_text
        if mcfg.get("system_supported", True):
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": query}]
        else:
            msgs = [{"role": "user", "content": f"{system}\n\n{query}"}]
    else:
        raise ValueError(mode)

    kw = {}
    if "enable_thinking" in mcfg:
        kw["enable_thinking"] = mcfg["enable_thinking"]
    prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    # chat template already injects special tokens; do NOT add more.
    ids = tok(prompt_str, add_special_tokens=False).input_ids
    return ids, rule_text, True


def find_sub_span(h, n):
    N, M = len(h), len(n)
    for s in range(N - M + 1):
        if h[s:s + M] == n:
            return s
    return -1


def locate_positions(tok, ids, rule_text, contrast_word, mode, positions):
    """
    Locate requested anchors as absolute token indices in `ids`.
      contrast_token  : last subword of the swapped word, scoped to the clause span
      rule_clause_end : last token of the rule-text span (rule modes)
      sentence_end    : last real token of the stimulus
      post_instruction: last token of the full templated prompt (system_user only)
    """
    L = len(ids)
    last = L - 1

    # locate the rule-text span (used by contrast scoping and clause/sentence end)
    rc_start, rc_end = 0, last
    if mode == "raw_sentence":
        # whole sequence is the sentence; if BOS present it's at index 0.
        rc_start = 1 if L > 1 and ids[0] in getattr(tok, "all_special_ids", []) else 0
        rc_end = last
    else:
        found = False
        for variant in (rule_text, " " + rule_text, rule_text.rstrip(".")):
            nid = tok(variant, add_special_tokens=False).input_ids
            pos = find_sub_span(ids, nid)
            if pos != -1:
                rc_start, rc_end = pos, pos + len(nid) - 1
                found = True
                break
        if not found:
            rc_start, rc_end = 0, last  # fallback: treat whole seq as clause

    out = {}
    if "contrast_token" in positions:
        scope = ids[rc_start:rc_end + 1]
        ct = -1
        for variant in (" " + contrast_word, contrast_word):
            nid = tok(variant, add_special_tokens=False).input_ids
            pos = find_sub_span(scope, nid)
            if pos != -1:
                ct = rc_start + pos + len(nid) - 1
                break
        out["contrast_token"] = ct if ct != -1 else rc_end
    if "rule_clause_end" in positions:
        out["rule_clause_end"] = rc_end
    if "sentence_end" in positions:
        # last real token of the sentence span (raw: end of seq; templated: end of clause)
        out["sentence_end"] = rc_end if mode != "raw_sentence" else last
    if "post_instruction" in positions:
        out["post_instruction"] = last
    return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(mcfg, ocfg):
    kw = dict(
        torch_dtype=DTYPES["bfloat16"] if ocfg.get("bf16", True) else DTYPES[mcfg["dtype"]],
        device_map=mcfg.get("device", "cuda"),
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    attn = ocfg.get("attn_implementation", "sdpa")
    if attn:
        kw["attn_implementation"] = attn
    print(f"[load] {mcfg['hf_name']} attn={attn} dtype={kw['torch_dtype']}")
    tok = AutoTokenizer.from_pretrained(mcfg["hf_name"], trust_remote_code=mcfg.get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    try:
        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_name"], **kw)
    except Exception as e:
        print(f"[load] {attn} failed ({e}); retrying sdpa")
        kw["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_name"], **kw)
    model.eval()
    if ocfg.get("torch_compile", False):
        try:
            model = torch.compile(model, mode=ocfg.get("compile_mode", "reduce-overhead"))
            print("[load] torch.compile on")
        except Exception as e:
            print(f"[load] compile skipped: {e}")
    return tok, model


@torch.no_grad()
def batched_hidden_states(model, tok, id_lists, device, use_cache):
    """
    Manually right-pad a batch of id lists, run once, return list of
    [nl+1, seq_i, d] unpadded per item. Taking ids (not strings) keeps position
    indices identical to what locate_positions computed.
    """
    maxlen = max(len(x) for x in id_lists)
    pad_id = tok.pad_token_id
    input_ids, attn = [], []
    for x in id_lists:
        p = maxlen - len(x)
        input_ids.append(x + [pad_id] * p)
        attn.append([1] * len(x) + [0] * p)
    input_ids = torch.tensor(input_ids, device=device)
    attn = torch.tensor(attn, device=device)
    out = model(input_ids=input_ids, attention_mask=attn,
                output_hidden_states=True, use_cache=use_cache)
    hs = torch.stack(out.hidden_states, 0)  # [nl+1, B, seqmax, d]
    res = [hs[:, b, :len(id_lists[b]), :].float().cpu() for b in range(len(id_lists))]
    del out, hs, input_ids, attn
    return res


# --------------------------------------------------------------------------- #
def _auc(pos_scores, neg_scores):
    """Rank-based AUC = P(clean projection > corrupt projection). No threshold."""
    import numpy as _np
    a = _np.asarray(pos_scores); b = _np.asarray(neg_scores)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # Mann-Whitney U / (n*m)
    allv = _np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = _np.empty(len(allv), dtype=float)
    ranks[order] = _np.arange(1, len(allv) + 1)
    # average ranks for ties
    # (simple tie handling: recompute with scipy-like averaging)
    _, inv, counts = _np.unique(allv, return_inverse=True, return_counts=True)
    csum = _np.cumsum(counts)
    start = csum - counts + 1
    avg = (start + csum) / 2.0
    ranks = avg[inv]
    r1 = ranks[:len(a)].sum()
    u1 = r1 - len(a) * (len(a) + 1) / 2.0
    return float(u1 / (len(a) * len(b)))


def extract_store(rows, ckpt, tok, model, mcfg, dcfg, ecfg, ocfg, mode, positions, device):
    """Run the model over `rows`, cache per-row activations via `ckpt`, return a
    store: member -> position -> list of (meta, tensor[nl+1, d]). Resumable."""
    field = dcfg["rule_field_map"]; ctok_field = dcfg["contrast_token_field"]
    members = {"clean": field["clean"], "corrupt_may": field["corrupt_may"],
               "corrupt_neutral": field["corrupt_neutral"]}
    B = ocfg.get("batch_size", 8); empty_every = ocfg.get("empty_cache_every_n_batches", 4)
    raw_ast = ecfg.get("raw_add_special_tokens", True)
    frame_field = dcfg.get("frame_field")
    has_frames = frame_field is not None and rows and frame_field in rows[0]

    already = ckpt.done_ids()
    todo = [r for r in rows if r["id"] not in already]
    print(f"[ckpt:{ckpt.base}] {len(already)} cached, {len(todo)} to compute")

    work = [(r, m, mf) for r in todo for m, mf in members.items()]
    need_members = set(members); row_buf = defaultdict(dict); batch_i = 0
    for s in range(0, len(work), B):
        chunk = work[s:s + B]
        id_lists, locs, keys = [], [], []
        for r, m, mf in chunk:
            rule_text = r[mf]; cword = r[ctok_field[m]]
            ids, _, _ = build_prompt(
                tok, mcfg, mode, rule_text=rule_text,
                context=r.get(dcfg["context_field"]) if mode == "system_user" else None,
                query=r.get(dcfg["query_field"]) if mode == "system_user" else None,
                raw_add_special_tokens=raw_ast)
            id_lists.append(ids)
            locs.append(locate_positions(tok, ids, rule_text, cword, mode, positions))
            keys.append((r["id"], m, r))
        hs_list = batched_hidden_states(model, tok, id_lists, device, ocfg.get("use_cache", False))
        for (rid, m, r), hs, pos in zip(keys, hs_list, locs):
            row_buf[rid][m] = {p: hs[:, pos[p], :].clone() for p in positions}
            meta = {"frame": r.get(frame_field) if has_frames else None,
                    "category": r.get("category"), "topic": r.get("topic"),
                    "lexeme_set": r.get("lexeme_set"), "id": rid}
            if need_members.issubset(row_buf[rid].keys()):
                ckpt.save_row(rid, row_buf.pop(rid), meta)
        batch_i += 1
        if device.startswith("cuda") and empty_every and batch_i % empty_every == 0:
            torch.cuda.empty_cache(); gc.collect()
        if batch_i % 5 == 0:
            print(f"  batch {batch_i} ({min(s+B,len(work))}/{len(work)} items)")
    ckpt.finalize()

    cached = ckpt.load_all()
    store = defaultdict(lambda: defaultdict(list))
    for rid, payload, meta in cached:
        for m, posmap in payload.items():
            for p, vec in posmap.items():
                store[m][p].append((meta, vec))
    return store, has_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--preset", default=None, help="rule_following | concept | concept_raw")
    ap.add_argument("--language", required=True,
                    help="language code/name for this dataset, e.g. en, yoruba, igbo")
    ap.add_argument("--concept", required=True,
                    help="concept type for this dataset, e.g. obligation, negation")
    ap.add_argument("--data", default=None, help="local train json; if omitted, pull train split per config.hf")
    ap.add_argument("--test-data", default=None, help="local test json; if omitted, pull test split per config.hf")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg, mcfg = load_cfg(args.config, args.model, preset=args.preset)
    dcfg, ecfg, ocfg = cfg["data"], cfg["extraction"], cfg["optim"]
    mode = ecfg["stimulus_mode"]
    positions = ecfg["positions"]
    device = mcfg.get("device", "cuda")
    torch.manual_seed(ecfg.get("seed", 0))
    preset_tag = ecfg.get("_active_preset", mode)
    # canonical run identity: concept / language / model__preset
    lang, concept = args.language, args.concept
    run_tag = f"{args.model}__{preset_tag}"
    group_path = f"{concept}/{lang}/{run_tag}"          # HF path prefix + local subdir
    ckpt_ns = f"{concept}__{lang}__{run_tag}"           # flat checkpoint namespace
    print(f"[cfg] concept={concept} language={lang} preset={preset_tag} mode={mode} "
          f"positions={positions} contrasts={ecfg['contrasts']}")

    # ---- data: pull train and (optionally) test split ----
    from_hub = cfg["hf"].get("dataset_load", "hub") == "hub" and args.data is None
    def _load(split):
        if args.data and split == "train":
            path = args.data
        elif args.test_data and split == "test":
            path = args.test_data
        else:
            path = f"{concept}_{lang}_{split}.json"
            if from_hub:
                hf_io.pull_dataset(cfg, path, concept=concept, language=lang, split=split)
            elif not os.path.exists(path):
                return None
        if not os.path.exists(path):
            return None
        rr = json.load(open(path))
        return rr[:args.limit] if args.limit else rr

    train_rows = _load("train")
    if train_rows is None:
        raise SystemExit("[data] no train split found (need --data, --test-data, or HF splits)")
    test_rows = _load("test")
    print(f"[data] train={len(train_rows)} rows"
          + (f", test={len(test_rows)} rows" if test_rows else " (no test split)"))

    tok, model = load_model(mcfg, ocfg)

    # extract activations for each split into its own checkpoint namespace
    ckpt_train = RowCheckpoint(cfg, f"{ckpt_ns}__train")
    store_tr, has_frames_tr = extract_store(train_rows, ckpt_train, tok, model, mcfg,
                                            dcfg, ecfg, ocfg, mode, positions, device)
    store_te = None
    if test_rows:
        ckpt_test = RowCheckpoint(cfg, f"{ckpt_ns}__test")
        store_te, _ = extract_store(test_rows, ckpt_test, tok, model, mcfg,
                                    dcfg, ecfg, ocfg, mode, positions, device)

    del model; gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- DIM math ----
    p0 = positions[0]
    n_layers_p1 = store_tr["clean"][p0][0][1].shape[0]
    d_model = store_tr["clean"][p0][0][1].shape[1]

    def by_frame(store, member, position):
        by = defaultdict(list)
        for meta, vec in store[member][position]:
            by[meta.get("frame")].append(vec)
        return {f: torch.stack(v, 0) for f, v in by.items()}

    def all_of(store, member, position):
        return torch.stack([v for _, v in store[member][position]], 0)

    results = {"model": args.model, "hf_name": mcfg["hf_name"], "preset": preset_tag,
               "concept": concept, "language": lang, "stimulus_mode": mode,
               "n_layers_incl_embed": n_layers_p1, "d_model": d_model,
               "positions": positions, "contrasts": ecfg["contrasts"],
               "n_train": len(train_rows), "n_test": len(test_rows) if test_rows else 0,
               "directions": {}}
    candidate_tensors = {}
    contrast_pairs = {"must_may": ("clean", "corrupt_may"), "must_neutral": ("clean", "corrupt_neutral")}

    frame_vals = set(m.get("frame") for _, _, m in
                     [(None, None, meta) for meta, _ in store_tr["clean"][p0]])
    use_frames = has_frames_tr and len([f for f in frame_vals if f is not None]) > 1

    def cohens_d(a, b):
        va, vb = a.var(unbiased=True), b.var(unbiased=True); n1, n2 = len(a), len(b)
        sp = math.sqrt(((n1-1)*va + (n2-1)*vb)/max(n1+n2-2,1) + 1e-12)
        return float((a.mean()-b.mean())/(sp+1e-12))

    def eval_split(store, unit, m_clean, m_corr, position):
        """Per-layer Cohen's d and AUC of clean vs corrupt projections onto `unit`."""
        ca, ka = all_of(store, m_clean, position), all_of(store, m_corr, position)
        d_list, auc_list = [], []
        for l in range(ca.shape[1]):
            pc = (ca[:, l, :] * unit[l]).sum(-1)
            pk = (ka[:, l, :] * unit[l]).sum(-1)
            d_list.append(cohens_d(pc, pk))
            auc_list.append(_auc(pc.numpy(), pk.numpy()))
        return d_list, auc_list

    for contrast in ecfg["contrasts"]:
        m_clean, m_corr = contrast_pairs[contrast]
        for position in positions:
            # --- fit direction on TRAIN only ---
            if position == "contrast_token" and ecfg.get("contrast_token_agg") == "per_frame_mean" and use_frames:
                cf, rf = by_frame(store_tr, m_clean, position), by_frame(store_tr, m_corr, position)
                dims, per_frame = [], {}
                for f in cf:
                    dims.append(cf[f].mean(0) - rf[f].mean(0)); per_frame[f] = {"n": cf[f].shape[0]}
                dim = torch.stack(dims, 0).mean(0); agg = "per_frame_mean"
            else:
                dim = all_of(store_tr, m_clean, position).mean(0) - all_of(store_tr, m_corr, position).mean(0)
                agg = "pooled"; per_frame = None
            norms = dim.norm(dim=-1); unit = dim / (norms.unsqueeze(-1) + 1e-8)
            name = f"{contrast}__{position}"
            candidate_tensors[name] = unit if ecfg["normalize_directions"] else dim

            # --- evaluate on TRAIN (in-sample) and TEST (held-out) with the SAME train unit ---
            d_tr, auc_tr = eval_split(store_tr, unit, m_clean, m_corr, position)
            bl = int(np.argmax(np.abs(d_tr)))
            entry = {"contrast": contrast, "position": position, "agg": agg,
                     "per_layer_norm": norms.tolist(), "argmax_norm_layer": int(norms.argmax()),
                     "frames": per_frame,
                     "in_sample": {"per_layer_cohens_d": d_tr, "per_layer_auc": auc_tr,
                                   "best_layer": bl, "best_cohens_d": d_tr[bl], "best_auc": auc_tr[bl]}}
            if store_te is not None:
                d_te, auc_te = eval_split(store_te, unit, m_clean, m_corr, position)
                # report held-out at the layer chosen on train (honest), and its own best
                bl_te = int(np.argmax(np.abs(d_te)))
                entry["held_out"] = {"per_layer_cohens_d": d_te, "per_layer_auc": auc_te,
                                     "at_train_best_layer": {"layer": bl, "cohens_d": d_te[bl], "auc": auc_te[bl]},
                                     "own_best_layer": {"layer": bl_te, "cohens_d": d_te[bl_te], "auc": auc_te[bl_te]}}
            results["directions"][name] = entry

    # transfer check (train frames only; unchanged logic)
    if cfg["transfer_check"]["enabled"] and use_frames and "contrast_token" in positions:
        tc = cfg["transfer_check"]; src, tgt = set(tc["source_frames"]), set(tc["target_frames"]); transfer = {}
        for contrast in ecfg["contrasts"]:
            m_clean, m_corr = contrast_pairs[contrast]; position = "contrast_token"
            cf_c, cf_k = by_frame(store_tr, m_clean, position), by_frame(store_tr, m_corr, position)
            def fdim(fr):
                ds = [cf_c[f].mean(0)-cf_k[f].mean(0) for f in fr if f in cf_c]
                return torch.stack(ds,0).mean(0) if ds else None
            sd, td = fdim(src), fdim(tgt)
            if sd is not None and td is not None:
                cos = torch.nn.functional.cosine_similarity(sd, td, dim=-1)
                transfer[contrast] = {"per_layer_cosine": cos.tolist(), "mean_cosine": float(cos.mean())}
        results["transfer_check"] = transfer
    else:
        results["transfer_check"] = None
        if cfg["transfer_check"]["enabled"] and not use_frames:
            print("[transfer] skipped: dataset has <2 frames (expected for concept/cross-lingual stimuli)")

    # ---- save (local mirrors the HF group path: <out>/<concept>/<lang>/<model>__<preset>/) ----
    local_dir = os.path.join(cfg["output"]["dir"], concept, lang, run_tag)
    os.makedirs(local_dir, exist_ok=True)
    dir_path = os.path.join(local_dir, "dim_candidates.pt")
    rep_path = os.path.join(local_dir, "dim_report.json")
    torch.save(candidate_tensors, dir_path)
    json.dump(results, open(rep_path, "w"), indent=2)

    has_test = store_te is not None
    print(f"\n=== DIM summary [{group_path}] ===")
    if has_test:
        print(f"  {'direction':30s} {'layer':>5} | {'in-sample':>18} | {'held-out @train-L':>18} | {'held-out own-best':>18}")
        print(f"  {'':30s} {'':>5} | {'d':>8} {'AUC':>8} | {'d':>8} {'AUC':>8} | {'L':>4} {'d':>6} {'AUC':>5}")
        for name, r in results["directions"].items():
            ins = r["in_sample"]; ho = r["held_out"]; L = ins["best_layer"]
            at = ho["at_train_best_layer"]; ob = ho["own_best_layer"]
            print(f"  {name:30s} {L:>5} | {ins['best_cohens_d']:>8.3f} {ins['best_auc']:>8.3f} | "
                  f"{at['cohens_d']:>8.3f} {at['auc']:>8.3f} | {ob['layer']:>4} {ob['cohens_d']:>6.2f} {ob['auc']:>5.2f}")
    else:
        print(f"  (train only; no test split)  {'layer':>5} {'d':>8} {'AUC':>8}")
        for name, r in results["directions"].items():
            ins = r["in_sample"]
            print(f"  {name:30s} {ins['best_layer']:>5} {ins['best_cohens_d']:>8.3f} {ins['best_auc']:>8.3f}")
    if results["transfer_check"]:
        print("\n=== transfer (source -> target frames, contrast_token) ===")
        for c, t in results["transfer_check"].items():
            print(f"  {c:14s} mean cos={t['mean_cosine']:+.3f}")

    if not args.no_push:
        try:
            if cfg["checkpoint"].get("upload_row_cache_to_hf", False):
                hf_io.push(cfg, "activations", group_path, ckpt_train.rows_dir)
            hf_io.push(cfg, "directions", group_path, dir_path)
            hf_io.push(cfg, "results", group_path, rep_path)
        except SystemExit as e:
            print(f"[hf] push skipped: {e}")
    print(f"\n[done] {rep_path}")


if __name__ == "__main__":
    main()
