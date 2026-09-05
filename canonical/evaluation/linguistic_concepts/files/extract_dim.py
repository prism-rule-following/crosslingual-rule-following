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

--dataset-variants selects which train/test partition(s) of the canonical dataset to
pull and score, per (language, model, dataset_variant, position, contrast):
  frame    : train/test split on disjoint frame_ids, same scenarios  (default)
  scenario : train/test split on disjoint scenarios, same frames
Ground truth for the held-out AUC is which text variant (must/may/neutral) was
fed in -- not a judged label, so there is no judge/API cost at this stage.

NOTE: post_instruction is "last prompt token before generation". This is only a
stable anchor when thinking is disabled -- with thinking on, a <think> scaffold
token sits at that boundary instead, so post_instruction would end up pointing at
the reasoning trace rather than the pre-answer position. load_cfg() refuses to
run with post_instruction + enable_thinking=true for this reason.

--models / --languages / --dataset-variants each take one or more values, and the
sweep runs every (model, language, variant) combination in one process. The loop
nests language/variant INSIDE model -- each model is loaded (and torch.compile'd)
exactly once and reused across every language/variant, then freed before the next
model loads. This is a compute-exhaustion fix as much as a memory one: reloading
an 8B model per language would dominate wall-clock time for no benefit, since
nothing about the model depends on language/variant.

Usage:
    export HF_TOKEN=hf_...
    python extract_dim.py --models qwen3-8b --languages en de ig --dataset-variants frame scenario --preset rule_following
    python extract_dim.py --models qwen3-8b llama3.1-8b --languages en --preset concept
    python extract_dim.py --models llama3.1-8b --languages en ig --preset concept_raw --no-push --continue-on-error
"""

import os, json, argparse, math, gc
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from checkpoint import RowCheckpoint
import hf_io

from dotenv import load_dotenv

load_dotenv()

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

VALID_MODES = {"system_user", "user_only", "raw_sentence"}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_cfg(path, model_key, preset=None):
    cfg = json.load(open(path))
    if model_key not in cfg["models"]:
        raise SystemExit(f"model '{model_key}' not in config: {list(cfg['models'])}")
    mcfg = cfg["models"][model_key]

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
    # post_instruction is defined as "last prompt token before generation". With
    # thinking enabled, the chat template inserts a <think> scaffold token at
    # exactly that boundary, so the position silently shifts from "about to
    # answer" to "about to reason" -- a different extraction site, not a style
    # choice. Refuse rather than let it pass quietly.
    if "post_instruction" in positions and mcfg.get("enable_thinking", False):
        raise SystemExit(
            f"[config] model '{model_key}' has enable_thinking=true but positions "
            f"include 'post_instruction'. post_instruction = last prompt token "
            f"before generation; with thinking on, that token is the <think> "
            f"scaffold opening the reasoning trace, not the pre-answer boundary. "
            f"Set enable_thinking=false for this model."
        )
    # sentence_end is the concept-mode anchor; rule_clause_end is the rule-mode anchor.
    if "sentence_end" in positions and mode == "system_user":
        print("[config][warn] 'sentence_end' with system_user is unusual; "
              "'rule_clause_end' is the intended rule-mode anchor.")
    if "rule_clause_end" in positions and mode == "raw_sentence":
        print("[config][warn] 'rule_clause_end' in raw_sentence mode behaves like "
              "'sentence_end' (whole sequence is the clause).")

    return cfg, mcfg


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


def run_one(cfg, mcfg, model_key, tok, model, concept, lang, variant, limit, no_push):
    """Run extraction + DIM fit/score for one (model, language, dataset_variant)
    combo, reusing an already-loaded (tok, model). Returns the local report path.

    Caller owns the model's lifetime -- this function frees only what it
    allocates for THIS combo (row stores, candidate tensors, results dict) so
    memory doesn't accumulate across a sweep, but it never touches tok/model."""
    dcfg, ecfg, ocfg = cfg["data"], cfg["extraction"], cfg["optim"]
    mode = ecfg["stimulus_mode"]
    positions = ecfg["positions"]
    device = mcfg.get("device", "cuda")
    torch.manual_seed(ecfg.get("seed", 0))
    preset_tag = ecfg.get("_active_preset", mode)
    run_tag = f"{model_key}__{preset_tag}"
    # combined (all contrasts x positions) group path -- kept for tooling (e.g.
    # steering_poc.py) that wants one dim_candidates.pt/dim_report.json per run.
    # Per-contrast x per-position split paths are pushed separately below.
    group_path = f"{concept}/{variant}/{lang}/{run_tag}"
    ckpt_ns = f"{concept}__{variant}__{lang}__{run_tag}"  # flat checkpoint namespace;
    # variant is baked in here (not just group_path) because frame- and scenario-split
    # files share row ids for overlapping records -- a shared cache dir would silently
    # treat one variant's cached row as "done" for the other.
    print(f"[cfg] concept={concept} language={lang} variant={variant} model={model_key} "
          f"preset={preset_tag} mode={mode} positions={positions} contrasts={ecfg['contrasts']}")

    # ---- data: pull train and (optionally) test split ----
    from_hub = cfg["hf"].get("dataset_load", "hub") == "hub"
    def _load(split):
        # local naming mirrors the hub layout: <split>.json for frame,
        # <split>_scenario.json for scenario (e.g. obligation_ig_test_scenario.json)
        file_split = split if variant == "frame" else f"{split}_scenario"
        path = f"{concept}_{lang}_{file_split}.json"
        if from_hub:
            hf_io.pull_dataset(cfg, path, concept=concept, language=lang,
                               split=split, dataset_variant=variant)
        elif not os.path.exists(path):
            return None
        if not os.path.exists(path):
            return None
        rr = json.load(open(path))
        return rr[:limit] if limit else rr

    train_rows = _load("train")
    if train_rows is None:
        print(f"[skip] {group_path}: no train split found (set config.hf.dataset_load"
              f"='local' with a local file, or check HF dataset_files)")
        return None
    test_rows = _load("test")
    print(f"[data] train={len(train_rows)} rows"
          + (f", test={len(test_rows)} rows" if test_rows else " (no test split)"))

    # extract activations for each split into its own checkpoint namespace
    ckpt_train = RowCheckpoint(cfg, f"{ckpt_ns}__train")
    store_tr, has_frames_tr = extract_store(train_rows, ckpt_train, tok, model, mcfg,
                                            dcfg, ecfg, ocfg, mode, positions, device)
    store_te = None
    if test_rows:
        ckpt_test = RowCheckpoint(cfg, f"{ckpt_ns}__test")
        store_te, _ = extract_store(test_rows, ckpt_test, tok, model, mcfg,
                                    dcfg, ecfg, ocfg, mode, positions, device)
    # NOTE: the model is NOT freed here -- it is reused for every remaining
    # (language, variant) combo under this model in the sweep. Only the
    # per-combo row stores (host-RAM tensors) get cleaned up, at the end of
    # this function, so RAM doesn't accumulate run over run.

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

    results = {"model": model_key, "hf_name": mcfg["hf_name"], "preset": preset_tag,
               "concept": concept, "language": lang, "dataset_variant": variant,
               "stimulus_mode": mode,
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

    # ---- save combined (all contrasts x positions) --------------------------
    # local mirrors the HF group path: <out>/<concept>/<variant>/<lang>/<model>__<preset>/
    local_dir = os.path.join(cfg["output"]["dir"], concept, variant, lang, run_tag)
    os.makedirs(local_dir, exist_ok=True)
    dir_path = os.path.join(local_dir, "dim_candidates.pt")
    rep_path = os.path.join(local_dir, "dim_report.json")
    torch.save(candidate_tensors, dir_path)
    json.dump(results, open(rep_path, "w"), indent=2)

    # ---- save split copies, one dir per (contrast, position) ----------------
    # HF layout: <concept>/<variant>/<contrast>/<position>/<lang>/<model>__<preset>/
    # These exist purely for HF-side navigation/browsability -- must_may and
    # must_neutral (and each position) are otherwise only distinguishable by key
    # inside the combined file. The combined file remains the source of truth;
    # steering_poc.py and any other consumer of the full per-run bundle should
    # keep reading it from `group_path`, not from a split path.
    split_paths = []  # (contrast, position, split_group_path, cand_path, rep_path)
    for name, entry in results["directions"].items():
        contrast, position = name.split("__", 1)
        split_group_path = f"{concept}/{variant}/{contrast}/{position}/{lang}/{run_tag}"
        split_dir = os.path.join(cfg["output"]["dir"], concept, variant, contrast, position, lang, run_tag)
        os.makedirs(split_dir, exist_ok=True)
        split_cand_path = os.path.join(split_dir, "dim_candidates.pt")
        split_rep_path = os.path.join(split_dir, "dim_report.json")
        torch.save({name: candidate_tensors[name]}, split_cand_path)
        split_report = {k: v for k, v in results.items() if k not in ("directions", "transfer_check")}
        split_report["contrast"], split_report["position"] = contrast, position
        split_report["directions"] = {name: entry}
        # transfer_check doesn't decompose per (contrast, position); see the combined report.
        json.dump(split_report, open(split_rep_path, "w"), indent=2)
        split_paths.append((contrast, position, split_group_path, split_cand_path, split_rep_path))

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

    if not no_push:
        try:
            if cfg["checkpoint"].get("upload_row_cache_to_hf", False):
                hf_io.push(cfg, "activations", group_path, ckpt_train.rows_dir)
            # combined bundle (pipeline-compat: steering_poc.py reads this)
            hf_io.push(cfg, "directions", group_path, dir_path)
            # results (per-model, per-layer AUC for frame- and scenario-generalization)
            # live under an AUC/ folder in the results repo.
            hf_io.push(cfg, "results", f"AUC/{group_path}", rep_path)
            # split copies, one per (contrast, position), for clean HF navigation
            for contrast, position, split_group_path, split_cand_path, split_rep_path in split_paths:
                hf_io.push(cfg, "directions", split_group_path, split_cand_path)
                hf_io.push(cfg, "results", f"AUC/{split_group_path}", split_rep_path)
        except SystemExit as e:
            print(f"[hf] push skipped: {e}")
    print(f"[done] {rep_path}")
    if split_paths:
        print(f"[done] + {len(split_paths)} split (contrast, position) copies under "
              f"{concept}/{variant}/<contrast>/<position>/{lang}/{run_tag}/")

    # ---- per-combo cleanup (host RAM) ----------------------------------------
    # store_tr/store_te hold every row's activations for every position x member
    # already moved to CPU; candidate_tensors/results hold the derived directions
    # and report. None of this is needed once this combo is saved/pushed, and
    # letting it accumulate across a long sweep is exactly the RAM-exhaustion
    # failure mode a multi-language/multi-model run risks. The model itself is
    # deliberately NOT touched here -- see the docstring above.
    del store_tr, store_te, candidate_tensors, results
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        print(f"[mem] cuda allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
    return rep_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="one or more model keys from config.models, e.g. --models qwen3-8b llama3.1-8b")
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--preset", default=None, help="rule_following | concept | concept_raw")
    ap.add_argument("--languages", nargs="+", required=True,
                    help="one or more language codes, e.g. --languages en de ig")
    ap.add_argument("--concept", required=True,
                    help="concept type for this dataset, e.g. obligation, negation")
    ap.add_argument("--dataset-variants", nargs="+", choices=["frame", "scenario"], default=["frame"],
                    help="frame: train/test split on disjoint frame_ids (same scenarios). "
                         "scenario: train/test split on disjoint scenarios (same frames). "
                         "Pass both to run the full sweep in one process.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="log and skip a failing (model, language, variant) combo "
                         "instead of aborting the whole sweep (e.g. one language OOMs "
                         "or is missing on the hub).")
    args = ap.parse_args()

    combos = [(m, l, v) for m in args.models for l in args.languages for v in args.dataset_variants]
    print(f"[sweep] {len(args.models)} model(s) x {len(args.languages)} language(s) x "
          f"{len(args.dataset_variants)} variant(s) = {len(combos)} runs")

    completed, failures = [], []
    for model_key in args.models:
        cfg, mcfg = load_cfg(args.config, model_key, preset=args.preset)
        device = mcfg.get("device", "cuda")

        print(f"\n{'=' * 70}\n[load model] {model_key}\n{'=' * 70}")
        tok, model = load_model(mcfg, cfg["optim"])
        try:
            for lang in args.languages:
                for variant in args.dataset_variants:
                    tag = f"{args.concept}/{variant}/{lang}/{model_key}"
                    print(f"\n--- [{tag}] ---")
                    try:
                        rep_path = run_one(cfg, mcfg, model_key, tok, model, args.concept,
                                           lang, variant, args.limit, args.no_push)
                        (completed if rep_path else failures).append((tag, rep_path or "no train split"))
                    except Exception as e:
                        if not args.continue_on_error:
                            raise
                        print(f"[error] {tag} failed: {e!r} -- continuing (--continue-on-error)")
                        failures.append((tag, repr(e)))
                        # an exception mid-batch can leave partially-built CUDA
                        # tensors referenced from the traceback frame; clear what
                        # we can before the next combo reuses this model.
                        gc.collect()
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
        finally:
            # always unload the model before moving to the next one (or exiting),
            # even if a combo raised -- this is the main defense against GPU
            # memory exhaustion across a multi-model sweep.
            print(f"[unload model] {model_key}")
            del tok, model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"\n[sweep] {len(completed)}/{len(combos)} succeeded, {len(failures)}/{len(combos)} failed/skipped")
    for tag, info in failures:
        print(f"  - {tag}: {info}")
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
