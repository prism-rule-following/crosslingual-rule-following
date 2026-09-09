#!/usr/bin/env python3
"""
Difference-in-means extraction for the RULE ACTIVE vs REVOKED contrast -- a
different concept and a different source dataset from extract_dim.py (which
does must/may/neutral obligation contrasts from nunaa/canonical_obligation_dataset).
This one uses crosslingual-rule-following/canonical-dataset, one test.jsonl per
language, with no train/test split of its own -- there is exactly one file per
language holding the whole set, so THIS script does its own split.

The split holds out pair_type (the status-word pair used to express "active" vs
"revoked": active/cancelled, on/off, true/false, valid/invalid, enabled/
disabled) rather than splitting randomly -- train never sees the held-out
wording. This tests whether the direction generalizes to an unseen way of
LEXICALIZING the contrast, not just to unseen instances of wordings it already
saw; it's the closest analog here to the obligation pipeline's lexeme_set
holdout. See --test-pair-types.

Contrast:  rule_text (active)  vs  non_rule_text (revoked)
Both fields are already fully-formed "<rule_clause>. Rule status: <word>."
strings; only the trailing status word differs (active/cancelled, on/off,
true/false, valid/invalid, enabled/disabled, depending on the row's pair_type).
`active_status`/`revoked_status` give that word directly, so it's used as the
contrast_token anchor with no need for a separate token field.

`system_rule`/`system_non_rule` in the source data are exactly
`f"{context} Rule: {rule_text}"` / `f"{context} Rule: {non_rule_text}"` -- i.e.
identical to what build_prompt()'s system_user mode already constructs, so the
same prompt/position machinery from extract_dim.py applies unchanged; this
script imports it rather than re-implementing it. What's NOT reused is
extract_store(), which is hardcoded in extract_dim.py to exactly 3 members
(clean/corrupt_may/corrupt_neutral) -- this contrast has exactly 2 (active/
revoked), so a small dedicated extraction loop is used instead of forcing a
fake third member through the 3-member shape.

Positions: contrast_token, rule_clause_end, post_instruction (same anchors,
same meanings, same enable_thinking + post_instruction guardrail as
extract_dim.py -- see load_cfg() there).

Like extract_dim.py: models loaded once and reused across every language under
them, per-combo host RAM freed each iteration, model cache evicted from disk
between models, results batched into one HF commit per model (not one per
combo) to stay under HF's 128 commits/hour cap.

Unlike extract_dim.py: this dataset has ~2340 rows/language (~10x the
obligation dataset's ~238), so its row-level activation checkpoint -- meant
only to survive a crash mid-combo, not to be a durable archive -- would
otherwise accumulate to ~170GB across a full sweep and exhaust disk on the
very first combo (this happened). A combo's row cache is deleted once its DIM
math is saved, by default (see --keep-row-cache); pass --no-checkpoint to skip
checkpointing entirely instead (no disk writes at all, but a crash mid-combo
redoes that whole combo's extraction rather than resuming it).

HF layout (results/directions repos, config.hf.directions_repo/results_repo --
same repos extract_dim.py uses, disambiguated by the "active_revoked" concept
segment): there is only one contrast here, so unlike extract_dim.py's
concept/variant/contrast/position/... path there is no contrast segment to add:
  directions: active_revoked/<position>/<language>/<model>/dim_candidates.pt
  results:    AUC/active_revoked/<position>/<language>/<model>/dim_report.json

Usage:
  python extract_dim_active_revoked.py --models qwen3-8b llama3.1-8b \
    --languages en de hi ig it ko ru tr ur yo am sw ta \
    --test-pair-types enabled_disabled --continue-on-error
"""
import os, json, argparse, math, gc, shutil
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from checkpoint import RowCheckpoint
import hf_io
from extract_dim import (
    DTYPES, build_prompt, locate_positions, load_model, batched_hidden_states, _auc,
)

load_dotenv()

POSITIONS = ["contrast_token", "rule_clause_end", "post_instruction"]
CONCEPT = "active_revoked"


ALL_PAIR_TYPES = {"active_cancelled", "on_off", "true_false", "valid_invalid", "enabled_disabled"}


# --------------------------------------------------------------------------- #
def load_language_rows(cfg, language, test_pair_types):
    """Pull the whole per-language test.jsonl (there is no pre-split train/test
    for this dataset -- every row for a language lives in one file) and split
    it by pair_type: train never sees the status-word pair(s) used for test.
    This is a generalization test, not an instance holdout -- it asks whether
    the active/revoked direction transfers to an unseen way of LEXICALIZING
    "active" vs "revoked" (active/cancelled, on/off, true/false, valid/invalid,
    enabled/disabled), analogous to the obligation pipeline's lexeme_set holdout."""
    unknown = set(test_pair_types) - ALL_PAIR_TYPES
    if unknown:
        raise SystemExit(f"[data] unknown pair_type(s) in --test-pair-types: {unknown}; "
                          f"valid values: {sorted(ALL_PAIR_TYPES)}")
    from huggingface_hub import hf_hub_download
    repo = cfg["hf"]["active_revoked_dataset_repo"]
    token = os.environ.get(cfg["hf"].get("token_env", "HF_TOKEN"))
    fp = hf_hub_download(repo, f"data/{language}/test.jsonl", repo_type="dataset", token=token)
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    test_set = set(test_pair_types)
    train_rows = [r for r in rows if r["pair_type"] not in test_set]
    test_rows = [r for r in rows if r["pair_type"] in test_set]
    if not train_rows or not test_rows:
        raise SystemExit(f"[data] {language}: train={len(train_rows)} test={len(test_rows)} rows "
                          f"after splitting on pair_type={sorted(test_set)} -- one side is empty, "
                          f"check --test-pair-types against the dataset's actual pair_type values")
    return train_rows, test_rows


def check_thinking_guardrail(mcfg, model_key):
    # Same reasoning as extract_dim.py's load_cfg(): post_instruction is "last
    # prompt token before generation," which only holds when thinking is off --
    # with it on, that token is the <think> scaffold opening the reasoning
    # trace, not the pre-answer boundary. POSITIONS here is fixed and always
    # includes post_instruction, so this check is unconditional (not gated on
    # a configurable positions list like extract_dim.py's).
    if mcfg.get("enable_thinking", False):
        raise SystemExit(
            f"[config] model '{model_key}' has enable_thinking=true. This script's "
            f"positions always include post_instruction, which requires thinking off "
            f"(see extract_dim.py's load_cfg for the full explanation). Set "
            f"enable_thinking=false for this model."
        )


# --------------------------------------------------------------------------- #
def extract_store(rows, ckpt, tok, model, mcfg, ocfg, device):
    """Run the model over `rows`, return store: member -> position -> list of
    (meta, tensor[nl+1, d]). Members are exactly "active" and "revoked" -- see
    module docstring for why this doesn't reuse extract_dim.py's 3-member
    extract_store().

    The in-memory `store` is built directly from each row's freshly-computed
    activations as they complete, NOT by reading them back from `ckpt` --
    `ckpt.save_row()`/`done_ids()` are purely a resumability side-channel here
    (write-through cache: skip rows already on disk, persist new ones for a
    future resume). This matters when checkpointing is disabled
    (config.checkpoint.enabled=false, or --no-checkpoint): RowCheckpoint's
    methods all safely no-op in that case (no directories, no writes), so a
    version of this function that assembled `store` FROM ckpt.load_all() -- as
    extract_dim.py's does -- would silently come back empty. Building `store`
    from row_buf directly makes checkpointing truly optional rather than a
    hidden dependency of the extraction itself."""
    B = ocfg.get("batch_size", 8); empty_every = ocfg.get("empty_cache_every_n_batches", 4)
    members = {"active": ("rule_text", "active_status"), "revoked": ("non_rule_text", "revoked_status")}

    store = defaultdict(lambda: defaultdict(list))
    # seed with anything already on disk from a prior (interrupted) run of
    # this exact combo -- a no-op read when checkpointing is disabled.
    for rid, payload, meta in ckpt.load_all():
        for m, posmap in payload.items():
            for p, vec in posmap.items():
                store[m][p].append((meta, vec))

    already = ckpt.done_ids()
    todo = [r for r in rows if r["id"] not in already]
    print(f"[ckpt:{ckpt.base}] {len(already)} cached, {len(todo)} to compute")

    work = [(r, m) for r in todo for m in members]
    need_members = set(members); row_buf = defaultdict(dict); batch_i = 0
    for s in range(0, len(work), B):
        chunk = work[s:s + B]
        id_lists, locs, keys = [], [], []
        for r, m in chunk:
            text_field, tok_field = members[m]
            rule_text = r[text_field]; cword = r[tok_field]
            # mcfg (not {}) so enable_thinking actually reaches apply_chat_template --
            # the guardrail in main() only checks the config says thinking is off;
            # it has to actually be threaded through here to be enforced.
            ids, _, _ = build_prompt(tok, mcfg, "system_user", rule_text=rule_text,
                                     context=r["context"], query=r["user_query"])
            id_lists.append(ids)
            locs.append(locate_positions(tok, ids, rule_text, cword, "system_user", POSITIONS))
            keys.append((r["id"], m, r))
        hs_list = batched_hidden_states(model, tok, id_lists, device, ocfg.get("use_cache", False))
        for (rid, m, r), hs, pos in zip(keys, hs_list, locs):
            row_buf[rid][m] = {p: hs[:, pos[p], :].clone() for p in POSITIONS}
            meta = {"category": r.get("category"), "topic": r.get("topic"),
                    "pair_type": r.get("pair_type"), "grammar_type": r.get("grammar_type"), "id": rid}
            if need_members.issubset(row_buf[rid].keys()):
                payload = row_buf.pop(rid)
                ckpt.save_row(rid, payload, meta)  # no-op if checkpointing disabled
                for m2, posmap in payload.items():
                    for p2, vec in posmap.items():
                        store[m2][p2].append((meta, vec))
        batch_i += 1
        if device.startswith("cuda") and empty_every and batch_i % empty_every == 0:
            torch.cuda.empty_cache(); gc.collect()
        if batch_i % 5 == 0:
            print(f"  batch {batch_i} ({min(s+B,len(work))}/{len(work)} items)")
    ckpt.finalize()
    return store


def cohens_d(a, b):
    va, vb = a.var(unbiased=True), b.var(unbiased=True); n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1-1)*va + (n2-1)*vb)/max(n1+n2-2,1) + 1e-12)
    return float((a.mean()-b.mean())/(sp+1e-12))


def all_of(store, member, position):
    return torch.stack([v for _, v in store[member][position]], 0)


def eval_split(store, unit, position):
    ca, ka = all_of(store, "active", position), all_of(store, "revoked", position)
    d_list, auc_list = [], []
    for l in range(ca.shape[1]):
        pc = (ca[:, l, :] * unit[l]).sum(-1)
        pk = (ka[:, l, :] * unit[l]).sum(-1)
        d_list.append(cohens_d(pc, pk))
        auc_list.append(_auc(pc.numpy(), pk.numpy()))
    return d_list, auc_list


# --------------------------------------------------------------------------- #
def run_one(cfg, mcfg, model_key, tok, model, lang, test_pair_types, seed, limit, no_push,
           keep_row_cache, no_checkpoint):
    device = mcfg.get("device", "cuda")
    torch.manual_seed(seed)
    # the pair_type holdout is baked into the checkpoint namespace (not just
    # results metadata) because which rows are "train" vs "test" depends on it --
    # reusing a namespace across a different --test-pair-types selection would
    # mean rows cached under the old split's __train dir don't get looked up
    # for the new split's __test dir (and vice versa), forcing silent recompute
    # rather than wrong results, but the naming should make the config explicit.
    held_out = "_".join(sorted(test_pair_types))
    ckpt_ns = f"{CONCEPT}__{lang}__{model_key}__ho_{held_out}"

    train_rows, test_rows = load_language_rows(cfg, lang, test_pair_types)
    if limit:
        train_rows, test_rows = train_rows[:limit], test_rows[:limit]
    train_pair_types = sorted(ALL_PAIR_TYPES - set(test_pair_types))
    print(f"[data] {lang}: train={len(train_rows)} test={len(test_rows)} rows "
          f"(train pair_types={train_pair_types}, test pair_types={sorted(test_pair_types)})")

    # --no-checkpoint disables RowCheckpoint entirely (no directories, no writes,
    # no resumability) rather than just skipping the post-combo cleanup below --
    # useful if you'd rather not touch disk for row activations at all and are
    # fine re-running a whole combo from scratch if it's interrupted. Only
    # affects the checkpoint's own config; cfg itself (output dir, HF repos,
    # etc.) is untouched.
    ckpt_cfg = {**cfg, "checkpoint": {**cfg["checkpoint"], "enabled": False}} if no_checkpoint else cfg
    ckpt_train = RowCheckpoint(ckpt_cfg, f"{ckpt_ns}__train")
    store_tr = extract_store(train_rows, ckpt_train, tok, model, mcfg, cfg["optim"], device)
    ckpt_test = RowCheckpoint(ckpt_cfg, f"{ckpt_ns}__test")
    store_te = extract_store(test_rows, ckpt_test, tok, model, mcfg, cfg["optim"], device)

    p0 = POSITIONS[0]
    n_layers_p1 = store_tr["active"][p0][0][1].shape[0]
    d_model = store_tr["active"][p0][0][1].shape[1]

    results = {"model": model_key, "hf_name": mcfg["hf_name"], "concept": CONCEPT,
               "language": lang, "contrast": "active_revoked",
               "n_layers_incl_embed": n_layers_p1, "d_model": d_model,
               "positions": POSITIONS, "n_train": len(train_rows), "n_test": len(test_rows),
               "train_pair_types": train_pair_types, "test_pair_types": sorted(test_pair_types),
               "directions": {}}
    candidate_tensors = {}

    for position in POSITIONS:
        dim = all_of(store_tr, "active", position).mean(0) - all_of(store_tr, "revoked", position).mean(0)
        norms = dim.norm(dim=-1); unit = dim / (norms.unsqueeze(-1) + 1e-8)
        candidate_tensors[position] = unit

        d_tr, auc_tr = eval_split(store_tr, unit, position)
        bl = int(np.argmax(np.abs(d_tr)))
        d_te, auc_te = eval_split(store_te, unit, position)
        bl_te = int(np.argmax(np.abs(d_te)))
        results["directions"][position] = {
            "position": position, "per_layer_norm": norms.tolist(), "argmax_norm_layer": int(norms.argmax()),
            "in_sample": {"per_layer_cohens_d": d_tr, "per_layer_auc": auc_tr,
                          "best_layer": bl, "best_cohens_d": d_tr[bl], "best_auc": auc_tr[bl]},
            "held_out": {"per_layer_cohens_d": d_te, "per_layer_auc": auc_te,
                         "at_train_best_layer": {"layer": bl, "cohens_d": d_te[bl], "auc": auc_te[bl]},
                         "own_best_layer": {"layer": bl_te, "cohens_d": d_te[bl_te], "auc": auc_te[bl_te]}},
        }

    print(f"\n=== DIM summary [{CONCEPT}/{lang}/{model_key}] ===")
    print(f"  {'position':18s} {'layer':>5} | {'in-sample':>18} | {'held-out @train-L':>18} | {'held-out own-best':>18}")
    print(f"  {'':18s} {'':>5} | {'d':>8} {'AUC':>8} | {'d':>8} {'AUC':>8} | {'L':>4} {'d':>6} {'AUC':>5}")
    for position, r in results["directions"].items():
        ins, ho = r["in_sample"], r["held_out"]; L = ins["best_layer"]
        at, ob = ho["at_train_best_layer"], ho["own_best_layer"]
        print(f"  {position:18s} {L:>5} | {ins['best_cohens_d']:>8.3f} {ins['best_auc']:>8.3f} | "
              f"{at['cohens_d']:>8.3f} {at['auc']:>8.3f} | {ob['layer']:>4} {ob['cohens_d']:>6.2f} {ob['auc']:>5.2f}")

    # ---- save one dir per position -- no combined file, same reasoning as
    # extract_dim.py: candidate_tensors/results["directions"] already hold
    # everything, keyed by position, so a combined file would just duplicate it.
    out_dir_base = cfg["output"]["dir"]
    split_paths = []  # (position, group_path, cand_path, rep_path)
    for position, entry in results["directions"].items():
        group_path = f"{CONCEPT}/{position}/{lang}/{model_key}"
        local_dir = os.path.join(out_dir_base, CONCEPT, position, lang, model_key)
        os.makedirs(local_dir, exist_ok=True)
        cand_path = os.path.join(local_dir, "dim_candidates.pt")
        rep_path = os.path.join(local_dir, "dim_report.json")
        torch.save({position: candidate_tensors[position]}, cand_path)
        rep = {k: v for k, v in results.items() if k != "directions"}
        rep["directions"] = {position: entry}
        json.dump(rep, open(rep_path, "w"), indent=2)
        split_paths.append((position, group_path, cand_path, rep_path))

    if not no_push:
        local_root = out_dir_base
        patterns = [f"{gp}/*" for _, gp, _, _ in split_paths]

        def _try(label, fn):
            try:
                fn()
            except Exception as e:
                print(f"[hf][warn] {label} push failed, local files kept at "
                      f"{local_root}/{CONCEPT}/*/{lang}/{model_key}/ for a later retry: {e!r}")

        _try("directions", lambda: hf_io.push_batch(
            cfg, "directions", local_root, patterns,
            commit_message=f"active_revoked directions: {len(split_paths)} positions for {lang}/{model_key}"))
        _try("results", lambda: hf_io.push_batch(
            cfg, "results", local_root, patterns, path_in_repo="AUC",
            commit_message=f"active_revoked results: {len(split_paths)} positions for {lang}/{model_key}"))

    if not keep_row_cache and not no_checkpoint:
        # (no_checkpoint already means nothing was ever written to disk here)
        # The row-level checkpoint exists so a crash MID-extraction can resume
        # without redoing work already done for THIS combo -- it is not meant
        # to be a durable archive of every combo's raw activations. This
        # dataset has ~2340 rows/language (vs ~238 for the obligation
        # dataset extract_dim.py was built around); at 2 members x 3 positions
        # x float32, that's ~3.6MB/row -- ~8.5GB per (language, model) combo,
        # ~170GB for a full 10-language x 2-model sweep if never freed. Since
        # nothing else can free disk mid-sweep, this filled the very first
        # combo's quota and every subsequent combo failed identically for the
        # rest of the run. The direction tensors + reports this combo produced
        # are already saved under dim_out/ (and pushed, if not --no-push)
        # before this point, so the raw per-row cache is safe to drop now.
        for ckpt, label in ((ckpt_train, "train"), (ckpt_test, "test")):
            try:
                size = sum(f.stat().st_size for f in Path(ckpt.base).rglob("*") if f.is_file())
                shutil.rmtree(ckpt.base, ignore_errors=True)
                print(f"[ckpt] cleared {label} row cache for {lang}/{model_key}: freed {size/1e9:.2f}GB")
            except Exception as e:
                print(f"[ckpt][warn] could not clear {label} row cache at {ckpt.base}: {e!r}")

    del store_tr, store_te, candidate_tensors, results
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return f"{CONCEPT}/{lang}/{model_key}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="one or more model keys from config.models, e.g. --models qwen3-8b llama3.1-8b")
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--languages", nargs="+", required=True,
                    help="one or more language codes present in the active-revoked dataset, "
                         "e.g. --languages en de hi ig it ko ru tr ur yo am sw ta")
    ap.add_argument("--test-pair-types", nargs="+", default=["enabled_disabled"],
                    choices=sorted(ALL_PAIR_TYPES),
                    help="pair_type value(s) held out entirely for test -- train never sees "
                         "them. Tests generalization to an unseen active/revoked wording, not "
                         "just unseen instances (default: train on active_cancelled/on_off/"
                         "true_false/valid_invalid, test on enabled_disabled)")
    ap.add_argument("--seed", type=int, default=0, help="torch.manual_seed for this run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="log and skip a failing (model, language) combo instead of "
                         "aborting the whole sweep")
    ap.add_argument("--keep-model-cache", action="store_true",
                    help="don't evict a model's downloaded weights from the local HF cache "
                         "once its combos are done (see extract_dim.py's flag of the same name)")
    ap.add_argument("--keep-row-cache", action="store_true",
                    help="don't delete a combo's row-level activation checkpoint once its "
                         "DIM math is saved. Default is to delete: this dataset has ~2340 "
                         "rows/language, and the row cache is ~3.6MB/row (2 members x 3 "
                         "positions x float32) -- ~8.5GB per (language, model) combo, ~170GB "
                         "for a full sweep if kept, which will exhaust disk on most machines. "
                         "The row cache still protects a crash mid-combo (still resumable while "
                         "that combo is in progress); this only clears it AFTER a combo's "
                         "outputs are safely saved. Pass this only if you have disk to spare "
                         "and want to inspect raw per-row activations later.")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="disable row-level checkpointing entirely for this run -- no "
                         "directories, no writes, no resumability. A crash mid-combo means "
                         "redoing that whole combo's extraction from scratch rather than "
                         "resuming. Use this if you'd rather not touch disk for row "
                         "activations at all; --keep-row-cache is irrelevant when this is set "
                         "since there's nothing written to keep.")
    args = ap.parse_args()

    cfg_all = json.load(open(args.config))
    combos = [(m, l) for m in args.models for l in args.languages]
    print(f"[sweep] {len(args.models)} model(s) x {len(args.languages)} language(s) = {len(combos)} runs")

    completed, failures = [], []
    for model_key in args.models:
        mcfg = cfg_all["models"][model_key]
        check_thinking_guardrail(mcfg, model_key)
        device = mcfg.get("device", "cuda")

        print(f"\n{'=' * 70}\n[load model] {model_key}\n{'=' * 70}")
        tok, model = load_model(mcfg, cfg_all["optim"])
        try:
            for lang in args.languages:
                tag = f"{CONCEPT}/{lang}/{model_key}"
                print(f"\n--- [{tag}] ---")
                try:
                    done_id = run_one(cfg_all, mcfg, model_key, tok, model, lang,
                                      args.test_pair_types, args.seed, args.limit, args.no_push,
                                      args.keep_row_cache, args.no_checkpoint)
                    completed.append((tag, done_id))
                except Exception as e:
                    if not args.continue_on_error:
                        raise
                    print(f"[error] {tag} failed: {e!r} -- continuing (--continue-on-error)")
                    failures.append((tag, repr(e)))
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
        finally:
            print(f"[unload model] {model_key}")
            del tok, model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if not args.keep_model_cache:
                hf_io.evict_model_cache(mcfg["hf_name"])

    print(f"\n[sweep] {len(completed)}/{len(combos)} succeeded, {len(failures)}/{len(combos)} failed")
    for tag, info in failures:
        print(f"  - {tag}: {info}")
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
