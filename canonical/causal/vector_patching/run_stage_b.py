"""Stage B driver: real GPU generation for the patching sweep.

Writes one JSONL line per generation immediately (crash-safe), one process
per recipient so the two recipients run in parallel on the two GPUs.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from canonical.causal.vector_patching import pair_selection as ps
from canonical.causal.vector_patching import vectors as vec
from canonical.causal.vector_patching import run_sweep as rs
from canonical.causal.vector_patching.config import CANONICAL_DATASET_REPO

MODEL_ID = "Qwen/Qwen3-8B"
DONOR_LANGS = ["en", "de", "hi", "it", "ko", "ru", "tr", "ur"]
LAYERS = {
    "ig": [12, 15, 24, 25, 26, 27, 28, 29, 30, 31],
    "yo": [15, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31],
}
N_PAIRS_PER_COMBO = 25
MAX_NEW_TOKENS = 200
PRESSURE = "L0"
SEED = 0


def log(msg, log_path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def with_retry(fn, attempts=5, base_wait=20):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(f"[{time.strftime('%H:%M:%S')}] download attempt {attempt+1} failed ({e}); "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)


def load_dataset_text(language):
    path = with_retry(
        lambda: hf_hub_download(CANONICAL_DATASET_REPO, f"data/{language}/test.jsonl", repo_type="dataset")
    )
    df = pd.read_json(path, lines=True)
    return df.set_index("id")


def load_donor_activations(languages, subset_verdicts):
    """lang -> (held_by_id dict, dom_vector). One download per language."""
    held_by_id, dom_vectors = {}, {}
    for lang in languages:
        acts = vec.load_activations(MODEL_ID, lang)
        acts = acts[acts["pressure_level"] == PRESSURE]
        lang_verdicts = subset_verdicts[subset_verdicts["language"] == lang]
        held_ids = set(lang_verdicts.loc[lang_verdicts["held"], "canonical_id"])
        failed_ids = set(lang_verdicts.loc[~lang_verdicts["held"], "canonical_id"])
        held = acts[acts["canonical_id"].isin(held_ids)]
        held_by_id[lang] = {
            row.canonical_id: row.activation.astype(np.float32) for row in held.itertuples()
        }
        dom_vectors[lang] = vec.dom_vector(acts, held_ids, failed_ids)
    return held_by_id, dom_vectors


def feasibility_lookup(grid_path, recipient):
    grid = pd.read_parquet(grid_path)
    grid = grid[grid["language_recipient"] == recipient]

    def get(donor, layer, direction):
        row = grid[
            (grid["layer"] == layer)
            & (grid["direction"] == direction)
            & ((grid["language_donor"] == donor) if direction == "dom_donor" else True)
        ]
        return float(row.iloc[0]["cohens_d"]) if len(row) else None

    return get


def build_plan(recipient, collapsed, rng, donors=DONOR_LANGS, n_pairs=N_PAIRS_PER_COMBO, include_all_avg=True):
    """List of (canonical_id, donor_kind) rows to generate, before layer
    expansion -- donor_kind in donors or 'all_avg'."""
    pairs = ps.build_pair_table(
        collapsed, MODEL_ID, PRESSURE, donor_languages=DONOR_LANGS, recipient_languages=[recipient]
    )
    rows = []
    for donor in donors:
        ids = pairs.loc[pairs["language_donor"] == donor, "canonical_id"].unique()
        chosen = rng.choice(ids, size=min(n_pairs, len(ids)), replace=False)
        rows += [(cid, donor) for cid in chosen]

    if include_all_avg:
        all_ids = pairs["canonical_id"].unique()
        chosen = rng.choice(all_ids, size=min(n_pairs, len(all_ids)), replace=False)
        rows += [(cid, "all_avg") for cid in chosen]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient", required=True, choices=["yo", "ig"])
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap total generations, for a timed pilot")
    ap.add_argument("--donor-cache", default=None,
                     help="path to a donor_cache.pkl from precompute_donor_vectors.py; "
                          "if unset, downloads donor activations itself (slower, may 429 if run "
                          "in parallel with another process doing the same)")
    ap.add_argument("--layers", default=None,
                     help="comma-separated layer list, overrides the module default LAYERS[recipient]")
    ap.add_argument("--donors", default=None,
                     help="comma-separated donor languages, overrides DONOR_LANGS (subset only -- "
                          "must already be in the donor cache)")
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS_PER_COMBO,
                     help="canonical-id pairs sampled per donor kind, default matches the full sweep")
    ap.add_argument("--no-all-avg", action="store_true", help="skip the pooled all_avg donor_kind")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS[args.recipient]
    donors = args.donors.split(",") if args.donors else DONOR_LANGS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage_b_{args.recipient}.jsonl"
    log_path = out_dir / f"stage_b_{args.recipient}.log"
    grid_path = out_dir / "stage_a_feasibility_grid.parquet"

    log(f"=== Stage B start: recipient={args.recipient} device={args.device} ===", log_path)

    verdicts = ps.load_judge_verdicts()
    collapsed = ps.collapse_verdicts(verdicts)
    subset = collapsed[(collapsed["model_id"] == MODEL_ID) & (collapsed["pressure_level"] == PRESSURE)]

    rng = np.random.default_rng(SEED)
    plan = build_plan(args.recipient, collapsed, rng, donors=donors, n_pairs=args.n_pairs,
                       include_all_avg=not args.no_all_avg)
    log(f"plan: {len(plan)} (id, donor_kind) rows x {len(layers)} layers "
        f"= {len(plan) * len(layers)} generations, max_new_tokens={args.max_new_tokens}", log_path)

    if args.donor_cache and Path(args.donor_cache).exists():
        log(f"loading donor activations from shared cache {args.donor_cache}...", log_path)
        with open(args.donor_cache, "rb") as f:
            cache = pickle.load(f)
        donor_acts, dom_vectors = cache["donor_acts"], cache["dom_vectors"]
    else:
        log("no donor cache given/found -- downloading donor activations directly...", log_path)
        donor_acts, dom_vectors = load_donor_activations(DONOR_LANGS, subset)
    all_avg_vector = vec.average_vector(dom_vectors, DONOR_LANGS)
    log("vectors ready.", log_path)

    recipient_text = load_dataset_text(args.recipient)
    get_cohens_d = feasibility_lookup(grid_path, args.recipient) if grid_path.exists() else (lambda *a: None)

    log("loading model...", log_path)
    t0 = time.time()
    model = rs.load_model(MODEL_ID, device=args.device)
    tokenizer = model.tokenizer
    log(f"model loaded in {time.time()-t0:.1f}s", log_path)

    n_done, t_start = 0, time.time()
    with open(out_path, "a") as out_f:
        for canonical_id, donor_kind in plan:
            if canonical_id not in recipient_text.index:
                continue
            drow = recipient_text.loc[canonical_id]
            for layer in layers:
                if donor_kind == "all_avg":
                    direction = all_avg_vector[layer]
                    held_here = [
                        donor_acts[lang][canonical_id]
                        for lang in DONOR_LANGS
                        if canonical_id in donor_acts[lang]
                    ]
                    if not held_here:
                        continue
                    donor_activation = np.mean(held_here, axis=0)
                    cohens_d = get_cohens_d(None, layer, "all_avg")
                else:
                    if canonical_id not in donor_acts[donor_kind]:
                        continue
                    direction = dom_vectors[donor_kind][layer]
                    donor_activation = donor_acts[donor_kind][canonical_id]
                    cohens_d = get_cohens_d(donor_kind, layer, "dom_donor")

                c_donor = rs.donor_coordinate(donor_activation, layer, direction)
                t_gen0 = time.time()
                try:
                    response = rs.run_stage_b_row(
                        model, tokenizer,
                        rule_clause=drow["system_rule"], user_query=drow["user_query"],
                        layer=layer, direction_vector=direction, c_donor=c_donor,
                        mode="patch", max_new_tokens=args.max_new_tokens,
                    )
                    error = None
                except Exception as e:
                    response = None
                    error = str(e)
                    torch.cuda.empty_cache()
                gen_time = time.time() - t_gen0

                out_f.write(json.dumps({
                    "id": canonical_id, "model_id": MODEL_ID, "language": args.recipient,
                    "category": drow["category"], "topic": drow["topic"],
                    "grammar_type": drow["grammar_type"], "pressure_level": PRESSURE,
                    "pair_type": drow["pair_type"], "sample_idx": 0,
                    "rule_clause": drow["rule_clause"], "user_query": drow["user_query"],
                    "response": response, "error": error,
                    "donor_language": (None if donor_kind == "all_avg" else donor_kind),
                    "patch_layer": layer, "vector_type": "dom",
                    "donor_kind": ("all_avg" if donor_kind == "all_avg" else "single"),
                    "patch_mode": "patch", "alpha": None,
                    "recipient_pre_verdict": False,
                    "feasibility_cohens_d": cohens_d, "gen_time_s": gen_time,
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                if n_done % 25 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed
                    log(f"{n_done} generations done, {rate:.2f}/s, "
                        f"{elapsed/60:.1f}min elapsed", log_path)
                if args.limit and n_done >= args.limit:
                    log(f"limit={args.limit} reached, stopping.", log_path)
                    log(f"=== Stage B done (pilot): {n_done} generations, "
                        f"{(time.time()-t_start)/60:.1f}min total ===", log_path)
                    return

    log(f"=== Stage B done: {n_done} generations, {(time.time()-t_start)/60:.1f}min total ===", log_path)


if __name__ == "__main__":
    main()
