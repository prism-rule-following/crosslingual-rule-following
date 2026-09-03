"""All-language Qwen3-8B dom-vs-w patching runner (plan exp2-all-langs §2).

One worker per GPU, recipients processed in waves from a frozen manifest.
Shared patch layers for every recipient: [15, 24, 27, 29, 31]. Each prompt
gets one baseline row and one dom + one w row per layer. Resumable by
(arm, prompt_id, patch_layer); fails loudly on any preflight violation.
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

from canonical.causal.vector_patching import probe_vectors as pv
from canonical.causal.vector_patching import run_qwen_w_stage_b as qw
from canonical.causal.vector_patching import run_sweep as rs
from canonical.causal.vector_patching.config import CANONICAL_DATASET_REPO

MODEL_ID = "Qwen/Qwen3-8B"
DONOR_LANGUAGE = "en"
PRESSURE = "L0"
SHARED_LAYERS = [15, 24, 27, 29, 31]
MAX_NEW_TOKENS = 768
RECIPIENTS = ["de", "hi", "ig", "it", "ko", "ru", "tr", "ur", "yo"]


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def run_baseline(model, tokenizer, drow):
    return qw.run_baseline(model, tokenizer, drow)


def response_record(drow, recipient, response, error, arm, layer, vector_type,
                    gen_time_s, pre_verdict):
    patched = arm != "baseline"
    return {
        "id": drow.name,
        "model_id": MODEL_ID,
        "language": recipient,
        "category": drow["category"],
        "topic": drow["topic"],
        "grammar_type": drow["grammar_type"],
        "pressure_level": PRESSURE,
        "pair_type": drow["pair_type"],
        "sample_idx": 0,
        "rule_clause": drow["rule_clause"],
        "user_query": drow["user_query"],
        "response": response,
        "error": error,
        "arm": arm,
        "donor_language": DONOR_LANGUAGE if patched else None,
        "patch_layer": layer,
        "vector_type": vector_type,
        "donor_kind": "single" if patched else "baseline",
        "patch_mode": "patch" if patched else "none",
        "alpha": None,
        "recipient_pre_verdict": pre_verdict,
        "feasibility_cohens_d": None,
        "gen_time_s": gen_time_s,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def run_patch(model, tokenizer, drow, layer, direction, donor_activation):
    c_donor = rs.donor_coordinate(donor_activation, layer, direction)
    return rs.run_stage_b_row(
        model, tokenizer,
        rule_clause=drow["system_rule"], user_query=drow["user_query"],
        layer=layer, direction_vector=direction, c_donor=c_donor,
        mode="patch", max_new_tokens=MAX_NEW_TOKENS,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--recipient", required=True, choices=RECIPIENTS)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--donor-cache", required=True)
    ap.add_argument("--layers", default=",".join(str(x) for x in SHARED_LAYERS))
    ap.add_argument("--arms", default="baseline,dom,w")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.recipient not in manifest["recipients"]:
        raise ValueError(f"recipient {args.recipient} not in manifest {manifest['manifest_id']}")
    if manifest["model_id"] != MODEL_ID or manifest["donor_language"] != DONOR_LANGUAGE:
        raise ValueError("manifest model/donor mismatch")
    id_category = manifest["id_category"]
    ids = manifest["selected_ids"]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if set(layers) != set(SHARED_LAYERS):
        raise ValueError(f"unexpected layer set {layers}; must be {SHARED_LAYERS}")
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = set(arms) - {"baseline", "dom", "w"}
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage_b_all_{args.recipient}.jsonl"
    log_path = out_dir / f"stage_b_all_{args.recipient}.log"
    qw.log(
        f"=== all-langs run: recipient={args.recipient} device={args.device} "
        f"manifest={manifest['manifest_id']} arms={arms} layers={layers} ===",
        log_path,
    )

    recipient_text = qw.load_dataset_text(args.recipient)
    for cid in ids:
        if cid not in recipient_text.index:
            raise ValueError(f"recipient {args.recipient} missing manifest id {cid}")
        if recipient_text.loc[cid, "category"] != id_category[cid]:
            raise ValueError(f"recipient {args.recipient} category mismatch on {cid}")

    with open(args.donor_cache, "rb") as f:
        cache = pickle.load(f)
    donor_acts, dom_vectors = cache["donor_acts"], cache["dom_vectors"]
    if DONOR_LANGUAGE not in dom_vectors:
        raise ValueError(f"donor cache missing {DONOR_LANGUAGE} dom vector")
    dom_direction = dom_vectors[DONOR_LANGUAGE]
    if dom_direction.ndim != 2:
        raise ValueError(f"dom direction must be (n_layers, d_model), got {dom_direction.shape}")
    en_acts = donor_acts[DONOR_LANGUAGE]
    missing_donor = [cid for cid in ids if cid not in en_acts]
    if missing_donor:
        raise ValueError(f"manifest ids missing English held activations: {missing_donor}")
    for layer in layers:
        if layer >= len(dom_direction):
            raise ValueError(f"layer {layer} out of range for dom vector ({len(dom_direction)})")

    w_directions = {layer: pv.load_probe_direction(MODEL_ID, DONOR_LANGUAGE, layer) for layer in layers}
    for layer, direction in w_directions.items():
        if direction.shape != dom_direction[layer].shape:
            raise ValueError(
                f"layer {layer}: w shape {direction.shape} != dom shape {dom_direction[layer].shape}"
            )

    qw.log("vectors validated; loading model...", log_path)
    t0 = time.time()
    model = rs.load_model(MODEL_ID, device=args.device)
    tokenizer = model.tokenizer
    qw.log(f"model loaded in {time.time() - t0:.1f}s", log_path)

    completed = qw.load_completed(out_path)
    if completed:
        qw.log(f"resuming: {len(completed)} completed rows already in {out_path}", log_path)

    planned = len(ids) * (
        (1 if "baseline" in arms else 0) + len(layers) * sum(arm != "baseline" for arm in arms)
    )
    qw.log(f"plan: {len(ids)} ids x arms={arms} x {len(layers)} layers = {planned} rows", log_path)

    n_done = 0
    t_start = time.time()
    with open(out_path, "a") as out_f:
        for arm in arms:
            arm_layers = [None] if arm == "baseline" else layers
            for cid in ids:
                drow = recipient_text.loc[cid]
                donor_activation = en_acts[cid]
                pre_verdict = manifest["recipient_verdicts"].get(args.recipient, {}).get(cid)
                if pre_verdict is not None:
                    pre_verdict = pre_verdict["held"]
                for layer in arm_layers:
                    key = qw.row_key(arm, cid, layer)
                    if key in completed:
                        continue
                    direction = None if arm == "baseline" else (
                        dom_direction[layer] if arm == "dom" else w_directions[layer]
                    )
                    t_gen0 = time.time()
                    try:
                        if arm == "baseline":
                            response = run_baseline(model, tokenizer, drow)
                        else:
                            response = run_patch(
                                model, tokenizer, drow, layer, direction, donor_activation
                            )
                        error = None
                    except Exception as exc:
                        response = None
                        error = str(exc)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    gen_time_s = time.time() - t_gen0
                    row = response_record(
                        drow, args.recipient, response, error, arm, layer,
                        "none" if arm == "baseline" else arm, gen_time_s, pre_verdict,
                    )
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed.add(key)
                    n_done += 1
                    if n_done % 25 == 0:
                        elapsed = time.time() - t_start
                        qw.log(
                            f"{n_done} new rows, {n_done / elapsed:.3f}/s, "
                            f"{elapsed / 60:.1f}min elapsed",
                            log_path,
                        )
                    if args.limit and n_done >= args.limit:
                        qw.log(f"limit={args.limit} reached; output is resumable.", log_path)
                        return

    qw.log(
        f"=== all-langs run done: {n_done} new rows, "
        f"{(time.time() - t_start) / 60:.1f}min ===",
        log_path,
    )


if __name__ == "__main__":
    main()