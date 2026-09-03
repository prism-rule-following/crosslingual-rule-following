"""Run the paired Qwen3-8B dom-vs-w patching experiment.

The run is deliberately narrow: English donor probes, yo/ig recipients,
the preselected recipient-specific layers, and 25 matched IDs per recipient.
Each recipient writes baseline, dom-control, and w rows incrementally so a
partial run can be resumed safely.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from canonical.causal.vector_patching import pair_selection as ps
from canonical.causal.vector_patching import probe_vectors as pv
from canonical.causal.vector_patching import run_stage_b as old_stage_b
from canonical.causal.vector_patching import run_sweep as rs
from canonical.causal.vector_patching.config import CANONICAL_DATASET_REPO

MODEL_ID = "Qwen/Qwen3-8B"
DONOR_LANGUAGE = "en"
PRESSURE = "L0"
SEED = 0
N_PAIRS_PER_RECIPIENT = 25
MAX_NEW_TOKENS = 768
LAYERS = old_stage_b.LAYERS


def log(msg, log_path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def with_retry(fn, attempts=5, base_wait=20):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(
                f"[{time.strftime('%H:%M:%S')}] download attempt {attempt + 1} "
                f"failed ({exc}); retrying in {wait}s",
                flush=True,
            )
            time.sleep(wait)


def load_dataset_text(language):
    path = with_retry(
        lambda: hf_hub_download(
            CANONICAL_DATASET_REPO,
            f"data/{language}/test.jsonl",
            repo_type="dataset",
        )
    )
    return pd.read_json(path, lines=True).set_index("id")


def build_paired_ids(recipient, collapsed, n_pairs=N_PAIRS_PER_RECIPIENT):
    rng = np.random.default_rng(SEED)
    plan = old_stage_b.build_plan(
        recipient,
        collapsed,
        rng,
        donors=[DONOR_LANGUAGE],
        n_pairs=n_pairs,
        include_all_avg=False,
    )
    return [canonical_id for canonical_id, _ in plan]


def row_key(arm, canonical_id, layer):
    return arm, canonical_id, layer


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            completed.add(row_key(row["arm"], row["id"], row.get("patch_layer")))
    return completed


def response_record(drow, recipient, response, error, arm, layer, vector_type, gen_time_s):
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
        "recipient_pre_verdict": False,
        "feasibility_cohens_d": None,
        "gen_time_s": gen_time_s,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def run_baseline(model, tokenizer, drow):
    prompt = rs.format_chat_prompt(tokenizer, drow["system_rule"], drow["user_query"])
    tokens = model.to_tokens(prompt)
    n_prompt = tokens.shape[-1]
    with torch.no_grad():
        out = model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            stop_at_eos=True,
            verbose=False,
        )
    return model.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)


def run_arm(model, tokenizer, drow, arm, layer, direction, donor_activation=None):
    if arm == "baseline":
        return run_baseline(model, tokenizer, drow)
    c_donor = rs.donor_coordinate(donor_activation, layer, direction)
    return rs.run_stage_b_row(
        model,
        tokenizer,
        rule_clause=drow["system_rule"],
        user_query=drow["user_query"],
        layer=layer,
        direction_vector=direction,
        c_donor=c_donor,
        mode="patch",
        max_new_tokens=MAX_NEW_TOKENS,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient", required=True, choices=["yo", "ig"])
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS_PER_RECIPIENT)
    ap.add_argument("--layers", default=None)
    ap.add_argument(
        "--arms",
        default="baseline,dom,w",
        help="comma-separated arms from baseline,dom,w",
    )
    args = ap.parse_args()

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = set(arms) - {"baseline", "dom", "w"}
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS[args.recipient]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage_b_qwen_w_{args.recipient}.jsonl"
    log_path = out_dir / f"stage_b_qwen_w_{args.recipient}.log"
    log(
        f"=== Qwen w Stage B start: recipient={args.recipient} device={args.device} "
        f"max_new_tokens={MAX_NEW_TOKENS} ===",
        log_path,
    )

    verdicts = ps.load_judge_verdicts()
    collapsed = ps.collapse_verdicts(verdicts)
    subset = collapsed[
        (collapsed["model_id"] == MODEL_ID)
        & (collapsed["pressure_level"] == PRESSURE)
    ]
    ids = build_paired_ids(args.recipient, collapsed, n_pairs=args.n_pairs)
    planned_generations = len(ids) * (
        (1 if "baseline" in arms else 0) + len(layers) * sum(arm != "baseline" for arm in arms)
    )
    log(
        f"paired plan: {len(ids)} IDs x arms={arms} x layers={len(layers)} "
        f"= {planned_generations} generations",
        log_path,
    )

    recipient_text = load_dataset_text(args.recipient)
    missing = [canonical_id for canonical_id in ids if canonical_id not in recipient_text.index]
    if missing:
        raise ValueError(f"paired IDs missing from {args.recipient} dataset: {missing[:5]}")

    log("loading English activations and dom vector...", log_path)
    donor_acts, dom_vectors = old_stage_b.load_donor_activations([DONOR_LANGUAGE], subset)
    missing_donor = [canonical_id for canonical_id in ids if canonical_id not in donor_acts[DONOR_LANGUAGE]]
    if missing_donor:
        raise ValueError(f"paired IDs missing English activations: {missing_donor[:5]}")
    dom_direction = dom_vectors[DONOR_LANGUAGE]

    log("loading English w probe directions...", log_path)
    w_directions = {
        layer: pv.load_probe_direction(MODEL_ID, DONOR_LANGUAGE, layer)
        for layer in layers
    }
    for layer, direction in w_directions.items():
        if direction.shape != dom_direction[layer].shape:
            raise ValueError(
                f"layer {layer}: w shape {direction.shape} != dom shape {dom_direction[layer].shape}"
            )
    log("vectors ready; loading model...", log_path)
    t0 = time.time()
    model = rs.load_model(MODEL_ID, device=args.device)
    tokenizer = model.tokenizer
    log(f"model loaded in {time.time() - t0:.1f}s", log_path)

    completed = load_completed(out_path)
    if completed:
        log(f"resuming: {len(completed)} completed rows already in {out_path}", log_path)

    n_done = 0
    t_start = time.time()
    with open(out_path, "a") as out_f:
        for arm in arms:
            arm_layers = [None] if arm == "baseline" else layers
            for canonical_id in ids:
                drow = recipient_text.loc[canonical_id]
                donor_activation = donor_acts[DONOR_LANGUAGE].get(canonical_id)
                for layer in arm_layers:
                    key = row_key(arm, canonical_id, layer)
                    if key in completed:
                        continue
                    direction = None if arm == "baseline" else (
                        dom_direction[layer] if arm == "dom" else w_directions[layer]
                    )
                    t_gen0 = time.time()
                    try:
                        response = run_arm(
                            model,
                            tokenizer,
                            drow,
                            arm,
                            layer,
                            direction,
                            donor_activation=donor_activation,
                        )
                        error = None
                    except Exception as exc:
                        response = None
                        error = str(exc)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    gen_time_s = time.time() - t_gen0
                    row = response_record(
                        drow,
                        args.recipient,
                        response,
                        error,
                        arm,
                        layer,
                        "none" if arm == "baseline" else arm,
                        gen_time_s,
                    )
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed.add(key)
                    n_done += 1
                    if n_done % 25 == 0:
                        elapsed = time.time() - t_start
                        log(
                            f"{n_done} new generations done, {n_done / elapsed:.3f}/s, "
                            f"{elapsed / 60:.1f}min elapsed",
                            log_path,
                        )
                    if args.limit and n_done >= args.limit:
                        log(f"limit={args.limit} reached; output is resumable.", log_path)
                        return

    log(
        f"=== Qwen w Stage B done: {n_done} new generations, "
        f"{(time.time() - t_start) / 60:.1f}min ===",
        log_path,
    )


if __name__ == "__main__":
    main()
