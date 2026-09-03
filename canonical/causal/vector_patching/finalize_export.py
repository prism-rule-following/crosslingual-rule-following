"""Packages all nine recipients' Stage B JSONL output into the documented
response schema, runs the structural audit, and uploads to HF. Run once,
after every recipient finishes.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from huggingface_hub import hf_hub_download

from canonical.causal.vector_patching.config import EXPORT_REPO
from canonical.causal.vector_patching.export_responses import (
    build_response_row, export_to_hf, sanity_check_response,
)

RECIPIENTS = ["de", "hi", "ig", "it", "ko", "ru", "tr", "ur", "yo"]
SHARED_LAYERS = [15, 24, 27, 29, 31]
MAX_NEW_TOKENS = 768
README_PATH = Path(__file__).resolve().parent / "README.md"


def sanity_check(response, expected_language):
    if response is None:
        return {"still_target_language": None, "non_degenerate": None}
    return sanity_check_response(response, expected_language)


def audit(manifest, rows_by_recipient):
    problems = []
    stats = {}
    manifest_ids = set(manifest["selected_ids"])
    id_category = manifest["id_category"]
    for lang, rows in rows_by_recipient.items():
        stats[lang] = {"n": len(rows)}
        expected = 25 + 2 * len(SHARED_LAYERS) * 25
        if len(rows) != expected:
            problems.append(f"{lang}: {len(rows)} rows, expected {expected}")
        arm_counts = Counter(r["arm"] for r in rows)
        if arm_counts["baseline"] != 25:
            problems.append(f"{lang}: baseline {arm_counts['baseline']} != 25")
        for arm in ("dom", "w"):
            if arm_counts[arm] != len(SHARED_LAYERS) * 25:
                problems.append(f"{lang}: {arm} {arm_counts[arm]} != {len(SHARED_LAYERS) * 25}")
        keys = [(r["arm"], r["id"], r["patch_layer"]) for r in rows]
        if len(set(keys)) != len(keys):
            problems.append(f"{lang}: duplicate (arm, id, patch_layer) keys")
        errors = sum(1 for r in rows if r.get("error"))
        stats[lang]["errors"] = errors
        if errors:
            problems.append(f"{lang}: {errors} generation errors")
        empty = sum(1 for r in rows if not (r.get("response") or "").strip())
        if empty:
            problems.append(f"{lang}: {empty} empty responses")
        bad_ids = [r["id"] for r in rows if r["id"] not in manifest_ids]
        if bad_ids:
            problems.append(f"{lang}: {len(bad_ids)} ids outside manifest")
        bad_cat = [r["id"] for r in rows if r["category"] != id_category[r["id"]]]
        if bad_cat:
            problems.append(f"{lang}: {len(bad_cat)} category mismatches vs manifest")
        leaked = sum(1 for r in rows if (r.get("response") or "") and "<|" in r["response"])
        if leaked:
            problems.append(f"{lang}: {leaked} rows with special-token leakage")
        bad_tokens = sum(1 for r in rows if r.get("max_new_tokens") != MAX_NEW_TOKENS)
        if bad_tokens:
            problems.append(f"{lang}: {bad_tokens} rows with wrong max_new_tokens")
        bad_model = sum(1 for r in rows if r["model_id"] != manifest["model_id"])
        if bad_model:
            problems.append(f"{lang}: {bad_model} rows with wrong model_id")
        bad_donor = sum(1 for r in rows if r["arm"] != "baseline" and r["donor_language"] != manifest["donor_language"])
        if bad_donor:
            problems.append(f"{lang}: {bad_donor} patched rows with wrong donor_language")
        bad_layers = {r["patch_layer"] for r in rows if r["arm"] != "baseline"}
        if bad_layers != set(SHARED_LAYERS):
            problems.append(f"{lang}: patched layer set {sorted(bad_layers)} != {SHARED_LAYERS}")
        null_verdict = [r["id"] for r in rows if r.get("recipient_pre_verdict") is None]
        stats[lang]["null_pre_verdict"] = len(null_verdict)
    return problems, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/exp2_out")
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    with open(args.manifest) as f:
        manifest = json.load(f)

    rows_by_recipient = {}
    for lang in RECIPIENTS:
        path = out_dir / f"stage_b_all_{lang}.jsonl"
        with open(path) as f:
            rows_by_recipient[lang] = [json.loads(line) for line in f]
        print(f"{lang}: {len(rows_by_recipient[lang])} rows", flush=True)

    problems, stats = audit(manifest, rows_by_recipient)
    if problems:
        print("STRUCTURAL AUDIT FAILED:", flush=True)
        for p in problems:
            print(" -", p, flush=True)
        sys.exit(1)
    print(f"STRUCTURAL AUDIT OK: {sum(len(v) for v in rows_by_recipient.values())} rows", flush=True)

    all_rows = []
    for lang in RECIPIENTS:
        for r in rows_by_recipient[lang]:
            checks = sanity_check(r["response"], lang)
            all_rows.append(build_response_row(
                canonical_id=r["id"], model_id=r["model_id"], language=r["language"],
                category=r["category"], topic=r["topic"], grammar_type=r["grammar_type"],
                pressure_level=r["pressure_level"], pair_type=r["pair_type"],
                sample_idx=r["sample_idx"], rule_clause=r["rule_clause"],
                user_query=r["user_query"], response=r["response"],
                donor_language=r["donor_language"], patch_layer=r["patch_layer"],
                vector_type=r["vector_type"], donor_kind=r["donor_kind"],
                patch_mode=r["patch_mode"], alpha=r["alpha"],
                recipient_pre_verdict=r["recipient_pre_verdict"],
                feasibility_cohens_d=r["feasibility_cohens_d"],
                still_target_language=checks["still_target_language"],
                non_degenerate=checks["non_degenerate"],
            ))

    ts = time.strftime("%Y%m%d_%H%M%S")
    path_in_repo = f"qwen3-8b/exp2_all_langs_{ts}.parquet"
    dest = export_to_hf(all_rows, path_in_repo)
    print(f"uploaded {len(all_rows)} rows to {dest}", flush=True)

    print("verifying upload by reading it back from HF...", flush=True)
    local_check_path = hf_hub_download(EXPORT_REPO, path_in_repo, repo_type="dataset")
    remote_df = pd.read_parquet(local_check_path)
    if len(remote_df) != len(all_rows):
        print(f"VERIFICATION FAILED: uploaded {len(all_rows)} rows, "
              f"read back {len(remote_df)}. DO NOT delete local data.", flush=True)
        sys.exit(1)
    if set(remote_df["id"]) != {r["id"] for r in all_rows}:
        print("VERIFICATION FAILED: id sets don't match. DO NOT delete local data.", flush=True)
        sys.exit(1)
    print(f"VERIFIED: {len(remote_df)} rows read back match exactly. "
          f"SAFE_TO_CLEANUP", flush=True)

    hf_url = f"https://huggingface.co/datasets/{EXPORT_REPO}/blob/main/{path_in_repo}"
    section = [
        "",
        "## Upload log",
        "",
        f"- **{ts}** ({len(all_rows)} total rows, 0 generation errors, "
        f"verified by read-back after upload)",
        f"  - manifest: `{manifest['manifest_id']}` (25 shared IDs, "
        f"{manifest['category_order']})",
        f"  - HF: [{path_in_repo}]({hf_url})",
    ]
    for lang, s in stats.items():
        section.append(
            f"  - `{lang}`: {s['n']} rows ({s['errors']} errors, "
            f"{s['null_pre_verdict']} null pre-verdicts)"
        )
    with open(README_PATH, "a") as f:
        f.write("\n".join(section) + "\n")
    print(f"appended upload log to {README_PATH}", flush=True)


if __name__ == "__main__":
    main()