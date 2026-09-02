"""Packages the two recipients' Stage B JSONL output into the documented
response schema and uploads to HF. Run once, after both sweeps finish.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from huggingface_hub import hf_hub_download

from canonical.causal.vector_patching.config import EXPORT_REPO
from canonical.causal.vector_patching.export_responses import build_response_row, export_to_hf

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/exp2_out")
RECIPIENTS = ["ig", "yo"]
README_PATH = Path(__file__).resolve().parent / "README.md"


def sanity_check(response, expected_language):
    if response is None:
        return {"still_target_language": None, "non_degenerate": None}
    import langdetect
    try:
        still_target_language = langdetect.detect(response) == expected_language
    except langdetect.lang_detect_exception.LangDetectException:
        still_target_language = False
    stripped = response.strip()
    non_degenerate = len(stripped) >= 5 and len(set(stripped)) > 1
    return {"still_target_language": still_target_language, "non_degenerate": non_degenerate}


def main():
    all_rows = []
    per_recipient_stats = {}
    for recipient in RECIPIENTS:
        path = OUT_DIR / f"stage_b_{recipient}.jsonl"
        with open(path) as f:
            raw_rows = [json.loads(line) for line in f]
        n_errors = sum(1 for r in raw_rows if r.get("error"))
        per_recipient_stats[recipient] = {
            "n": len(raw_rows), "errors": n_errors,
            "donor_kinds": dict(Counter(r["donor_kind"] for r in raw_rows)),
            "layers": sorted(set(r["patch_layer"] for r in raw_rows)),
        }
        print(f"{recipient}: {len(raw_rows)} rows, {n_errors} errors", flush=True)
        for r in raw_rows:
            checks = sanity_check(r["response"], recipient)
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
    path_in_repo = f"qwen3-8b/exp2_yo_ig_{ts}.parquet"
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
        f"- **{ts}** ({len(all_rows)} total rows, "
        f"{sum(s['errors'] for s in per_recipient_stats.values())} generation errors "
        f"across both recipients, verified by read-back after upload)",
        f"  - HF: [{path_in_repo}]({hf_url})",
    ]
    for recipient, stats in per_recipient_stats.items():
        section.append(
            f"  - `{recipient}`: {stats['n']} rows ({stats['errors']} errors), "
            f"layers {stats['layers']}, donor_kind counts {stats['donor_kinds']}"
        )
    with open(README_PATH, "a") as f:
        f.write("\n".join(section) + "\n")
    print(f"appended upload log to {README_PATH}", flush=True)


if __name__ == "__main__":
    main()
