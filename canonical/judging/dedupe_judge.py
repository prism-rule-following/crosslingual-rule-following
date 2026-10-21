"""
Removes duplicate rows from a judge's results.jsonl -- keeps ONE row per
(id, model_id, language, sample_idx), preferring a successful (error is
None) row over a failed one, and the last occurrence on a genuine tie.

RUN:
    python -m canonical.judging.dedupe_judge --config canonical/judging/configs/judge_deepseek.json
"""

import os

from canonical.judging.common import load_rows, merge_best, write_rows
from canonical.judging.config import parse_config_arg

if __name__ == "__main__":
    config = parse_config_arg("Dedupe a judge's results.jsonl.")
    in_file = config.results_file
    out_file = os.path.join(config.output_dir, "results_deduped.jsonl")

    rows = load_rows(in_file, missing_ok=False)
    print(f"Read {len(rows)} rows from {in_file}")

    final_rows = merge_best(rows)
    print(
        f"After dedup: {len(final_rows)} unique rows ({len(rows) - len(final_rows)} duplicates removed)"
    )
    if config.expected_rows is not None:
        print(f"({config.expected_rows} is the expected total once the run is complete)")

    write_rows(out_file, final_rows)
    print(f"\nWrote {out_file}")
    print(f"Once confirmed: mv {out_file} {in_file}")
