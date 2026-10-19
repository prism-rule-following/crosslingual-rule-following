"""
Merges results.jsonl (original run) + retry.jsonl (retry pass) into ONE
clean final file, exactly one row per (id, model_id, language, sample_idx)
-- preferring a successful retry over an original failure, and never
duplicating a row that succeeded the first time.

RUN (no API calls, just file processing):
    python -m canonical.judging.consolidate --config canonical/judging/configs/judge_gpt_mini.json

Output: {output_dir}/final.jsonl -- the one to use for metrics and to
upload to HF.
"""

from canonical.judging.common import load_rows, merge_best, write_rows
from canonical.judging.config import parse_config_arg

if __name__ == "__main__":
    config = parse_config_arg("Merge a judge's original run + retry pass.")
    original_rows = load_rows(config.results_file)
    retry_rows = load_rows(config.retry_file)
    print(f"Original file: {len(original_rows)} rows")
    print(f"Retry file:    {len(retry_rows)} rows")

    # retry rows come last, so they win ties (they're the more recent attempt)
    final_rows = merge_best(original_rows + retry_rows)
    print(f"\nConsolidated: {len(final_rows)} unique rows")

    still_failed = [r for r in final_rows if r.get("error") is not None]
    if final_rows:
        print(
            f"Still failed after consolidation: {len(still_failed)} ({len(still_failed)/len(final_rows):.2%})"
        )
    if still_failed:
        print(
            "These need another retry pass -- re-run retry_failed_rows.py, it will pick these up automatically."
        )

    n = write_rows(config.final_file, final_rows)
    print(f"\nWrote {n} rows to {config.final_file}")
