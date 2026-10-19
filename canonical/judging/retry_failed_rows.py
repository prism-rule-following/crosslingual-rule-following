"""
Retries ONLY the rows that are missing or failed in a judge's results.jsonl
-- does not re-pay for rows that already succeeded. Works for any judge:
the backend is picked from `judge` in the config.

Completion is keyed on (id, model_id, language, sample_idx), NOT id alone
-- id is shared across models, languages and samples, so an id-only check
would mark all of them "done" the moment any one succeeded.

RUN (from the repo root):
    caffeinate -i python -m canonical.judging.retry_failed_rows \
        --config canonical/judging/configs/judge_gpt_mini.json

Writes ONLY the retried rows to {output_dir}/retry.jsonl (doesn't touch
results.jsonl). Run consolidate.py afterward to merge the two into
final.jsonl.
"""

from canonical.judging.common import (
    composite_key,
    load_done_keys,
    load_responses,
    make_judge,
    run_judging,
)
from canonical.judging.config import parse_config_arg

if __name__ == "__main__":
    config = parse_config_arg("Retry failed/missing rows for one judge.")
    judge = make_judge(config)
    markers = getattr(judge, "permanent_failure_markers", ())

    succeeded_keys = load_done_keys(config.results_file, markers)
    # in case this script itself gets interrupted and resumed
    already_retried = load_done_keys(config.retry_file, markers)
    done_keys = succeeded_keys | already_retried
    print(f"{len(succeeded_keys)} rows already done in the original run.")
    print(f"{len(already_retried)} rows already done in a prior retry pass.")

    dataset = load_responses(config)
    rows_to_retry = [row for row in dataset if composite_key(row) not in done_keys]
    run_judging(
        rows_to_retry,
        judge,
        config.retry_file,
        config.retry_max_workers,
        progress_every=25,
    )
    print(
        "Next: run consolidate.py to merge the original successes + these retries into one clean final file."
    )
