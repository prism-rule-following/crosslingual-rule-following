"""
Merges several results files for one judge (e.g. OpenRouter Gemini, its
split-key instance files, and the direct-Vertex run) into one clean final
file -- exactly one row per (id, model_id, language, sample_idx), same
composite-key logic used everywhere else.

Sources are `merge_sources` in the config (relative to results_dir); a
later source wins ties. Output is the config's own final.jsonl.

RUN (no API calls, pure file processing):
    python -m canonical.judging.merge_gemini_sources --config canonical/judging/configs/judge_gemini.json
"""

import os

from canonical.judging.common import load_rows, merge_best, write_rows
from canonical.judging.config import parse_config_arg

if __name__ == "__main__":
    config = parse_config_arg("Merge multiple results files into one final.jsonl.")
    if not config.merge_sources:
        raise SystemExit("merge_sources is empty in this config -- nothing to merge.")

    all_rows = []
    for source in config.merge_sources:
        path = os.path.join(config.results_dir, source)
        rows = load_rows(path)
        print(f"{path}: {len(rows)} rows")
        all_rows.extend(rows)

    final_rows = merge_best(all_rows)
    print(f"\nTotal rows read across all sources: {len(all_rows)}")
    print(f"Consolidated: {len(final_rows)} unique rows")
    if config.expected_rows is not None:
        print(f"Expected once fully complete: {config.expected_rows}")

    still_failed = [r for r in final_rows if r.get("error") is not None]
    if final_rows:
        print(
            f"Still failed after consolidation: {len(still_failed)} ({len(still_failed)/len(final_rows):.2%})"
        )

    os.makedirs(config.output_dir, exist_ok=True)
    n = write_rows(config.final_file, final_rows)
    print(f"\nWrote {n} rows to {config.final_file}")
