"""
Removes duplicate rows from any judge's results file -- keeps ONE row per
(id, model_id, language, sample_idx), preferring a successful (error is
None) row over a failed one, and the last occurrence on a genuine tie.

RUN:
    python dedupe_judge.py gpt_mini
    python dedupe_judge.py deepseek
    python dedupe_judge.py gemini
"""

import json
import os
import sys

if len(sys.argv) != 2 or sys.argv[1] not in ("gpt_mini", "deepseek", "gemini"):
    raise SystemExit("Usage: python dedupe_judge.py [gpt_mini|deepseek|gemini]")

JUDGE_NAME = sys.argv[1]
IN_FILE = os.path.join("results", JUDGE_NAME, "results.jsonl")
OUT_FILE = os.path.join("results", JUDGE_NAME, "results_deduped.jsonl")


def composite_key(row):
    return (row["id"], row["model_id"], row["language"], row["sample_idx"])


rows = []
with open(IN_FILE, encoding="utf-8") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Read {len(rows)} rows from {IN_FILE}")

best_by_key = {}
for row in rows:
    key = composite_key(row)
    existing = best_by_key.get(key)
    if existing is None or existing.get("error") is not None or row.get("error") is None:
        best_by_key[key] = row

final_rows = list(best_by_key.values())
print(f"After dedup: {len(final_rows)} unique rows ({len(rows) - len(final_rows)} duplicates removed)")
print(f"(140400 is the correct full-dataset total once the run is complete)")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for row in final_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nWrote {OUT_FILE}")
print(f"Once confirmed: mv {OUT_FILE} {IN_FILE}")
