"""
Merges results.jsonl (original run) + retry.jsonl (retry pass) into ONE
clean final file, exactly one row per (id, model_id, language, sample_idx)
-- preferring a successful retry over an original failure, and never
duplicating a row that succeeded the first time.

RUN (no API calls, no .env needed, just file processing):
    python consolidate.py

Output: results/gpt_mini/final.jsonl -- this is the one to actually use
for metrics, and the one to upload to HF.
"""

import json
import os

OUTPUT_DIR = os.path.join("results", "gpt_mini")
ORIGINAL_FILE = os.path.join(OUTPUT_DIR, "results.jsonl")
RETRY_FILE = os.path.join(OUTPUT_DIR, "retry.jsonl")
FINAL_FILE = os.path.join(OUTPUT_DIR, "final.jsonl")


def composite_key(row):
    return (row["id"], row["model_id"], row["language"], row["sample_idx"])


def load_rows(path):
    rows = []
    if not os.path.exists(path):
        print(f"{path} not found -- skipping (fine if you haven't run a retry pass yet).")
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


original_rows = load_rows(ORIGINAL_FILE)
retry_rows = load_rows(RETRY_FILE)
print(f"Original file: {len(original_rows)} rows")
print(f"Retry file:    {len(retry_rows)} rows")

# Keep the BEST row per composite key: prefer a successful (error is None)
# row over a failed one, and prefer retry results over original ones when
# both succeeded (retry is the more recent attempt).
best_by_key = {}

for row in original_rows:
    key = composite_key(row)
    best_by_key[key] = row

for row in retry_rows:
    key = composite_key(row)
    existing = best_by_key.get(key)
    if existing is None or existing.get("error") is not None or row.get("error") is None:
        # no existing row, OR existing failed, OR this retry succeeded -- take the retry
        best_by_key[key] = row

final_rows = list(best_by_key.values())
print(f"\nConsolidated: {len(final_rows)} unique rows")

still_failed = [r for r in final_rows if r.get("error") is not None]
print(f"Still failed after consolidation: {len(still_failed)} ({len(still_failed)/len(final_rows):.2%})")
if still_failed:
    print("These need another retry pass -- re-run retry_failed_rows.py, it will pick these up automatically.")

with open(FINAL_FILE, "w", encoding="utf-8") as f:
    for row in final_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nWrote {len(final_rows)} rows to {FINAL_FILE}")
