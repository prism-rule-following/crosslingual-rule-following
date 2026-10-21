"""
Merges ALL Gemini sources (OpenRouter single-process, OpenRouter split-key
instances, and the direct-Google-Cloud Vertex run) into one clean final
file -- exactly one row per (id, model_id, language, sample_idx), same
composite-key logic used everywhere else in this project.

RUN (no API calls, pure file processing):
    python merge_gemini_sources.py

Output: results/gemini/final.jsonl -- this is the one to use for metrics
and the eventual HF upload, same role as results/gpt_mini/final.jsonl.
"""

import json
import os

SOURCES = [
    os.path.join("results", "gemini", "results.jsonl"),
    os.path.join("results", "gemini", "results_instance1.jsonl"),
    os.path.join("results", "gemini", "results_instance2.jsonl"),
    os.path.join("results", "gemini_vertex", "results.jsonl"),
]
FINAL_FILE = os.path.join("results", "gemini", "final.jsonl")


def composite_key(row):
    return (row["id"], row["model_id"], row["language"], row["sample_idx"])


def load_rows(path):
    rows = []
    if not os.path.exists(path):
        print(f"{path} not found -- skipping (fine if you never used that source).")
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


best_by_key = {}
total_read = 0
for source in SOURCES:
    rows = load_rows(source)
    total_read += len(rows)
    print(f"{source}: {len(rows)} rows")
    for row in rows:
        key = composite_key(row)
        existing = best_by_key.get(key)
        # prefer a successful row over a failed one; if both succeeded or
        # both failed, the later source in the list wins (arbitrary but
        # deterministic tie-break)
        if existing is None or existing.get("error") is not None or row.get("error") is None:
            best_by_key[key] = row

final_rows = list(best_by_key.values())
print(f"\nTotal rows read across all sources: {total_read}")
print(f"Consolidated: {len(final_rows)} unique rows")
print(f"Expected once fully complete: 140400")

still_failed = [r for r in final_rows if r.get("error") is not None]
print(f"Still failed after consolidation: {len(still_failed)} ({len(still_failed)/len(final_rows):.2%})" if final_rows else "")

with open(FINAL_FILE, "w", encoding="utf-8") as f:
    for row in final_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nWrote {len(final_rows)} rows to {FINAL_FILE}")
