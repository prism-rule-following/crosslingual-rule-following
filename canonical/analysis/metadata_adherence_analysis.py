"""
Breaks down 3-judge consensus adherence by every metadata dimension in the
dataset: category, topic, grammar_type, pressure_level, pressure_name,
pair_type -- using the same full-coverage consensus logic as
three_judge_agreement.py (3-valid majority, 2-valid fallback when DeepSeek
has no verdict, genuinely unresolved rows excluded rather than guessed).

Also breaks down by model_id (Qwen vs Llama) alone, AND each dimension
split by model -- so you can see whether a given category/topic/pressure
pattern holds for both models or is specific to one.

RUN:
    python metadata_adherence_analysis.py results/gpt_mini/final.jsonl results/gemini/final.jsonl results/deepseek/final.jsonl

Prints one aggregate breakdown table per dimension (sorted worst-first),
then a standalone model breakdown, then each dimension re-broken-down
per model.
"""

import json
import sys
from collections import Counter, defaultdict

if len(sys.argv) != 4:
    raise SystemExit("Usage: python metadata_adherence_analysis.py <judge_A> <judge_B> <judge_C>  (paths to final.jsonl files)")

FILE_A, FILE_B, FILE_C = sys.argv[1], sys.argv[2], sys.argv[3]

DIMENSIONS = ["category", "topic", "grammar_type", "pressure_level", "pressure_name", "pair_type"]


def composite_key(row):
    return (row["id"], row["model_id"], row["language"], row["sample_idx"])


def load_by_key(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[composite_key(row)] = row
    return rows


print("Loading files...")
rows_a = load_by_key(FILE_A)
rows_b = load_by_key(FILE_B)
rows_c = load_by_key(FILE_C)
print(f"{FILE_A}: {len(rows_a)} rows")
print(f"{FILE_B}: {len(rows_b)} rows")
print(f"{FILE_C}: {len(rows_c)} rows")

common_keys = set(rows_a.keys()) & set(rows_b.keys()) & set(rows_c.keys())
print(f"Present in all three files: {len(common_keys)}\n")


def get_verdict(rows_dict, key):
    row = rows_dict.get(key)
    if row is None:
        return None, None
    v = row.get("verdict")
    return (v if v in ("HELD", "VIOLATED") else None), row


# Build full-coverage consensus for every row, same logic as three_judge_agreement.py
results = []
for key in common_keys:
    v_a, ra = get_verdict(rows_a, key)
    v_b, rb = get_verdict(rows_b, key)
    v_c, rc = get_verdict(rows_c, key)
    verdicts = [v for v in (v_a, v_b, v_c) if v is not None]
    n_valid = len(verdicts)
    any_row = ra or rb or rc

    metadata = {dim: any_row.get(dim) for dim in DIMENSIONS}
    model_id = any_row.get("model_id")

    if n_valid == 3:
        counts = Counter(verdicts)
        consensus, count = counts.most_common(1)[0]
        unanimous = (count == 3)
        results.append({"metadata": metadata, "model_id": model_id, "resolved": True,
                         "consensus": consensus, "unanimous": unanimous})
    elif n_valid == 2:
        if verdicts[0] == verdicts[1]:
            results.append({"metadata": metadata, "model_id": model_id, "resolved": True,
                             "consensus": verdicts[0], "unanimous": None})
        else:
            results.append({"metadata": metadata, "model_id": model_id, "resolved": False,
                             "consensus": None, "unanimous": None})
    else:
        results.append({"metadata": metadata, "model_id": model_id, "resolved": False,
                         "consensus": None, "unanimous": None})

n_resolved = sum(1 for r in results if r["resolved"])
print(f"Total resolved (has a consensus): {n_resolved}/{len(results)} ({n_resolved/len(results)*100:.2f}%)\n")


def print_breakdown(rows_subset, group_key_fn, title):
    print("=" * 80)
    print(title)
    print("=" * 80)
    by_val = defaultdict(lambda: {"total": 0, "resolved": 0, "held": 0})
    for r in rows_subset:
        val = group_key_fn(r)
        if val is None:
            val = "(none)"
        d = by_val[val]
        d["total"] += 1
        if r["resolved"]:
            d["resolved"] += 1
            if r["consensus"] == "HELD":
                d["held"] += 1

    rows_out = []
    for val, d in by_val.items():
        held_pct = d["held"] / d["resolved"] * 100 if d["resolved"] else 0
        rows_out.append((val, d["total"], d["resolved"], held_pct))
    rows_out.sort(key=lambda x: x[3])  # worst HELD% first

    print(f"{'value':<35} {'total':<8} {'resolved':<10} {'HELD%':<8}")
    for val, total, resolved, held_pct in rows_out:
        print(f"{str(val):<35} {total:<8} {resolved:<10} {held_pct:<8.2f}")
    print()


# ---- Aggregate breakdown by each metadata dimension (both models combined) ----
for dim in DIMENSIONS:
    print_breakdown(results, lambda r, d=dim: r["metadata"].get(d), f"BY {dim.upper()} (both models combined)")

# ---- Standalone model breakdown ----
print_breakdown(results, lambda r: r["model_id"], "BY MODEL")

# ---- Each dimension, split by model ----
models_present = sorted(set(r["model_id"] for r in results))
for dim in DIMENSIONS:
    print("#" * 80)
    print(f"# {dim.upper()} -- SPLIT BY MODEL")
    print("#" * 80)
    for model in models_present:
        subset = [r for r in results if r["model_id"] == model]
        print_breakdown(subset, lambda r, d=dim: r["metadata"].get(d), f"  {model} -- BY {dim.upper()}")
