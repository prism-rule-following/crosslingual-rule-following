"""
Three-judge consensus analysis, with a full-coverage fallback: every one of the
140,400 rows gets a consensus verdict where possible --
  - 3 valid verdicts  -> majority vote (always resolves: HELD/VIOLATED never ties
    among 3 judges) + whether it was unanimous (3-0) or split (2-1)
  - exactly 2 valid verdicts (e.g. DeepSeek jailbreak-blocked, GPT+Gemini both
    have real verdicts) -> falls back to 2-judge agreement; if they agree, that's
    the consensus; if they disagree, the row is genuinely UNRESOLVED (not forced
    into a verdict) and counted separately
  - fewer than 2 valid verdicts -> cannot form any consensus, counted separately
    (should be ~0 given GPT and Gemini are both 100% complete)

RUN:
    python three_judge_agreement.py results/gpt_mini/final.jsonl results/gemini/final.jsonl results/deepseek/final.jsonl

Works for any three judges' final.jsonl files, in any order, and correctly
handles ANY of them having missing verdicts, not just the third argument.
"""

import json
import sys
from collections import Counter, defaultdict

if len(sys.argv) != 4:
    raise SystemExit("Usage: python three_judge_agreement.py <judge_A> <judge_B> <judge_C>  (paths to final.jsonl files)")

FILE_A, FILE_B, FILE_C = sys.argv[1], sys.argv[2], sys.argv[3]
JUDGE_NAMES = [FILE_A, FILE_B, FILE_C]


def composite_key(row):
    return (row["id"], row["model_id"], row["language"], row["sample_idx"])


def load_by_key(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[composite_key(row)] = row
    return rows


rows_a = load_by_key(FILE_A)
rows_b = load_by_key(FILE_B)
rows_c = load_by_key(FILE_C)
print(f"{FILE_A}: {len(rows_a)} rows")
print(f"{FILE_B}: {len(rows_b)} rows")
print(f"{FILE_C}: {len(rows_c)} rows")

all_keys = set(rows_a.keys()) | set(rows_b.keys()) | set(rows_c.keys())
common_keys = set(rows_a.keys()) & set(rows_b.keys()) & set(rows_c.keys())
print(f"Union of all keys across the three files: {len(all_keys)}")
print(f"Present in all three files: {len(common_keys)}\n")


def get_verdict(rows_dict, key):
    row = rows_dict.get(key)
    if row is None:
        return None, None
    v = row.get("verdict")
    return (v if v in ("HELD", "VIOLATED") else None), row


# For every row present in all three files (should be ~140,400), classify by
# how many judges actually produced a usable verdict, then resolve a
# consensus verdict with full coverage.
results = []  # each: dict with key, language, model_id, n_valid, consensus, resolved(bool), unanimous(bool or None)

for key in common_keys:
    v_a, ra = get_verdict(rows_a, key)
    v_b, rb = get_verdict(rows_b, key)
    v_c, rc = get_verdict(rows_c, key)
    verdicts = [v for v in (v_a, v_b, v_c) if v is not None]
    n_valid = len(verdicts)
    any_row = ra or rb or rc
    language = any_row["language"]
    model_id = any_row["model_id"]

    if n_valid == 3:
        counts = Counter(verdicts)
        consensus, count = counts.most_common(1)[0]
        unanimous = (count == 3)
        results.append({"key": key, "language": language, "model_id": model_id,
                         "n_valid": 3, "resolved": True, "consensus": consensus,
                         "unanimous": unanimous})
    elif n_valid == 2:
        if verdicts[0] == verdicts[1]:
            results.append({"key": key, "language": language, "model_id": model_id,
                             "n_valid": 2, "resolved": True, "consensus": verdicts[0],
                             "unanimous": None})
        else:
            results.append({"key": key, "language": language, "model_id": model_id,
                             "n_valid": 2, "resolved": False, "consensus": None,
                             "unanimous": None})
    else:
        results.append({"key": key, "language": language, "model_id": model_id,
                         "n_valid": n_valid, "resolved": False, "consensus": None,
                         "unanimous": None})

n_three_valid = sum(1 for r in results if r["n_valid"] == 3)
n_two_valid = sum(1 for r in results if r["n_valid"] == 2)
n_lt_two = sum(1 for r in results if r["n_valid"] < 2)
n_resolved = sum(1 for r in results if r["resolved"])
n_unresolved = sum(1 for r in results if not r["resolved"])

print("="*70)
print("COVERAGE")
print("="*70)
print(f"Rows with all 3 judges valid:        {n_three_valid}")
print(f"Rows with exactly 2 judges valid:     {n_two_valid}  (fallback to 2-judge agreement)")
print(f"Rows with fewer than 2 judges valid:  {n_lt_two}  (should be ~0)")
print(f"Total resolved (has a consensus):     {n_resolved}  ({n_resolved/len(results)*100:.2f}% of all rows)")
print(f"Total UNRESOLVED (genuine split, no consensus possible): {n_unresolved}  ({n_unresolved/len(results)*100:.2f}%)\n")

# ---- Unanimity, among the 3-valid subset only ----
three_valid_results = [r for r in results if r["n_valid"] == 3]
n_unanimous = sum(1 for r in three_valid_results if r["unanimous"])
print("="*70)
print("UNANIMOUS AGREEMENT (among rows where all 3 judges had a verdict)")
print("="*70)
print(f"Unanimous (3-0):     {n_unanimous}  ({n_unanimous/len(three_valid_results)*100:.2f}%)")
print(f"Split (2-1):         {len(three_valid_results)-n_unanimous}  ({(len(three_valid_results)-n_unanimous)/len(three_valid_results)*100:.2f}%)\n")

# ---- Full-coverage consensus HELD/VIOLATED breakdown ----
resolved_results = [r for r in results if r["resolved"]]
held_n = sum(1 for r in resolved_results if r["consensus"] == "HELD")
violated_n = len(resolved_results) - held_n
print("="*70)
print("FULL-COVERAGE CONSENSUS (3-way majority + 2-way fallback combined)")
print("="*70)
print(f"HELD:     {held_n}  ({held_n/len(resolved_results)*100:.2f}% of resolved rows)")
print(f"VIOLATED: {violated_n}  ({violated_n/len(resolved_results)*100:.2f}% of resolved rows)")
print(f"(out of {len(results)} total rows: {len(resolved_results)} resolved, {n_unresolved} unresolved)\n")

# ---- By language ----
print("="*70)
print("BY LANGUAGE")
print("="*70)
by_lang = defaultdict(lambda: {"total": 0, "resolved": 0, "held": 0, "three_valid": 0, "unanimous": 0})
for r in results:
    d = by_lang[r["language"]]
    d["total"] += 1
    if r["resolved"]:
        d["resolved"] += 1
        if r["consensus"] == "HELD":
            d["held"] += 1
    if r["n_valid"] == 3:
        d["three_valid"] += 1
        if r["unanimous"]:
            d["unanimous"] += 1

print(f"{'lang':<6} {'resolved%':<11} {'HELD% (of resolved)':<20} {'unanimous% (of 3-valid)':<24}")
for lang in sorted(by_lang.keys(), key=lambda l: -(by_lang[l]['held']/by_lang[l]['resolved'] if by_lang[l]['resolved'] else 0)):
    d = by_lang[lang]
    resolved_pct = d["resolved"] / d["total"] * 100 if d["total"] else 0
    held_pct = d["held"] / d["resolved"] * 100 if d["resolved"] else 0
    unan_pct = d["unanimous"] / d["three_valid"] * 100 if d["three_valid"] else 0
    print(f"{lang:<6} {resolved_pct:<11.2f} {held_pct:<20.2f} {unan_pct:<24.2f}")

# ---- By model ----
print(f"\n{'='*70}")
print("BY MODEL")
print("="*70)
by_model = defaultdict(lambda: {"total": 0, "resolved": 0, "held": 0, "three_valid": 0, "unanimous": 0})
for r in results:
    d = by_model[r["model_id"]]
    d["total"] += 1
    if r["resolved"]:
        d["resolved"] += 1
        if r["consensus"] == "HELD":
            d["held"] += 1
    if r["n_valid"] == 3:
        d["three_valid"] += 1
        if r["unanimous"]:
            d["unanimous"] += 1

for model, d in by_model.items():
    resolved_pct = d["resolved"] / d["total"] * 100 if d["total"] else 0
    held_pct = d["held"] / d["resolved"] * 100 if d["resolved"] else 0
    unan_pct = d["unanimous"] / d["three_valid"] * 100 if d["three_valid"] else 0
    print(f"  {model:<40} resolved={resolved_pct:.2f}%  HELD%={held_pct:.2f}%  unanimous%={unan_pct:.2f}%")
