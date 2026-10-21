"""
Compares two judges' final.jsonl files on the same rows: overall/language/
model HELD-VIOLATED breakdown restricted to rows where BOTH judges agree,
plus the agreement rate itself (overall, by language, by model) -- and a
sample of disagreement rows for manual reading.

RUN:
    python two_judge_agreement.py results/gpt_mini/final.jsonl results/gemini/final.jsonl

Works for any two judges' final.jsonl files, not just these two specifically.
"""

import json
import sys
from collections import Counter, defaultdict

if len(sys.argv) != 3:
    raise SystemExit("Usage: python two_judge_agreement.py <judge_A_final.jsonl> <judge_B_final.jsonl>")

FILE_A, FILE_B = sys.argv[1], sys.argv[2]


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
print(f"{FILE_A}: {len(rows_a)} rows")
print(f"{FILE_B}: {len(rows_b)} rows")

common_keys = set(rows_a.keys()) & set(rows_b.keys())
only_a = set(rows_a.keys()) - set(rows_b.keys())
only_b = set(rows_b.keys()) - set(rows_a.keys())
print(f"Common rows (present in both, comparable): {len(common_keys)}")
if only_a:
    print(f"  WARNING: {len(only_a)} rows only in {FILE_A}")
if only_b:
    print(f"  WARNING: {len(only_b)} rows only in {FILE_B}")
print()

# Only compare rows where BOTH judges produced a real verdict
comparable = []
for key in common_keys:
    ra, rb = rows_a[key], rows_b[key]
    if ra["verdict"] in ("HELD", "VIOLATED") and rb["verdict"] in ("HELD", "VIOLATED"):
        comparable.append((key, ra, rb))

print(f"Rows with a real verdict from both judges: {len(comparable)}\n")

agree = [(k, ra, rb) for k, ra, rb in comparable if ra["verdict"] == rb["verdict"]]
disagree = [(k, ra, rb) for k, ra, rb in comparable if ra["verdict"] != rb["verdict"]]

print("="*70)
print("OVERALL AGREEMENT")
print("="*70)
print(f"Agree:    {len(agree)}  ({len(agree)/len(comparable)*100:.2f}%)")
print(f"Disagree: {len(disagree)}  ({len(disagree)/len(comparable)*100:.2f}%)\n")

held_agree = sum(1 for _, ra, _ in agree if ra["verdict"] == "HELD")
violated_agree = sum(1 for _, ra, _ in agree if ra["verdict"] == "VIOLATED")
print("Among AGREED rows -- HELD/VIOLATED breakdown (both judges concur):")
print(f"  HELD:     {held_agree}  ({held_agree/len(agree)*100:.2f}%)")
print(f"  VIOLATED: {violated_agree}  ({violated_agree/len(agree)*100:.2f}%)\n")

# ---- Agreement rate by language ----
print("="*70)
print("AGREEMENT RATE BY LANGUAGE")
print("="*70)
by_lang_total = defaultdict(int)
by_lang_agree = defaultdict(int)
by_lang_held_among_agreed = defaultdict(int)
for key, ra, rb in comparable:
    lang = ra["language"]
    by_lang_total[lang] += 1
    if ra["verdict"] == rb["verdict"]:
        by_lang_agree[lang] += 1
        if ra["verdict"] == "HELD":
            by_lang_held_among_agreed[lang] += 1

for lang in sorted(by_lang_total.keys(), key=lambda l: -(by_lang_agree[l]/by_lang_total[l])):
    total = by_lang_total[lang]
    agree_n = by_lang_agree[lang]
    held_n = by_lang_held_among_agreed[lang]
    print(f"  {lang:<6} agreement={agree_n:>6}/{total:<6} ({agree_n/total*100:5.2f}%)   "
          f"HELD% among agreed={held_n/agree_n*100:5.2f}%" if agree_n else f"  {lang:<6} no agreed rows")

# ---- Agreement rate by model ----
print(f"\n{'='*70}")
print("AGREEMENT RATE BY MODEL")
print("="*70)
by_model_total = defaultdict(int)
by_model_agree = defaultdict(int)
for key, ra, rb in comparable:
    model = ra["model_id"]
    by_model_total[model] += 1
    if ra["verdict"] == rb["verdict"]:
        by_model_agree[model] += 1

for model in by_model_total:
    total = by_model_total[model]
    agree_n = by_model_agree[model]
    print(f"  {model:<40} agreement={agree_n:>6}/{total:<6} ({agree_n/total*100:.2f}%)")

# ---- Agreement rate by model AND language jointly ----
print(f"\n{'='*70}")
print("AGREEMENT RATE BY MODEL x LANGUAGE (the real joint breakdown)")
print("="*70)
by_ml_total = defaultdict(int)
by_ml_agree = defaultdict(int)
by_ml_held_agreed = defaultdict(int)
for key, ra, rb in comparable:
    mk = (ra["model_id"], ra["language"])
    by_ml_total[mk] += 1
    if ra["verdict"] == rb["verdict"]:
        by_ml_agree[mk] += 1
        if ra["verdict"] == "HELD":
            by_ml_held_agreed[mk] += 1

for model in sorted(set(m for m, l in by_ml_total.keys())):
    print(f"\n  {model}")
    langs_for_model = sorted([l for m, l in by_ml_total.keys() if m == model],
                               key=lambda l: -(by_ml_agree[(model, l)] / by_ml_total[(model, l)]))
    for lang in langs_for_model:
        mk = (model, lang)
        total = by_ml_total[mk]
        agree_n = by_ml_agree[mk]
        held_n = by_ml_held_agreed[mk]
        print(f"    {lang:<6} agreement={agree_n:>5}/{total:<5} ({agree_n/total*100:5.2f}%)   "
              f"HELD% among agreed={held_n/agree_n*100:5.2f}%" if agree_n else f"    {lang:<6} no agreed rows")

# ---- Sample disagreement rows for manual reading ----
print(f"\n{'='*70}")
print(f"SAMPLE DISAGREEMENT ROWS (first 5 of {len(disagree)})")
print("="*70)
for key, ra, rb in disagree[:5]:
    print(f"\nid={ra['id']}  language={ra['language']}  category={ra['category']}  model={ra['model_id']}")
    print(f"  Judge A ({ra['judge_model']}): {ra['verdict']}")
    print(f"  Judge B ({rb['judge_model']}): {rb['verdict']}")
    print(f"  response: {ra['response'][:200]}")
