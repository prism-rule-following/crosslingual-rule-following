"""
Reproduces the same "Analysis of Results" breakdown structure used for the
GPT report (overall HELD/VIOLATED, language-wise, model-wise, model x
language) -- for any judge's final.jsonl.

RUN:
    python analysis_report.py results/gemini/final.jsonl
    python analysis_report.py results/gpt_mini/final.jsonl
    python analysis_report.py results/deepseek/final.jsonl   (once merged)
"""

import json
import sys
from collections import Counter, defaultdict

if len(sys.argv) != 2:
    raise SystemExit("Usage: python analysis_report.py path/to/final.jsonl")

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
print(f"Source: {sys.argv[1]}. {len(rows)} records.\n")

# ---- 1. Overall ----
held = sum(1 for r in rows if r["verdict"] == "HELD")
violated = sum(1 for r in rows if r["verdict"] == "VIOLATED")
total = held + violated
print("1. Overall Results")
print(f"  HELD:     {held:>7}  ({held/total*100:.2f}%)")
print(f"  VIOLATED: {violated:>7}  ({violated/total*100:.2f}%)")
print(f"  Total:    {total:>7}\n")

# ---- 2. Language-wise ----
by_lang = defaultdict(lambda: {"HELD": 0, "VIOLATED": 0})
for r in rows:
    if r["verdict"] in ("HELD", "VIOLATED"):
        by_lang[r["language"]][r["verdict"]] += 1

print("2. Overall Language-wise Results")
lang_sorted = sorted(by_lang.items(), key=lambda x: -(x[1]["HELD"] / (x[1]["HELD"] + x[1]["VIOLATED"])))
for lang, counts in lang_sorted:
    t = counts["HELD"] + counts["VIOLATED"]
    pct = counts["HELD"] / t * 100 if t else 0
    print(f"  {lang:<6} HELD={counts['HELD']:>6}  VIOLATED={counts['VIOLATED']:>6}  Total={t:>6}  HELD%={pct:.2f}%")
print()

# ---- 3. Model-wise ----
by_model = defaultdict(lambda: {"HELD": 0, "VIOLATED": 0})
for r in rows:
    if r["verdict"] in ("HELD", "VIOLATED"):
        by_model[r["model_id"]][r["verdict"]] += 1

print("3. Model-wise Results")
for model, counts in by_model.items():
    t = counts["HELD"] + counts["VIOLATED"]
    pct = counts["HELD"] / t * 100 if t else 0
    print(f"  {model:<40} HELD={counts['HELD']:>6}  VIOLATED={counts['VIOLATED']:>6}  Total={t:>6}  HELD%={pct:.2f}%")
print()

# ---- 4. Model x Language ----
print("6. Detailed Model x Language Results")
by_model_lang = defaultdict(lambda: defaultdict(lambda: {"HELD": 0, "VIOLATED": 0}))
for r in rows:
    if r["verdict"] in ("HELD", "VIOLATED"):
        by_model_lang[r["model_id"]][r["language"]][r["verdict"]] += 1

for model, lang_counts in by_model_lang.items():
    print(f"\n  {model}")
    for lang, counts in sorted(lang_counts.items(), key=lambda x: -(x[1]["HELD"]/(x[1]["HELD"]+x[1]["VIOLATED"]))):
        t = counts["HELD"] + counts["VIOLATED"]
        pct = counts["HELD"] / t * 100 if t else 0
        print(f"    {lang:<6} HELD={counts['HELD']:>6}  VIOLATED={counts['VIOLATED']:>6}  Total={t:>6}  HELD%={pct:.2f}%")

# ---- Extra: judge_model breakdown (worth checking on every file, not just Gemini's) ----
print(f"\n\njudge_model values present: {dict(Counter(r['judge_model'] for r in rows))}")
