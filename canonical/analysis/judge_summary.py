"""Aggregate the LLM-judge verdicts into per (model, language, pressure) HELD rates.

Reads the raw judge JSONL once and writes a small JSON that the geometry and
plotting stages join against. Verdicts other than HELD/VIOLATED (e.g. AMBIGUOUS)
are counted separately and excluded from the rate denominator.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

REPO = "crosslingual-rule-following/judge-results-active-only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="gpt_mini/results.jsonl")
    ap.add_argument("--out", default="canonical/results/rule_geometry/judge_summary.json")
    ap.add_argument("--work", default="/tmp/rule_geometry_work/judge")
    args = ap.parse_args()

    load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, args.file, repo_type="dataset",
                           local_dir=args.work, token=os.getenv("HF_TOKEN"))

    counts = defaultdict(lambda: defaultdict(int))
    verdicts = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            v = row.get("verdict")
            verdicts[v] += 1
            k = (row["model_id"], row["language"], row["pressure_level"],
                 row["category"], row["grammar_type"])
            counts[k][v] += 1

    print("verdict totals:", dict(verdicts))

    def rate(sel):
        held = sum(c.get("HELD", 0) for k, c in counts.items() if sel(k))
        viol = sum(c.get("VIOLATED", 0) for k, c in counts.items() if sel(k))
        n = held + viol
        return (100.0 * held / n, n) if n else (None, 0)

    models = sorted({k[0] for k in counts})
    langs = sorted({k[1] for k in counts})
    levels = sorted({k[2] for k in counts})
    grammars = sorted({k[4] for k in counts})
    cats = sorted({k[3] for k in counts})

    out = {}
    for m in models:
        out[m] = {}
        for l in langs:
            overall, n_all = rate(lambda k: k[0] == m and k[1] == l)
            l0, n_l0 = rate(lambda k: k[0] == m and k[1] == l and k[2] == "L0")
            out[m][l] = {
                "held_overall": overall, "n_overall": n_all,
                "held_L0": l0, "n_L0": n_l0,
                "by_pressure": {p: rate(lambda k: k[0] == m and k[1] == l and k[2] == p)[0]
                                for p in levels},
                "by_grammar_L0": {g: rate(lambda k: k[0] == m and k[1] == l
                                          and k[2] == "L0" and k[4] == g)[0]
                                  for g in grammars},
                "by_category": {c: rate(lambda k: k[0] == m and k[1] == l and k[3] == c)[0]
                                for c in cats},
            }
            print(f"{m.split('/')[-1]:24} {l}  overall {overall:.1f}%  L0 {l0:.1f}%  (n={n_all})")

    out["_meta"] = {"verdict_totals": dict(verdicts), "levels": levels,
                    "grammars": grammars, "categories": cats}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
