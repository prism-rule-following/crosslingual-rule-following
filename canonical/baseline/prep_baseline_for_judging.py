"""Prepare no-rule baseline responses for the repo's judging pipeline (canonical/judging).

Problem it fixes: the judge prompt prints  RULE_TEXT = row['system'].  In a baseline row
`system` is only the persona ("You are an assistant."), so the judges would be asked whether
the response follows a rule that is not there. This script restores the ORIGINAL rule system
prompt (the one the baseline row was derived from) as `system`. The model never saw the rule;
the judges only need it to know what to check. The response, user_query and the ACTIVE
checker/rubric are untouched, and the persona the model really saw is kept as `baseline_system`.

Join is exact:  baseline id  norule::<src_id>_clean  ->  source row id <src_id>, field system_rule.

Output layout = what the judging code reads ({model}/{lang}.parquet under a directory):
  <out>/meta-llama__Llama-3.1-8B-Instruct/en.parquet, <out>/Qwen__Qwen3-8B/yo.parquet, ...

  python prep_baseline_for_judging.py --in /workspace/baseline_out --out /workspace/baseline_judge_in
  (offline test: add --source-json canonical/data/v2/judgment_rules_expanded.verified.json, English only)
"""
import argparse, glob, json, os, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="/workspace/baseline_out")
ap.add_argument("--out", default="/workspace/baseline_judge_in")
ap.add_argument("--source-repo", default="crosslingual-rule-following/canonical-dataset")
ap.add_argument("--source-json", default=None, help="local {pairs:[...]} json instead of HF (English only)")
a = ap.parse_args()

def source_rules(lang):
    if a.source_json:
        pairs = json.load(open(a.source_json, encoding="utf-8"))["pairs"]
        return {p["id"]: p["system_rule"] for p in pairs if p.get("language", "en") == lang}
    from datasets import load_dataset
    ds = load_dataset(a.source_repo, data_files=f"data/{lang}/test.jsonl")
    ds = ds["train"] if "train" in ds else ds[next(iter(ds))]
    return {r["id"]: r["system_rule"] for r in ds}

files = sorted(glob.glob(f"{a.inp}/*_norule_responses.jsonl"))
if not files: sys.exit(f"no *_norule_responses.jsonl in {a.inp}")
cache, total, missing = {}, 0, 0
for f in files:
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    lang, model = rows[0]["language"], rows[0]["model_id"]
    if lang not in cache: cache[lang] = source_rules(lang)
    out_rows = []
    for r in rows:
        src_id = r["id"].removeprefix("norule::").removesuffix("_clean")
        rule_sys = cache[lang].get(src_id)
        if rule_sys is None: missing += 1; continue
        r["baseline_system"] = r["system"]   # what the model actually saw
        r["system"] = rule_sys               # what the judges treat as the rule
        out_rows.append(r)
    d = os.path.join(a.out, model.replace("/", "__")); os.makedirs(d, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(os.path.join(d, f"{lang}.parquet"), index=False)
    total += len(out_rows)
    print(f"{model} | {lang}: {len(out_rows)} rows -> {d}/{lang}.parquet")
print(f"\nwrote {total} rows; unmatched ids: {missing}")
if missing: sys.exit("!! some baseline ids did not match a source row; do not judge until fixed")
