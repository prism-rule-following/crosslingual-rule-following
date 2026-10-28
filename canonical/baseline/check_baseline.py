"""Sanity-check no-rule baseline outputs.
Usage: python check_baseline.py [dir] [n_samples] [n_prompts]   (defaults: /workspace/baseline_out 5 26)
Reads every *_norule_responses.jsonl in dir."""
import glob, json, random, statistics, sys, collections

d = sys.argv[1] if len(sys.argv) > 1 else "/workspace/baseline_out"
EXPECT_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EXPECT_PROMPTS = int(sys.argv[3]) if len(sys.argv) > 3 else 26
files = sorted(glob.glob(f"{d}/*_norule_responses.jsonl"))
if not files:
    sys.exit(f"no *_norule_responses.jsonl in {d} yet")
random.seed(0)
for f in files:
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    lang, model = rows[0]["language"], rows[0]["model_id"]
    prompts = {r["id"] for r in rows}
    per = collections.Counter(r["id"] for r in rows)
    lens = [len(r["response"]) for r in rows]
    empty = sum(1 for r in rows if not r["response"].strip())
    by = collections.defaultdict(list)
    for r in rows: by[r["id"]].append(r["response"].strip())
    dup = sum(len(v) - len(set(v)) for v in by.values())
    print(f"\n=== {model} | {lang} | {len(rows)} rows (expect {EXPECT_PROMPTS*EXPECT_SAMPLES})")
    print(f"  prompts: {len(prompts)} (expect {EXPECT_PROMPTS}) | samples per prompt: {sorted(set(per.values()))} (expect [{EXPECT_SAMPLES}])")
    print(f"  system prompts: {sorted({r['system'] for r in rows})}")
    print(f"  empty: {empty} | exact-duplicate samples: {dup} | chars min/median/mean/max: "
          f"{min(lens)}/{int(statistics.median(lens))}/{int(statistics.mean(lens))}/{max(lens)}")
    print("  prompts per category:", dict(collections.Counter(r['category'] for r in rows if r['sample_idx'] == 0)))
    bad = [k for k, v in per.items() if v != EXPECT_SAMPLES]
    if len(prompts) != EXPECT_PROMPTS or bad or empty:
        print("  !! CHECK: count mismatch (wrong n_samples? old n=5 files left over?) or empty responses")
    for r in random.sample(rows, 2):
        print(f"  -- [{r['category']}/{r['topic']}] Q: {r['user_query'][:100]}\n     A: {r['response'][:300].replace(chr(10),' ')}")
