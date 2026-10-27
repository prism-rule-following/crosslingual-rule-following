"""No-rule baseline vs rule-run analysis (judge agreement, per-judge, per-model, per-language).

Inputs: per-judge verdict jsonl for the baseline (3 files) and for the rule run filtered to L0 (3 files).
Rows need: id, model_id, language, sample_idx, category, topic, pressure_level, verdict.
Unit of analysis = base-query cell (language, category, topic): 26 per language. Samples/wordings are
repeats inside a cell, not new scenarios.

  python analyze.py --baseline gemini=B/gemini.jsonl gpt_mini=B/gpt.jsonl deepseek=B/ds.jsonl \
                    --rule     gemini=R/gemini.jsonl gpt_mini=R/gpt.jsonl deepseek=R/ds.jsonl --out OUT_DIR
"""
import argparse, itertools, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", nargs=3, required=True, help="name=path x3")
ap.add_argument("--rule", nargs=3, required=True, help="name=path x3")
ap.add_argument("--out", required=True)
ap.add_argument("--pressure", default="L0")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
KEY = ["id", "model_id", "language", "sample_idx"]
CELL = ["model_id", "language", "category", "topic"]
JUDGES = [x.split("=")[0] for x in a.baseline]


def load(specs, cond):
    longs = []
    for sp in specs:
        name, path = sp.split("=", 1)
        d = pd.DataFrame(map(json.loads, open(path, encoding="utf-8")))
        d["sample_idx"] = d["sample_idx"].astype(int)
        d["id"] = d["id"].str.replace("norule::", "", regex=False)
        d = d[d.pressure_level == a.pressure].copy()
        d["v"] = d.verdict.map({"HELD": 1.0, "VIOLATED": 0.0})  # errors -> NaN
        d["judge"] = name
        longs.append(d)
    L = pd.concat(longs, ignore_index=True)
    meta = L.drop_duplicates(KEY)[KEY + ["category", "topic", "pressure_level"]]
    wide = L.pivot_table(index=KEY, columns="judge", values="v", aggfunc="first", dropna=False).reset_index()
    for j in JUDGES:
        if j not in wide: wide[j] = np.nan
    wide = wide.merge(meta, on=KEY, how="left")
    wide["cond"] = cond
    return wide


B = load(a.baseline, "baseline")
R = load(a.rule, "rule")
# rule rows: keep only languages present in baseline AND in rule (en, yo) for paired work
D = pd.concat([B, R], ignore_index=True)
D["n_valid"] = D[JUDGES].notna().sum(axis=1)
vals = D[JUDGES]
D["maj"] = np.where(D.n_valid == 3, (vals.sum(axis=1) >= 2).astype(float),
            np.where((D.n_valid == 2) & (vals.sum(axis=1) % 2 == 0), vals.mean(axis=1), np.nan))
D["unanimous"] = (D.n_valid == 3) & (vals.nunique(axis=1) == 1)
D.to_pickle(os.path.join(a.out, "_wide.pkl")) if False else None
out = {}


def kappa(x, y):
    m = x.notna() & y.notna(); x, y = x[m].values, y[m].values
    po = (x == y).mean()
    p1x, p1y = x.mean(), y.mean()
    pe = p1x * p1y + (1 - p1x) * (1 - p1y)
    return po, (np.nan if pe == 1 else (po - pe) / (1 - pe)), len(x)


def fleiss(df):
    m = df[df.n_valid == 3]
    if not len(m): return np.nan
    h = m[JUDGES].sum(axis=1).values; n = 3
    Pi = ((h * (h - 1)) + ((n - h) * (n - h - 1))) / (n * (n - 1))
    p1 = h.sum() / (n * len(h)); pe = p1 ** 2 + (1 - p1) ** 2
    return np.nan if pe == 1 else (Pi.mean() - pe) / (1 - pe)


# ---- 1. judge agreement -------------------------------------------------------------------
rows = []
for (cond, model, lang), g in D.groupby(["cond", "model_id", "language"]):
    r = dict(cond=cond, model=model, lang=lang, n=len(g), unanimous=g.unanimous.mean(),
             resolved=g.maj.notna().mean(), fleiss_kappa=fleiss(g))
    for j1, j2 in itertools.combinations(JUDGES, 2):
        po, k, _ = kappa(g[j1], g[j2]); r[f"agree_{j1}_{j2}"] = po; r[f"kappa_{j1}_{j2}"] = k
    rows.append(r)
agree = pd.DataFrame(rows); agree.to_csv(f"{a.out}/judge_agreement.csv", index=False)

# disagreement by category
dis = (D[D.n_valid >= 2].assign(split=lambda x: ~x.unanimous & (x.n_valid == 3))
       .groupby(["cond", "language", "category"]).split.mean().unstack("cond").round(3))
dis.to_csv(f"{a.out}/judge_disagreement_by_category.csv")

# ---- 2. per judge + consensus adherence and rule effect ------------------------------------
SCORERS = JUDGES + ["maj"]


def cell_rates(df, s):
    return df.groupby(CELL)[s].mean()


def boot_ci(d):
    d = d.dropna().values
    if len(d) == 0: return np.nan, np.nan, np.nan
    bs = rng.choice(d, (a.boot, len(d))).mean(axis=1)
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


from scipy import stats
rows = []
for model, lang in itertools.product(sorted(D.model_id.unique()), ["en", "yo"]):
    for s in SCORERS:
        b = cell_rates(D[(D.cond == "baseline") & (D.model_id == model) & (D.language == lang)], s)
        r = cell_rates(D[(D.cond == "rule") & (D.model_id == model) & (D.language == lang)], s)
        c = pd.concat([b.rename("b"), r.rename("r")], axis=1).dropna()
        diff = c.r - c.b
        m, lo, hi = boot_ci(diff)
        nz = diff[diff != 0]
        sp = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan
        try: wp = stats.wilcoxon(nz).pvalue if len(nz) else np.nan
        except Exception: wp = np.nan
        rows.append(dict(model=model, lang=lang, scorer=s, cells=len(c), baseline=c.b.mean(), rule=c.r.mean(),
                         diff=m, ci_lo=lo, ci_hi=hi, up=int((diff > 0).sum()), down=int((diff < 0).sum()),
                         sign_p=sp, wilcoxon_p=wp))
eff = pd.DataFrame(rows); eff.to_csv(f"{a.out}/rule_effect_by_judge.csv", index=False)

# judge leniency (raw rates)
len_ = (D.melt(id_vars=["cond", "model_id", "language"], value_vars=JUDGES, var_name="judge", value_name="v")
        .dropna().groupby(["cond", "model_id", "language", "judge"]).v.mean().unstack("judge").round(3))
len_.to_csv(f"{a.out}/judge_leniency.csv")

# ---- 3. category tables (consensus) -----------------------------------------------------------
cat = (D.groupby(["model_id", "language", "category", "cond"]).maj.mean().unstack("cond")
       .assign(rule_minus_baseline=lambda x: x.rule - x.baseline).round(3))
cat.to_csv(f"{a.out}/category_effect_consensus.csv")

# per-judge category effect (shows if any single judge drives a category)
pj = []
for s in JUDGES:
    t = D.groupby(["language", "category", "cond"])[s].mean().unstack("cond")
    t["judge"] = s; t["diff"] = t.rule - t.baseline; pj.append(t.reset_index())
pd.concat(pj).round(3).to_csv(f"{a.out}/category_effect_by_judge.csv", index=False)

# ---- 4. swahili baseline (descriptive only) ------------------------------------------------
sw = D[(D.cond == "baseline") & (D.language == "sw")].groupby(["model_id", "category"]).maj.mean().unstack("model_id").round(3)
sw.to_csv(f"{a.out}/baseline_swahili_descriptive.csv")
print("ok", a.out)
for f in ["judge_agreement", "rule_effect_by_judge"]:
    print("\n", f); print(pd.read_csv(f"{a.out}/{f}.csv").round(3).to_string())
