"""Capability baseline analysis: can the models PERFORM the act (referral / reasoned refusal) when asked directly?

Compares, per model x language x category (cells = the 4 base queries per category, 8 per language):
   baseline  : plain question, no rule, no request           (no-rule baseline, data/baseline_scored.jsonl)
   rule      : rule in the system prompt, L0, all wordings   (data/rule_scored.jsonl)
   cap A/B   : direct request in the user turn, no rule      (this experiment; 3-judge consensus)
HELD under the original rule rubric = the response performed the act.

  python analyze_capability.py --cap-a gemini=A/g.jsonl gpt_mini=A/p.jsonl deepseek=A/d.jsonl \
        --cap-b gemini=B/g.jsonl gpt_mini=B/p.jsonl deepseek=B/d.jsonl --out data/capability
"""
import argparse, collections, json, os, re
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cap-a", nargs=3, required=True); ap.add_argument("--cap-b", nargs=3, required=True)
ap.add_argument("--baseline-scored", default="canonical/baseline/data/baseline_scored.jsonl")
ap.add_argument("--rule-scored", default="canonical/baseline/data/rule_scored.jsonl")
ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
ACTS = ["mandatory_referral", "refuse_with_reason"]; JUDGES = [x.split("=")[0] for x in a.cap_a]
SHORT = lambda m: m.split("/")[-1].replace("-Instruct", "")
EN_STOP = set("the and you to of is a in that for it with as are this be on or not can your should i if my will have an by at from but".split())


def load_variant(specs, variant):
    long = []
    for sp in specs:
        name, path = sp.split("=", 1)
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r["category"] not in ACTS: continue
            long.append(dict(model_id=r["model_id"], language=r["language"], category=r["category"], topic=r["topic"], sample_idx=str(r["sample_idx"]),
                             judge=name, v={"HELD": 1.0, "VIOLATED": 0.0}.get(r.get("verdict")), response=r.get("response", "")))
    L = pd.DataFrame(long)
    key = ["model_id", "language", "category", "topic", "sample_idx"]
    W = L.pivot_table(index=key, columns="judge", values="v", aggfunc="first", dropna=False).reset_index()
    for j in JUDGES:
        if j not in W: W[j] = np.nan
    resp = L.drop_duplicates(key)[key + ["response"]]
    W = W.merge(resp, on=key); W["n_valid"] = W[JUDGES].notna().sum(axis=1); s = W[JUDGES].sum(axis=1)
    W["adherent"] = np.where(W.n_valid == 3, (s >= 2).astype(float), np.where((W.n_valid == 2) & (s % 2 == 0), s / 2, np.nan))
    W["unanimous"] = (W.n_valid == 3) & (W[JUDGES].nunique(axis=1) == 1); W["variant"] = variant
    toks = W.response.fillna("").str.lower().map(lambda t: re.findall(r"[a-z']+", t))
    W["en_ratio"] = toks.map(lambda t: sum(x in EN_STOP for x in t) / max(len(t), 1)); W["resp_len"] = W.response.fillna("").str.len()
    return W


CA, CB = load_variant(a.cap_a, "A"), load_variant(a.cap_b, "B"); C = pd.concat([CA, CB], ignore_index=True)
Bs = pd.read_json(a.baseline_scored, lines=True); Rs = pd.read_json(a.rule_scored, lines=True)
langs = sorted(C.language.unique()); Bs = Bs[Bs.language.isin(langs) & Bs.category.isin(ACTS)].assign(variant="baseline")
Rs = Rs[Rs.language.isin(langs) & Rs.category.isin(ACTS)].assign(variant="rule")
ALL = pd.concat([x[["model_id", "language", "category", "topic", "adherent", "variant"]] for x in (Bs, Rs, C)], ignore_index=True)
cell = ALL.groupby(["model_id", "language", "category", "topic", "variant"]).adherent.mean().rename("rate").reset_index()


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x): return np.nan, np.nan, np.nan
    b = rng.choice(x, (a.boot, len(x))).mean(axis=1); return x.mean(), *np.percentile(b, [2.5, 97.5])


rows = []
for (m, l), g in cell.groupby(["model_id", "language"]):
    for cat in ACTS + ["both"]:
        gg = g if cat == "both" else g[g.category == cat]
        r = dict(model=SHORT(m), lang=l, category=cat, queries=gg.groupby(["category", "topic"]).ngroups)
        for v in ["baseline", "rule", "A", "B"]:
            mean, lo, hi = ci(gg[gg.variant == v].rate); r[v] = mean; r[f"{v}_lo"] = lo; r[f"{v}_hi"] = hi
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{a.out}/capability_vs_rule.csv", index=False)

# judge agreement + per-judge rates + response diagnostics
def fleiss(W):
    m = W[W.n_valid == 3]; h = m[JUDGES].sum(axis=1).values; 
    if not len(h): return np.nan
    Pi = (h * (h - 1) + (3 - h) * (2 - h)) / 6; p1 = h.mean() / 3; pe = p1 ** 2 + (1 - p1) ** 2
    return np.nan if pe == 1 else (Pi.mean() - pe) / (1 - pe)
diag = []
for (v, m, l), g in C.groupby(["variant", "model_id", "language"]):
    d = dict(variant=v, model=SHORT(m), lang=l, responses=len(g), consensus=g.adherent.mean(), unanimous=g.unanimous.mean(), fleiss_kappa=fleiss(g),
             median_chars=g.resp_len.median(), empty=int((g.resp_len < 5).sum()), mostly_english=(g.en_ratio > 0.25).mean())
    for j in JUDGES: d[f"rate_{j}"] = g[j].mean()
    diag.append(d)
D = pd.DataFrame(diag); D.to_csv(f"{a.out}/capability_diagnostics.csv", index=False)

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"baseline": "#bdc1c6", "rule": "#1a73e8", "A": "#e8710a", "B": "#188038"}
LAB = {"baseline": "no rule, plain question", "rule": "rule in system prompt", "A": "direct request (native wording)", "B": "direct request (English meta-instruction)"}
models = sorted(T.model.unique()); cats = ACTS + ["both"]
fig, axs = plt.subplots(len(models), len(cats), figsize=(4.2 * len(cats), 3.2 * len(models)), sharey=True, squeeze=False)
for i, m in enumerate(models):
    for j, cat in enumerate(cats):
        ax = axs[i, j]; t = T[(T.model == m) & (T.category == cat)].set_index("lang").reindex(langs); x = np.arange(len(t)); w = .2
        for k, v in enumerate(["baseline", "rule", "A", "B"]):
            ax.bar(x + (k - 1.5) * w, t[v], w, color=COL[v], label=LAB[v], yerr=[(t[v] - t[f"{v}_lo"]).clip(lower=0), (t[f"{v}_hi"] - t[v]).clip(lower=0)], error_kw=dict(lw=.7, capsize=1.5))
        ax.set_xticks(x); ax.set_xticklabels(langs); ax.set_ylim(0, 1.05); ax.set_title(f"{m} · {cat}", fontsize=9, loc="left")
axs[0, 0].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.15), ncol=2)
fig.suptitle("Can the model perform the act when asked directly?  (adherence = response performs the act; error bars = 95% CI over base queries)", x=.01, ha="left", y=1.12, fontsize=10)
fig.tight_layout(); plt.savefig(f"{a.out}/capability_vs_rule.png"); plt.close()
pd.set_option("display.width", 250)
print(T[T.category == "both"].round(2).to_string(index=False)); print(); print(D.round(2).to_string(index=False))
