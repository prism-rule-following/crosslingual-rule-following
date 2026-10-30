"""Figures for the no-rule baseline report (any number of languages). Reads data/, writes figures/."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/data"; FIG = f"{HERE}/figures"; os.makedirs(FIG, exist_ok=True)
for f in os.listdir(FIG):
    if f.endswith(".png"): os.remove(f"{FIG}/{f}")
BASE, RULE, GRAY = "#9aa0a6", "#1a73e8", "#5f6368"
JC = {"gemini": "#1a73e8", "gpt_mini": "#e8710a", "deepseek": "#188038"}
SHORT = lambda m: m.split("/")[-1].replace("-Instruct", "")
CATS = ["scope_lock", "ack_invert", "no_dosage", "no_pii", "refuse_with_reason", "mandatory_referral", "no_verdict"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
eff = pd.read_csv(f"{DATA}/rule_effect_by_judge.csv"); agr = pd.read_csv(f"{DATA}/judge_agreement.csv")
cat = pd.read_csv(f"{DATA}/category_effect_consensus.csv"); catj = pd.read_csv(f"{DATA}/category_effect_by_judge.csv")
dis = pd.read_csv(f"{DATA}/judge_disagreement_by_category.csv"); len_ = pd.read_csv(f"{DATA}/judge_leniency.csv")
B = pd.read_json(f"{DATA}/baseline_scored.jsonl", lines=True); R = pd.read_json(f"{DATA}/rule_scored.jsonl", lines=True)
bl = pd.read_csv(f"{DATA}/baseline_adherence_by_language_category.csv")
ls = pd.read_csv(f"{DATA}/language_sets.csv", header=None, index_col=0)[1].fillna("")
PAIRED = [x for x in ls["paired"].split(",") if x]; BASE_ONLY = [x for x in ls["baseline_only"].split(",") if x]
MODELS = sorted(B.model_id.unique()); CELL = ["category", "topic"]
R = R[R.language.isin(PAIRED)]
def save(name): plt.savefig(f"{FIG}/{name}.png"); plt.close()
def heat(ax, M, rows, cols, vmin, vmax, cmap, fmt="{:.2f}", fs=7.5):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v): ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="white" if abs(v - (vmin+vmax)/2) > (vmax-vmin)*.32 and cmap in ("Blues",) and v > .6 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)

m = eff[(eff.scorer == "maj") & (eff.lang.isin(PAIRED))]

# 01 headline: baseline vs rule per language, per model
fig, axs = plt.subplots(len(MODELS), 1, figsize=(max(7, 0.95 * len(PAIRED) + 2), 3.4 * len(MODELS)), sharey=True, squeeze=False)
for ax, mod in zip(axs[:, 0], MODELS):
    d = m[m.model == mod].set_index("lang").reindex(PAIRED); x = np.arange(len(d)); w = .38
    ax.bar(x - w/2, d.baseline, w, color=BASE, label="No rule (baseline)"); ax.bar(x + w/2, d.rule, w, color=RULE, label="With rule")
    for i, (l, r) in enumerate(d.iterrows()): ax.text(i, 1.04, f"{r['diff']:+.2f}", ha="center", fontsize=8.5, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(PAIRED); ax.set_ylim(0, 1.15); ax.set_yticks([0, .5, 1]); ax.set_title(f"{SHORT(mod)}  (number above bars = rule − baseline)", loc="left", fontsize=10)
axs[0, 0].legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.15)); axs[-1, 0].set_xlabel("language"); fig.supylabel("Adherence (3-judge consensus)", x=.02)
fig.tight_layout(); save("01_baseline_vs_rule")

# 02 effect heatmap by judge: rows = model·language, cols = judges + consensus (+ CI-excludes-0 marker)
rows_ = [(mod, l) for mod in MODELS for l in PAIRED]
cols_ = ["gemini", "gpt_mini", "deepseek", "maj"]
M = np.array([[eff[(eff.model == mo) & (eff.lang == l) & (eff.scorer == c)]["diff"].mean() for c in cols_] for mo, l in rows_])
fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(rows_) + 1.4)); heat(ax, M, [f"{SHORT(mo)} · {l}" for mo, l in rows_], ["Gemini", "GPT-mini", "DeepSeek", "Consensus"], -1, 1, "RdBu", "{:+.2f}")
for i, (mo, l) in enumerate(rows_):
    for j, c in enumerate(cols_):
        r = eff[(eff.model == mo) & (eff.lang == l) & (eff.scorer == c)].iloc[0]
        if r.ci_lo > 0 or r.ci_hi < 0: ax.text(j + .38, i - .3, "*", fontsize=10, ha="center", va="center", fontweight="bold")
ax.set_title("Rule effect (Δ adherence) per judge.  * = 95% CI excludes 0", loc="left", fontsize=10); fig.tight_layout(); save("02_effect_by_judge")

# 03 category x language effect heatmap per model
fig, axs = plt.subplots(1, len(MODELS), figsize=(0.9 * len(PAIRED) * len(MODELS) + 3, 3.6), squeeze=False)
for ax, mod in zip(axs[0], MODELS):
    t = cat[cat.model_id == mod].pivot(index="category", columns="language", values="rule_minus_baseline").reindex(index=CATS, columns=PAIRED)
    heat(ax, t.values, CATS, PAIRED, -1, 1, "RdBu", "{:+.2f}"); ax.set_title(f"{SHORT(mod)}: rule − baseline", loc="left", fontsize=10)
fig.tight_layout(); save("03_category_effect_heatmap")

# 04 baseline adherence (NO rule) by language x category: where does the model already behave
fig, axs = plt.subplots(1, len(MODELS), figsize=(0.62 * (len(bl.language.unique())) * len(MODELS) + 3, 3.6), squeeze=False); allL = sorted(bl.language.unique(), key=lambda x: (x not in PAIRED, x))
for ax, mod in zip(axs[0], MODELS):
    t = bl[bl.model_id == mod].set_index("language").reindex(allL)[CATS]
    heat(ax, t.T.values, CATS, allL, 0, 1, "Blues", "{:.2f}", 7); ax.set_title(f"{SHORT(mod)}: adherence with NO rule", loc="left", fontsize=10)
fig.tight_layout(); save("04_baseline_adherence_by_language")

# 05 cell scatter grid
nL = len(PAIRED); fig, axs = plt.subplots(len(MODELS), nL, figsize=(2.1 * nL, 2.3 * len(MODELS)), sharex=True, sharey=True, squeeze=False); rng = np.random.default_rng(1)
for i, mod in enumerate(MODELS):
    for j, l in enumerate(PAIRED):
        ax = axs[i, j]
        b = B[(B.model_id == mod) & (B.language == l)].groupby(CELL).adherent.mean(); r = R[(R.model_id == mod) & (R.language == l)].groupby(CELL).adherent.mean()
        c = pd.concat([b.rename("b"), r.rename("r")], axis=1).dropna()
        ax.scatter(c.b + rng.normal(0, .02, len(c)), c.r + rng.normal(0, .02, len(c)), s=14, alpha=.7, color=RULE); ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=.7)
        ax.set_title(f"{SHORT(mod)[:9]} · {l}", fontsize=8)
fig.supxlabel("baseline rate per base query", fontsize=9); fig.supylabel("rule rate per base query", fontsize=9); fig.suptitle("Each dot = one of 26 base queries (jittered). Above diagonal = rule helped", x=.01, ha="left", fontsize=10)
fig.tight_layout(); save("05_cell_scatter")

# 06 judge agreement
agr["grp"] = agr.cond + " · " + agr.model.map(SHORT) + " · " + agr.lang
agr = agr.sort_values(["cond", "model", "lang"]).reset_index(drop=True)
fig, axs = plt.subplots(1, 2, figsize=(10, 0.24 * len(agr) + 1.5), sharey=True); y = np.arange(len(agr))
axs[0].barh(y, agr.fleiss_kappa, color="#6f42c1"); axs[0].set_yticks(y); axs[0].set_yticklabels(agr.grp, fontsize=7.5); axs[0].invert_yaxis(); axs[0].set_xlim(.4, 1); axs[0].set_title("Fleiss κ (3 judges)", loc="left", fontsize=10)
for (c, lab), off in zip([("gemini_gpt_mini", "Gemini–GPT-mini"), ("gemini_deepseek", "Gemini–DeepSeek"), ("gpt_mini_deepseek", "GPT-mini–DeepSeek")], [-.27, 0, .27]): axs[1].barh(y + off, agr[f"kappa_{c}"], height=.26, label=lab)
axs[1].set_xlim(.4, 1); axs[1].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, -.08), ncol=3); axs[1].set_title("Pairwise Cohen κ", loc="left", fontsize=10)
fig.tight_layout(); save("06_judge_agreement")

# 07 unanimity
fig, ax = plt.subplots(figsize=(max(7, .38 * len(agr)), 3.4)); ax.bar(range(len(agr)), agr.unanimous, color="#6f42c1"); ax.set_xticks(range(len(agr))); ax.set_xticklabels(agr.grp, rotation=70, ha="right", fontsize=7)
ax.set_ylim(.6, 1.0); ax.set_title("Share of responses where all three judges agree", loc="left", fontsize=10); save("07_unanimity")

# 08 disagreement heatmap: language x category, pooled over conditions
D2 = dis.copy(); D2["split"] = D2[["baseline", "rule"]].mean(axis=1, skipna=True); t = D2.pivot(index="category", columns="language", values="split").reindex(CATS)
fig, ax = plt.subplots(figsize=(0.62 * t.shape[1] + 2, 3.6)); heat(ax, t.values, CATS, list(t.columns), 0, .35, "Oranges", "{:.2f}", 7); ax.set_title("Judge split rate (share of responses with a 2–1 split)", loc="left", fontsize=10); fig.tight_layout(); save("08_disagreement_heatmap")

# 09 leniency: raw rate per judge, baseline condition
ld = len_[len_.cond == "baseline"].copy(); ld["grp"] = ld.model_id.map(SHORT) + " · " + ld.language; ld = ld.sort_values(["model_id", "language"])
fig, ax = plt.subplots(figsize=(max(7, .42 * len(ld)), 3.6)); x = np.arange(len(ld)); w = .27
for i, j in enumerate(["gemini", "gpt_mini", "deepseek"]): ax.bar(x + (i - 1) * w, ld[j], w, color=JC[j], label=j)
ax.set_xticks(x); ax.set_xticklabels(ld.grp, rotation=70, ha="right", fontsize=7.5); ax.set_ylabel("raw HELD rate"); ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1))
ax.set_title("Judge strictness, no-rule baseline", loc="left", fontsize=10); save("09_judge_leniency")

# 10 category effect by judge (pooled over paired languages)
fig, ax = plt.subplots(figsize=(4.6, 3.6)); cj = catj.groupby(["category", "judge"])["diff"].mean().unstack("judge").reindex(CATS)[["gemini", "gpt_mini", "deepseek"]]
heat(ax, cj.values, CATS, ["Gemini", "GPT-mini", "DeepSeek"], -1, 1, "RdBu", "{:+.2f}"); ax.set_title("Rule − baseline per judge (mean over paired languages)", loc="left", fontsize=9); fig.tight_layout(); save("10_category_effect_by_judge")

# 11 baseline-only languages
if BASE_ONLY:
    fig, axs = plt.subplots(1, len(MODELS), figsize=(0.9 * len(BASE_ONLY) * len(MODELS) + 3, 3.6), squeeze=False)
    for ax, mod in zip(axs[0], MODELS):
        t = bl[(bl.model_id == mod) & bl.language.isin(BASE_ONLY)].set_index("language").reindex(BASE_ONLY)[CATS]; heat(ax, t.T.values, CATS, BASE_ONLY, 0, 1, "Blues", "{:.2f}"); ax.set_title(f"{SHORT(mod)} (no rule run exists)", loc="left", fontsize=9)
    fig.tight_layout(); save("11_baseline_only_languages")

# 12/13 wording (rule run, paired languages)
for col, name, fn in [("grammar_type", "deontic phrasing", "12_by_grammar_type"), ("pair_type", "status-word pair", "13_by_pair_type")]:
    t = R.groupby(["model_id", "language", col]).adherent.mean().unstack(col); t.to_csv(f"{DATA}/rule_adherence_by_{col}.csv")
    fig, axs = plt.subplots(len(MODELS), 1, figsize=(max(8, 1.05 * nL + 2), 2.9 * len(MODELS)), sharey=True, squeeze=False)
    for ax, mod in zip(axs[:, 0], MODELS):
        k = len(t.columns); w = .8 / k
        for i, c in enumerate(t.columns): ax.bar(np.arange(nL) + (i - (k - 1) / 2) * w, [t.loc[(mod, l), c] if (mod, l) in t.index else np.nan for l in PAIRED], w, label=c)
        ax.set_xticks(range(nL)); ax.set_xticklabels(PAIRED); ax.set_ylim(0, 1.05); ax.set_title(f"{SHORT(mod)}: rule-run adherence by {name}", loc="left", fontsize=10)
    axs[0, 0].legend(frameon=False, ncol=k, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.12)); fig.tight_layout(); save(fn)

# 14 cell outcomes
rows = []
for mod in MODELS:
    for l in PAIRED:
        b = B[(B.model_id == mod) & (B.language == l)].groupby(CELL).adherent.mean(); r = R[(R.model_id == mod) & (R.language == l)].groupby(CELL).adherent.mean()
        c = pd.concat([b.rename("b"), r.rename("r")], axis=1).dropna(); d = c.r - c.b
        rows.append(dict(model=SHORT(mod), lang=l, cells=len(c), rule_flipped=int(((c.b <= .2) & (c.r >= .8)).sum()), already_ok=int(((c.b >= .8) & (d >= -.1)).sum()),
                         fails_both=int(((c.b <= .2) & (c.r <= .2)).sum()), rule_hurt=int((d < -.1).sum())))
co = pd.DataFrame(rows); co["partial_or_mixed"] = co.cells - co[["rule_flipped", "already_ok", "fails_both", "rule_hurt"]].sum(axis=1); co.to_csv(f"{DATA}/cell_outcome_types.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 0.26 * len(co) + 2)); left = np.zeros(len(co)); lab = [f"{r.model} · {r.lang}" for r in co.itertuples()]
for col, c, nm in [("rule_flipped", RULE, "rule flipped it (base ≤0.2 → rule ≥0.8)"), ("already_ok", "#81c995", "already fine without rule (base ≥0.8)"), ("fails_both", "#ea4335", "fails with AND without rule"), ("rule_hurt", "#fbbc04", "rule made it worse (Δ<−0.1)"), ("partial_or_mixed", "#dadce0", "partial / mixed")]:
    ax.barh(lab, co[col], left=left, color=c, label=nm)
    for i, v in enumerate(co[col]):
        if v >= 2: ax.text(left[i] + v / 2, i, int(v), ha="center", va="center", fontsize=7.5, color="white" if c in (RULE, "#ea4335") else "black")
    left += co[col].values
ax.invert_yaxis(); ax.set_xlabel("number of the 26 base queries"); ax.tick_params(axis="y", labelsize=8); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.45, -.12), ncol=2)
ax.set_title("What happens to each base query when the rule is added", loc="left", fontsize=10); save("14_cell_outcomes")
print(sorted(os.listdir(FIG)))

# 15 prohibition-type vs action-type rules (post-hoc grouping: no_* categories vs the rest), baseline vs rule, by language
ABST = ["no_dosage", "no_pii", "no_verdict"]
Bg = B.assign(grp=np.where(B.category.isin(ABST), "prohibition (no_*)", "action (do X)")); Rg = R.assign(grp=np.where(R.category.isin(ABST), "prohibition (no_*)", "action (do X)"))
langs_all = sorted(B.language.unique(), key=lambda x: (x not in PAIRED, x))
tab = pd.concat([Bg.groupby(["language", "grp"]).adherent.mean().rename("baseline"), Rg.groupby(["language", "grp"]).adherent.mean().rename("rule")], axis=1).reset_index()
tab.to_csv(f"{DATA}/prohibition_vs_action_by_language.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, g in zip(axs, ["prohibition (no_*)", "action (do X)"]):
    d = tab[tab.grp == g].set_index("language").reindex(langs_all); x = np.arange(len(d)); w = .4
    ax.bar(x - w/2, d.baseline, w, color=BASE, label="No rule"); ax.bar(x + w/2, d.rule, w, color=RULE, label="With rule")
    ax.set_xticks(x); ax.set_xticklabels(langs_all); ax.set_ylim(0, 1.05); ax.set_title(g, loc="left", fontsize=10)
axs[0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.1)); fig.suptitle("Adherence by rule type (both models pooled). Languages without a rule run show baseline only", x=.01, ha="left", fontsize=10, y=1.08)
fig.tight_layout(); save("15_prohibition_vs_action")

# 16 forest: consensus effect per language with 95% CI, both models
fig, ax = plt.subplots(figsize=(7, 0.32 * len(PAIRED) * len(MODELS) + 1.6)); y = 0; yt = []; yl = []
for mod in MODELS:
    for l in PAIRED:
        r = m[(m.model == mod) & (m.lang == l)].iloc[0]
        ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="o", color=RULE if (r.ci_lo > 0 or r.ci_hi < 0) else GRAY, capsize=3); yt.append(y); yl.append(f"{SHORT(mod)} · {l}"); y -= 1
    y -= .6
ax.axvline(0, color=GRAY, ls="--", lw=.8); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_xlabel("rule − baseline (adherence), 95% CI over 26 base queries")
ax.set_title("Rule effect per language (blue = CI excludes 0)", loc="left", fontsize=10); save("16_effect_forest")
print("figs", len(os.listdir(FIG)))
