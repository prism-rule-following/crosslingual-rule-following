"""Figures for the no-rule baseline report. Reads data/*.csv|jsonl, writes figures/*.png."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/data"; FIG = f"{HERE}/figures"; os.makedirs(FIG, exist_ok=True)
BASE, RULE, GRAY = "#9aa0a6", "#1a73e8", "#5f6368"
JC = {"gemini": "#1a73e8", "gpt_mini": "#e8710a", "deepseek": "#188038"}
MODELS = {"Qwen/Qwen3-8B": "Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
CATS = ["scope_lock", "ack_invert", "no_dosage", "no_pii", "refuse_with_reason", "mandatory_referral", "no_verdict"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
eff = pd.read_csv(f"{DATA}/rule_effect_by_judge.csv"); agr = pd.read_csv(f"{DATA}/judge_agreement.csv")
cat = pd.read_csv(f"{DATA}/category_effect_consensus.csv"); catj = pd.read_csv(f"{DATA}/category_effect_by_judge.csv")
dis = pd.read_csv(f"{DATA}/judge_disagreement_by_category.csv"); len_ = pd.read_csv(f"{DATA}/judge_leniency.csv")
B = pd.read_json(f"{DATA}/baseline_scored.jsonl", lines=True); R = pd.read_json(f"{DATA}/rule_scored.jsonl", lines=True)

# 1 headline: baseline vs rule, consensus, CI
m = eff[eff.scorer == "maj"]
fig, ax = plt.subplots(figsize=(7, 4)); w = .36; x = np.arange(len(m))
ax.bar(x - w/2, m.baseline, w, color=BASE, label="No rule (baseline)")
ax.bar(x + w/2, m.rule, w, color=RULE, label="With rule")
for i, (_, r) in enumerate(m.iterrows()):
    ax.text(i - w/2, r.baseline + .02, f"{r.baseline:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, r.rule + .02, f"{r.rule:.2f}", ha="center", fontsize=9)
    ax.text(i, 1.12, f"Δ {r['diff']:+.2f}\n[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", ha="center", fontsize=8.5, color=GRAY)
ax.set_xticks(x); ax.set_xticklabels([f"{MODELS[r.model]}\n{r.lang}" for _, r in m.iterrows()])
ax.set_ylim(0, 1.3); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_ylabel("Adherence rate (3-judge consensus)")
ax.set_title("The rule drives adherence in English, not in Yoruba (L0)", loc="left", fontsize=11); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
fig.savefig(f"{FIG}/01_baseline_vs_rule.png"); plt.close()

# 2 forest: effect by judge
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
for ax, lang in zip(axs, ["en", "yo"]):
    d = eff[(eff.lang == lang)]; ylab = []; y = 0
    for mod in MODELS:
        for s, c in [("gemini", JC["gemini"]), ("gpt_mini", JC["gpt_mini"]), ("deepseek", JC["deepseek"]), ("maj", "black")]:
            r = d[(d.model == mod) & (d.scorer == s)].iloc[0]
            ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="D" if s == "maj" else "o", color=c, capsize=3)
            ylab.append(f"{MODELS[mod]} · {'consensus' if s=='maj' else s}"); y -= 1
        y -= .5
    ax.axvline(0, color=GRAY, lw=.8, ls="--"); ax.set_yticks([0,-1,-2,-3,-5.5+0,-6.5,-7.5,-8.5][:0]); 
    ys = [0,-1,-2,-3,-4.5,-5.5,-6.5,-7.5]; ax.set_yticks(ys); ax.set_yticklabels(ylab if lang == "en" else [""]*8, fontsize=8.5)
    ax.set_title(f"{lang}: rule − baseline (95% CI over 26 cells)", loc="left", fontsize=10); ax.set_xlabel("Δ adherence")
fig.tight_layout(); fig.savefig(f"{FIG}/02_effect_by_judge_forest.png"); plt.close()

# 3 heatmap category effect
fig, axs = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
k = 0
for mod in MODELS:
    for lang in ["en", "yo"]:
        ax = axs[k]; k += 1
        d = cat[(cat.model_id == mod) & (cat.language == lang)].set_index("category").reindex(CATS)
        M = d[["baseline", "rule", "rule_minus_baseline"]].values
        im = ax.imshow(M[:, :2], cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(CATS)):
            for j in range(2): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > .6 else "black")
            ax.text(2.1, i, f"{M[i,2]:+.2f}", va="center", fontsize=8.5, color=("#188038" if M[i,2] > .1 else "#c5221f" if M[i,2] < -.1 else GRAY), fontweight="bold" if abs(M[i,2]) > .1 else None)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "rule"]); ax.set_xlim(-.5, 2.8); ax.set_title(f"{MODELS[mod]} · {lang}", fontsize=10)
        ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS)
fig.suptitle("Adherence by category (consensus); right number = rule − baseline", x=.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/03_category_heatmap.png"); plt.close()

# 4 dumbbell by category (en)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, mod in zip(axs, MODELS):
    d = cat[(cat.model_id == mod) & (cat.language == "en")].set_index("category").reindex(CATS)
    for i, (c, r) in enumerate(d.iterrows()):
        ax.plot([r.baseline, r.rule], [i, i], color="#c8cdd2", lw=3, zorder=1)
        ax.scatter(r.baseline, i, color=BASE, s=55, zorder=2); ax.scatter(r.rule, i, color=RULE, s=55, zorder=2)
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS); ax.invert_yaxis(); ax.set_xlim(-.05, 1.05); ax.set_title(f"{MODELS[mod]} · English", loc="left", fontsize=10); ax.set_xlabel("Adherence")
axs[0].scatter([], [], color=BASE, label="No rule"); axs[0].scatter([], [], color=RULE, label="With rule"); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.06))
fig.tight_layout(); fig.savefig(f"{FIG}/04_category_dumbbell_en.png"); plt.close()

# 5 cell scatter
fig, axs = plt.subplots(1, 4, figsize=(11, 3.2), sharex=True, sharey=True); k = 0
CELL = ["category", "topic"]
for mod in MODELS:
    for lang in ["en", "yo"]:
        ax = axs[k]; k += 1
        b = B[(B.model_id == mod) & (B.language == lang)].groupby(CELL).adherent.mean()
        r = R[(R.model_id == mod) & (R.language == lang)].groupby(CELL).adherent.mean()
        c = pd.concat([b.rename("b"), r.rename("r")], axis=1).dropna().reset_index()
        rng = np.random.default_rng(1)
        ax.scatter(c.b + rng.normal(0, .015, len(c)), c.r + rng.normal(0, .015, len(c)), s=28, alpha=.7, color=RULE)
        ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=.8)
        ax.set_title(f"{MODELS[mod]} · {lang}", fontsize=10); ax.set_xlabel("baseline rate (per cell)")
axs[0].set_ylabel("rule rate (per cell)")
fig.suptitle("26 base-query cells per panel (jittered). Above the diagonal = rule helped", x=.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/05_cell_scatter.png"); plt.close()

# 6 judge agreement
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
agr["grp"] = agr.cond + " · " + agr.model.map(MODELS) + " · " + agr.lang
pairs = [("gemini_gpt_mini", "Gemini–GPT-mini"), ("gemini_deepseek", "Gemini–DeepSeek"), ("gpt_mini_deepseek", "GPT-mini–DeepSeek")]
y = np.arange(len(agr))
axs[0].barh(y, agr.fleiss_kappa, color="#6f42c1"); axs[0].set_yticks(y); axs[0].set_yticklabels(agr.grp, fontsize=8); axs[0].invert_yaxis()
axs[0].set_xlim(.5, 1); axs[0].set_title("Fleiss κ (3 judges)", loc="left", fontsize=10)
for i, v in enumerate(agr.fleiss_kappa): axs[0].text(v + .005, i, f"{v:.2f}", va="center", fontsize=8)
for (c, lab), off in zip(pairs, [-.25, 0, .25]):
    axs[1].barh(y + off, agr[f"kappa_{c}"], height=.25, label=lab)
axs[1].set_xlim(.5, 1); axs[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=3); axs[1].set_title("Pairwise Cohen κ", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/06_judge_agreement.png"); plt.close()

# 7 unanimity
fig, ax = plt.subplots(figsize=(7, 3.2)); ax.bar(range(len(agr)), agr.unanimous, color="#6f42c1")
ax.set_xticks(range(len(agr))); ax.set_xticklabels(agr.grp, rotation=60, ha="right", fontsize=8); ax.set_ylim(.7, 1.0)
for i, v in enumerate(agr.unanimous): ax.text(i, v + .004, f"{v:.0%}", ha="center", fontsize=8)
ax.set_title("Share of responses where all three judges agree", loc="left", fontsize=10); fig.savefig(f"{FIG}/07_unanimity.png"); plt.close()

# 8 disagreement by category
dd = dis.copy(); dd["row"] = dd.language + " · " + dd.category
fig, ax = plt.subplots(figsize=(7, 5)); y = np.arange(len(dd)); 
ax.barh(y - .2, dd.baseline.fillna(0), .4, color=BASE, label="baseline"); ax.barh(y + .2, dd.rule.fillna(0), .4, color=RULE, label="rule")
ax.set_yticks(y); ax.set_yticklabels(dd.row, fontsize=8); ax.invert_yaxis(); ax.legend(frameon=False)
ax.set_title("Judge split rate by category (share of responses with a 2–1 split)", loc="left", fontsize=10); fig.savefig(f"{FIG}/08_disagreement_by_category.png"); plt.close()

# 9 leniency
ld = len_.copy(); ld["grp"] = ld.cond + " · " + ld.model_id.map(MODELS) + " · " + ld.language
fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(ld)); w = .26
for i, j in enumerate(["gemini", "gpt_mini", "deepseek"]): ax.bar(x + (i - 1) * w, ld[j], w, color=JC[j], label=j)
ax.set_xticks(x); ax.set_xticklabels(ld.grp, rotation=60, ha="right", fontsize=8); ax.set_ylabel("Raw HELD rate"); ax.legend(frameon=False, ncol=3)
ax.set_title("Judge strictness: raw adherence rate per judge", loc="left", fontsize=10); fig.savefig(f"{FIG}/09_judge_leniency.png"); plt.close()

# 10 per-judge category effect heatmap (en)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, lang in zip(axs, ["en", "yo"]):
    d = catj[catj.language == lang].pivot(index="category", columns="judge", values="diff").reindex(CATS)[["gemini", "gpt_mini", "deepseek"]]
    ax.imshow(d.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(CATS)):
        for j in range(3): ax.text(j, i, f"{d.values[i,j]:+.2f}", ha="center", va="center", fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(d.columns); ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS); ax.set_title(f"{lang}: rule − baseline per judge", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/10_category_effect_by_judge.png"); plt.close()

# 11 swahili
sw = pd.read_csv(f"{DATA}/baseline_swahili_descriptive.csv").set_index("category").reindex(CATS)
fig, ax = plt.subplots(figsize=(7, 3.4)); x = np.arange(len(sw)); w = .38
ax.bar(x - w/2, sw["Qwen/Qwen3-8B"], w, color="#8ab4f8", label="Qwen3-8B"); ax.bar(x + w/2, sw["meta-llama/Llama-3.1-8B-Instruct"], w, color="#f6ae74", label="Llama-3.1-8B")
ax.set_xticks(x); ax.set_xticklabels(CATS, rotation=30, ha="right", fontsize=8.5); ax.set_ylabel("Adherence with NO rule"); ax.legend(frameon=False)
ax.set_title("Swahili baseline (no rule run exists to compare)", loc="left", fontsize=10); fig.savefig(f"{FIG}/11_swahili_baseline.png"); plt.close()
print(sorted(os.listdir(FIG)))

# 12/13 deontic wording (rule run only)
for col, name, fn in [("grammar_type", "deontic phrasing", "12_by_grammar_type"), ("pair_type", "status-word pair", "13_by_pair_type")]:
    t = R.groupby(["model_id", "language", col]).adherent.mean().unstack(col); t.to_csv(f"{DATA}/rule_adherence_by_{col}.csv")
    fig, ax = plt.subplots(figsize=(9, 3.4)); grp = [(m_, l_) for m_ in MODELS for l_ in ["en", "yo"]]; k = len(t.columns); w = .8 / k
    for i, c in enumerate(t.columns):
        ax.bar(np.arange(4) + (i - (k-1)/2) * w, [t.loc[g_, c] for g_ in grp], w, label=c)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{MODELS[m_]}\n{l_}" for m_, l_ in grp]); ax.set_ylim(0, 1.05); ax.set_ylabel("Adherence (consensus)")
    ax.legend(frameon=False, ncol=k, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2)); ax.set_title(f"Rule-run adherence by {name} (L0)", loc="left", fontsize=10)
    fig.savefig(f"{FIG}/{fn}.png"); plt.close()

# 14 cell outcome types
rows = []
for mod in MODELS:
    for lang in ["en", "yo"]:
        b = B[(B.model_id == mod) & (B.language == lang)].groupby(CELL).adherent.mean(); r = R[(R.model_id == mod) & (R.language == lang)].groupby(CELL).adherent.mean()
        c = pd.concat([b.rename("b"), r.rename("r")], axis=1).dropna(); d = c.r - c.b
        rows.append(dict(model=MODELS[mod], lang=lang, cells=len(c), rule_flipped=int(((c.b <= .2) & (c.r >= .8)).sum()), already_ok=int((c.b >= .8).sum() - ((c.b >= .8) & (d < -.1)).sum()),
                         fails_both=int(((c.b <= .2) & (c.r <= .2)).sum()), rule_hurt=int((d < -.1).sum())))
co = pd.DataFrame(rows); co["partial_or_mixed"] = co.cells - co[["rule_flipped", "already_ok", "fails_both", "rule_hurt"]].sum(axis=1); co.to_csv(f"{DATA}/cell_outcome_types.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 3.4)); left = np.zeros(len(co)); lab = [f"{r.model}\n{r.lang}" for r in co.itertuples()]
for col, c, nm in [("rule_flipped", RULE, "rule flipped it (base ≤0.2 → rule ≥0.8)"), ("already_ok", "#81c995", "already fine without rule (base ≥0.8)"), ("fails_both", "#ea4335", "fails with AND without rule (both ≤0.2)"), ("rule_hurt", "#fbbc04", "rule made it worse (Δ<−0.1)"), ("partial_or_mixed", "#dadce0", "partial / mixed")]:
    ax.barh(lab, co[col], left=left, color=c, label=nm)
    for i, v in enumerate(co[col]):
        if v: ax.text(left[i] + v / 2, i, int(v), ha="center", va="center", fontsize=9, color="white" if c in (RULE, "#ea4335") else "black")
    left += co[col].values
ax.invert_yaxis(); ax.set_xlabel("number of the 26 base-query cells"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.45, -.2), ncol=2)
ax.set_title("What happens to each base query when the rule is added", loc="left", fontsize=10); fig.savefig(f"{FIG}/14_cell_outcomes.png"); plt.close()
print(co.to_string()); print(pd.read_csv(f"{DATA}/rule_adherence_by_grammar_type.csv").round(3).to_string()); print(pd.read_csv(f"{DATA}/rule_adherence_by_pair_type.csv").round(3).to_string())
