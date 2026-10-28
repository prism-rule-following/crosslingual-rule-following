"""Figures for the sampling-stability analysis, drawn from run.py's output folder.

RUN from the repo root after run.py:
    .venv/bin/python -m canonical.analysis.sampling_stability.figures \
        --results <run.py --out dir> --out <figure dir> [--upload]

Writes <out>/<name>.pdf (for the paper) and <out>/<name>.png, plus INDEX.md
listing each figure's analysis, suggested placement and a draft caption.
--upload pushes the folder to crv-analysis under its own path.

Conventions: only numeric judge scores are shown. The adherence score is
P(HELD) x 100; main-text figures use the 3-judge mean adherence score (responses
all three judges scored), with each judge's own score alongside.
"""

import argparse
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from canonical.analysis.sampling_stability.data import CATEGORIES, JUDGES, LANGUAGES, PRESSURES  # noqa: E402

UPLOAD_REPO = "crosslingual-rule-following/crv-analysis"

# Reference palette (dataviz skill): categorical slots in fixed order, one-hue
# sequential blue ramp, blue<->red diverging with a gray midpoint.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2f"])

CAT_LABELS = {"no_verdict": "No verdict", "no_pii": "No PII", "no_dosage": "No dosage", "scope_lock": "Scope lock",
              "mandatory_referral": "Referral", "refuse_with_reason": "Refusal w/ reason"}
MODEL_LABELS = {"qwen": "Qwen3-8B", "llama": "Llama-3.1-8B"}
MODELS = ["qwen", "llama"]
SCORE_LABELS = {"mean": "3-judge mean", "gpt": "GPT-5.4-mini", "gemini": "Gemini-2.5-Flash",
                "deepseek": "DeepSeek-V4-Pro"}
SCORES = ["mean"] + JUDGES
FOCUS = ["yo", "ig"]
PRIMARY = "mean"
ADH = "Adherence score"  # P(HELD) x 100

FIGURES = []  # (name, analysis, placement, caption)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def save(fig, out, name, analysis, placement, caption):
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, analysis, placement, caption))


def pct(x):
    return 100 * np.asarray(x, dtype=float)


def heatmap(ax, mat, cmap, vmin, vmax, xlabels, ylabels):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def cell_matrix(t, value):
    """category x language matrix from a long table with CELL columns."""
    return t.pivot_table(index="category", columns="language", values=value).reindex(index=CATEGORIES, columns=LANGUAGES)


def focus_legend(ax, **kw):
    ax.legend(handles=[plt.Line2D([], [], color=ORANGE, lw=1.6, label="Yoruba, Igbo"),
                       plt.Line2D([], [], color=BLUE, lw=1.6, label="other 8 languages")], ncol=2, **kw)


# ------------------------------------------------------------------
# A2 headline gap per seed
# ------------------------------------------------------------------
def fig_headline_gap(T, out, languages, name, placement):
    per_seed = T["headline_gap_per_seed"]
    rows = [(l, m) for l in languages for m in MODELS]
    fig, axes = plt.subplots(1, len(SCORES), figsize=(7.0, 0.32 * len(rows) + 0.9), sharey=True, sharex=True)
    for ax, score in zip(axes, SCORES):
        d = per_seed[per_seed.score == score]
        # The samples often sit within ~1 point of each other, which is too close to see at this scale,
        # so the spread is also printed to the right of each row.
        right = blended_transform_factory(ax.transAxes, ax.transData)
        for i, (lang, model) in enumerate(rows):
            g = pct(d[(d.language == lang) & (d.model == model)].gap)
            color = ORANGE if lang in FOCUS else BLUE
            ax.plot([g.min(), g.max()], [i, i], color=color, lw=1.2, alpha=0.5, solid_capstyle="round")
            ax.scatter([g.mean()], [i], s=34, marker="D", facecolors="none", edgecolors=INK, lw=0.9, zorder=3)
            ax.scatter(g, np.full(len(g), i), s=12, color=color, alpha=0.8, lw=0, zorder=4)
            ax.text(1.04, i, f"{g.max() - g.min():.1f}", transform=right, va="center", fontsize=6.5, color=INK_2)
        ax.margins(x=0.1)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_title(SCORE_LABELS[score], loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [f"{l.upper()} · {MODEL_LABELS[m]}" for l, m in rows])
    axes[0].invert_yaxis()
    fig.supxlabel(f"Withholding − constructed gap in mean {ADH.lower()} (points)", fontsize=7.5, color=INK_2)
    handles = [axes[0].scatter([], [], s=12, color=INK_2, alpha=0.8, lw=0, label="each of 5 samples"),
               axes[0].scatter([], [], s=34, marker="D", facecolors="none", edgecolors=INK, lw=0.9, label="mean")]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    fig.text(1, 1.012, "number at right of each row: range across the 5 samples (points)", ha="right", va="bottom",
             fontsize=7, color=INK_2)
    save(fig, out, name, "A2 headline gap per seed", placement,
         "Mean adherence score (P(HELD) x 100) of the withholding categories (no verdict, no PII, no dosage, scope "
         "lock) minus the constructed categories (referral, refusal with reason), computed separately in each of "
         "the five samples. Dots: samples; line: range; hollow diamond: mean; number at right: max minus min across "
         "samples. Left: 3-judge mean score (responses all three judges scored); right: each judge's own score.")


# ------------------------------------------------------------------
# A3 cell stability
# ------------------------------------------------------------------
def fig_cell_sd(T, out):
    cs = T["cell_stability"]
    d = cs[cs.score == PRIMARY]
    vmax = pct(d.sd).max()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        mat = pct(cell_matrix(d[d.model == model], "sd").to_numpy())
        im = heatmap(ax, mat, SEQ, 0, vmax, [l.upper() for l in LANGUAGES], [CAT_LABELS[c] for c in CATEGORIES])
        r, c = np.unravel_index(np.nanargmax(mat), mat.shape)
        ax.text(c, r, f"{mat[r, c]:.1f}", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
        ax.set_title(MODEL_LABELS[model], loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("SD across 5 samples (points)")
    cb.outline.set_visible(False)
    save(fig, out, "fig_cell_stability_sd_heatmap", "A3 cell stability", "Main",
         "Standard deviation, across the five samples, of each category x language x model cell's mean adherence "
         "score (3-judge mean, points). One shared scale for both models; the largest SD in each panel is labelled.")

    top = T["cell_stability_top"]
    t = top[top.score == PRIMARY].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for i, r in t.iterrows():
        color = ORANGE if r.language in FOCUS else BLUE
        ax.plot(pct([r["min"], r["max"]]), [i, i], color=color, lw=1.6, solid_capstyle="round")
        ax.scatter(pct([r["mean"]]), [i], s=22, color=INK, zorder=3, lw=0)
    ax.set_yticks(range(len(t)), [f"{CAT_LABELS[c]} · {l.upper()} · {MODEL_LABELS[m]}"
                                  for c, l, m in zip(t.category, t.language, t.model)])
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean {ADH.lower()} — dot: mean, bar: min–max across 5 samples")
    ax.grid(axis="y", visible=False)
    focus_legend(ax, loc="lower left", bbox_to_anchor=(0, 1))
    save(fig, out, "fig_cell_stability_top10", "A3 cell stability", "Appendix",
         "The ten cells whose mean adherence score (3-judge mean) varies most across the five samples: mean and "
         "min-max range of the five sample means. Orange: Yoruba and Igbo.")


# ------------------------------------------------------------------
# Adherence-score stability point-range (all cells)
# ------------------------------------------------------------------
def fig_score_pointrange(T, out):
    cs = T["cell_stability"]
    d = cs[cs.score == PRIMARY]
    fig, axes = plt.subplots(2, len(CATEGORIES), figsize=(7.2, 3.4), sharey=True, sharex=True)
    x = np.arange(len(LANGUAGES))
    for r, model in enumerate(MODELS):
        for c, cat in enumerate(CATEGORIES):
            ax = axes[r, c]
            g = d[(d.model == model) & (d.category == cat)].set_index("language").reindex(LANGUAGES)
            colors = [ORANGE if l in FOCUS else BLUE for l in LANGUAGES]
            ax.vlines(x, pct(g["mean"] - g["sd"]), pct(g["mean"] + g["sd"]), colors=colors, lw=1)
            ax.scatter(x, pct(g["mean"]), s=9, c=colors, zorder=3, lw=0)
            ax.set_xticks(x, [l.upper() for l in LANGUAGES], rotation=90)
            ax.set_ylim(-3, 103)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(CAT_LABELS[cat], fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n{ADH}")
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, out, "fig_adherence_pointrange", "A3 cell stability", "Appendix",
         "Mean adherence score (3-judge mean) across the five samples with +/-1 SD bars, per category, language "
         "and model. Orange: Yoruba and Igbo.")


# ------------------------------------------------------------------
# A4 cross-model replication
# ------------------------------------------------------------------
def fig_cross_model(T, out):
    m = T["cell_means_per_seed"]
    d = m[m.score == PRIMARY]
    cm = T["cross_model_per_seed"]
    cm = cm[cm.score == PRIMARY].set_index("sample_idx")
    seeds = sorted(d.sample_idx.unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(7.2, 1.75), sharex=True, sharey=True)
    for ax, s in zip(axes, seeds):
        p = d[d.sample_idx == s].pivot_table(index=["category", "language"], columns="model", values="value").reset_index()
        focus = p.language.isin(FOCUS)
        ax.plot([0, 100], [0, 100], color=AXIS, lw=0.8, zorder=1)
        ax.scatter(pct(p.loc[~focus, "llama"]), pct(p.loc[~focus, "qwen"]), s=9, color=BLUE, lw=0, alpha=0.8,
                   label="other 8 languages")
        ax.scatter(pct(p.loc[focus, "llama"]), pct(p.loc[focus, "qwen"]), s=9, color=ORANGE, lw=0,
                   label="Yoruba, Igbo")
        ax.text(0.04, 0.96, f"r = {cm.loc[s, 'pearson']:.2f}", transform=ax.transAxes, va="top", color=INK)
        ax.set_title(f"Sample {s + 1}", loc="left")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_aspect("equal")
        ax.set_xlabel("Llama score")
    axes[0].set_ylabel("Qwen score")
    axes[-1].legend(loc="lower right", bbox_to_anchor=(1, 1.15), ncol=2)
    save(fig, out, "fig_cross_model_per_seed", "A4 cross-model replication", "Appendix",
         "Qwen vs Llama mean adherence score (3-judge mean) for each category x language cell, one panel per "
         "sample, with the identity line and Pearson r. Correlation does not imply equal absolute scores.")


# ------------------------------------------------------------------
# A5 decomposition
# ------------------------------------------------------------------
TERMS = ["category", "language", "category_x_language", "model"]
TERM_LABELS = {"category": "Category", "language": "Language", "category_x_language": "Category × language",
               "model": "Model", "other": "Other interactions"}


def fig_decomposition(T, out):
    ps = T["decomposition_per_seed"]
    fig, axes = plt.subplots(1, len(SCORES), figsize=(7.2, 2.0), sharey=True, sharex=True)
    for ax, score in zip(axes, SCORES):
        d = ps[ps.score == score]
        other = d[~d.term.isin(TERMS)].groupby("sample_idx").share.sum()
        for i, term in enumerate(TERMS + ["other"]):
            s = pct(other.values if term == "other" else d[d.term == term].share)
            # Hollow mean marker under the sample dots: the samples are often too close to separate otherwise.
            ax.errorbar([s.mean()], [i], xerr=[s.std(ddof=1)], fmt="D", mfc="none", mec=INK, color=INK, ms=6,
                        mew=0.9, elinewidth=1, capsize=2, zorder=3)
            ax.scatter(s, np.full(len(s), i), s=12, color=BLUE, alpha=0.8, lw=0, zorder=4)
        ax.set_title(textwrap.fill(SCORE_LABELS[score], 18), loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(TERMS) + 1), [TERM_LABELS[t] for t in TERMS + ["other"]])
    axes[0].invert_yaxis()
    fig.supxlabel("Share of between-cell variance in mean adherence score (%)", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, out, "fig_decomposition_per_seed", "A5 decomposition", "Main (compact) or appendix",
         "Balanced-ANOVA shares of the variance in cell mean adherence scores (equal cell weights), computed per "
         "sample, for the 3-judge mean and each judge. Blue dots: the five samples; hollow diamond and bar: "
         "mean +/- SD. Answer inversion excluded.")


# ------------------------------------------------------------------
# A1 per-score matrices
# ------------------------------------------------------------------
def fig_matrices(T, out):
    m = T["matrices_long"]
    xl, yl = [l.upper() for l in LANGUAGES], [CAT_LABELS[c] for c in CATEGORIES]
    fig, axes = plt.subplots(len(SCORES), 2, figsize=(7.0, 1.6 * len(SCORES) + 0.4), sharex=True, sharey=True)
    for r, score in enumerate(SCORES):
        for c, model in enumerate(MODELS):
            d = m[(m.score == score) & (m.model == model)]
            im = heatmap(axes[r, c], pct(cell_matrix(d, "value").to_numpy()), SEQ, 0, 100, xl, yl)
            axes[r, c].set_title(f"{SCORE_LABELS[score]} · {MODEL_LABELS[model]}", loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(f"Mean {ADH.lower()}")
    cb.outline.set_visible(False)
    save(fig, out, "fig_matrices_adherence_by_judge", "A1 per-judge matrices", "Appendix",
         "Mean adherence score (P(HELD) x 100, mean of the five sample means) per category x language, for the "
         "3-judge mean and each judge, on one 0-100 scale. Usable n per cell is in tables/matrices_long.csv.")

    dm = T["judge_difference_matrices"]
    pairs = ["gpt-gemini", "gpt-deepseek", "deepseek-gemini"]
    lim = pct(dm["diff"].abs()).max()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(7.0, 1.6 * len(pairs) + 0.4), sharex=True, sharey=True)
    for r, pair in enumerate(pairs):
        a, b = pair.split("-")
        for c, model in enumerate(MODELS):
            g = dm[(dm.pair == pair) & (dm.model == model)]
            im = heatmap(axes[r, c], pct(cell_matrix(g, "diff").to_numpy()), DIV, -lim, lim, xl, yl)
            axes[r, c].set_title(f"{SCORE_LABELS[a]} − {SCORE_LABELS[b]} · {MODEL_LABELS[model]}", loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Difference in mean adherence score (points)")
    cb.outline.set_visible(False)
    save(fig, out, "fig_judge_difference_heatmaps", "A1 per-judge matrices", "Appendix",
         "Judge-minus-judge differences in mean adherence score per cell. Red: first judge scores adherence "
         "higher; blue: lower.")


# ------------------------------------------------------------------
# A6 judge confidence
# ------------------------------------------------------------------
def fig_confidence(T, out):
    q = T["confidence_vs_within_prompt_sd_by_quintile"]
    f = T["confidence_vs_within_prompt_sd"].set_index("judge")
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(7.0, 1.9), sharey=True)
    for ax, judge in zip(axes, JUDGES):
        g = q[q.judge == judge].sort_values("conf_quintile")
        ax.plot(g.conf_quintile.astype(int), pct(g.mean_within_prompt_sd), color=BLUE, lw=2, marker="o", ms=4)
        ax.set_title(SCORE_LABELS[judge], loc="left")
        ax.text(0.97, 0.95, f"Spearman ρ = {f.loc[judge, 'spearman_conf_vs_within_prompt_sd']:.2f}",
                transform=ax.transAxes, ha="right", va="top", color=INK)
        ax.set_xticks(range(1, 6), ["1\nleast", "2", "3", "4", "5\nmost"])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("SD of adherence score\nacross samples (points)")
    fig.supxlabel("Prompt's mean confidence score, quintile within judge", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, out, "fig_within_prompt_sd_vs_confidence", "A6 judge confidence", "Main",
         "Mean within-prompt SD of the judge's adherence score across the five samples, by quintile of the judge's "
         "mean confidence score (confidence_score, 100 x (1 - normalized entropy)) for that prompt; quintiles are "
         "within judge. Spearman rho over prompts. The two are partly linked by construction (scores near 0 or "
         "100 carry high confidence and little room to vary); the confidence score is uncalibrated and is not "
         "accuracy.")

    cd = T["confidence_distribution"]
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(7.0, 2.3), sharey=True)
    for ax, judge in zip(axes, JUDGES):
        g = cd[(cd.judge == judge) & (cd.grouping == "language")].set_index("language").reindex(LANGUAGES)
        y = np.arange(len(LANGUAGES))
        colors = [ORANGE if l in FOCUS else BLUE for l in LANGUAGES]
        ax.hlines(y, pct(g.p10), pct(g["median"]), colors=colors, lw=1.4)
        ax.scatter(pct(g.p25), y, s=10, c=colors, lw=0, zorder=3)
        ax.scatter(pct(g["median"]), y, s=20, color=INK, lw=0, zorder=4)
        ax.set_title(SCORE_LABELS[judge], loc="left")
        ax.set_xlabel("Confidence score")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(LANGUAGES)), [l.upper() for l in LANGUAGES])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, out, "fig_confidence_by_language", "A6 judge confidence", "Appendix",
         "Distribution of the judge confidence score by language: line from 10th percentile to median, small dot "
         "25th percentile, black dot median. Faceted by judge; scales are not comparable across judges.")

    fig, axes = plt.subplots(1, len(JUDGES), figsize=(7.4, 2.0), sharey=True)
    for ax, judge in zip(axes, JUDGES):
        g = cd[(cd.judge == judge) & (cd.grouping == "categoryxlanguage")]
        mat = pct(cell_matrix(g, "mean").to_numpy())
        im = heatmap(ax, mat, SEQ, np.nanmin(mat), np.nanmax(mat), [l.upper() for l in LANGUAGES],
                     [CAT_LABELS[c] for c in CATEGORIES])
        ax.set_title(SCORE_LABELS[judge], loc="left")
        ax.tick_params(axis="x", rotation=90)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
        if judge == JUDGES[-1]:
            cb.set_label("mean confidence score", fontsize=6.5)
        cb.outline.set_visible(False)
    save(fig, out, "fig_confidence_heatmap", "A6 judge confidence", "Appendix",
         "Mean judge confidence score per category x language (0 = split 50/50 between HELD and VIOLATED, "
         "100 = fully sure), one panel and color scale per judge.")


# ------------------------------------------------------------------
# A7 unscored responses
# ------------------------------------------------------------------
REASONS = [("filtered", BLUE), ("tied", ORANGE), ("invalid", AQUA), ("uncertain", YELLOW), ("failure", MAGENTA)]


def fig_missing(T, out, labels):
    r = T["missing_reasons_by_pressure"]
    n_per_level = labels.groupby("pressure_level").size().reindex(PRESSURES)
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(7.0, 2.0), sharey=True)
    for ax, judge in zip(axes, JUDGES):
        g = r[(r.judge == judge) & (r.reason_code != "ok")].groupby("reason_code")[PRESSURES].sum()
        bottom = np.zeros(len(PRESSURES))
        for reason, color in REASONS:
            if reason not in g.index:
                continue
            h = pct(g.loc[reason].to_numpy() / n_per_level.to_numpy())
            ax.bar(PRESSURES, h, bottom=bottom, color=color, width=0.7, edgecolor="white", linewidth=0.8, label=reason)
            bottom += h
        ax.set_title(SCORE_LABELS[judge], loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Pressure level")
    axes[0].set_ylabel(f"Unscored responses (%)\nn = {n_per_level.iloc[0]:,} per level")
    handles, lbls = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in lbls:
                handles.append(h)
                lbls.append(l)
    fig.legend(handles, lbls, loc="upper center", ncol=len(lbls), bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, out, "fig_missing_by_pressure", "A7 missing-score rate", "Main (small figure)",
         "Percentage of responses without a usable judge score, by pressure level and judge, split by reason "
         "(provider content filter, exact probability tie, invalid output, uncertain request, failure). "
         "Unscored responses are excluded, never scored as 0.")

    fig, axes = plt.subplots(1, len(JUDGES), figsize=(7.2, 2.0), sharey=True)
    mats = {j: pct(labels.assign(m=labels[f"{j}_status"].ne("ok")).pipe(cell_matrix_from_labels).to_numpy())
            for j in JUDGES}
    vmax = max(np.nanmax(v) for v in mats.values())
    for ax, judge in zip(axes, JUDGES):
        im = heatmap(ax, mats[judge], SEQ, 0, vmax, [l.upper() for l in LANGUAGES], [CAT_LABELS[c] for c in CATEGORIES])
        ax.set_title(SCORE_LABELS[judge], loc="left")
        ax.tick_params(axis="x", rotation=90)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Unscored (%)")
    cb.outline.set_visible(False)
    save(fig, out, "fig_missing_heatmap", "A7 missing-score rate", "Appendix",
         "Percentage of responses without a usable score per category x language, pooled over models and samples.")


def cell_matrix_from_labels(df):
    return df.groupby(["category", "language"]).m.mean().unstack().reindex(index=CATEGORIES, columns=LANGUAGES)


# ------------------------------------------------------------------
# A8 pressure sensitivity
# ------------------------------------------------------------------
def fig_pressure(T, out):
    p = T["pressure_profiles"]
    p = p[p.weighting == "pooled"]
    fig, axes = plt.subplots(len(SCORES), 2, figsize=(5.6, 1.45 * len(SCORES) + 0.3), sharex=True)
    x = np.arange(len(PRESSURES))
    for r, score in enumerate(SCORES):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            g = p[(p.score == score) & (p.model == model)].set_index("variant")
            if "bound_missing_as_0" in g.index:
                ax.fill_between(x, pct(g.loc["bound_missing_as_0", PRESSURES]),
                                pct(g.loc["bound_missing_as_100", PRESSURES]), color=BLUE, alpha=0.15, lw=0,
                                label="bounds (unscored = 0 … 100)")
            ax.plot(x, pct(g.loc["available_case", PRESSURES]), color=BLUE, lw=2, marker="o", ms=3,
                    label="available-case (all scored responses)")
            if "complete_case" in g.index:
                ax.plot(x, pct(g.loc["complete_case", PRESSURES]), color=ORANGE, lw=2, marker="o", ms=3,
                        label="complete-case (scored by all 3 judges)")
            ax.set_xticks(x, PRESSURES)
            ax.set_title(f"{SCORE_LABELS[score]} · {MODEL_LABELS[model]}", loc="left")
            if c == 0:
                ax.set_ylabel(ADH)
    handles, lbls = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, out, "fig_pressure_sensitivity", "A8 pressure sensitivity", "Main or appendix",
         "Mean adherence score by pressure level (pooled over responses, mean of the five sample means). For each "
         "judge: all responses that judge scored (blue), only responses all three judges scored (orange), and the "
         "extreme band if every unscored response had scored 0 or 100. The 3-judge mean is defined only on "
         "responses all three judges scored, so it has a single line. Orderings with and without L3 are in "
         "tables/pressure_orderings.csv.")


# ------------------------------------------------------------------
# A9 within-prompt variability per language
# ------------------------------------------------------------------
def fig_within_prompt(T, out):
    w = T["within_prompt_sd"]
    w = w[w.grouping == "categoryxlanguage"]
    vmax = pct(w.mean_sd).max()
    fig, axes = plt.subplots(1, len(SCORES), figsize=(7.4, 2.0), sharey=True)
    for ax, score in zip(axes, SCORES):
        mat = pct(cell_matrix(w[w.score == score], "mean_sd").to_numpy())
        im = heatmap(ax, mat, SEQ, 0, vmax, [l.upper() for l in LANGUAGES], [CAT_LABELS[c] for c in CATEGORIES])
        ax.set_title(SCORE_LABELS[score], loc="left")
        ax.tick_params(axis="x", rotation=90)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Mean within-prompt SD (points)")
    cb.outline.set_visible(False)
    save(fig, out, "fig_within_prompt_sd_heatmap", "A9 within-prompt variability", "Appendix",
         "How much a single prompt's adherence score changes across its five samples: the SD over samples, "
         "averaged over the prompts in each category x language cell (both models pooled). One shared scale. "
         "Prompts need at least 4 scored samples (dropped counts in tables/within_prompt_dropped_prompts.csv).")


# ------------------------------------------------------------------
# A10 variance components per language
# ------------------------------------------------------------------
def fig_variance(T, out):
    vc = T["variance_components"]
    langs = LANGUAGES + ["all"]
    comps = [("var_item", "between items", BLUE), ("var_prompt", "prompt within item", ORANGE),
             ("var_residual", "between samples (residual)", AQUA)]
    fig, axes = plt.subplots(1, len(SCORES), figsize=(7.2, 2.6), sharey=True)
    for ax, score in zip(axes, SCORES):
        g = vc[vc.score == score].set_index("language").reindex(langs)
        tot = g[["var_item", "var_prompt", "var_residual"]].sum(axis=1)
        left = np.zeros(len(langs))
        for col, lbl, color in comps:
            w = pct(g[col] / tot)
            ax.barh(range(len(langs)), w, left=left, color=color, height=0.7, edgecolor="white", linewidth=0.8,
                    label=lbl)
            left += w
        ax.set_title(SCORE_LABELS[score], loc="left")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(langs)), [l.upper() if l != "all" else "All" for l in langs])
    axes[0].invert_yaxis()
    fig.supxlabel("Share of variance in adherence score (%)", fontsize=7.5, color=INK_2)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    save(fig, out, "fig_variance_components", "A10 variance components", "Appendix",
         "Variance in the adherence score per language, after removing category x model x pressure means, split "
         "into differences between base items, between prompts of the same item, and between the five samples of "
         "the same prompt (nested ANOVA, method of moments; models pooled). 'All' pools the ten languages.")


# ------------------------------------------------------------------
# A11 cluster bootstrap
# ------------------------------------------------------------------
def fig_bootstrap(T, out):
    b = T["bootstrap"]
    b = b[b.score == PRIMARY]
    seeds = T["headline_gap_summary"]
    seeds = seeds[seeds.score == PRIMARY].set_index(["language", "model"])
    rows = [(l, m) for l in FOCUS for m in MODELS]
    intervals = [("seed range (min–max of 5 samples)", ORANGE, None),
                 ("bootstrap 95% CI · 66 items", BLUE, "item"),
                 ("bootstrap 95% CI · 22 rule tasks", AQUA, "task")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    for i, (lang, model) in enumerate(rows):
        for k, (lbl, color, unit) in enumerate(intervals):
            y = i + (k - 1) * 0.22
            if unit is None:
                s = seeds.loc[(lang, model)]
                lo, hi, pt = s["min"], s["max"], s["mean"]
            else:
                g = b[(b.statistic == "gap") & (b.cluster_unit == unit) & (b.language_or_term == lang) & (b.model == model)].iloc[0]
                lo, hi, pt = g.ci_low, g.ci_high, g.point
            ax.plot(pct([lo, hi]), [y, y], color=color, lw=1.8, solid_capstyle="round", label=lbl if i == 0 else None)
            ax.scatter(pct([pt]), [y], s=12, color=INK, zorder=3, lw=0)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(len(rows)), [f"{l.upper()} · {MODEL_LABELS[m]}" for l, m in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Withholding − constructed gap (points)")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    dec = T["decomposition_summary"]
    dec = dec[dec.score == PRIMARY].set_index("term")
    for i, term in enumerate(TERMS):
        for k, (lbl, color, unit) in enumerate(intervals):
            y = i + (k - 1) * 0.22
            if unit is None:
                lo, hi, pt = dec.loc[term, "min"], dec.loc[term, "max"], dec.loc[term, "mean"]
            else:
                g = b[(b.statistic == "share") & (b.cluster_unit == unit) & (b.language_or_term == term)].iloc[0]
                lo, hi, pt = g.ci_low, g.ci_high, g.point
            ax.plot(pct([lo, hi]), [y, y], color=color, lw=1.8, solid_capstyle="round")
            ax.scatter(pct([pt]), [y], s=12, color=INK, zorder=3, lw=0)
    ax.set_yticks(range(len(TERMS)), [TERM_LABELS[t] for t in TERMS])
    ax.invert_yaxis()
    ax.set_xlabel("Share of between-cell variance (%)")
    ax.grid(axis="y", visible=False)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, "fig_bootstrap_vs_seed_range", "A11 cluster bootstrap", "Appendix",
         "Generation noise (range across the five samples) next to item-level noise (95% cluster-bootstrap "
         "intervals, resampling base items or whole rule tasks within category), 3-judge mean adherence score. "
         "The two intervals measure different sources of variability.")


def write_index(out):
    lines = ["# Sampling-stability figures", "",
             "Generated by `canonical/analysis/sampling_stability/figures.py` from the tables in the sibling "
             "analysis folder. Each figure is saved as PDF (paper) and PNG. Scope: the 5-sample run only; "
             "answer inversion excluded (not judged in this run). Only numeric judge scores are used: the adherence "
             "score (P(HELD) x 100) and the confidence score; binary verdicts are not used. Main-text figures use "
             "the 3-judge mean adherence score.", "",
             "| Figure | Analysis | Suggested placement | Caption draft |", "|---|---|---|---|"]
    lines += [f"| `{n}` | {a} | {p} | {c} |" for n, a, p, c in FIGURES]
    with open(os.path.join(out, "INDEX.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="run.py --out folder (with tables/ and data/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--path-in-repo", default="sampling-stability-scores-figures-2026-10-28")
    args = ap.parse_args()

    style()
    os.makedirs(args.out, exist_ok=True)
    tdir = os.path.join(args.results, "tables")
    T = {f[:-4]: pd.read_csv(os.path.join(tdir, f)) for f in os.listdir(tdir) if f.endswith(".csv")}
    labels = pd.read_parquet(os.path.join(args.results, "data", "labels_long.parquet"))

    fig_headline_gap(T, args.out, FOCUS, "fig_headline_gap_per_seed", "Main")
    fig_headline_gap(T, args.out, LANGUAGES, "fig_headline_gap_per_seed_all_languages", "Appendix")
    fig_cell_sd(T, args.out)
    fig_score_pointrange(T, args.out)
    fig_cross_model(T, args.out)
    fig_decomposition(T, args.out)
    fig_matrices(T, args.out)
    fig_confidence(T, args.out)
    fig_missing(T, args.out, labels)
    fig_pressure(T, args.out)
    fig_within_prompt(T, args.out)
    fig_variance(T, args.out)
    fig_bootstrap(T, args.out)
    write_index(args.out)
    print(f"wrote {len(FIGURES)} figures to {args.out}")

    if args.upload:
        HfApi().upload_folder(repo_id=UPLOAD_REPO, repo_type="dataset", folder_path=args.out,
                              path_in_repo=args.path_in_repo,
                              commit_message="Sampling-stability figures, numeric scores only (camera-ready)")
        print(f"uploaded to {UPLOAD_REPO}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
