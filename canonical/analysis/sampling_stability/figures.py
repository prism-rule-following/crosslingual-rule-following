"""Figures for the sampling-stability analysis, drawn from run.py's output folder.

RUN from the repo root after run.py (needs seaborn and SciencePlots):
    .venv/bin/python -m canonical.analysis.sampling_stability.figures \
        --results <run.py --out dir> --out <figure dir> [--upload]

Writes <out>/<name>.pdf (for the paper; vector, fonts embedded, editable in
Inkscape) and <out>/<name>.png, plus INDEX.md listing each figure's analysis,
suggested placement and a draft caption. --upload pushes the folder to
crv-analysis under its own path.

Conventions:
  - Only numeric judge scores are shown. The adherence score is P(HELD) x 100;
    main-text figures use the 3-judge mean (responses all three judges scored),
    with each judge's own score alongside.
  - Every error bar or band is labelled in the figure. Sampling uncertainty is
    always the variation across the five repeated samples (SD or min-max);
    with five samples these spreads are themselves imprecise.
  - One palette everywhere (Okabe-Ito, colour-blind safe): vermilion = Yoruba
    and Igbo, blue = the other eight languages, greys for analysis variants,
    the remaining Okabe-Ito hues for stacked components.
"""

import argparse
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401  (registers the "science" styles)
import seaborn as sns  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from canonical.analysis.sampling_stability.data import CATEGORIES, CONSTRUCTED, JUDGES, LANGUAGES, PRESSURES  # noqa: E402

UPLOAD_REPO = "crosslingual-rule-following/crv-analysis"

# Okabe-Ito palette, one role per colour across all figures.
FOCUS_C, OTHER_C = "#D55E00", "#0072B2"  # Yoruba/Igbo, other eight languages
COMPONENTS = ["#56B4E9", "#E69F00", "#009E73", "#F0E442", "#CC79A7"]  # stacked parts
INK, GREY, LIGHT, GRIDC = "#1a1a1a", "#6e6e6e", "#c8c8c8", "#e8e8e8"
SEQ = sns.color_palette("Blues", as_cmap=True)
DIV = sns.color_palette("vlag", as_cmap=True)

CAT_LABELS = {"no_verdict": "No verdict", "no_pii": "No PII", "no_dosage": "No dosage", "scope_lock": "Scope lock",
              "mandatory_referral": "Referral", "refuse_with_reason": "Refusal w/ reason"}
MODEL_LABELS = {"qwen": "Qwen3-8B", "llama": "Llama-3.1-8B"}
MODELS = ["qwen", "llama"]
SCORE_LABELS = {"mean": "3-judge mean", "gpt": "GPT-5.4-mini", "gemini": "Gemini-2.5-Flash",
                "deepseek": "DeepSeek-V4-Pro"}
SCORES = ["mean"] + JUDGES
FOCUS = ["yo", "ig"]
PRIMARY = "mean"
N_SEEDS = 5
FULL_W = 7.0  # inches, two-column width

FIGURES = []  # (name, analysis, placement, caption)


def style():
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False, "xtick.minor.visible": False, "ytick.minor.visible": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": 1.5, "errorbar.capsize": 2,
        "figure.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out, name, analysis, placement, caption):
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, analysis, placement, caption))


def pct(x):
    return 100 * np.asarray(x, dtype=float)


def lang_color(lang):
    return FOCUS_C if lang in FOCUS else OTHER_C


def language_handles():
    return [Line2D([], [], color=FOCUS_C, marker="o", lw=0, label="Yoruba, Igbo"),
            Line2D([], [], color=OTHER_C, marker="o", lw=0, label="other 8 languages")]


def cell_frame(t, value, scale=100):
    """category x language DataFrame (display labels) from a long table with CELL columns."""
    m = t.pivot_table(index="category", columns="language", values=value).reindex(index=CATEGORIES, columns=LANGUAGES)
    m = m * scale
    m.index = [CAT_LABELS[c] for c in m.index]
    m.columns = [l.upper() for l in m.columns]
    return m


def mark_extreme(mat, which):
    """Annotation array that labels only the single max / min / largest-magnitude cell."""
    v = mat.to_numpy()
    flat = {"max": np.nanargmax(v), "min": np.nanargmin(v), "absmax": np.nanargmax(np.abs(v))}[which]
    annot = np.full(v.shape, "", dtype=object)
    r, c = np.unravel_index(flat, v.shape)
    annot[r, c] = f"{v[r, c]:.0f}" if abs(v[r, c]) >= 10 else f"{v[r, c]:.1f}"
    return annot


def heatmap(ax, mat, cmap, vmin, vmax, annot=None, center=None, show_y=True):
    sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=center, cbar=False, linewidths=0.5,
                linecolor="white", annot=annot if annot is not None else False, fmt="",
                annot_kws={"fontsize": 8, "fontweight": "bold"}, square=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    if not show_y:
        ax.set_yticklabels([])
    return ax.collections[0]


def colorbar(fig, mappable, axes, label):
    cb = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(label)
    cb.outline.set_linewidth(0.6)
    cb.ax.minorticks_off()
    return cb


# ------------------------------------------------------------------
# A2 headline gap per seed
# ------------------------------------------------------------------
def fig_headline_gap(T, out, languages, name, placement):
    per_seed = T["headline_gap_per_seed"]
    rows = [(l, m) for l in languages for m in MODELS]
    fig, axes = plt.subplots(1, len(SCORES), figsize=(FULL_W, 0.3 * len(rows) + 1.3), sharey=True, sharex=True)
    for ax, score in zip(axes, SCORES):
        d = per_seed[per_seed.score == score]
        right = blended_transform_factory(ax.transAxes, ax.transData)
        for i, (lang, model) in enumerate(rows):
            g = pct(d[(d.language == lang) & (d.model == model)].gap)
            c = lang_color(lang)
            ax.plot([g.min(), g.max()], [i, i], color=c, lw=2.4, alpha=0.35, solid_capstyle="round")
            ax.scatter(g, np.full(len(g), i), s=11, color=c, lw=0, zorder=4)
            ax.scatter([g.mean()], [i], s=30, marker="D", facecolors="white", edgecolors=INK, lw=0.9, zorder=3)
            # Direct label: the spread across samples, which is too narrow to read off the axis.
            ax.text(1.03, i, f"{g.max() - g.min():.1f}", transform=right, va="center", fontsize=7.5, color=GREY)
        ax.margins(x=0.12)
        ax.axvline(0, color=INK, lw=0.6, zorder=1)
        ax.set_title(SCORE_LABELS[score])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [f"{l.upper()} · {MODEL_LABELS[m]}" for l, m in rows])
    axes[0].invert_yaxis()
    fig.supxlabel("Withholding − constructed gap in mean adherence score (points)", fontsize=10)
    handles = [Line2D([], [], color=GREY, marker="o", ms=3.5, lw=0, label=f"each sample (n = {N_SEEDS})"),
               Line2D([], [], color=GREY, lw=2.4, alpha=0.35, label="min–max across samples"),
               Line2D([], [], color=INK, marker="D", mfc="white", ms=5, lw=0, label="mean of samples")]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3)
    fig.text(1.0, 1.005, "grey number: min–max range (points)", ha="right", va="bottom", fontsize=8, color=GREY)
    save(fig, out, name, "A2 headline gap per seed", placement,
         "Mean adherence score (P(HELD) x 100) of the withholding categories (no verdict, no PII, no dosage, scope "
         "lock) minus the constructed categories (referral, refusal with reason), computed separately in each of "
         "the five repeated samples. Dots: individual samples; bar and grey number: min-max range across samples; "
         "diamond: mean. Left: 3-judge mean score (responses all three judges scored); right: each judge's own "
         "score.")


# ------------------------------------------------------------------
# A3 cell stability
# ------------------------------------------------------------------
def fig_cell_sd(T, out):
    cs = T["cell_stability"]
    d = cs[cs.score == PRIMARY]
    mats = {m: cell_frame(d[d.model == m], "sd") for m in MODELS}
    vmax = max(np.nanmax(v.to_numpy()) for v in mats.values())
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    for k, (ax, model) in enumerate(zip(axes, MODELS)):
        im = heatmap(ax, mats[model], SEQ, 0, vmax, annot=mark_extreme(mats[model], "max"), show_y=k == 0)
        ax.set_title(MODEL_LABELS[model], loc="left")
    colorbar(fig, im, axes, f"SD across {N_SEEDS} samples (points)")
    save(fig, out, "fig_cell_stability_sd_heatmap", "A3 cell stability", "Main",
         "Standard deviation, across the five repeated samples, of each category x language x model cell's mean "
         "adherence score (3-judge mean, points); the largest value per panel is labelled. One shared scale. With "
         "five samples each SD is itself imprecise; read the pattern rather than single cells.")

    top = T["cell_stability_top"]
    t = top[top.score == PRIMARY].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, r in t.iterrows():
        c = lang_color(r.language)
        ax.plot(pct([r["min"], r["max"]]), [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter(pct([r["mean"]]), [i], s=22, color=INK, zorder=3, lw=0)
        ax.text(pct(r["max"]) + 0.6, i, f"{100 * (r['max'] - r['min']):.1f}", va="center", fontsize=7.5, color=GREY)
    ax.set_yticks(range(len(t)), [f"{CAT_LABELS[c]} · {l.upper()} · {MODEL_LABELS[m]}"
                                  for c, l, m in zip(t.category, t.language, t.model)])
    ax.invert_yaxis()
    ax.set_xlabel("Mean adherence score")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.08)
    ax.legend(handles=language_handles() + [
        Line2D([], [], color=GREY, lw=2.2, label=f"min–max of {N_SEEDS} samples"),
        Line2D([], [], color=INK, marker="o", lw=0, ms=4, label="mean")],
        loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    save(fig, out, "fig_cell_stability_top10", "A3 cell stability", "Appendix",
         "The ten cells whose mean adherence score (3-judge mean) varies most across the five repeated samples. "
         "Bar: min-max of the five sample means (range printed at right); dot: mean.")


# ------------------------------------------------------------------
# Adherence-score point-range (all cells)
# ------------------------------------------------------------------
def fig_score_pointrange(T, out):
    cs = T["cell_stability"]
    d = cs[cs.score == PRIMARY]
    fig, axes = plt.subplots(2, len(CATEGORIES), figsize=(FULL_W + 0.4, 4.2), sharey=True, sharex=True)
    x = np.arange(len(LANGUAGES))
    focus = np.isin(LANGUAGES, FOCUS)
    for r, model in enumerate(MODELS):
        for c, cat in enumerate(CATEGORIES):
            ax = axes[r, c]
            g = d[(d.model == model) & (d.category == cat)].set_index("language").reindex(LANGUAGES)
            for mask, color in [(~focus, OTHER_C), (focus, FOCUS_C)]:
                ax.errorbar(x[mask], pct(g["mean"])[mask], yerr=pct(g["sd"])[mask], fmt="o", ms=3.2, color=color,
                            elinewidth=1, capsize=1.5)
            ax.set_xticks(x, [l.upper() for l in LANGUAGES], rotation=90, fontsize=7)
            ax.set_ylim(-3, 103)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(CAT_LABELS[cat], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nadherence score")
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    fig.legend(handles=language_handles(), loc="lower left", bbox_to_anchor=(0.07, 1.0), ncol=2)
    fig.text(1.0, 1.005, f"error bars: ±1 SD across {N_SEEDS} samples", ha="right", va="bottom", fontsize=8.5)
    save(fig, out, "fig_adherence_pointrange", "A3 cell stability", "Appendix",
         "Mean adherence score (3-judge mean) per category, language and model; point: mean of the five repeated "
         "samples, error bars: +/-1 SD across those samples (most are smaller than the marker).")


# ------------------------------------------------------------------
# A4 cross-model replication
# ------------------------------------------------------------------
def fig_cross_model(T, out):
    cs = T["cell_stability"]
    d = cs[cs.score == PRIMARY]
    p = d.pivot_table(index=["category", "language"], columns="model", values=["mean", "sd"])
    cm = T["cross_model_per_seed"]
    r = cm[cm.score == PRIMARY].pearson
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 100], [0, 100], color=GREY, lw=0.8, ls="--", zorder=1)
    for is_focus, color in [(False, OTHER_C), (True, FOCUS_C)]:
        q = p[p.index.get_level_values("language").isin(FOCUS) == is_focus]
        ax.errorbar(pct(q[("mean", "llama")]), pct(q[("mean", "qwen")]), xerr=pct(q[("sd", "llama")]),
                    yerr=pct(q[("sd", "qwen")]), fmt="o", ms=3.5, color=color, elinewidth=0.8, capsize=0,
                    alpha=0.9)
    # One direct label for the cluster of Yoruba/Igbo constructed-rule cells.
    idx = [(c, l) for c in CONSTRUCTED for l in FOCUS]
    xx, yy = pct(p.loc[idx, ("mean", "llama")].mean()), pct(p.loc[idx, ("mean", "qwen")].mean())
    ax.annotate("YO, IG · Referral,\nRefusal w/ reason", (xx + 4, yy - 2), xytext=(38, 6), textcoords="data",
                fontsize=7.5, color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.text(0.04, 0.96, f"Pearson r = {r.min():.2f}–{r.max():.2f}\nacross the {N_SEEDS} samples",
            transform=ax.transAxes, va="top", fontsize=8.5)
    ax.set(xlim=(-3, 103), ylim=(-3, 103), xlabel="Llama-3.1-8B adherence score", ylabel="Qwen3-8B adherence score")
    ax.set_aspect("equal")
    ax.legend(handles=language_handles(), loc="center left", bbox_to_anchor=(0.0, 0.62))
    save(fig, out, "fig_cross_model", "A4 cross-model replication", "Appendix",
         "Qwen vs Llama mean adherence score (3-judge mean) for each category x language cell. Point: mean of the "
         "five repeated samples; error bars: +/-1 SD across samples on each axis; dashed line: equal scores. The "
         "correlation is computed within each sample; its range across samples is shown.")


# ------------------------------------------------------------------
# A5 decomposition
# ------------------------------------------------------------------
TERMS = ["category", "language", "category_x_language", "model"]
TERM_LABELS = {"category": "Category", "language": "Language", "category_x_language": "Category × language",
               "model": "Model", "other": "Other interactions"}


def fig_decomposition(T, out):
    ps = T["decomposition_per_seed"]
    fig, axes = plt.subplots(1, len(SCORES), figsize=(FULL_W + 0.4, 2.4), sharey=True, sharex=True)
    for ax, score in zip(axes, SCORES):
        d = ps[ps.score == score]
        other = d[~d.term.isin(TERMS)].groupby("sample_idx").share.sum()
        for i, term in enumerate(TERMS + ["other"]):
            s = pct(other.values if term == "other" else d[d.term == term].share)
            ax.errorbar([s.mean()], [i], xerr=[s.std(ddof=1)], fmt="D", mfc="white", mec=INK, color=INK, ms=5,
                        mew=0.9, elinewidth=1.2, capsize=2.5, zorder=3)
            ax.scatter(s, np.full(len(s), i), s=10, color=OTHER_C, lw=0, zorder=4)
            if score == PRIMARY:
                ax.text(s.mean(), i - 0.32, f"{s.mean():.0f}%", ha="center", fontsize=7.5, color=GREY)
        ax.set_title(textwrap.fill(SCORE_LABELS[score], 18))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(TERMS) + 1), [TERM_LABELS[t] for t in TERMS + ["other"]])
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(TERMS) + 0.5, -0.8)
    fig.supxlabel("Share of between-cell variance in mean adherence score (%)", fontsize=10)
    fig.tight_layout()
    fig.legend(handles=[Line2D([], [], color=OTHER_C, marker="o", ms=3.5, lw=0, label=f"each sample (n = {N_SEEDS})"),
                        Line2D([], [], color=INK, marker="D", mfc="white", ms=5, label="mean ± 1 SD across samples")],
               loc="lower left", bbox_to_anchor=(0.12, 1.0), ncol=2)
    save(fig, out, "fig_decomposition_per_seed", "A5 decomposition", "Main (compact) or appendix",
         "Balanced-ANOVA shares of the variance in cell mean adherence scores (equal cell weights), computed "
         "separately in each repeated sample, for the 3-judge mean and each judge. Dots: individual samples; "
         "diamond and bar: mean +/-1 SD across samples (means printed for the 3-judge mean).")


# ------------------------------------------------------------------
# A1 per-score matrices
# ------------------------------------------------------------------
def fig_matrices(T, out):
    m = T["matrices_long"]
    fig, axes = plt.subplots(len(SCORES), 2, figsize=(FULL_W, 1.75 * len(SCORES) + 0.3), sharex=True)
    for r, score in enumerate(SCORES):
        for c, model in enumerate(MODELS):
            mat = cell_frame(m[(m.score == score) & (m.model == model)], "value")
            im = heatmap(axes[r, c], mat, SEQ, 0, 100, annot=mark_extreme(mat, "min"), show_y=c == 0)
            axes[r, c].set_title(f"{SCORE_LABELS[score]} · {MODEL_LABELS[model]}", loc="left", fontsize=9.5)
    colorbar(fig, im, axes, "Mean adherence score")
    save(fig, out, "fig_matrices_adherence_by_judge", "A1 per-judge matrices", "Appendix",
         "Mean adherence score (P(HELD) x 100, mean of the five repeated-sample means) per category x language, "
         "for the 3-judge mean and each judge, on one 0-100 scale; the lowest cell per panel is labelled. Usable n "
         "per cell is in tables/matrices_long.csv.")

    dm = T["judge_difference_matrices"]
    pairs = ["gpt-gemini", "gpt-deepseek", "deepseek-gemini"]
    lim = pct(dm["diff"].abs()).max()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(FULL_W, 1.75 * len(pairs) + 0.3), sharex=True)
    for r, pair in enumerate(pairs):
        a, b = pair.split("-")
        for c, model in enumerate(MODELS):
            mat = cell_frame(dm[(dm.pair == pair) & (dm.model == model)], "diff")
            im = heatmap(axes[r, c], mat, DIV, -lim, lim, annot=mark_extreme(mat, "absmax"), center=0, show_y=c == 0)
            axes[r, c].set_title(f"{SCORE_LABELS[a]} − {SCORE_LABELS[b]} · {MODEL_LABELS[model]}", loc="left",
                                 fontsize=9.5)
    colorbar(fig, im, axes, "Difference in mean adherence score (points)")
    save(fig, out, "fig_judge_difference_heatmaps", "A1 per-judge matrices", "Appendix",
         "Judge-minus-judge differences in mean adherence score per cell; the largest difference per panel is "
         "labelled. Red: the first judge scores adherence higher; blue: lower.")


# ------------------------------------------------------------------
# A6 judge confidence
# ------------------------------------------------------------------
def fig_confidence(T, out):
    q = T["confidence_vs_within_prompt_sd_by_quintile"]
    f = T["confidence_vs_within_prompt_sd"].set_index("judge")
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(FULL_W, 2.3), sharey=True)
    for ax, judge in zip(axes, JUDGES):
        g = q[q.judge == judge].sort_values("conf_quintile")
        y = pct(g.mean_within_prompt_sd)
        ax.plot(g.conf_quintile.astype(int), y, color=INK, marker="o", ms=4)
        ax.text(1, y[0] + 1.5, f"{y[0]:.0f}", ha="center", fontsize=7.5, color=GREY)
        ax.text(5, y[-1] + 1.5, f"{y[-1]:.0f}", ha="center", fontsize=7.5, color=GREY)
        ax.set_title(SCORE_LABELS[judge])
        ax.text(0.97, 0.95, f"Spearman ρ = {f.loc[judge, 'spearman_conf_vs_within_prompt_sd']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
        ax.set_xticks(range(1, 6), ["1\nleast", "2", "3", "4", "5\nmost"])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(f"SD across {N_SEEDS} samples\n(points, mean over prompts)")
    fig.supxlabel("Prompt's mean confidence score, quintile within judge", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig_within_prompt_sd_vs_confidence", "A6 judge confidence", "Main",
         "For each prompt, the SD of the judge's adherence score across the five repeated samples, averaged within "
         "quintiles of the judge's mean confidence score (confidence_score, 100 x (1 - normalized entropy)) for that "
         "prompt; quintiles are within judge, end values labelled. Spearman rho over prompts. The two are partly "
         "linked by construction (scores near 0 or 100 carry high confidence and little room to vary); the "
         "confidence score is uncalibrated and is not accuracy.")

    cd = T["confidence_distribution"]
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(FULL_W, 2.7), sharey=True)
    for ax, judge in zip(axes, JUDGES):
        g = cd[(cd.judge == judge) & (cd.grouping == "language")].set_index("language").reindex(LANGUAGES)
        y = np.arange(len(LANGUAGES))
        colors = [lang_color(l) for l in LANGUAGES]
        ax.hlines(y, pct(g.p10), pct(g["median"]), colors=colors, lw=2)
        ax.scatter(pct(g.p25), y, s=14, facecolors="white", edgecolors=colors, lw=1, zorder=3)
        ax.scatter(pct(g["median"]), y, s=20, color=INK, lw=0, zorder=4)
        ax.set_title(SCORE_LABELS[judge])
        ax.set_xlabel("Confidence score")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(LANGUAGES)), [l.upper() for l in LANGUAGES])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.legend(handles=[Line2D([], [], color=GREY, lw=2, label="10th percentile → median"),
                        Line2D([], [], color=GREY, marker="o", mfc="white", lw=0, label="25th percentile"),
                        Line2D([], [], color=INK, marker="o", lw=0, label="median")] + language_handles(),
               loc="lower left", bbox_to_anchor=(0.06, 1.0), ncol=5)
    save(fig, out, "fig_confidence_by_language", "A6 judge confidence", "Appendix",
         "Distribution of the judge confidence score by language over all scored responses: line from the 10th "
         "percentile to the median, open circle 25th percentile, filled circle median. Faceted by judge; x scales "
         "differ and are not comparable across judges.")

    fig, axes = plt.subplots(1, len(JUDGES), figsize=(FULL_W + 0.6, 2.4))
    for k, (ax, judge) in enumerate(zip(axes, JUDGES)):
        mat = cell_frame(cd[(cd.judge == judge) & (cd.grouping == "categoryxlanguage")], "mean")
        im = heatmap(ax, mat, SEQ, np.nanmin(mat.to_numpy()), np.nanmax(mat.to_numpy()),
                     annot=mark_extreme(mat, "min"), show_y=k == 0)
        ax.set_title(SCORE_LABELS[judge], loc="left")
        ax.tick_params(axis="x", labelsize=7.5, rotation=90)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.outline.set_linewidth(0.6)
        cb.ax.minorticks_off()
        if judge == JUDGES[-1]:
            cb.set_label("Mean confidence score")
    save(fig, out, "fig_confidence_heatmap", "A6 judge confidence", "Appendix",
         "Mean judge confidence score per category x language (0 = split 50/50 between HELD and VIOLATED, "
         "100 = fully sure); the least confident cell per panel is labelled. One panel and colour scale per judge.")


# ------------------------------------------------------------------
# A7 unscored responses
# ------------------------------------------------------------------
REASONS = list(zip(["filtered", "tied", "invalid", "uncertain", "failure"], COMPONENTS))
REASON_LABELS = {"filtered": "content filter", "tied": "exact tie", "invalid": "invalid output",
                 "uncertain": "uncertain request", "failure": "request failure"}


def fig_missing(T, out, labels):
    r = T["missing_reasons_by_pressure"]
    n_per_level = labels.groupby("pressure_level").size().reindex(PRESSURES)
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(FULL_W, 2.4), sharey=True)
    present = set()
    for ax, judge in zip(axes, JUDGES):
        g = r[(r.judge == judge) & (r.reason_code != "ok")].groupby("reason_code")[PRESSURES].sum()
        bottom = np.zeros(len(PRESSURES))
        for reason, color in REASONS:
            if reason not in g.index:
                continue
            present.add(reason)
            h = pct(g.loc[reason].to_numpy() / n_per_level.to_numpy())
            ax.bar(PRESSURES, h, bottom=bottom, color=color, width=0.7, edgecolor="white", linewidth=0.6)
            bottom += h
        for i, tot in enumerate(bottom):
            ax.text(i, tot + 0.4, f"{tot:.1f}", ha="center", va="bottom", fontsize=7, color=GREY)
        ax.set_title(SCORE_LABELS[judge])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Pressure level")
    axes[0].set_ylabel(f"Unscored responses (%)\nn = {n_per_level.iloc[0]:,} per level")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=REASON_LABELS[k]) for k, c in REASONS if k in present]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    save(fig, out, "fig_missing_by_pressure", "A7 missing-score rate", "Main (small figure)",
         "Percentage of responses without a usable judge score, by pressure level and judge, split by reason "
         "(provider content filter, exact probability tie, invalid output, uncertain request, failure); totals "
         "printed above bars. Unscored responses are excluded, never scored as 0.")

    mats = {j: cell_frame(labels.assign(m=labels[f"{j}_status"].ne("ok")), "m") for j in JUDGES}
    vmax = max(np.nanmax(v.to_numpy()) for v in mats.values())
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(FULL_W + 0.4, 2.4))
    for k, (ax, judge) in enumerate(zip(axes, JUDGES)):
        im = heatmap(ax, mats[judge], SEQ, 0, vmax, annot=mark_extreme(mats[judge], "max"), show_y=k == 0)
        ax.set_title(SCORE_LABELS[judge], loc="left")
        ax.tick_params(axis="x", labelsize=7.5, rotation=90)
    colorbar(fig, im, axes, "Unscored (%)")
    save(fig, out, "fig_missing_heatmap", "A7 missing-score rate", "Appendix",
         "Percentage of responses without a usable score per category x language, pooled over models and samples; "
         "the highest cell per panel is labelled. One shared scale.")


# ------------------------------------------------------------------
# A8 pressure sensitivity
# ------------------------------------------------------------------
def fig_pressure(T, out):
    p = T["pressure_profiles"]
    p = p[p.weighting == "pooled"]
    sd_cols = [f"{lvl}_sd" for lvl in PRESSURES]
    fig, axes = plt.subplots(len(SCORES), 2, figsize=(5.8, 1.6 * len(SCORES) + 0.4), sharex=True)
    x = np.arange(len(PRESSURES))
    for r, score in enumerate(SCORES):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            g = p[(p.score == score) & (p.model == model)].set_index("variant")
            if "bound_missing_as_0" in g.index:
                ax.fill_between(x, pct(g.loc["bound_missing_as_0", PRESSURES]),
                                pct(g.loc["bound_missing_as_100", PRESSURES]), color=LIGHT, alpha=0.45, lw=0)
            if "complete_case" in g.index:
                ax.errorbar(x + 0.08, pct(g.loc["complete_case", PRESSURES]), yerr=pct(g.loc["complete_case", sd_cols]),
                            color=GREY, ls="--", marker="s", ms=3, elinewidth=0.8)
            y = pct(g.loc["available_case", PRESSURES])
            ax.errorbar(x, y, yerr=pct(g.loc["available_case", sd_cols]), color=INK, marker="o", ms=3.5,
                        elinewidth=0.8)
            lo = int(np.argmin(y))
            ax.annotate(f"{y[lo]:.1f}", (x[lo], y[lo]), xytext=(7, -2), textcoords="offset points", ha="left",
                        va="center", fontsize=7.5, color=INK)
            ax.set_xticks(x, PRESSURES)
            ax.margins(y=0.18)
            ax.set_title(f"{SCORE_LABELS[score]} · {MODEL_LABELS[model]}", loc="left", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("Adherence score")
    fig.supxlabel("Pressure level", fontsize=10)
    fig.tight_layout()
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", ms=3.5, label="all responses the judge scored"),
                        Line2D([], [], color=GREY, ls="--", marker="s", ms=3, label="responses all 3 judges scored"),
                        plt.Rectangle((0, 0), 1, 1, color=LIGHT, alpha=0.45, label="bounds: unscored = 0 … 100"),
                        Line2D([], [], color=INK, marker="|", ms=8, lw=0,
                               label=f"error bars: ±1 SD across {N_SEEDS} samples")],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, out, "fig_pressure_sensitivity", "A8 pressure sensitivity", "Main or appendix",
         "Mean adherence score by pressure level (pooled over responses). Point: mean of the five repeated-sample "
         "means; error bars: +/-1 SD across samples; lowest level labelled. For each judge: all responses that "
         "judge scored (solid), only responses all three judges scored (dashed), and the extreme band if every "
         "unscored response had scored 0 or 100. The 3-judge mean is defined only on responses all three judges "
         "scored, so it has a single line. Orderings with and without L3 are in tables/pressure_orderings.csv.")


# ------------------------------------------------------------------
# A9 within-prompt variability per language
# ------------------------------------------------------------------
def fig_within_prompt(T, out):
    w = T["within_prompt_sd"]
    w = w[w.grouping == "categoryxlanguage"]
    mats = {s: cell_frame(w[w.score == s], "mean_sd") for s in SCORES}
    vmax = max(np.nanmax(v.to_numpy()) for v in mats.values())
    fig, axes = plt.subplots(1, len(SCORES), figsize=(FULL_W + 0.6, 2.4))
    for k, (ax, score) in enumerate(zip(axes, SCORES)):
        im = heatmap(ax, mats[score], SEQ, 0, vmax, annot=mark_extreme(mats[score], "max"), show_y=k == 0)
        ax.set_title(SCORE_LABELS[score], loc="left")
        ax.tick_params(axis="x", labelsize=7.5, rotation=90)
    colorbar(fig, im, axes, f"SD across {N_SEEDS} samples\n(points, mean over prompts)")
    save(fig, out, "fig_within_prompt_sd_heatmap", "A9 within-prompt variability", "Appendix",
         "How much a single prompt's adherence score changes across its five repeated samples: the SD over "
         "samples, averaged over the prompts in each category x language cell (both models pooled); the largest "
         "cell per panel is labelled. One shared scale. Prompts need at least 4 scored samples (dropped counts in "
         "tables/within_prompt_dropped_prompts.csv).")


# ------------------------------------------------------------------
# A10 variance components per language
# ------------------------------------------------------------------
def fig_variance(T, out):
    vc = T["variance_components"]
    langs = LANGUAGES + ["all"]
    comps = [("var_item", "between items"), ("var_prompt", "prompt within item"),
             ("var_residual", "between samples (residual)")]
    fig, axes = plt.subplots(1, len(SCORES), figsize=(FULL_W + 0.4, 3.0), sharey=True)
    for ax, score in zip(axes, SCORES):
        g = vc[vc.score == score].set_index("language").reindex(langs)
        tot = g[["var_item", "var_prompt", "var_residual"]].sum(axis=1)
        left = np.zeros(len(langs))
        for (col, _), color in zip(comps, COMPONENTS):
            w = pct(g[col] / tot)
            ax.barh(range(len(langs)), w, left=left, color=color, height=0.72, edgecolor="white", linewidth=0.6)
            left += w
        # Direct label: the residual (between-sample) share for the pooled row.
        ax.text(98, len(langs) - 1, f"{pct(g.loc['all', 'var_residual'] / tot['all']):.0f}%", ha="right",
                va="center", fontsize=7.5, color="white", fontweight="bold")
        ax.set_title(SCORE_LABELS[score])
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.axhline(len(langs) - 1.5, color=INK, lw=0.6)
    axes[0].set_yticks(range(len(langs)), [l.upper() if l != "all" else "All" for l in langs])
    axes[0].invert_yaxis()
    fig.supxlabel("Share of variance in adherence score (%)", fontsize=10)
    fig.tight_layout()
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=lbl) for (_, lbl), c in zip(comps, COMPONENTS)],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, out, "fig_variance_components", "A10 variance components", "Appendix",
         "Variance in the adherence score per language, after removing category x model x pressure means, split "
         "into differences between base items, between prompts of the same item, and between the five repeated "
         "samples of the same prompt (nested ANOVA, method of moments; models pooled). 'All' pools the ten "
         "languages; its between-sample share is printed.")


# ------------------------------------------------------------------
# A11 cluster bootstrap
# ------------------------------------------------------------------
def fig_bootstrap(T, out):
    b = T["bootstrap"]
    b = b[b.score == PRIMARY]
    seeds = T["headline_gap_summary"]
    seeds = seeds[seeds.score == PRIMARY].set_index(["language", "model"])
    rows = [(l, m) for l in FOCUS for m in MODELS]
    intervals = [(f"min–max across {N_SEEDS} samples", INK, 2.6, None),
                 ("95% bootstrap CI, 66 items", GREY, 1.8, "item"),
                 ("95% bootstrap CI, 22 rule tasks", LIGHT, 1.8, "task")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    for i, (lang, model) in enumerate(rows):
        for k, (_, color, lw, unit) in enumerate(intervals):
            y = i + (k - 1) * 0.24
            if unit is None:
                s = seeds.loc[(lang, model)]
                lo, hi, pt = s["min"], s["max"], s["mean"]
            else:
                g = b[(b.statistic == "gap") & (b.cluster_unit == unit) & (b.language_or_term == lang) & (b.model == model)].iloc[0]
                lo, hi, pt = g.ci_low, g.ci_high, g.point
            ax.plot(pct([lo, hi]), [y, y], color=color, lw=lw, solid_capstyle="butt")
            ax.scatter(pct([pt]), [y], s=12, color=INK, zorder=3, lw=0)
        ax.text(pct(seeds.loc[(lang, model), "mean"]), i - 0.42, f"{pct(seeds.loc[(lang, model), 'mean']):.1f}",
                ha="center", fontsize=7.5, color=GREY)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(rows)), [f"{l.upper()} · {MODEL_LABELS[m]}" for l, m in rows])
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.set_xlabel("Withholding − constructed gap (points)")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    dec = T["decomposition_summary"]
    dec = dec[dec.score == PRIMARY].set_index("term")
    for i, term in enumerate(TERMS):
        for k, (_, color, lw, unit) in enumerate(intervals):
            y = i + (k - 1) * 0.24
            if unit is None:
                lo, hi, pt = dec.loc[term, "min"], dec.loc[term, "max"], dec.loc[term, "mean"]
            else:
                g = b[(b.statistic == "share") & (b.cluster_unit == unit) & (b.language_or_term == term)].iloc[0]
                lo, hi, pt = g.ci_low, g.ci_high, g.point
            ax.plot(pct([lo, hi]), [y, y], color=color, lw=lw, solid_capstyle="butt")
            ax.scatter(pct([pt]), [y], s=12, color=INK, zorder=3, lw=0)
    ax.set_yticks(range(len(TERMS)), [TERM_LABELS[t] for t in TERMS])
    ax.set_ylim(len(TERMS) - 0.5, -0.8)
    ax.set_xlabel("Share of between-cell variance (%)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.legend(handles=[Line2D([], [], color=c, lw=lw, label=lbl) for lbl, c, lw, _ in intervals],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, out, "fig_bootstrap_vs_seed_range", "A11 cluster bootstrap", "Appendix",
         "Generation noise (min-max across the five repeated samples) next to item-level noise (95% percentile "
         "cluster-bootstrap intervals, resampling base items or whole rule tasks within category), 3-judge mean "
         "adherence score; dot: point estimate (gap values printed). The intervals measure different sources of "
         "variability.")


def write_index(out):
    lines = ["# Sampling-stability figures", "",
             "Generated by `canonical/analysis/sampling_stability/figures.py` (Matplotlib + seaborn + SciencePlots) "
             "from the tables in the sibling analysis folder. Each figure is saved as PDF (vector, for the paper and "
             "Inkscape) and PNG. Scope: the 5-sample run only; answer inversion excluded (not judged in this run). "
             "Only numeric judge scores are used: the adherence score (P(HELD) x 100) and the confidence score. "
             "Main-text figures use the 3-judge mean adherence score. All sampling uncertainty shown is the "
             "variation across the five repeated samples (SD or min-max, as labelled); with five samples these "
             "spreads are themselves imprecise.", "",
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
