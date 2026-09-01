"""Figures for LP4FM §4-§5 from the judge summary and the rule-geometry results.

Palette follows the validated three-slot categorical set (blue / amber / red,
all-pairs CVD ΔE 15.3, normal-vision 20.8) used here for the ordinal resource
band; amber sits below 3:1 on the light surface, so every band-coloured mark
also carries a visible direct label.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BAND_COLOR = {"high": "#2a78d6", "mid": "#eda100", "low": "#e34948"}
BANDS = {"ru": "high", "de": "high", "en": "high", "it": "high", "ko": "high",
         "tr": "mid", "hi": "mid", "ur": "mid",
         "yo": "low", "ig": "low"}
ORDER = ["ru", "de", "en", "it", "ko", "tr", "hi", "ur", "yo", "ig"]
PRESSURE = ["L0", "L1", "L2", "L3", "L4", "L5"]
PRESSURE_LABEL = {"L0": "L0\nneutral", "L1": "L1\nsocial", "L2": "L2\nauthority",
                  "L3": "L3\noverride", "L4": "L4\nemotion", "L5": "L5\nmulti-turn"}
MODEL_LABEL = {"Qwen__Qwen3-8B": "Qwen3-8B",
               "meta-llama__Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct"}
MODEL_KEY = {"Qwen__Qwen3-8B": "Qwen/Qwen3-8B",
             "meta-llama__Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"}

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def tidy(ax, ygrid=True):
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)


def band_legend(ax, loc="lower left"):
    handles = [plt.Line2D([], [], marker="s", ls="", ms=7, color=BAND_COLOR[b],
                          label=f"{b}-resource") for b in ("high", "mid", "low")]
    ax.legend(handles=handles, loc=loc, handletextpad=0.5, borderpad=0.3)


def sequential(v, lo, hi):
    t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
    return SEQ[int(round(np.clip(t, 0, 1) * (len(SEQ) - 1)))]


# ------------------------------------------------------------------- figures

def fig_held_by_language(judge, out):
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    for ax, m in zip(axes, models):
        vals = [judge[MODEL_KEY[m]][l]["held_L0"] for l in ORDER]
        cols = [BAND_COLOR[BANDS[l]] for l in ORDER]
        bars = ax.bar(range(len(ORDER)), vals, color=cols, width=0.68)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER)
        ax.set_ylim(0, 105)
        ax.set_title(MODEL_LABEL[m], color=INK, loc="left", pad=8)
        tidy(ax)
    axes[0].set_ylabel("rule held (%), neutral prompts")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=7, color=BAND_COLOR[b],
                          label=f"{b}-resource") for b in ("high", "mid", "low")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10),
               handletextpad=0.5, columnspacing=1.8)
    fig.text(0.5, -0.005, "dashed line = 50%", ha="center", fontsize=7.5, color=MUTED)
    fig.suptitle("Fig 3 · The same rule, the same task: adherence falls with "
                 "language resource level",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.04)
    fig.savefig(out / "fig3_held_by_language.png")
    fig.savefig(out / "fig3_held_by_language.pdf")
    plt.close(fig)


def fig_resistance(judge, out):
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, m in zip(axes, models):
        per_band = {b: [] for b in BAND_COLOR}
        for l in ORDER:
            ys = [judge[MODEL_KEY[m]][l]["by_pressure"][p] for p in PRESSURE]
            per_band[BANDS[l]].append(ys)
            ax.plot(range(len(PRESSURE)), ys, color=BAND_COLOR[BANDS[l]],
                    lw=0.9, alpha=0.28, zorder=1)
        for b, series in per_band.items():
            mean = np.nanmean(np.array(series, dtype=float), axis=0)
            ax.plot(range(len(PRESSURE)), mean, color=BAND_COLOR[b], lw=2.0, zorder=3)
            ax.scatter(range(len(PRESSURE)), mean, s=22, color=BAND_COLOR[b],
                       zorder=4, edgecolor=SURFACE, linewidth=1.2)
            ax.annotate(f"{b}-resource", (0, mean[0]),
                        xytext=(-8, 0), textcoords="offset points", ha="right",
                        color=BAND_COLOR[b], fontsize=8.5, va="center")
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xticks(range(len(PRESSURE)))
        ax.set_xticklabels([PRESSURE_LABEL[p] for p in PRESSURE], fontsize=7.5)
        ax.set_xlim(-1.5, len(PRESSURE) - 0.7)
        ax.set_ylim(20, 100)
        ax.set_title(MODEL_LABEL[m], color=INK, loc="left", pad=8)
        tidy(ax)
    axes[0].set_ylabel("rule held (%)")
    fig.suptitle("Fig 4 · Pressure erodes rules that were binding; where nothing "
                 "was binding, there is nothing to erode",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.05)
    fig.text(0.005, -0.06, "Faint lines: individual languages. Bold lines: band means. "
             "Dashed line = 50%.", ha="left", fontsize=7.5, color=MUTED)
    fig.savefig(out / "fig4_resistance_curves.png")
    fig.savefig(out / "fig4_resistance_curves.pdf")
    plt.close(fig)


def fig_cka_heatmap(results, out):
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 4.3))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, m in zip(axes, models):
        res = results[m]
        langs = res["langs"]
        idx = [langs.index(l) for l in ORDER if l in langs]
        names = [langs[i] for i in idx]
        M = np.array(res["cka_matrix"])[np.ix_(idx, idx)]
        im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.4,
                        color="#ffffff" if v > 0.66 else INK2)
        ax.set_title(f"{MODEL_LABEL[m]}  ·  layer {res['common_layer']}",
                     color=INK, loc="left", pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7.5)
    cb.set_label("CKA of rule-status difference vectors", fontsize=8, color=INK2)
    fig.suptitle("Fig 5a · In Qwen the rule's signature is shared across "
                 "high-resource languages and absent in Yoruba and Igbo; "
                 "in Llama it is uniform",
                 x=0.005, ha="left", fontsize=10.5, color=INK, y=1.02)
    fig.text(0.005, -0.02, "Each model is shown at the layer where rule status is "
             "most decodable in English. Shared colour scale.",
             ha="left", fontsize=7.5, color=MUTED)
    fig.savefig(out / "fig5a_cka_heatmap.png")
    fig.savefig(out / "fig5a_cka_heatmap.pdf")
    plt.close(fig)


def fig_layer_curves(results, out):
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, m in zip(axes, models):
        res = results[m]
        for l in ORDER:
            if l not in res["per_lang"]:
                continue
            y = np.array(res["per_lang"][l]["status_auc_xreal"], dtype=float)
            x = np.arange(len(y)) / (len(y) - 1)
            ax.plot(x, y, color=BAND_COLOR[BANDS[l]], lw=1.4,
                    alpha=0.95 if BANDS[l] != "high" else 0.75)
            if l in ("en", "yo", "ig", "hi"):
                k = int(np.nanargmax(y))
                ax.annotate(l, (x[k], y[k]), xytext=(4, 6),
                            textcoords="offset points", fontsize=8,
                            color=BAND_COLOR[BANDS[l]], zorder=6,
                            path_effects=[pe.withStroke(linewidth=2.6,
                                                        foreground=SURFACE)])
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.text(0.99, 0.507, "chance", fontsize=7.5, color=MUTED, va="bottom",
                ha="right", path_effects=[pe.withStroke(linewidth=2.6,
                                                        foreground=SURFACE)])
        ax.set_xlabel("relative depth (layer / total layers)")
        ax.set_ylim(0.42, 0.95)
        ax.set_title(MODEL_LABEL[m], color=INK, loc="left", pad=8)
        tidy(ax)
    axes[0].set_ylabel("rule status readable (AUC,\nheld-out status wording)")
    band_legend(axes[1], loc="upper right")
    fig.suptitle("Fig 5b · Where the rule is encoded: a mid-network peak that "
                 "low-resource languages never develop",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.04)
    fig.savefig(out / "fig5b_layer_curves.png")
    fig.savefig(out / "fig5b_layer_curves.pdf")
    plt.close(fig)


def fig_controls(results, out):
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, m in zip(axes, models):
        res = results[m]
        langs = [l for l in ORDER if l in res["per_lang"]]
        x = np.arange(len(langs))
        status = [res["per_lang"][l]["status_auc_xreal_peak"] for l in langs]
        cat = [res["per_lang"][l]["category_decode_acc"] for l in langs]
        ax.bar(x - 0.19, status, 0.36, color=[BAND_COLOR[BANDS[l]] for l in langs])
        ax.bar(x + 0.19, cat, 0.36, color="#d8d7d0", edgecolor=SURFACE, linewidth=1)
        chance_cat = res["per_lang"][langs[0]]["category_decode_chance"]
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.axhline(chance_cat, color=MUTED, lw=0.8, ls=(0, (1, 3)))
        ax.text(len(langs) - 0.4, 0.512, "chance, status", fontsize=7,
                color=MUTED, ha="right", va="bottom")
        ax.text(len(langs) - 0.4, chance_cat + 0.012, "chance, category",
                fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(langs)
        ax.set_ylim(0, 1.08)
        ax.set_title(MODEL_LABEL[m], color=INK, loc="left", pad=8)
        tidy(ax)
    axes[0].set_ylabel("held-out score")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=7, color="#2a78d6",
                          label="rule status (AUC, held-out wording)"),
               plt.Line2D([], [], marker="s", ls="", ms=7, color="#d8d7d0",
                          label="rule category (7-way accuracy)")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.10), handletextpad=0.5, columnspacing=2.0)
    fig.suptitle("Fig 5c · Control: Yoruba and Igbo prompts are represented "
                 "clearly — it is rule status specifically that is missing",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.04)
    fig.savefig(out / "fig5c_controls.png")
    fig.savefig(out / "fig5c_controls.pdf")
    plt.close(fig)



def place_labels(ax, xs, ys, labels, color=None):
    """Greedy non-overlapping label placement in axis-fraction space."""
    ax.figure.canvas.draw()
    to_axes = ax.transLimits.transform
    placed = []
    offsets = [(0, 10), (0, -14), (13, 0), (-13, 0), (11, 8), (-11, 8),
               (11, -10), (-11, -10)]
    for x, y, lab in zip(xs, ys, labels):
        ax_x, ax_y = to_axes((x, y))
        for dx, dy in offsets:
            cand = (ax_x + dx / 320.0, ax_y + dy / 260.0)
            if all(abs(cand[0] - q[0]) > 0.062 or abs(cand[1] - q[1]) > 0.055
                   for q in placed):
                break
        placed.append(cand)
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=8,
                    color=color if color else INK2)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def fig_link(results, judge, out):
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, m in zip(axes, models):
        res = results[m]
        langs = [l for l in ORDER if l in res["per_lang"]]
        xs = [res["per_lang"][l]["status_auc_xreal_peak"] for l in langs]
        ys = [judge[MODEL_KEY[m]][l]["held_L0"] for l in langs]
        r = spearman(xs, ys)
        b, a = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 10)
        ax.plot(gx, b * gx + a, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for l, x, y in zip(langs, xs, ys):
            ax.scatter(x, y, s=64, color=BAND_COLOR[BANDS[l]], zorder=3,
                       edgecolor=SURFACE, linewidth=1.6)
        place_labels(ax, xs, ys, langs)
        ax.set_xlabel("rule status readable (AUC, held-out wording)")
        ax.set_title(f"{MODEL_LABEL[m]}   Spearman ρ = {r:.2f}", color=INK,
                     loc="left", pad=8)
        ax.set_ylim(30, 105)
        tidy(ax)
    axes[0].set_ylabel("rule held (%), neutral prompts")
    band_legend(axes[0], loc="lower right")
    fig.suptitle("Fig 5d · Where the rule is not encoded, it is not obeyed",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.04)
    fig.savefig(out / "fig5d_representation_vs_behaviour.png")
    fig.savefig(out / "fig5d_representation_vs_behaviour.pdf")
    plt.close(fig)


def fig_transport(results, out):
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    for ax, m in zip(axes, models):
        res = results[m]
        langs = [l for l in ORDER if l in res["per_lang"] and l != "en"]
        x = np.arange(len(langs))
        direct = [res["per_lang"][l]["transport_direct"] for l in langs]
        proc = [res["per_lang"][l]["transport_procrustes"] for l in langs]
        ceil = [res["per_lang"][l]["in_language_ceiling"] for l in langs]
        ax.bar(x - 0.22, direct, 0.2, color="#9ec5f4", label="English direction, as is")
        ax.bar(x, proc, 0.2, color="#2a78d6", label="after rotation to English")
        ax.bar(x + 0.22, ceil, 0.2, color="#d8d7d0", label="in-language ceiling")
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xticks(x)
        ax.set_xticklabels(langs)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(MODEL_LABEL[m], color=INK, loc="left", pad=8)
        tidy(ax)
    axes[0].set_ylabel("status decoded (AUC)")
    axes[1].legend(loc="upper right", ncol=1)
    fig.suptitle("Fig 5e · An English rule direction reads rule status directly in "
                 "high-resource languages, and has nothing to read in low-resource ones",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.04)
    fig.savefig(out / "fig5e_transport.png")
    fig.savefig(out / "fig5e_transport.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="canonical/results/rule_geometry")
    ap.add_argument("--judge", default="canonical/results/rule_geometry/judge_summary.json")
    ap.add_argument("--out", default="canonical/results/figures")
    args = ap.parse_args()

    style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = json.loads(Path(args.judge).read_text())
    results = {}
    for m in MODEL_LABEL:
        p = Path(args.results) / f"{m}.json"
        if p.exists():
            results[m] = json.loads(p.read_text())

    fig_held_by_language(judge, out)
    fig_resistance(judge, out)
    print("wrote fig3, fig4")
    if len(results) == len(MODEL_LABEL):
        fig_cka_heatmap(results, out)
        fig_layer_curves(results, out)
        fig_controls(results, out)
        fig_link(results, judge, out)
        fig_transport(results, out)
        write_table(results, judge, out)
        print("wrote fig5a-5e")
    else:
        print(f"skipping fig5*: have {list(results)}")




def write_table(results, judge, out):
    """Markdown table of the §5 numbers, one row per language per model."""
    lines = ["| model | lang | band | HELD% (L0) | HELD% (all) | status AUC | "
             "category acc | CKA vs EN | EN dir agree | transport direct | "
             "transport rotated | in-lang ceiling |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODEL_LABEL:
        if m not in results:
            continue
        res = results[m]
        for l in ORDER:
            r = res["per_lang"].get(l)
            if not r:
                continue
            j = judge[MODEL_KEY[m]][l]
            lines.append(
                f"| {MODEL_LABEL[m]} | {l} | {BANDS[l]} | {j['held_L0']:.1f} | "
                f"{j['held_overall']:.1f} | {r['status_auc_xreal_peak']:.3f} | "
                f"{r['category_decode_acc']:.3f} | {r['cka_vs_en']:.3f} | "
                f"{r['en_direction_sign_agreement']:.3f} | {r['transport_direct']:.3f} | "
                f"{r['transport_procrustes']:.3f} | {r['in_language_ceiling']:.3f} |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'results_table.md'}")


if __name__ == "__main__":
    main()
