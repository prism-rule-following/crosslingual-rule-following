#!/usr/bin/env python3
"""
Cross-experiment analysis: obligation DIM (must vs may, must vs neutral) +
binding-state DIM (active vs revoked) + logit-lens semantic validation +
behavioral HELD-rate correlation, across both models, all 10 languages, both
obligation contrasts, and all 3 extraction positions.

The organizing question:
  Does the model represent that a rule is currently binding (active vs
  revoked) before, after, or together with representing its deontic force
  (must vs may / must vs neutral)? Does that ordering vary by language,
  model, contrast, or position -- and does representation strength predict
  actual behavioral rule-following?

NOT built here: category-level breakdown (Q4) and a per-(language, category)
HELD correlation. Both need per-category held-out AUC, which the pooled
dim_report.json files on HF don't have -- that requires a category-stratified
eval added to extract_dim.py and a rerun. A coarser, PER-LANGUAGE HELD
correlation has no such blocker (the judge data already has verdict per row,
independent of AUC granularity) and IS built here (see --judges below).

IMPORTANT -- contrast_token is not a clean "onset of representation" position,
and this script does NOT silently pick a different default to route around
that; it runs all three positions and labels the trivial cells explicitly so
you can compare them yourself.

contrast_token is anchored at the literal token where must/may (or active/
revoked) differ. The residual stream AT THAT TOKEN'S OWN POSITION trivially
carries "which word is here" at every layer, including layer 0 (the raw
embedding) -- that's just what residual streams do, independent of whether the
model has developed any downstream use of the concept. Whether that shows up
as a degenerate AUC~1.0-at-every-layer result depends on whether train/test
actually use a DIFFERENT word there:
  - obligation scenario-gen: train/test share the same frame, i.e. the exact
    same word -> AUC~1.0-everywhere at contrast_token is a tautology, not a
    finding. Flagged TRIVIAL below.
  - obligation frame-gen: train/test use different lexicalizations (that's
    the whole point of the per_frame_mean aggregation) -> a high AUC is a real
    claim about the embedding space generalizing across specific words.
  - binding (active/revoked): the pair_type holdout means train/test also use
    different words -> not trivial either.
rule_clause_end and post_instruction are never anchored to the differing
token, so this confound doesn't apply -- they're the positions that actually
speak to "does the concept persist/get integrated", which is what the
onset-timing question (Q3) is really asking. Confirmed empirically: at
contrast_token, AUC is ~1.0 at every layer including layer 0 for BOTH
obligation and binding; at rule_clause_end/post_instruction, onset layers and
peak AUCs vary meaningfully by language and are well below 1.0 for
post_instruction.

For each (model, language, contrast, position) this pulls, from the results
repo:
  - obligation, frame-generalization,    contrast in {must_may, must_neutral}
  - obligation, scenario-generalization, same contrast
  - active_revoked (binding state) -- pulled once per (model, language,
    position); it doesn't depend on the obligation contrast, so it's not
    re-fetched per contrast, just reused.
and computes, from each one's HELD-OUT per-layer AUC curve:
  - peak AUC and the layer it occurs at
  - "onset" layer: the first layer at which AUC reaches --auc-threshold and
    stays at or above it for --min-consecutive layers (a single crossing can
    be noise; this asks for the point past which the direction is reliably
    decodable, not just spikes once).

Q3 (timing): delta = obligation_scenario_onset - binding_onset. Positive means
binding is established before obligation; negative means the reverse; None
means one side never reliably crosses --auc-threshold for that (language,
position, contrast).

Q1's second half (semantic validation, not just separability) pulls the
logit-lens per_layer.json for the same contrast at the SAME peak layer used
for the AUC numbers -- both extraction scripts use identical layer indexing
(0=embedding, i=after decoder block i-1) for the direction tensor and the
per-layer AUC array, so a peak-AUC layer index is directly a valid tensor_index
into the logit-lens output.

Behavioral correlation (per-language, not per-category): pulls verdicts from
crosslingual-rule-following/judge-results-active-only/<judge>/results.jsonl
for all three judges by default (gpt_mini, deepseek, gemini). Each judge's own
HELD rate is computed independently (count(HELD) / count(HELD or VIOLATED)
for that judge, per model/language), then the per-judge rates are AVERAGED --
e.g. gpt_mini=0.90, deepseek=0.85, gemini=0.88 -> 0.877. This is not a
majority vote over individual responses and not a pooled tally of raw
verdicts; each judge contributes one rate, weighted equally regardless of how
many rows it judged. Plotted against each language's scenario-gen obligation
AUC.

Outputs (local, and one batched HF commit if --push), nested by contrast then
position:
  <out>/<contrast>/<position>/table_a_timing.csv   model, language, frame/
                                                    scenario/binding peak+
                                                    onset, delta, held_rate
  <out>/<contrast>/<position>/table_c_semantic.csv model, language, peak
                                                    layer, top tokens (one
                                                    row per (model, language) --
                                                    the peak-AUC layer only)
  <out>/<contrast>/<position>/table_c_semantic_all_layers.csv
                                                    same tokens for EVERY
                                                    layer (all already saved
                                                    by logit_lens_all_layers.py;
                                                    no scoring/onset detection
                                                    applied, just exported)
  <out>/<contrast>/<position>/fig1_obligation_auc_<model>.png
  <out>/<contrast>/<position>/fig2_binding_auc_<model>.png
  <out>/<contrast>/<position>/fig3_onset_scatter.png    (skipped when trivial)
  <out>/<contrast>/<position>/fig4_auc_vs_held_<model>.png

A missing combo (that extraction/push never completed) is skipped with a log
line, not fatal -- across 2 models x 10 languages x 2 contrasts x 3 positions,
partial coverage is the expected case.

Usage:
  python analyze_dim_results.py --models qwen3-8b llama3.1-8b \
    --languages en de hi ig it ko ru tr ur yo \
    --auc-threshold 0.75 --min-consecutive 3 --push
"""
import os, json, argparse, csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import hf_io

load_dotenv()

CONCEPT = "obligation"
CONTRASTS = ["must_may", "must_neutral"]
ALL_POSITIONS = ["contrast_token", "rule_clause_end", "post_instruction"]
# The only (generalization type, position) pair where train/test share the
# literal differing token, making AUC~1 at every layer a tautology rather
# than a finding -- see module docstring.
TRIVIAL = {("scenario", "contrast_token")}


# --------------------------------------------------------------------------- #
def pull_report(cfg, repo_kind, path):
    """Download one dim_report.json from the results repo; None if it's not
    there yet (that combo's extraction/push never completed)."""
    from huggingface_hub import hf_hub_download
    token = os.environ.get(cfg["hf"].get("token_env", "HF_TOKEN"))
    repo = cfg["hf"][repo_kind]
    try:
        fp = hf_hub_download(repo, path, repo_type="dataset", token=token)
        return json.load(open(fp))
    except Exception as e:
        print(f"[skip] {repo}/{path}: {e!r}")
        return None


def obligation_report(cfg, model_key, lang, variant, contrast, position, preset):
    path = f"AUC/{CONCEPT}/{variant}/{contrast}/{position}/{lang}/{model_key}__{preset}/dim_report.json"
    return pull_report(cfg, "results_repo", path)


def active_revoked_report(cfg, model_key, lang, position):
    path = f"AUC/active_revoked/{position}/{lang}/{model_key}/dim_report.json"
    return pull_report(cfg, "results_repo", path)


def logit_lens_report(cfg, model_key, lang, variant, contrast, position, preset):
    path = f"logit_lens/{CONCEPT}/{variant}/{contrast}/{position}/{lang}/{model_key}__{preset}/per_layer.json"
    return pull_report(cfg, "results_repo", path)


def compute_held_rates(cfg, judges, models, languages):
    """HELD rate per (model_key, language): compute each judge's OWN HELD rate
    independently (count(HELD) / count(HELD or VIOLATED) for that judge over
    that model/language's responses), then AVERAGE those per-judge rates
    across judges -- e.g. gpt_mini=0.90, deepseek=0.85, gemini=0.88 ->
    0.877. This is not a majority vote over individual responses and not a
    pooled tally of raw verdicts; each judge contributes one rate, weighted
    equally regardless of how many rows it judged.
    Returns {(model_key, language): (mean_rate_or_None, n_judges_averaged)}."""
    from huggingface_hub import hf_hub_download
    acfg = cfg["ablation"]
    token = os.environ.get(cfg["hf"].get("token_env", "HF_TOKEN"))
    hf_name_to_key = {mcfg["hf_name"]: key for key, mcfg in cfg["models"].items() if key in models}
    lang_set = set(languages)

    # per_judge_counts[judge][(model_key, lang)] = {"held": n, "violated": n}
    per_judge_counts = defaultdict(lambda: defaultdict(lambda: {"held": 0, "violated": 0}))
    for judge in judges:
        try:
            fp = hf_hub_download(acfg["baseline_repo"], f"{judge}/results.jsonl", repo_type="dataset", token=token)
        except Exception as e:
            print(f"[held][skip] judge {judge}: {e!r}"); continue
        n_rows = 0
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                mk = hf_name_to_key.get(r.get("model_id"))
                lang = r.get("language")
                if mk is None or lang not in lang_set:
                    continue
                v = r.get("verdict")
                if v == "HELD":
                    per_judge_counts[judge][(mk, lang)]["held"] += 1; n_rows += 1
                elif v == "VIOLATED":
                    per_judge_counts[judge][(mk, lang)]["violated"] += 1; n_rows += 1
        print(f"[held] judge={judge}: {n_rows} usable verdicts in scope")

    all_keys = set()
    for judge_counts in per_judge_counts.values():
        all_keys |= set(judge_counts.keys())

    rates = {}
    for key in all_keys:
        per_judge_rates = []
        for judge in judges:
            c = per_judge_counts.get(judge, {}).get(key)
            if not c:
                continue
            n = c["held"] + c["violated"]
            if n:
                per_judge_rates.append(c["held"] / n)
        rates[key] = (sum(per_judge_rates) / len(per_judge_rates), len(per_judge_rates)) if per_judge_rates else (None, 0)
    return rates


# --------------------------------------------------------------------------- #
def held_out_auc_curve(report, direction_key):
    """Extract the held-out per-layer AUC list from a dim_report.json's
    directions[direction_key]. None if that key or held_out is absent (e.g.
    the run had no test split)."""
    if report is None:
        return None
    entry = report.get("directions", {}).get(direction_key)
    if entry is None or "held_out" not in entry:
        return None
    return entry["held_out"]["per_layer_auc"]


def peak(curve):
    if not curve:
        return None, None
    l = max(range(len(curve)), key=lambda i: curve[i])
    return l, curve[l]


def onset_layer(curve, threshold, min_consecutive):
    """First layer from which AUC stays >= threshold for min_consecutive
    layers (or through the end of the curve, if fewer remain). A single
    high-AUC layer surrounded by noise doesn't count."""
    if not curve:
        return None
    n = len(curve)
    for l in range(n):
        window = curve[l:min(l + min_consecutive, n)]
        if window and all(a >= threshold for a in window):
            return l
    return None


# --------------------------------------------------------------------------- #
def analyze_combo(cfg, model_key, lang, contrast, position, preset, threshold, min_consec, held_lookup):
    frame_rep = obligation_report(cfg, model_key, lang, "frame", contrast, position, preset)
    scen_rep = obligation_report(cfg, model_key, lang, "scenario", contrast, position, preset)
    bind_rep = active_revoked_report(cfg, model_key, lang, position)

    name = f"{contrast}__{position}"
    frame_curve = held_out_auc_curve(frame_rep, name)
    scen_curve = held_out_auc_curve(scen_rep, name)
    bind_curve = held_out_auc_curve(bind_rep, position)

    frame_pl, frame_pa = peak(frame_curve)
    scen_pl, scen_pa = peak(scen_curve)
    bind_pl, bind_pa = peak(bind_curve)
    frame_onset = onset_layer(frame_curve, threshold, min_consec)
    scen_onset = onset_layer(scen_curve, threshold, min_consec)
    bind_onset = onset_layer(bind_curve, threshold, min_consec)
    delta = (scen_onset - bind_onset) if (scen_onset is not None and bind_onset is not None) else None

    held_rate, n_judges = held_lookup.get((model_key, lang), (None, 0))

    row = {
        "model": model_key, "language": lang, "contrast": contrast, "position": position,
        "frame_trivial": ("frame", position) in TRIVIAL,
        "frame_peak_layer": frame_pl, "frame_peak_auc": frame_pa, "frame_onset_layer": frame_onset,
        "scenario_trivial": ("scenario", position) in TRIVIAL,
        "scenario_peak_layer": scen_pl, "scenario_peak_auc": scen_pa, "scenario_onset_layer": scen_onset,
        "binding_peak_layer": bind_pl, "binding_peak_auc": bind_pa, "binding_onset_layer": bind_onset,
        "delta_onset_scenario_minus_binding": delta,
        "held_rate": held_rate, "held_n_judges": n_judges,
    }
    return row, frame_curve, scen_curve, bind_curve


def semantic_row(model_key, lang, ll, peak_layer, topk_show=8):
    """Peak-layer-only summary row (existing headline table). Takes the
    already-fetched full logit-lens sweep (`ll`, every layer) so it doesn't
    re-download it -- see all_layer_rows() for the full sweep this is sampled
    from."""
    if peak_layer is None:
        return {"model": model_key, "language": lang, "peak_layer": None,
                "top_plus": "", "top_minus": ""}
    if ll is None:
        return {"model": model_key, "language": lang, "peak_layer": peak_layer,
                "top_plus": "(no logit-lens data)", "top_minus": ""}
    hit = next((r for r in ll if r.get("tensor_index") == peak_layer), None)
    if hit is None or hit.get("empty"):
        return {"model": model_key, "language": lang, "peak_layer": peak_layer,
                "top_plus": "(empty direction at this layer)", "top_minus": ""}
    return {"model": model_key, "language": lang, "peak_layer": peak_layer,
            "top_plus": " ".join(hit["top_plus"][:topk_show]),
            "top_minus": " ".join(hit["top_minus"][:topk_show])}


def all_layer_rows(model_key, lang, ll):
    """Every layer's logit-lens tokens, not just the peak-AUC one -- the full
    sweep is already computed and saved by logit_lens_all_layers.py (confirmed:
    37 layers stored for Qwen, 33 for Llama, per combo); this just exports all
    of it instead of sampling one row, so a real semantic-onset question (at
    which layer do the tokens first read as deontic, not just separate the
    labels) can be answered by inspection later without re-running extraction.
    No scoring/threshold is applied here -- that's a separate, harder problem
    (see conversation) and out of scope for this export."""
    if ll is None:
        return []
    out_rows = []
    for r in ll:
        out_rows.append({
            "model": model_key, "language": lang,
            "tensor_index": r.get("tensor_index"), "resid_after_block": r.get("resid_after_block"),
            "empty": r.get("empty", False),
            "top_plus": " ".join(r.get("top_plus", [])),
            "top_minus": " ".join(r.get("top_minus", [])),
            "top_random_baseline": " ".join(r.get("top_random_baseline", [])),
        })
    return out_rows


# --------------------------------------------------------------------------- #
def plot_auc_by_layer(curves_by_lang_solid, curves_by_lang_dashed, title, out_path):
    """curves_by_lang_* : {language: [auc_per_layer]} or None entries skipped."""
    langs = sorted(set(curves_by_lang_solid) | set(curves_by_lang_dashed or {}))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lang in enumerate(langs):
        color = cmap(i % 10)
        c1 = curves_by_lang_solid.get(lang)
        if c1:
            ax.plot(range(len(c1)), c1, "-", color=color, label=lang)
        if curves_by_lang_dashed:
            c2 = curves_by_lang_dashed.get(lang)
            if c2:
                ax.plot(range(len(c2)), c2, "--", color=color, alpha=0.6)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("layer index (0=embedding)"); ax.set_ylabel("held-out AUC")
    ax.set_title(title); ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[fig] {out_path}")


def plot_onset_scatter(rows, out_path):
    models = sorted(set(r["model"] for r in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), squeeze=False)
    for ax, model_key in zip(axes[0], models):
        pts = [(r["binding_onset_layer"], r["scenario_onset_layer"], r["language"])
               for r in rows if r["model"] == model_key
               and r["binding_onset_layer"] is not None and r["scenario_onset_layer"] is not None]
        if pts:
            xs, ys, labels = zip(*pts)
            ax.scatter(xs, ys, s=40)
            for x, y, lab in pts:
                ax.annotate(lab, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
            lo, hi = min(xs + ys) - 1, max(xs + ys) + 1
            ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", linewidth=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        missing = [r["language"] for r in rows if r["model"] == model_key
                   and (r["binding_onset_layer"] is None or r["scenario_onset_layer"] is None)]
        if missing:
            ax.text(0.02, 0.98, f"no reliable onset: {', '.join(missing)}",
                    transform=ax.transAxes, fontsize=7, va="top", color="firebrick")
        ax.set_xlabel("binding-state onset layer"); ax.set_ylabel("obligation (scenario) onset layer")
        ax.set_title(model_key)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[fig] {out_path}")


def plot_auc_vs_held(rows, out_path):
    """Per-language scatter: scenario-gen obligation AUC vs behavioral HELD
    rate, one panel per model. NOT the category-level Figure 6 from the full
    analysis plan -- this is the coarser, per-language version that has no
    data blocker (see module docstring)."""
    models = sorted(set(r["model"] for r in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), squeeze=False)
    for ax, model_key in zip(axes[0], models):
        pts = [(r["scenario_peak_auc"], r["held_rate"], r["language"])
               for r in rows if r["model"] == model_key
               and r["scenario_peak_auc"] is not None and r["held_rate"] is not None]
        if pts:
            xs, ys, labels = zip(*pts)
            ax.scatter(xs, ys, s=40)
            for x, y, lab in pts:
                ax.annotate(lab, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax.set_xlim(0.4, 1.02); ax.set_ylim(0, 1.02)
        missing = [r["language"] for r in rows if r["model"] == model_key
                   and (r["scenario_peak_auc"] is None or r["held_rate"] is None)]
        if missing:
            ax.text(0.02, 0.02, f"missing: {', '.join(missing)}",
                    transform=ax.transAxes, fontsize=7, va="bottom", color="firebrick")
        ax.set_xlabel("scenario-gen obligation peak AUC"); ax.set_ylabel("behavioral HELD rate")
        ax.set_title(model_key)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[fig] {out_path}")


# --------------------------------------------------------------------------- #
def run_position_contrast(cfg, models, languages, contrast, position, preset,
                          threshold, min_consec, out_root, held_lookup):
    out = out_root / contrast / position
    out.mkdir(parents=True, exist_ok=True)
    scen_trivial = ("scenario", position) in TRIVIAL
    frame_trivial = ("frame", position) in TRIVIAL

    rows, semantic_rows, sweep_rows = [], [], []
    frame_curves, scen_curves, bind_curves = {}, {}, {}
    for model_key in models:
        frame_curves[model_key], scen_curves[model_key], bind_curves[model_key] = {}, {}, {}
        for lang in languages:
            print(f"--- {contrast}/{position}: {model_key}/{lang} ---")
            row, fc, sc, bc = analyze_combo(cfg, model_key, lang, contrast, position, preset,
                                            threshold, min_consec, held_lookup)
            rows.append(row)
            if fc: frame_curves[model_key][lang] = fc
            if sc: scen_curves[model_key][lang] = sc
            if bc: bind_curves[model_key][lang] = bc
            # fetch the full per-layer logit-lens sweep ONCE, feed both the
            # peak-layer summary and the full-layer export from it
            ll = logit_lens_report(cfg, model_key, lang, "scenario", contrast, position, preset)
            semantic_rows.append(semantic_row(model_key, lang, ll, row["scenario_peak_layer"]))
            sweep_rows.extend(all_layer_rows(model_key, lang, ll))

    if not rows:
        print(f"[{contrast}/{position}] no data at all -- skipping outputs")
        return rows

    table_a_path = out / "table_a_timing.csv"
    with open(table_a_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[table] {table_a_path}")

    print(f"\n=== Table A summary [{contrast}/{position}]"
          f"{' -- SCENARIO COLUMN IS TRIVIAL (same word train/test)' if scen_trivial else ''}"
          f"{' -- FRAME COLUMN IS TRIVIAL' if frame_trivial else ''} ===")
    print(f"  {'model':14s} {'language':4s}  {'frame':>16s}  {'scenario':>16s}  {'binding':>16s}  {'delta':>6s}  {'held':>6s}")
    for r in rows:
        f_s = f"L{r['frame_onset_layer']}/{r['frame_peak_auc']:.2f}" if r["frame_peak_auc"] is not None else "n/a"
        s_s = f"L{r['scenario_onset_layer']}/{r['scenario_peak_auc']:.2f}" if r["scenario_peak_auc"] is not None else "n/a"
        b_s = f"L{r['binding_onset_layer']}/{r['binding_peak_auc']:.2f}" if r["binding_peak_auc"] is not None else "n/a"
        d = r["delta_onset_scenario_minus_binding"]
        h = f"{r['held_rate']:.2f}" if r["held_rate"] is not None else "n/a"
        print(f"  {r['model']:14s} {r['language']:4s}  {f_s:>16s}  {s_s:>16s}  {b_s:>16s}  {str(d):>6s}  {h:>6s}")

    table_c_path = out / "table_c_semantic.csv"
    with open(table_c_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(semantic_rows[0].keys()))
        w.writeheader(); w.writerows(semantic_rows)
    print(f"[table] {table_c_path} (peak-layer-only summary)")

    if sweep_rows:
        table_c_all_path = out / "table_c_semantic_all_layers.csv"
        with open(table_c_all_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader(); w.writerows(sweep_rows)
        print(f"[table] {table_c_all_path} ({len(sweep_rows)} rows -- every layer, "
              f"every (model, language), no scoring applied)")

    for model_key in models:
        plot_auc_by_layer(scen_curves[model_key], frame_curves[model_key],
                          f"Obligation ({contrast}) AUC by layer -- {model_key} @ {position}\n"
                          f"solid=scenario-gen{' [TRIVIAL]' if scen_trivial else ''}, "
                          f"dashed=frame-gen{' [TRIVIAL]' if frame_trivial else ''}",
                          out / f"fig1_obligation_auc_{model_key}.png")
        plot_auc_by_layer(bind_curves[model_key], None,
                          f"Binding state (active vs revoked) AUC by layer -- {model_key} @ {position}",
                          out / f"fig2_binding_auc_{model_key}.png")
    if scen_trivial:
        print(f"[{contrast}/{position}] NOTE: skipping onset-scatter figure -- the obligation "
              f"side (scenario-gen) is a tautology at this position.")
    else:
        plot_onset_scatter(rows, out / "fig3_onset_scatter.png")

    if any(r["held_rate"] is not None for r in rows):
        plot_auc_vs_held(rows, out / "fig4_auc_vs_held.png")
    else:
        print(f"[{contrast}/{position}] NOTE: no HELD-rate data matched -- skipping fig4 "
              f"(check --judges and that judge results cover these models/languages).")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--config", default="hyperparameters.json")
    ap.add_argument("--preset", default="rule_following")
    ap.add_argument("--contrasts", nargs="+", default=CONTRASTS, choices=CONTRASTS,
                    help="must_may = obligation vs permission; must_neutral = obligation vs "
                         "descriptive-norm. Default: both, run and saved separately.")
    ap.add_argument("--positions", nargs="+", default=ALL_POSITIONS, choices=ALL_POSITIONS,
                    help="which of the 3 extraction positions to analyze. Default: all three, "
                         "run and saved separately -- see module docstring for why contrast_token's "
                         "scenario-gen numbers are a tautology and shouldn't be compared at face "
                         "value against rule_clause_end/post_instruction.")
    ap.add_argument("--judges", nargs="+", default=["gpt_mini", "deepseek", "gemini"],
                    choices=["gpt_mini", "deepseek", "gemini"],
                    help="which judges to consolidate (majority vote per response, not a pooled "
                         "tally -- see compute_held_rates docstring) for the behavioral HELD-rate "
                         "correlation. Default: all three.")
    ap.add_argument("--auc-threshold", type=float, default=0.75)
    ap.add_argument("--min-consecutive", type=int, default=3)
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[held] pooling verdicts from judges={args.judges} for models={args.models} "
          f"languages={args.languages}")
    held_lookup = compute_held_rates(cfg, args.judges, args.models, args.languages)
    print(f"[held] resolved HELD rate for {len(held_lookup)} (model, language) pairs")

    for contrast in args.contrasts:
        for position in args.positions:
            run_position_contrast(cfg, args.models, args.languages, contrast, position, args.preset,
                                  args.auc_threshold, args.min_consecutive, out_root, held_lookup)

    if args.push:
        hf_io.push_batch(cfg, "results", str(out_root.parent), [f"{out_root.name}/*"],
                         path_in_repo="analysis",
                         commit_message=f"analysis: DIM timing + semantic + HELD correlation "
                                        f"({', '.join(args.contrasts)} x {', '.join(args.positions)})")


if __name__ == "__main__":
    main()
