"""Pre-registered checks (C1-C7) and analyses (A1-A11) on the 5-sample run.

Conventions (plan section 2):
  - Cell = category x language x model. Rates are computed over the cell's
    prompts within a seed first, then summarized across seeds.
  - Track A = the judges' reported HELD/VIOLATED verdicts ({j}_held), per judge
    and as consensus labels.
  - Track B = the adherence score P(HELD) ({j}_p), a continuous score: averaged,
    never thresholded, never pooled across judges.
  - Unjudged responses are NaN and drop out of the denominator; tables carry n.
  - Bootstrap resamples base items or rule tasks (stratified by category), never responses.
Every function takes the wide table from data.py and returns {name: DataFrame}.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from canonical.analysis.sampling_stability.data import (
    CATEGORIES,
    CONSTRUCTED,
    JUDGE_MODELS,
    JUDGES,
    LANGUAGES,
    PRESSURES,
    REASON_CODES,
    WITHHOLDING,
)

CELL = ["category", "language", "model"]
FOCUS_LANGUAGES = ["yo", "ig"]
MIN_SEEDS_FOR_FLIP = 4
# cons = three-judge consensus (primary); pair = GPT+Gemini agreement.
CONSENSUS_LABELS = ["cons", "pair"]
CONSENSUS_NAMES = {"cons": "3-judge consensus (primary)", "pair": "GPT+Gemini agreement"}


def track_specs():
    """(track, judge, column) triples."""
    return [("A", j, f"{j}_held") for j in CONSENSUS_LABELS + JUDGES] + [("B", j, f"{j}_p") for j in JUDGES]


def judge_model(judge):
    return CONSENSUS_NAMES.get(judge) or JUDGE_MODELS[judge]


def seed_cell_rates(df, col, keys=CELL):
    g = df.groupby(list(keys) + ["sample_idx"], observed=True)[col]
    return g.agg(rate="mean", n_usable="count", n_total="size").reset_index()


def cell_array(rates):
    """(category, language, model) -> numpy array in fixed order."""
    piv = rates.set_index(CELL)["rate"]
    idx = pd.MultiIndex.from_product([CATEGORIES, LANGUAGES, ["qwen", "llama"]], names=CELL)
    return piv.reindex(idx).to_numpy().reshape(len(CATEGORIES), len(LANGUAGES), 2)


# ------------------------------------------------------------------
# A1 per-judge, per-model matrices
# ------------------------------------------------------------------
def a1_matrices(df):
    tables, per_seed = [], []
    for track, judge, col in track_specs():
        r = seed_cell_rates(df, col)
        per_seed.append(r.assign(track=track, judge=judge))
        s = r.groupby(CELL).agg(value=("rate", "mean"), n_usable=("n_usable", "sum"), n_total=("n_total", "sum"))
        tables.append(s.reset_index().assign(track=track, judge=judge, judge_model=judge_model(judge)))
    long = pd.concat(tables, ignore_index=True)
    diffs = []
    for track in ["A", "B"]:
        piv = long[(long.track == track) & long.judge.isin(JUDGES)].pivot_table(index=CELL, columns="judge", values="value")
        for a, b in [("gpt", "gemini"), ("gpt", "deepseek"), ("deepseek", "gemini")]:
            diffs.append((piv[a] - piv[b]).rename("diff").reset_index().assign(track=track, pair=f"{a}-{b}"))
    return {"matrices_long": long, "judge_difference_matrices": pd.concat(diffs, ignore_index=True),
            "cell_rates_per_seed": pd.concat(per_seed, ignore_index=True)}


# ------------------------------------------------------------------
# A2 headline gap per seed: withholding minus constructed
# ------------------------------------------------------------------
def _gap(arr):
    wi = [CATEGORIES.index(x) for x in WITHHOLDING]
    ci = [CATEGORIES.index(x) for x in CONSTRUCTED]
    return np.nanmean(arr[wi], axis=0) - np.nanmean(arr[ci], axis=0)  # (language, model)


def a2_headline_gap(df):
    rows = []
    for track, judge, col in track_specs():
        for seed, rs in seed_cell_rates(df, col).groupby("sample_idx"):
            gap = _gap(cell_array(rs))
            for li, lang in enumerate(LANGUAGES):
                for mi, model in enumerate(["qwen", "llama"]):
                    rows.append(dict(track=track, judge=judge, language=lang, model=model, sample_idx=seed,
                                     gap=gap[li, mi]))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["track", "judge", "language", "model"]).agg(
        mean=("gap", "mean"), sd=("gap", "std"), min=("gap", "min"), max=("gap", "max"),
        n_seeds=("gap", "count"), n_positive=("gap", lambda x: int((x > 0).sum())),
    ).reset_index()
    summary["sign_holds_all_seeds"] = (summary.n_positive == summary.n_seeds) | (summary.n_positive == 0)
    summary["focus_language"] = summary.language.isin(FOCUS_LANGUAGES)
    return {"headline_gap_per_seed": per_seed, "headline_gap_summary": summary}


# ------------------------------------------------------------------
# A3 cell stability
# ------------------------------------------------------------------
def a3_cell_stability(df, top_n=10):
    out = []
    for track, judge, col in track_specs():
        s = seed_cell_rates(df, col).groupby(CELL).agg(
            mean=("rate", "mean"), sd=("rate", "std"), min=("rate", "min"), max=("rate", "max"),
            n_seeds=("rate", "count"), n_usable=("n_usable", "sum")).reset_index()
        out.append(s.assign(track=track, judge=judge))
    cells = pd.concat(out, ignore_index=True)
    cells["range"] = cells["max"] - cells["min"]
    top = (cells.sort_values("sd", ascending=False).groupby(["track", "judge"]).head(top_n)
           .sort_values(["track", "judge", "sd"], ascending=[True, True, False]))
    overview = cells.groupby(["track", "judge"]).agg(
        n_cells=("sd", "size"), median_sd=("sd", "median"), mean_sd=("sd", "mean"), max_sd=("sd", "max"),
        max_range=("range", "max"),
    ).reset_index()
    return {"cell_stability": cells, "cell_stability_top": top, "cell_stability_overview": overview}


# ------------------------------------------------------------------
# A4 cross-model replication per seed
# ------------------------------------------------------------------
def a4_cross_model(df):
    rows = []
    for track, judge, col in track_specs():
        for seed, rs in seed_cell_rates(df, col).groupby("sample_idx"):
            piv = rs.pivot_table(index=["category", "language"], columns="model", values="rate").dropna()
            rows.append(dict(track=track, judge=judge, sample_idx=seed, n_cells=len(piv),
                             pearson=stats.pearsonr(piv.qwen, piv.llama)[0],
                             spearman=stats.spearmanr(piv.qwen, piv.llama)[0]))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["track", "judge"]).agg(
        pearson_mean=("pearson", "mean"), pearson_min=("pearson", "min"), pearson_max=("pearson", "max"),
        spearman_mean=("spearman", "mean"), spearman_min=("spearman", "min"), spearman_max=("spearman", "max"),
        n_cells=("n_cells", "first"), n_seeds=("pearson", "count"),
    ).reset_index()
    return {"cross_model_per_seed": per_seed, "cross_model_summary": summary}


# ------------------------------------------------------------------
# A5 decomposition per seed (balanced ANOVA on equally weighted cell means)
# ------------------------------------------------------------------
def anova_shares(y):
    """Sum-of-squares shares for a balanced (category, language, model) array."""
    C, L, M = y.shape
    m = y.mean()
    a = y.mean((1, 2)) - m
    b = y.mean((0, 2)) - m
    g = y.mean((0, 1)) - m
    ab = y.mean(2) - m - a[:, None] - b[None, :]
    ag = y.mean(1) - m - a[:, None] - g[None, :]
    bg = y.mean(0) - m - b[:, None] - g[None, :]
    abg = y - m - a[:, None, None] - b[None, :, None] - g[None, None, :] - ab[:, :, None] - ag[:, None, :] - bg[None]
    ss = {
        "category": L * M * (a ** 2).sum(),
        "language": C * M * (b ** 2).sum(),
        "model": C * L * (g ** 2).sum(),
        "category_x_language": M * (ab ** 2).sum(),
        "category_x_model": L * (ag ** 2).sum(),
        "language_x_model": C * (bg ** 2).sum(),
        "category_x_language_x_model": (abg ** 2).sum(),
    }
    total = ((y - m) ** 2).sum()
    return {k: v / total for k, v in ss.items()}


def a5_decomposition(df):
    rows = []
    for track, judge, col in track_specs():
        for seed, rs in seed_cell_rates(df, col).groupby("sample_idx"):
            for term, share in anova_shares(cell_array(rs)).items():
                rows.append(dict(track=track, judge=judge, sample_idx=seed, term=term, share=share))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["track", "judge", "term"]).agg(
        mean=("share", "mean"), sd=("share", "std"), min=("share", "min"), max=("share", "max"),
    ).reset_index()
    return {"decomposition_per_seed": per_seed, "decomposition_summary": summary}


# ------------------------------------------------------------------
# A6 judge confidence (Track B)
# ------------------------------------------------------------------
def _auc(score, label):
    """P(score of a positive > score of a negative); 0.5 = no relation."""
    score, label = np.asarray(score), np.asarray(label).astype(bool)
    if label.all() or not label.any():
        return np.nan
    u = stats.mannwhitneyu(score[label], score[~label]).statistic
    return u / (label.sum() * (~label).sum())


def a6_confidence(df):
    dist = []
    for j in JUDGES:
        for keys in [["language"], ["category"], ["category", "language"]]:
            t = df.groupby(keys)[f"{j}_cert"].agg(
                n="count", mean="mean", median="median",
                p10=lambda x: x.quantile(0.10), p25=lambda x: x.quantile(0.25)).reset_index()
            dist.append(t.assign(judge=j, judge_model=judge_model(j), grouping="x".join(keys)))
    dist = pd.concat(dist, ignore_index=True)

    c = df[df[[f"{j}_held" for j in JUDGES]].notna().all(axis=1)].copy()
    c["unanimous"] = c[[f"{j}_held" for j in JUDGES]].nunique(axis=1).eq(1)
    unan = []
    for j in JUDGES:
        for lang, g in [("all", c)] + list(c.groupby("language")):
            unan.append(dict(judge=j, language=lang, n=len(g), share_unanimous=g.unanimous.mean(),
                             mean_cert_unanimous=g.loc[g.unanimous, f"{j}_cert"].mean(),
                             mean_cert_split=g.loc[~g.unanimous, f"{j}_cert"].mean(),
                             auc_cert_predicts_unanimous=_auc(g[f"{j}_cert"], g.unanimous)))

    flips, flip_q = [], []
    for j in JUDGES:
        p = df.groupby(["prompt", "category", "language", "model"]).agg(
            n_usable=(f"{j}_held", "count"), frac_held=(f"{j}_held", "mean"), mean_cert=(f"{j}_cert", "mean"))
        p = p[p.n_usable >= MIN_SEEDS_FOR_FLIP].reset_index()
        p["mixed"] = (p.frac_held > 0) & (p.frac_held < 1)
        flips.append(dict(judge=j, n_prompts=len(p), share_mixed=p.mixed.mean(),
                          auc_low_cert_predicts_flip=_auc(-p.mean_cert, p.mixed),
                          spearman_cert_vs_flip=stats.spearmanr(p.mean_cert, p.mixed)[0]))
        # Within-judge quintiles describe the relation; they are not a decision threshold.
        p["cert_quintile"] = pd.qcut(p.mean_cert.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        q = p.groupby("cert_quintile", observed=True).agg(n=("mixed", "size"), share_mixed=("mixed", "mean"),
                                                          cert_min=("mean_cert", "min"), cert_max=("mean_cert", "max"))
        flip_q.append(q.reset_index().assign(judge=j))
    return {
        "confidence_distribution": dist,
        "confidence_vs_unanimity": pd.DataFrame(unan),
        "confidence_vs_flip": pd.DataFrame(flips),
        "confidence_vs_flip_by_quintile": pd.concat(flip_q, ignore_index=True),
    }


# ------------------------------------------------------------------
# A7 missing labels with reason codes (also C6)
# ------------------------------------------------------------------
def a7_missing(df):
    reason_rows, rate_rows, tests = [], [], []
    for j in JUDGES:
        status = df[f"{j}_status"]
        missing = status.ne("ok")
        reason = status.map(lambda s: "ok" if s == "ok" else REASON_CODES.get(s, "other"))
        r = (pd.crosstab([reason.rename("reason_code"), status.rename("status")], df.pressure_level)
             .reset_index().assign(judge=j, judge_model=judge_model(j)))
        r["total"] = r[PRESSURES].sum(axis=1)
        reason_rows.append(r)
        for factor in ["pressure_level", "category", "language", "model", "sample_idx"]:
            t = df.groupby(factor).agg(n=(f"{j}_status", "size")).reset_index()
            t["n_unjudged"] = missing.groupby(df[factor]).sum().values
            t["pct_unjudged"] = 100 * t.n_unjudged / t.n
            rate_rows.append(t.rename(columns={factor: "level"}).assign(factor=factor, judge=j))
            ct = pd.crosstab(df[factor], missing)
            if ct.shape[1] == 2:
                chi2, pval, dof, _ = stats.chi2_contingency(ct)
                v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
                tests.append(dict(judge=j, factor=factor, chi2=chi2, dof=dof, p_value=pval, cramers_v=v))
    complete = df[[f"{j}_status" for j in JUDGES]].eq("ok").all(axis=1)
    overall = pd.DataFrame(
        [dict(judge=j, judge_model=judge_model(j), n=len(df), n_unjudged=int(df[f"{j}_status"].ne("ok").sum()))
         for j in JUDGES]
        + [dict(judge="all_three_usable", judge_model="", n=len(df), n_unjudged=int((~complete).sum()))]
        + [dict(judge=k, judge_model=judge_model(k), n=len(df), n_unjudged=int(df[f"{k}_held"].isna().sum()))
           for k in CONSENSUS_LABELS]
    )
    overall["pct_unjudged"] = 100 * overall.n_unjudged / overall.n
    overall["n_usable"] = overall.n - overall.n_unjudged
    return {
        "missing_overall": overall,
        "missing_reasons_by_pressure": pd.concat(reason_rows, ignore_index=True),
        "missing_rate_by_factor": pd.concat(rate_rows, ignore_index=True),
        "missing_dependence_tests": pd.DataFrame(tests),
    }


# ------------------------------------------------------------------
# A8 pressure sensitivity
# ------------------------------------------------------------------
def _pressure_profile(df, col, weighting):
    """Rate per (model, pressure) for each seed, then the mean over seeds.
    weighting="pooled": over responses; "cells": equal category x language weights."""
    if weighting == "pooled":
        r = df.groupby(["model", "pressure_level", "sample_idx"])[col].mean()
    else:
        r = (df.groupby(["model", "pressure_level", "sample_idx", "category", "language"])[col].mean()
             .groupby(["model", "pressure_level", "sample_idx"]).mean())
    return r.groupby(["model", "pressure_level"]).mean().unstack("pressure_level")


def _ordering(row, levels):
    vals = row[levels].sort_values()
    return dict(ordering=" < ".join(vals.index), lowest=vals.index[0], second_lowest=vals.index[1],
                spread_pp=100 * (vals.iloc[-1] - vals.iloc[0]))


def a8_pressure(df):
    rows, profiles = [], []
    for track, judge, col in track_specs():
        value_cols = [f"{j}_{'held' if track == 'A' else 'p'}" for j in JUDGES]
        variants = {"available_case": df, "complete_case": df[df[value_cols].notna().all(axis=1)]}
        if track == "A" and judge not in CONSENSUS_LABELS:
            variants["bound_missing_as_held"] = df.assign(**{col: df[col].fillna(1.0)})
            variants["bound_missing_as_violated"] = df.assign(**{col: df[col].fillna(0.0)})
        for vname, sub in variants.items():
            for weighting in ["pooled", "cells"]:
                for model, row in _pressure_profile(sub, col, weighting).iterrows():
                    for levels, tag in [(PRESSURES, "all_levels"), ([p for p in PRESSURES if p != "L3"], "excluding_L3")]:
                        rows.append(dict(track=track, judge=judge, variant=vname, weighting=weighting,
                                         levels=tag, model=model, **_ordering(row, levels)))
                    profiles.append(dict(track=track, judge=judge, variant=vname, weighting=weighting,
                                         model=model, **row.to_dict()))
    return {"pressure_orderings": pd.DataFrame(rows), "pressure_profiles": pd.DataFrame(profiles)}


# ------------------------------------------------------------------
# A9 within-prompt flip rate
# ------------------------------------------------------------------
def a9_flip_rate(df, min_seeds=MIN_SEEDS_FOR_FLIP):
    keys = ["prompt", "category", "language", "model"]
    class_rows, drop_rows, sd_rows = [], [], []
    for track, judge, col in track_specs():
        p = df.groupby(keys).agg(n_usable=(col, "count"), frac=(col, "mean"), sd=(col, "std")).reset_index()
        kept = p[p.n_usable >= min_seeds]
        drop_rows.append(dict(track=track, judge=judge, n_prompts=len(p), n_kept=len(kept), n_dropped=len(p) - len(kept)))
        if track == "A":
            kept = kept.assign(cls=np.select([kept.frac == 1, kept.frac == 0], ["always_held", "always_violated"], "mixed"))
            for grouping in [["category"], ["language"], ["category", "language"], ["model"], []]:
                t = (kept.groupby(grouping + ["cls"]).size().unstack("cls", fill_value=0)
                     if grouping else kept.cls.value_counts().to_frame().T)
                t = t.reindex(columns=["always_held", "always_violated", "mixed"], fill_value=0)
                n = t.sum(axis=1)
                t = t.div(n, axis=0).add_prefix("share_").assign(n_prompts=n).reset_index(drop=not grouping)
                class_rows.append(t.assign(judge=judge, grouping="x".join(grouping) or "all"))
        else:
            for grouping in [["category"], ["language"], ["category", "language"], []]:
                if grouping:
                    t = kept.groupby(grouping).agg(mean_sd_p=("sd", "mean"), median_sd_p=("sd", "median"),
                                                   n_prompts=("sd", "size")).reset_index()
                else:
                    t = pd.DataFrame([dict(mean_sd_p=kept.sd.mean(), median_sd_p=kept.sd.median(), n_prompts=len(kept))])
                sd_rows.append(t.assign(judge=judge, grouping="x".join(grouping) or "all"))
    return {"flip_classes": pd.concat(class_rows, ignore_index=True), "flip_dropped_prompts": pd.DataFrame(drop_rows),
            "flip_sd_p_held": pd.concat(sd_rows, ignore_index=True)}


# ------------------------------------------------------------------
# A10 variance components
# ------------------------------------------------------------------
LANGUAGE_GROUPS = {"yo_ig": ["yo", "ig"], "other8": [l for l in LANGUAGES if l not in ("yo", "ig")]}


def moment_components(d, col):
    """Nested ANOVA (method-of-moments) after removing category x language x model
    x pressure means: item -> prompt (id x model x language) -> sample."""
    d = d[d[col].notna()]
    d = d.assign(r=d[col] - d.groupby(CELL + ["pressure_level"])[col].transform("mean"))
    N = len(d)
    pm = d.groupby("prompt").r.agg(["mean", "count", "var"])
    pm["item"] = d.groupby("prompt").item.first()
    im = d.groupby("item").r.agg(["mean", "count"])
    P, I = len(pm), len(im)
    ss_e = ((pm["count"] - 1) * pm["var"].fillna(0)).sum()
    ss_p = (pm["count"] * (pm["mean"] - im.loc[pm["item"], "mean"].to_numpy()) ** 2).sum()
    ss_i = (im["count"] * (im["mean"] - d.r.mean()) ** 2).sum()
    ms_e, ms_p, ms_i = ss_e / (N - P), ss_p / (P - I), ss_i / (I - 1)
    s_e = ms_e
    s_p = max((ms_p - ms_e) / (N / P), 0.0)
    s_i = max((ms_i - ms_p) / (N / I), 0.0)
    tot = s_e + s_p + s_i
    return dict(method="moments", n=N, n_prompts=P, n_items=I, var_item=s_i, var_prompt=s_p, var_residual=s_e,
                icc_item=s_i / tot, icc_prompt=(s_i + s_p) / tot)


def glmm_components(d, col):
    """Logistic mixed model held ~ category*language*model + pressure + (1|item/prompt),
    fitted by variational Bayes (statsmodels BinomialBayesMixedGLM). Latent-scale ICCs."""
    import scipy.sparse as sp
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = d[d[col].notna()]
    fe_cell = pd.get_dummies(d.category + "|" + d.language + "|" + d.model, drop_first=True, dtype=float)
    fe_pressure = pd.get_dummies(d.pressure_level, drop_first=True, dtype=float)
    exog = np.column_stack([np.ones(len(d)), fe_cell.to_numpy(), fe_pressure.to_numpy()])
    item_codes = d.item.astype("category").cat.codes.to_numpy()
    prompt_codes = d.prompt.astype("category").cat.codes.to_numpy()
    n_i, n_p = item_codes.max() + 1, prompt_codes.max() + 1
    rows = np.arange(len(d))
    z = sp.hstack([
        sp.csr_matrix((np.ones(len(d)), (rows, item_codes)), shape=(len(d), n_i)),
        sp.csr_matrix((np.ones(len(d)), (rows, prompt_codes)), shape=(len(d), n_p)),
    ]).tocsr()
    ident = np.concatenate([np.zeros(n_i, int), np.ones(n_p, int)])
    model = BinomialBayesMixedGLM(d[col].to_numpy(), exog, z, ident, vcp_p=1, fe_p=2)
    res = model.fit_vb(fit_method="L-BFGS-B", minim_opts={"maxiter": 2000})
    s_i, s_p = np.exp(2 * res.vcp_mean)
    tot = s_i + s_p + np.pi ** 2 / 3
    return dict(method="glmm_vb_latent", n=len(d), n_prompts=int(n_p), n_items=int(n_i), var_item=s_i,
                var_prompt=s_p, var_residual=np.pi ** 2 / 3, icc_item=s_i / tot, icc_prompt=(s_i + s_p) / tot)


def a10_variance_components(df, glmm_judges=("cons", "gpt", "gemini", "deepseek")):
    rows = []
    for track, judge, col in track_specs():
        for gname, langs in LANGUAGE_GROUPS.items():
            d = df[df.language.isin(langs)]
            fits = [moment_components(d, col)]
            if track == "A" and judge in glmm_judges:
                fits.append(glmm_components(d, col))
            for f in fits:
                rows.append(dict(track=track, judge=judge, language_group=gname, **f))
    return {"variance_components": pd.DataFrame(rows)}


# ------------------------------------------------------------------
# A11 cluster bootstrap (Track A)
# ------------------------------------------------------------------
def _unit_arrays(df, col, unit):
    """Sum of held and usable count per (unit, language, model), pooled over seeds."""
    g = df.groupby([unit, "language", "model"])[col].agg(["sum", "count"])
    units = sorted(df[unit].unique())
    idx = pd.MultiIndex.from_product([units, LANGUAGES, ["qwen", "llama"]])
    g = g.reindex(idx, fill_value=0)
    S = g["sum"].to_numpy().reshape(len(units), len(LANGUAGES), 2)
    N = g["count"].to_numpy().reshape(len(units), len(LANGUAGES), 2)
    cats = np.array([CATEGORIES.index(u.split("|")[0]) for u in units])
    return S, N, cats


def _rates_from_weights(W, S, N, cats):
    onehot = np.eye(len(CATEGORIES))[cats]  # (units, C)
    num = np.einsum("bu,uc,ulm->bclm", W, onehot, S)
    den = np.einsum("bu,uc,ulm->bclm", W, onehot, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def a11_bootstrap(df, n_boot=1000, seed=0):
    """unit="item": 66 base items (category x topic x framing), as in the plan.
    unit="task": 22 rule tasks (category x topic); the three framings share one
    user query, so they are resampled together. Both stratified by category."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, judge, col in [s for s in track_specs() if s[0] == "A"]:
        for unit in ["item", "task"]:
            S, N, cats = _unit_arrays(df, col, unit)
            W = np.zeros((n_boot, len(cats)))
            for c in np.unique(cats):
                members = np.flatnonzero(cats == c)
                draws = rng.choice(members, size=(n_boot, len(members)), replace=True)
                for b in range(n_boot):
                    W[b] += np.bincount(draws[b], minlength=len(cats))
            point = _rates_from_weights(np.ones((1, len(cats))), S, N, cats)[0]
            boot = _rates_from_weights(W, S, N, cats)
            results = {}
            pg, bg = _gap(point), np.stack([_gap(x) for x in boot])
            for li, lang in enumerate(LANGUAGES):
                for mi, model in enumerate(["qwen", "llama"]):
                    results[("gap", lang, model)] = (pg[li, mi], bg[:, li, mi])
            ps, bs = anova_shares(point), [anova_shares(x) for x in boot]
            for term in ps:
                results[("share", term, "")] = (ps[term], np.array([b[term] for b in bs]))
            for (kind, a, b), (pt, dist) in results.items():
                rows.append(dict(judge=judge, cluster_unit=unit, n_units=len(cats), statistic=kind,
                                 language_or_term=a, model=b, point=pt, boot_sd=np.nanstd(dist, ddof=1),
                                 ci_low=np.nanpercentile(dist, 2.5), ci_high=np.nanpercentile(dist, 97.5),
                                 share_positive=np.nanmean(dist > 0)))
    return {"bootstrap": pd.DataFrame(rows)}


# ------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------
def c1_coverage(df):
    rows = []
    for (model, lang), g in df.groupby(["model", "language"]):
        rows.append(dict(model=model, language=lang, n_responses=len(g), n_prompts=g.id.nunique(),
                         samples_per_prompt=str(sorted(g.groupby("id").size().unique())),
                         n_items=g.item.nunique(), n_tasks=g.task.nunique(), n_framings=g.grammar_type.nunique(),
                         n_status_wordings=g.rule_status.nunique(), pressure_levels=str(sorted(g.pressure_level.unique()))))
    return {"coverage": pd.DataFrame(rows)}


def c2_label_origin(df):
    """Reported verdict vs the judge's own score: verdict is the weighted verdict, P(HELD) > 0.5."""
    rows = []
    for j in JUDGES:
        ok = df[f"{j}_status"].eq("ok")
        thr = (df.loc[ok, f"{j}_p"] > 0.5).astype(float)
        rows.append(dict(judge=j, judge_model=judge_model(j), n_ok=int(ok.sum()),
                         share_verdict_equals_p_held_gt_half=float((thr == df.loc[ok, f"{j}_held"]).mean())))
    return {"label_origin": pd.DataFrame(rows)}


def c3_judge_agreement(df):
    pair = []
    for a, b in [("deepseek", "gemini"), ("gpt", "gemini"), ("gpt", "deepseek")]:
        for lang, g in [("all", df)] + list(df.groupby("language")):
            both = g[[f"{a}_held", f"{b}_held"]].dropna()
            pair.append(dict(pair=f"{a}-{b}", language=lang, n=len(both),
                             agreement=(both.iloc[:, 0] == both.iloc[:, 1]).mean(),
                             held_rate_first=both.iloc[:, 0].mean(), held_rate_second=both.iloc[:, 1].mean()))
    rates = [dict(judge=j, judge_model=judge_model(j), held_rate=df[f"{j}_held"].mean(), n_usable=int(df[f"{j}_held"].count()))
             for j in JUDGES + CONSENSUS_LABELS]
    return {"judge_pairwise_agreement": pd.DataFrame(pair), "held_rate_by_judge": pd.DataFrame(rates)}


def c5_qwen_batch(df):
    """GPT/DeepSeek scored Qwen tr/ur/yo in a later addendum run. Gemini scored all
    cells in one run, so (judge - Gemini) on the same responses isolates a scoring-
    batch effect from response/language differences."""
    rows = []
    for j in ["gpt", "deepseek"]:
        for (model, lang), g in df.groupby(["model", "language"]):
            both = g[[f"{j}_p", "gemini_p"]].dropna()
            rows.append(dict(judge=j, model=model, language=lang,
                             batch="addendum" if (g[f"{j}_rev"] == "86ef3aa").all() else "original",
                             run_fingerprint=g[f"{j}_fp"].mode().iat[0][:12],
                             pct_unjudged=100 * g[f"{j}_held"].isna().mean(), held_rate=g[f"{j}_held"].mean(),
                             mean_p_held=g[f"{j}_p"].mean(), mean_cert=g[f"{j}_cert"].mean(),
                             mean_p_minus_gemini=(both.iloc[:, 0] - both.iloc[:, 1]).mean(),
                             gemini_pct_unjudged=100 * g.gemini_held.isna().mean()))
    cells = pd.DataFrame(rows)
    addendum = sorted(cells.query("batch == 'addendum'").language.unique())
    did = []
    for j, c in cells.groupby("judge"):
        for metric in ["mean_p_minus_gemini", "pct_unjudged", "held_rate", "mean_cert"]:
            diff = c.query("model == 'qwen'").set_index("language")[metric] - c.query("model == 'llama'").set_index("language")[metric]
            did.append(dict(judge=j, metric=metric, addendum_languages=",".join(addendum),
                            qwen_minus_llama_addendum_langs=diff[addendum].mean(),
                            qwen_minus_llama_original_langs=diff.drop(addendum).mean(),
                            difference_in_differences=diff[addendum].mean() - diff.drop(addendum).mean()))
    return {"qwen_batch_cells": cells, "qwen_batch_did": pd.DataFrame(did)}


def c7_reasoning(conf_dir):
    rows = []
    for path in ["combined-2026-10-23/summary.json", "gemini-2026-10-26/summary.json"]:
        with open(os.path.join(conf_dir, path)) as fh:
            s = json.load(fh)
        for u in s["usage_and_status"]["reported_usage"]:
            n = sum(c["n"] for c in s["usage_and_status"]["counts"] if c["judge_id"] == u["judge_id"])
            rows.append(dict(judge_id=u["judge_id"], n_records=n, completion_tokens=u["completion_tokens"],
                             completion_tokens_per_record=u["completion_tokens"] / n))
    t = pd.DataFrame(rows)
    t["reasons_before_verdict"] = t.completion_tokens_per_record > 20
    return {"judge_reasoning_tokens": t}
