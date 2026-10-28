"""Pre-registered checks (C1-C7) and analyses (A1-A11) on the 5-sample run.

Conventions (plan section 2):
  - Only the numeric judge scores are analysed: the adherence score P(HELD) and
    the confidence score. The binary verdict is never used; scores are averaged,
    never thresholded.
  - Scores: "mean" = 3-judge mean adherence score (primary; responses all three
    judges scored), plus each judge's own P(HELD).
  - Cell = category x language x model. Cell means are computed over the cell's
    prompts within a seed first, then summarized across seeds.
  - Unscored responses are NaN and drop out of the denominator; tables carry n.
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
MIN_SEEDS_PER_PROMPT = 4
SCORES = ["mean"] + JUDGES
SCORE_NAMES = {"mean": "3-judge mean (primary)", **JUDGE_MODELS}


def score_specs():
    """(score, column) pairs: the 3-judge mean first, then each judge."""
    return [(s, f"{s}_p") for s in SCORES]


def seed_cell_means(df, col, keys=CELL):
    g = df.groupby(list(keys) + ["sample_idx"], observed=True)[col]
    return g.agg(value="mean", n_usable="count", n_total="size").reset_index()


def cell_array(means):
    """(category, language, model) -> numpy array in fixed order."""
    piv = means.set_index(CELL)["value"]
    idx = pd.MultiIndex.from_product([CATEGORIES, LANGUAGES, ["qwen", "llama"]], names=CELL)
    return piv.reindex(idx).to_numpy().reshape(len(CATEGORIES), len(LANGUAGES), 2)


# ------------------------------------------------------------------
# A1 per-score, per-model matrices
# ------------------------------------------------------------------
def a1_matrices(df):
    tables, per_seed = [], []
    for score, col in score_specs():
        r = seed_cell_means(df, col)
        per_seed.append(r.assign(score=score))
        s = r.groupby(CELL).agg(value=("value", "mean"), n_usable=("n_usable", "sum"), n_total=("n_total", "sum"))
        tables.append(s.reset_index().assign(score=score, score_name=SCORE_NAMES[score]))
    long = pd.concat(tables, ignore_index=True)
    piv = long[long.score.isin(JUDGES)].pivot_table(index=CELL, columns="score", values="value")
    diffs = [(piv[a] - piv[b]).rename("diff").reset_index().assign(pair=f"{a}-{b}")
             for a, b in [("gpt", "gemini"), ("gpt", "deepseek"), ("deepseek", "gemini")]]
    return {"matrices_long": long, "judge_difference_matrices": pd.concat(diffs, ignore_index=True),
            "cell_means_per_seed": pd.concat(per_seed, ignore_index=True)}


# ------------------------------------------------------------------
# A2 headline gap per seed: withholding minus constructed
# ------------------------------------------------------------------
def _gap(arr):
    wi = [CATEGORIES.index(x) for x in WITHHOLDING]
    ci = [CATEGORIES.index(x) for x in CONSTRUCTED]
    return np.nanmean(arr[wi], axis=0) - np.nanmean(arr[ci], axis=0)  # (language, model)


def a2_headline_gap(df):
    rows = []
    for score, col in score_specs():
        for seed, rs in seed_cell_means(df, col).groupby("sample_idx"):
            gap = _gap(cell_array(rs))
            for li, lang in enumerate(LANGUAGES):
                for mi, model in enumerate(["qwen", "llama"]):
                    rows.append(dict(score=score, language=lang, model=model, sample_idx=seed, gap=gap[li, mi]))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["score", "language", "model"]).agg(
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
    for score, col in score_specs():
        s = seed_cell_means(df, col).groupby(CELL).agg(
            mean=("value", "mean"), sd=("value", "std"), min=("value", "min"), max=("value", "max"),
            n_seeds=("value", "count"), n_usable=("n_usable", "sum")).reset_index()
        out.append(s.assign(score=score))
    cells = pd.concat(out, ignore_index=True)
    cells["range"] = cells["max"] - cells["min"]
    top = (cells.sort_values("sd", ascending=False).groupby("score").head(top_n)
           .sort_values(["score", "sd"], ascending=[True, False]))
    overview = cells.groupby("score").agg(
        n_cells=("sd", "size"), median_sd=("sd", "median"), mean_sd=("sd", "mean"), max_sd=("sd", "max"),
        max_range=("range", "max"),
    ).reset_index()
    return {"cell_stability": cells, "cell_stability_top": top, "cell_stability_overview": overview}


# ------------------------------------------------------------------
# A4 cross-model replication per seed
# ------------------------------------------------------------------
def a4_cross_model(df):
    rows = []
    for score, col in score_specs():
        for seed, rs in seed_cell_means(df, col).groupby("sample_idx"):
            piv = rs.pivot_table(index=["category", "language"], columns="model", values="value").dropna()
            rows.append(dict(score=score, sample_idx=seed, n_cells=len(piv),
                             pearson=stats.pearsonr(piv.qwen, piv.llama)[0],
                             spearman=stats.spearmanr(piv.qwen, piv.llama)[0]))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("score").agg(
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
    for score, col in score_specs():
        for seed, rs in seed_cell_means(df, col).groupby("sample_idx"):
            for term, share in anova_shares(cell_array(rs)).items():
                rows.append(dict(score=score, sample_idx=seed, term=term, share=share))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["score", "term"]).agg(
        mean=("share", "mean"), sd=("share", "std"), min=("share", "min"), max=("share", "max"),
    ).reset_index()
    return {"decomposition_per_seed": per_seed, "decomposition_summary": summary}


# ------------------------------------------------------------------
# A6 judge confidence
# ------------------------------------------------------------------
def prompt_variability(df, col, min_seeds=MIN_SEEDS_PER_PROMPT):
    """Per prompt (id x model x language): SD of the score across its samples."""
    p = df.groupby(["prompt", "category", "language", "model"]).agg(
        n_usable=(col, "count"), mean=(col, "mean"), sd=(col, "std")).reset_index()
    return p, p[p.n_usable >= min_seeds]


def a6_confidence(df):
    dist = []
    for j in JUDGES:
        for keys in [["language"], ["category"], ["category", "language"]]:
            t = df.groupby(keys)[f"{j}_cert"].agg(
                n="count", mean="mean", median="median",
                p10=lambda x: x.quantile(0.10), p25=lambda x: x.quantile(0.25)).reset_index()
            dist.append(t.assign(judge=j, judge_model=JUDGE_MODELS[j], grouping="x".join(keys)))
    dist = pd.concat(dist, ignore_index=True)

    # Spread between the three judges' scores on the same response (all three scored).
    c = df[df.mean_p.notna()]
    spread = c[[f"{j}_p" for j in JUDGES]].std(axis=1)
    agree = []
    for j in JUDGES:
        for lang, g in [("all", c)] + list(c.groupby("language")):
            agree.append(dict(judge=j, language=lang, n=len(g), mean_judge_spread=spread[g.index].mean(),
                              spearman_conf_vs_judge_spread=stats.spearmanr(g[f"{j}_cert"], spread[g.index])[0]))

    var, var_q = [], []
    for j in JUDGES:
        _, p = prompt_variability(df, f"{j}_p")
        p = p.assign(mean_conf=df.groupby("prompt")[f"{j}_cert"].mean().reindex(p.prompt).to_numpy())
        var.append(dict(judge=j, n_prompts=len(p), mean_within_prompt_sd=p.sd.mean(),
                        spearman_conf_vs_within_prompt_sd=stats.spearmanr(p.mean_conf, p.sd)[0]))
        # Within-judge quintiles describe the relation; they are not a decision threshold.
        p["conf_quintile"] = pd.qcut(p.mean_conf.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        q = p.groupby("conf_quintile", observed=True).agg(
            n=("sd", "size"), mean_within_prompt_sd=("sd", "mean"), median_within_prompt_sd=("sd", "median"),
            conf_min=("mean_conf", "min"), conf_max=("mean_conf", "max"))
        var_q.append(q.reset_index().assign(judge=j))
    return {
        "confidence_distribution": dist,
        "confidence_vs_judge_spread": pd.DataFrame(agree),
        "confidence_vs_within_prompt_sd": pd.DataFrame(var),
        "confidence_vs_within_prompt_sd_by_quintile": pd.concat(var_q, ignore_index=True),
    }


# ------------------------------------------------------------------
# A7 unscored responses with reason codes (also C6)
# ------------------------------------------------------------------
def a7_missing(df):
    reason_rows, rate_rows, tests = [], [], []
    for j in JUDGES:
        status = df[f"{j}_status"]
        missing = status.ne("ok")
        reason = status.map(lambda s: "ok" if s == "ok" else REASON_CODES.get(s, "other"))
        r = (pd.crosstab([reason.rename("reason_code"), status.rename("status")], df.pressure_level)
             .reset_index().assign(judge=j, judge_model=JUDGE_MODELS[j]))
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
    overall = pd.DataFrame(
        [dict(score=j, score_name=JUDGE_MODELS[j], n=len(df), n_unjudged=int(df[f"{j}_status"].ne("ok").sum()))
         for j in JUDGES]
        + [dict(score="mean", score_name=SCORE_NAMES["mean"], n=len(df), n_unjudged=int(df.mean_p.isna().sum()))]
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
    """Mean score per (model, pressure) for each seed, then the mean over seeds.
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
                spread_points=100 * (vals.iloc[-1] - vals.iloc[0]))


def a8_pressure(df):
    rows, profiles = [], []
    complete = df[df.mean_p.notna()]
    for score, col in score_specs():
        # The 3-judge mean is defined only on complete cases, so it has one variant.
        variants = {"available_case": df}
        if score in JUDGES:
            variants["complete_case"] = complete
            variants["bound_missing_as_0"] = df.assign(**{col: df[col].fillna(0.0)})
            variants["bound_missing_as_100"] = df.assign(**{col: df[col].fillna(1.0)})
        for vname, sub in variants.items():
            for weighting in ["pooled", "cells"]:
                for model, row in _pressure_profile(sub, col, weighting).iterrows():
                    for levels, tag in [(PRESSURES, "all_levels"), ([p for p in PRESSURES if p != "L3"], "excluding_L3")]:
                        rows.append(dict(score=score, variant=vname, weighting=weighting,
                                         levels=tag, model=model, **_ordering(row, levels)))
                    profiles.append(dict(score=score, variant=vname, weighting=weighting, model=model, **row.to_dict()))
    return {"pressure_orderings": pd.DataFrame(rows), "pressure_profiles": pd.DataFrame(profiles)}


# ------------------------------------------------------------------
# A9 within-prompt variability across samples
# ------------------------------------------------------------------
def a9_within_prompt(df, min_seeds=MIN_SEEDS_PER_PROMPT):
    drop_rows, sd_rows = [], []
    for score, col in score_specs():
        p, kept = prompt_variability(df, col, min_seeds)
        drop_rows.append(dict(score=score, n_prompts=len(p), n_kept=len(kept), n_dropped=len(p) - len(kept)))
        for grouping in [["category"], ["language"], ["category", "language"], ["model"], []]:
            agg = dict(mean_sd=("sd", "mean"), median_sd=("sd", "median"), p90_sd=("sd", lambda x: x.quantile(0.9)),
                       n_prompts=("sd", "size"))
            t = kept.groupby(grouping).agg(**agg).reset_index() if grouping else kept.assign(_=0).groupby("_").agg(**agg).reset_index(drop=True)
            sd_rows.append(t.assign(score=score, grouping="x".join(grouping) or "all"))
    return {"within_prompt_sd": pd.concat(sd_rows, ignore_index=True),
            "within_prompt_dropped_prompts": pd.DataFrame(drop_rows)}


# ------------------------------------------------------------------
# A10 variance components per language
# ------------------------------------------------------------------
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


def a10_variance_components(df):
    rows = []
    for score, col in score_specs():
        for lang in LANGUAGES + ["all"]:
            d = df if lang == "all" else df[df.language == lang]
            rows.append(dict(score=score, language=lang, **moment_components(d, col)))
    return {"variance_components": pd.DataFrame(rows)}


# ------------------------------------------------------------------
# A11 cluster bootstrap
# ------------------------------------------------------------------
def _unit_arrays(df, col, unit):
    """Score sum and usable count per (unit, language, model), pooled over seeds."""
    g = df.groupby([unit, "language", "model"])[col].agg(["sum", "count"])
    units = sorted(df[unit].unique())
    idx = pd.MultiIndex.from_product([units, LANGUAGES, ["qwen", "llama"]])
    g = g.reindex(idx, fill_value=0)
    S = g["sum"].to_numpy().reshape(len(units), len(LANGUAGES), 2)
    N = g["count"].to_numpy().reshape(len(units), len(LANGUAGES), 2)
    cats = np.array([CATEGORIES.index(u.split("|")[0]) for u in units])
    return S, N, cats


def _means_from_weights(W, S, N, cats):
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
    for score, col in score_specs():
        for unit in ["item", "task"]:
            S, N, cats = _unit_arrays(df, col, unit)
            W = np.zeros((n_boot, len(cats)))
            for c in np.unique(cats):
                members = np.flatnonzero(cats == c)
                draws = rng.choice(members, size=(n_boot, len(members)), replace=True)
                for b in range(n_boot):
                    W[b] += np.bincount(draws[b], minlength=len(cats))
            point = _means_from_weights(np.ones((1, len(cats))), S, N, cats)[0]
            boot = _means_from_weights(W, S, N, cats)
            results = {}
            pg, bg = _gap(point), np.stack([_gap(x) for x in boot])
            for li, lang in enumerate(LANGUAGES):
                for mi, model in enumerate(["qwen", "llama"]):
                    results[("gap", lang, model)] = (pg[li, mi], bg[:, li, mi])
            ps, bs = anova_shares(point), [anova_shares(x) for x in boot]
            for term in ps:
                results[("share", term, "")] = (ps[term], np.array([b[term] for b in bs]))
            for (kind, a, b), (pt, dist) in results.items():
                rows.append(dict(score=score, cluster_unit=unit, n_units=len(cats), statistic=kind,
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


def c3_judge_agreement(df):
    """Agreement between the judges' adherence scores on the same responses."""
    pair = []
    for a, b in [("deepseek", "gemini"), ("gpt", "gemini"), ("gpt", "deepseek")]:
        for lang, g in [("all", df)] + list(df.groupby("language")):
            both = g[[f"{a}_p", f"{b}_p"]].dropna()
            x, y = both.iloc[:, 0], both.iloc[:, 1]
            pair.append(dict(pair=f"{a}-{b}", language=lang, n=len(both), pearson=stats.pearsonr(x, y)[0],
                             spearman=stats.spearmanr(x, y)[0], mean_abs_diff=(x - y).abs().mean(),
                             mean_first=x.mean(), mean_second=y.mean()))
    means = [dict(score=s, score_name=SCORE_NAMES[s], mean_score=df[col].mean(), n_usable=int(df[col].count()))
             for s, col in score_specs()]
    return {"judge_pairwise_agreement": pd.DataFrame(pair), "mean_score_by_judge": pd.DataFrame(means)}


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
                             pct_unjudged=100 * g[f"{j}_p"].isna().mean(),
                             mean_p_held=g[f"{j}_p"].mean(), mean_conf=g[f"{j}_cert"].mean(),
                             mean_p_minus_gemini=(both.iloc[:, 0] - both.iloc[:, 1]).mean(),
                             gemini_pct_unjudged=100 * g.gemini_p.isna().mean()))
    cells = pd.DataFrame(rows)
    addendum = sorted(cells.query("batch == 'addendum'").language.unique())
    did = []
    for j, c in cells.groupby("judge"):
        for metric in ["mean_p_minus_gemini", "pct_unjudged", "mean_p_held", "mean_conf"]:
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
