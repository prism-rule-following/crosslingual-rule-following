"""Paired baseline-vs-rule significance, per (model, language) and pooled.

Input: two files (.jsonl or .csv), one row per scored response, with columns
  language, category, topic, pressure_level, adherent   (adherent = 0/1)
and optionally model_id. 'baseline' = no-rule responses scored against the ACTIVE
checker ("would this have satisfied the rule"); 'rule' = the same cells from the
normal rule-in-system-prompt runs (active side).

Unit of analysis = the CELL (category x topic x pressure_level), NOT the response:
responses to the same prompt are not independent, so we average to a rate per cell
first, then test across cells (paired by cell). Reports mean difference with a
bootstrap CI over cells, a sign test and a Wilcoxon signed-rank test.

  python baseline_stats.py --baseline base_scored.jsonl --rule rule_scored.jsonl [--pressure L0]
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats

CELL = ["category", "topic", "pressure_level"]

def load(p):
    return pd.read_csv(p) if p.endswith(".csv") else pd.read_json(p, lines=True)

def cell_rates(df, name):
    g = ["model_id", "language"] if "model_id" in df.columns else ["language"]
    return df.groupby(g + CELL)["adherent"].mean().rename(name).reset_index()

def paired(d, label, rng, n_boot=10000):
    x = d["diff"].to_numpy(); n = len(x)
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    sign_p = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
    try:
        w_p = stats.wilcoxon(x).pvalue if (x != 0).any() else np.nan
    except ValueError:
        w_p = np.nan
    return dict(group=label, cells=n, baseline_rate=d["base"].mean(), rule_rate=d["rule"].mean(),
                mean_diff=x.mean(), ci95_lo=lo, ci95_hi=hi, cells_rule_higher=pos, cells_base_higher=neg,
                sign_p=sign_p, wilcoxon_p=w_p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True); ap.add_argument("--rule", required=True)
    ap.add_argument("--pressure", default=None, help="restrict to one pressure level, e.g. L0")
    a = ap.parse_args()
    b, r = load(a.baseline), load(a.rule)
    if a.pressure:
        b, r = b[b.pressure_level == a.pressure], r[r.pressure_level == a.pressure]
    g = ["model_id", "language"] if "model_id" in b.columns and "model_id" in r.columns else ["language"]
    m = cell_rates(b, "base").merge(cell_rates(r, "rule"), on=g + CELL, how="inner")
    if m.empty:
        raise SystemExit("no matching cells between the two files; check column values")
    m["diff"] = m["rule"] - m["base"]
    rng = np.random.default_rng(0)
    tag = lambda k: " / ".join(map(str, k if isinstance(k, tuple) else (k,)))
    rows = []
    # 1) each pressure level on its own: n = base queries (26)
    for k, sub in m.groupby(g + ["pressure_level"]):
        rows.append(paired(sub, tag(k), rng))
    # 2) all levels together. L0/L1/L4 are the SAME base queries with different wording, so they are
    #    not independent: average over levels within each base query (category, topic) first.
    base = ["category", "topic"]
    def collapse(df, by):
        c = df.groupby(by + base)[["base", "rule"]].mean().reset_index(); c["diff"] = c["rule"] - c["base"]; return c
    if m["pressure_level"].nunique() > 1:
        for k, sub in collapse(m, g).groupby(g):
            rows.append(paired(sub, tag(k) + " / ALL LEVELS (per base query)", rng))
    # 3) pooled over languages too (same queries, translated): unit = base query
    pg = ["model_id"] if "model_id" in g else []
    pooled_all = collapse(m, pg)
    for k, sub in (pooled_all.groupby(pg) if pg else [("ALL", pooled_all)]):
        rows.append(paired(sub, tag(k) + " / POOLED languages+levels", rng))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.3g}")
    print(out.to_string(index=False))
    print("\nper category (descriptive only, few base queries each):")
    print(m.groupby(g + ["category"])[["base", "rule", "diff"]].mean().to_string())
    print("\nNote: 'cells' = independent base queries. More samples or pressure wordings do not add scenarios.")

if __name__ == "__main__":
    main()
