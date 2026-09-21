"""Preliminary structural analysis of the all-language vector-patching run.

Pure pandas/numpy, no GPU. Computes script-level, degeneration-level, and
echo-level features for every row, then compares arms (baseline/dom/w)
within each language at the prompt level, across layers, and across
low/mid/high-resource tiers. These are mechanical screens, not adherence
verdicts -- judging is a separate pass.
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "crosslingual-rule-following/vector-patching-responses"
FILE = "qwen3-8b/exp2_all_langs_20260903_191445.parquet"
LANGUAGES = ["de", "hi", "ig", "it", "ko", "ru", "tr", "ur", "yo"]
LAYERS = [15, 24, 27, 29, 31]
TIERS = {
    "de": "high", "it": "high", "ko": "high", "ru": "high", "tr": "high",
    "hi": "mid", "ur": "mid", "ig": "low", "yo": "low",
}
TARGET_SCRIPT = {
    "de": "Latn", "it": "Latn", "tr": "Latn", "ig": "Latn", "yo": "Latn",
    "hi": "Deva", "ko": "Hang", "ru": "Cyrl", "ur": "Arab",
}

_SCRIPT_RANGES = {
    "Latn": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)],
    "Cyrl": [(0x400, 0x4FF)],
    "Deva": [(0x900, 0x97F)],
    "Hang": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "Arab": [(0x600, 0x6FF), (0x750, 0x77F)],
    "CJK": [(0x4E00, 0x9FFF), (0x3040, 0x30FF)],
}
_LATIN_UNITS = r"\d+\s*(?:mg|mcg|µg|ml|tablet\w*|pill\w*|dose\w*)"
_CYRILLIC_UNITS = r"\d+\s*(?:мг|мл|мкг|таблетк\w*|доз\w*)"


def script_fracs(text):
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for script, ranges in _SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] += 1
                break
    total = sum(counts.values()) or 1
    return {s: counts[s] / total for s in _SCRIPT_RANGES}


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def words(s):
    return len(re.findall(r"\S+", s or ""))


def max_ngram_repeat(s, n=3):
    toks = re.findall(r"\S+", s or "")
    if len(toks) < n:
        return 0
    return max(Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)).values())


def max_token_repeat(s):
    toks = re.findall(r"\S+", s or "")
    if not toks:
        return 0
    return max(Counter(toks).values())


def contains(s, needle):
    n = norm(needle)
    return len(n) >= 15 and n in norm(s)


def features(df):
    df = df.copy()
    df["words"] = df["response"].map(words)
    df["chars"] = df["response"].str.len()
    fracs = df["response"].map(script_fracs)
    df["cjk_frac"] = [f.get("CJK", 0.0) for f in fracs]
    df["target_frac"] = [f.get(TARGET_SCRIPT[l], 0.0) for l, f in zip(df["language"], fracs)]
    df["rep3"] = df["response"].map(max_ngram_repeat)
    df["rep_tok"] = df["response"].map(max_token_repeat)
    df["rep_loop"] = (df["rep3"] >= 20) | (df["rep_tok"] >= 10)
    df["query_echo"] = [contains(r, q) for r, q in zip(df["response"], df["user_query"])]
    df["rule_recite"] = [contains(r, q) for r, q in zip(df["response"], df["rule_clause"])]
    df["collapse"] = df["words"] <= 5
    df["dose_mention"] = df["response"].map(
        lambda s: bool(re.search(_LATIN_UNITS, s or "", re.I))
        or bool(re.search(_CYRILLIC_UNITS, s or "", re.I))
    )
    df["ack_long"] = df["words"] > 10
    return df


def pct(x):
    return round(100 * float(np.mean(x)), 1)


def med(x):
    return float(np.median(x)) if len(x) else None


def paired_delta(df, lang, metric, layer=None):
    base = df[(df.language == lang) & (df.vector_type == "none")]
    patched = df[(df.language == lang) & (df.vector_type == "w")]
    if layer is not None:
        patched = patched[patched.patch_layer == layer]
    joined = base.set_index("id")[metric].rename("b").to_frame().join(
        patched.groupby("id")[metric].agg("mean")
    )
    joined = joined.dropna()
    if len(joined) == 0:
        return None
    return joined[metric].to_numpy() - joined["b"].to_numpy()


def bootstrap_ci(deltas, n_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(n_iter)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    out_dir = Path("experimental/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    df = pd.read_parquet(path)
    df["patch_layer"] = df["patch_layer"].fillna(-1).astype(int)
    df = features(df)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "file": FILE}

    per_lang = {}
    for lang in LANGUAGES:
        sub = df[df.language == lang]
        rows = {}
        for arm in ["none", "dom", "w"]:
            a = sub[sub.vector_type == arm]
            rows[arm] = {
                "n": len(a),
                "median_words": med(a.words),
                "median_chars": med(a.chars),
                "target_script_frac": round(float(a.target_frac.mean()), 3),
                "cjk_frac": round(float(a.cjk_frac.mean()), 3),
                "off_script_pct": pct(a.target_frac < 0.5),
                "rep_loop_pct": pct(a.rep_loop),
                "median_rep3": med(a.rep3),
                "query_echo_pct": pct(a.query_echo),
                "rule_recite_pct": pct(a.rule_recite),
                "collapse_pct": pct(a.collapse),
            }
        per_lang[lang] = rows
    report["per_lang_per_arm"] = per_lang

    deltas = {}
    for lang in LANGUAGES:
        for metric in ["words", "target_frac", "cjk_frac", "rep_loop"]:
            for layer in LAYERS:
                d = paired_delta(df, lang, metric, layer=layer)
                if d is None or len(d) == 0:
                    continue
                lo, hi = bootstrap_ci(d)
                deltas[f"{lang}|{metric}|L{layer}"] = {
                    "n_prompts": int(len(d)),
                    "mean_delta": round(float(d.mean()), 3),
                    "ci95": [round(lo, 3), round(hi, 3)],
                    "sign": round(float(np.sign(d).mean()), 3),
                }
    report["paired_w_vs_baseline_deltas"] = deltas

    tiers = {}
    for tier in ["high", "mid", "low"]:
        sub = df[df.language.isin([l for l, t in TIERS.items() if t == tier])]
        tiers[tier] = {
            "n": len(sub),
            "languages": [l for l, t in TIERS.items() if t == tier],
        }
        for arm in ["none", "dom", "w"]:
            a = sub[sub.vector_type == arm]
            tiers[tier][arm] = {
                "median_words": med(a.words),
                "off_script_pct": pct(a.target_frac < 0.5),
                "cjk_frac": round(float(a.cjk_frac.mean()), 3),
                "rep_loop_pct": pct(a.rep_loop),
                "query_echo_pct": pct(a.query_echo),
                "rule_recite_pct": pct(a.rule_recite),
                "collapse_pct": pct(a.collapse),
            }
    report["by_tier"] = tiers

    layer_trend = {}
    for arm in ["none", "dom", "w"]:
        a = df[df.vector_type == arm]
        layer_trend[arm] = {}
        for layer in ([-1] if arm == "none" else LAYERS):
            l = a[a.patch_layer == layer]
            layer_trend[arm][str(layer)] = {
                "median_words": med(l.words),
                "cjk_frac": round(float(l.cjk_frac.mean()), 3),
                "rep_loop_pct": pct(l.rep_loop),
                "collapse_pct": pct(l.collapse),
                "query_echo_pct": pct(l.query_echo),
            }
    report["by_layer"] = layer_trend

    screens = {}
    ack = df[df.category == "ack_invert"]
    screens["ack_invert"] = {
        arm: {
            "median_words": med(a.words),
            "long_pct": pct(a.ack_long),
        } for arm, a in ack.groupby("vector_type")
    }
    dos = df[df.category == "no_dosage"]
    screens["no_dosage"] = {
        arm: {
            "dose_mention_pct": pct(a.dose_mention),
            "any_digits_pct": pct(a.response.str.contains(r"\d")),
        } for arm, a in dos.groupby("vector_type")
    }
    report["category_screens"] = screens

    samples = {}
    for lang in LANGUAGES:
        w_rows = df[(df.language == lang) & (df.vector_type == "w")]
        if len(w_rows):
            r = w_rows.iloc[0]
            samples[lang] = (r["response"] or "")[:200]
    report["samples"] = samples

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"all_langs_prelim_analysis_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")

    print("\n=== per language x arm (median words | off-script% | rep-loop% | echo% | recite%) ===")
    for lang in LANGUAGES:
        line = f"{lang:3s} ({TIERS[lang]:4s})"
        for arm in ["none", "dom", "w"]:
            r = per_lang[lang][arm]
            line += (f"  {arm:3s}: w{r['median_words']:>4.0f} off{r['off_script_pct']:>5.1f}% "
                     f"rep{r['rep_loop_pct']:>5.1f}% echo{r['query_echo_pct']:>5.1f}% "
                     f"rec{r['rule_recite_pct']:>5.1f}%")
        print(line)

    print("\n=== tier aggregates (baseline / dom / w) ===")
    for tier in ["high", "mid", "low"]:
        t = tiers[tier]
        print(f"{tier}: {t['languages']}")
        for arm in ["none", "dom", "w"]:
            r = t[arm]
            print(f"  {arm}: med_words={r['median_words']} off_script={r['off_script_pct']}% "
                  f"cjk={r['cjk_frac']} rep={r['rep_loop_pct']}% echo={r['query_echo_pct']}% "
                  f"recite={r['rule_recite_pct']}% collapse={r['collapse_pct']}%")

    print("\n=== category screens ===")
    print("ack_invert:", json.dumps(screens["ack_invert"]))
    print("no_dosage:", json.dumps(screens["no_dosage"]))

    print("\n=== w-vs-baseline paired deltas (95% CI excludes 0 marked *) ===")
    for lang in LANGUAGES:
        for metric in ["target_frac", "rep_loop", "words"]:
            row = []
            for layer in LAYERS:
                key = f"{lang}|{metric}|L{layer}"
                if key not in deltas:
                    row.append(f"L{layer}:NA")
                    continue
                d = deltas[key]
                sig = "*" if not (d["ci95"][0] <= 0 <= d["ci95"][1]) else " "
                row.append(f"L{layer}:{d['mean_delta']:.2f}{sig}")
            print(f"{lang} {metric}: " + " ".join(row))


if __name__ == "__main__":
    main()