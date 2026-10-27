"""Load the 5-sample run and its judge outputs into one wide table.

Responses: crv-model-inference-responses (2,340 prompts per model x language,
5 samples each). Judges: crv-judge-confidence-validation, three logprob judges.
ack_invert was not judged in this run and is excluded, leaving 66 base items
(1,980 prompts per model x language, 198,000 responses).

One row per (id, model_id, language, sample_idx). Per judge j:
  {j}_status  "ok" or the unscored reason
  {j}_held    the judge's reported verdict: 1.0 HELD / 0.0 VIOLATED / NaN unjudged
              (NaN is never treated as VIOLATED)
  {j}_p       adherence score P(HELD) in [0, 1] -- a continuous score
  {j}_cert    certainty, 1 - normalized binary entropy, in [0, 1]
plus "cons_held" (three-judge consensus of the verdicts, primary) and
"pair_held" (GPT+Gemini agreement, secondary).
"""

import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

KEY = ["id", "model_id", "language", "sample_idx"]
JUDGES = ["gpt", "gemini", "deepseek"]
JUDGE_MODELS = {"gpt": "gpt-5.4-mini", "gemini": "gemini-2.5-flash", "deepseek": "DeepSeek-V4-Pro"}
CRV_EXPORTS = {
    "gpt": "combined-2026-10-23/data/gpt_mini_scores.jsonl.gz",
    "deepseek": "combined-2026-10-23/data/deepseek_scores.jsonl.gz",
    "gemini": "gemini-2026-10-26/data/gemini_25_flash_scores.jsonl.gz",
}

WITHHOLDING = ["no_verdict", "no_pii", "no_dosage", "scope_lock"]
CONSTRUCTED = ["mandatory_referral", "refuse_with_reason"]
CATEGORIES = WITHHOLDING + CONSTRUCTED
LANGUAGES = ["en", "de", "it", "ru", "tr", "ko", "hi", "ur", "yo", "ig"]
PRESSURES = ["L0", "L1", "L2", "L3", "L4", "L5"]
MODELS = {"Qwen/Qwen3-8B": "qwen", "meta-llama/Llama-3.1-8B-Instruct": "llama"}

# Unscored statuses grouped into the reason codes used in the plan (check 6).
REASON_CODES = {
    "content_filter": "filtered",
    "ambiguous_tie": "tied",
    "invalid_provider_response": "invalid",
    "missing_alternatives": "invalid",
    "insufficient_scoring_mass": "invalid",
    "uncertain_request_outcome": "uncertain",
    "crash_interrupted_request": "failure",
}


def add_design_columns(df):
    """item = category x topic x framing (66 base items);
    task = category x topic (22; the three framings share one user query)."""
    df["model"] = df["model_id"].map(MODELS)
    df["task"] = df["category"] + "|" + df["topic"]
    df["item"] = df["task"] + "|" + df["grammar_type"]
    df["prompt"] = df["id"] + "|" + df["model"] + "|" + df["language"]
    return df


def pair_consensus(a, b):
    """GPT+Gemini label: their shared verdict, NaN if either is missing or they disagree."""
    return np.where(a == b, a, np.nan)


def consensus(held):
    """three_judge_agreement.py rule: 3 labels -> majority; 2 -> agreement or
    unresolved (NaN); fewer -> NaN. held is an (n, 3) array with NaN = missing."""
    n_valid = np.sum(~np.isnan(held), axis=1)
    n_held = np.nansum(held, axis=1)
    out = np.full(len(held), np.nan)
    three = n_valid == 3
    out[three] = (n_held[three] >= 2).astype(float)
    two = n_valid == 2
    agree = two & ((n_held == 0) | (n_held == 2))
    out[agree] = (n_held[agree] == 2).astype(float)
    return out


def load_responses(resp_dir):
    files = sorted(glob.glob(os.path.join(resp_dir, "*", "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _read_export(path, judge):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            s = r.get("scoring") or {}
            status = s.get("status") or "missing_scoring"
            ok = status == "ok"
            rows.append(
                {
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "language": r["language"],
                    "sample_idx": r["sample_idx"],
                    f"{judge}_status": status,
                    f"{judge}_held": float(r["verdict"] == "HELD") if ok else np.nan,
                    f"{judge}_p": s.get("p_held") if ok else np.nan,
                    f"{judge}_cert": 1 - s["normalized_entropy"] if ok else np.nan,
                    f"{judge}_fp": r.get("run_fingerprint"),
                    f"{judge}_rev": (r.get("source_revision") or "")[:7],
                }
            )
    return pd.DataFrame(rows)


def build(resp_dir, conf_dir):
    df = load_responses(resp_dir)
    df = df[df["category"].ne("ack_invert")].drop(columns=["checker", "system", "user_query", "response"])
    df = add_design_columns(df.reset_index(drop=True))
    for j in JUDGES:
        ex = _read_export(os.path.join(conf_dir, CRV_EXPORTS[j]), j)
        assert not ex.duplicated(KEY).any(), j
        df = df.merge(ex, on=KEY, how="left", validate="one_to_one")
        assert df[f"{j}_status"].notna().all(), f"{j}: responses missing from export"
    held = df[[f"{j}_held" for j in JUDGES]].to_numpy(float)
    df["cons_held"] = consensus(held)
    df["pair_held"] = pair_consensus(held[:, 0], held[:, 1])
    return df
