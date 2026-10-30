"""Run every check and analysis on the 5-sample run and write the result folder.

RUN from the repo root:
    python -W ignore -m canonical.analysis.sampling_stability.run --out <dir> [--upload]

Inputs are the latest revisions of crv-model-inference-responses and
crv-judge-confidence-validation (pin with --responses-revision / --judges-revision);
the revision actually used is recorded in summary.json. ack_invert is excluded
(not judged in this run). Only the numeric adherence and confidence scores are
analysed; the binary verdict is not used.
Output: <out>/tables/*.csv, <out>/data/labels_long.parquet, <out>/summary.json.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

from canonical.analysis.sampling_stability import data as D
from canonical.analysis.sampling_stability import metrics as M

ORG = "crosslingual-rule-following"
INPUTS = {
    "responses": (f"{ORG}/crv-model-inference-responses", None),
    "judges": (
        f"{ORG}/crv-judge-confidence-validation",
        list(D.CRV_EXPORTS.values())
        + ["combined-2026-10-23/summary.json", "gemini-2026-10-26/summary.json"],
    ),
}
UPLOAD_REPO = f"{ORG}/crv-analysis"
ANALYSES = [
    ("C1", M.c1_coverage),
    ("C3", M.c3_judge_agreement),
    ("C5", M.c5_qwen_batch),
    ("A1", M.a1_matrices),
    ("A2", M.a2_headline_gap),
    ("A3", M.a3_cell_stability),
    ("A4", M.a4_cross_model),
    ("A5", M.a5_decomposition),
    ("A6", M.a6_confidence),
    ("A7", M.a7_missing),
    ("A8", M.a8_pressure),
    ("A9", M.a9_within_prompt),
    ("A10", M.a10_variance_components),
    ("A11", M.a11_bootstrap),
]


def fetch_inputs(cache, revisions):
    """Download each input at the requested revision (None = latest).
    Returns local paths and the resolved commit SHAs."""
    api = HfApi()
    paths, resolved = {}, {}
    for name, (repo, patterns) in INPUTS.items():
        sha = api.dataset_info(repo, revision=revisions[name]).sha
        paths[name] = snapshot_download(
            repo,
            repo_type="dataset",
            revision=sha,
            allow_patterns=patterns,
            local_dir=os.path.join(cache, name),
        )
        resolved[name] = {"repo": repo, "revision": sha}
    return paths, resolved


def load_table(paths, cache, resolved):
    # Cache keyed by input revisions (and table schema) so a new upload is never read from a stale table.
    rev = "_".join(v["revision"][:7] for v in resolved.values())
    path = os.path.join(cache, f"crv_sampling_analysis_scores_{rev}.parquet")
    if not os.path.exists(path):
        D.build(paths["responses"], paths["judges"]).to_parquet(path)
    return pd.read_parquet(path)


def headline_summary(tables):
    """The numbers most likely to be quoted in the paper."""

    def rec(name, query=None):
        t = tables[name] if query is None else tables[name].query(query)
        return t.round(4).to_dict("records")

    return {
        "headline_gap_yo_ig": rec("headline_gap_summary", "focus_language"),
        "mean_score_by_judge": rec("mean_score_by_judge"),
        "cross_model": rec("cross_model_summary"),
        "cell_stability_overview": rec("cell_stability_overview"),
        "decomposition": rec(
            "decomposition_summary",
            "term in ['category', 'language', 'category_x_language', 'model']",
        ),
        "missing_overall": rec("missing_overall"),
        "pressure": rec(
            "pressure_orderings", "weighting == 'pooled' and levels == 'all_levels'"
        ),
        "within_prompt_sd_all": rec("within_prompt_sd", "grouping == 'all'"),
        "variance_components_all_languages": rec("variance_components", "language == 'all'"),
        "confidence_vs_within_prompt_sd": rec("confidence_vs_within_prompt_sd"),
        "confidence_vs_judge_spread": rec("confidence_vs_judge_spread", "language == 'all'"),
        "judge_reasoning_tokens": rec("judge_reasoning_tokens"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cache", default=None, help="input cache (default <out>/../inputs_cache)"
    )
    ap.add_argument("--responses-revision", default=None, help="default: latest")
    ap.add_argument("--judges-revision", default=None, help="default: latest")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--path-in-repo", default="sampling-stability-scores-2026-10-28")
    args = ap.parse_args()

    cache = args.cache or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "inputs_cache"
    )
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    paths, resolved = fetch_inputs(cache, {"responses": args.responses_revision, "judges": args.judges_revision})
    df = load_table(paths, cache, resolved)
    print(f"{len(df)} responses (ack_invert excluded)")

    tables = {}
    for tag, fn in ANALYSES:
        t0 = time.time()
        out = fn(df, n_boot=args.n_boot) if tag == "A11" else fn(df)
        tables.update(out)
        print(f"{tag:4s} {time.time() - t0:6.1f}s")
    tables.update(M.c7_reasoning(paths["judges"]))

    for name, t in tables.items():
        t.to_csv(os.path.join(args.out, "tables", f"{name}.csv"), index=False)
    keep = [
        "id",
        "model",
        "language",
        "sample_idx",
        "category",
        "topic",
        "grammar_type",
        "item",
        "task",
        "rule_status",
        "pressure_level",
        "mean_p",
    ]
    keep += [f"{j}_{k}" for j in D.JUDGES for k in ("status", "p", "cert")]
    df[keep].to_parquet(
        os.path.join(args.out, "data", "labels_long.parquet"), index=False
    )

    summary = {
        "inputs": resolved,
        "scope": "5-sample run only; ack_invert excluded (not judged in this run); numeric scores only "
        "(adherence P(HELD) and confidence), binary verdicts not used",
        "n_responses": len(df),
        "n_boot": args.n_boot,
        "results": headline_summary(tables),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(
            summary,
            fh,
            indent=1,
            default=lambda o: o.item() if isinstance(o, np.generic) else str(o),
        )
    print(f"wrote {len(tables)} tables to {args.out}")

    if args.upload:
        HfApi().upload_folder(
            repo_id=UPLOAD_REPO,
            repo_type="dataset",
            folder_path=args.out,
            path_in_repo=args.path_in_repo,
            commit_message="Sampling-stability and judge-confidence analysis (camera-ready)",
        )
        print(f"uploaded to {UPLOAD_REPO}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
