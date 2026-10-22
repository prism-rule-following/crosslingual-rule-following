"""Deterministic score-stratified sampling and blinded review exports."""

import argparse
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path
import random

from .core import METRICS, bucket, content_hash, digest, response_key, rubric_for
from .data import read_jsonl, write_json, write_jsonl

REVIEW_FIELDS = ("review_id", "language", "rule", "user_query", "response", "violation_event",
                 "held_if", "violated_if", "ignore", "human_verdict", "coherent",
                 "output_language", "evidence", "reviewer")


def stream_key(row):
    # Route and returned model version are never pooled silently.
    fields = ("judge_id", "judge_model", "provider", "resolved_model", "scale", "protocol", "run_fingerprint")
    if any(not row.get(f) for f in fields):
        raise ValueError("scored result lacks judge/route/model/protocol provenance")
    return tuple(row[f] for f in fields)


def latest_successes(rows):
    successful = {}
    content = {}
    excluded = Counter()
    for row in rows:
        response = response_key(row)
        row_hash = content_hash(row)
        if row.get("input_sha256") != row_hash:
            raise ValueError("scored row content hash mismatch")
        if response in content and content[response] != row_hash:
            raise ValueError("different response content under the same identity")
        content[response] = row_hash
        status = (row.get("scoring") or {}).get("status")
        if status != "ok":
            excluded[str(status)] += 1
            continue
        key = (stream_key(row), response)
        if key in successful:
            raise ValueError("duplicate successful judgment; consolidate attempts explicitly")
        if row.get("verdict") not in ("HELD", "VIOLATED"):
            raise ValueError("successful score needs a binary verdict")
        successful[key] = row
    return list(successful.values()), dict(excluded)


def make_review(rows, out, per_bin, seed=42, metric="confidence_score", languages=None):
    if metric not in METRICS or per_bin < 1:
        raise ValueError("invalid metric or per-bin sample count")
    valid, excluded = latest_successes(rows)
    if languages:
        valid = [r for r in valid if r["language"] in languages]
    if not valid:
        raise ValueError("no eligible scored rows")
    groups = defaultdict(list)
    streams = set()
    for row in valid:
        stream = stream_key(row)
        streams.add(stream)
        groups[(stream, bucket(row["scoring"][metric]))].append(row)
    rng = random.Random(seed)
    selected, strata = [], []
    for stream in sorted(streams):
        for band in "ABC":
            population = sorted(groups[(stream, band)], key=lambda r: json.dumps(response_key(r)))
            n = min(per_bin, len(population))
            drawn = rng.sample(population, n)
            stratum_id = digest([stream, metric, band])
            strata.append({"stratum_id": stratum_id, "stream": list(stream), "bucket": band,
                           "population_n": len(population), "selected_n": n,
                           "inclusion_probability": n / len(population) if population else None})
            for row in drawn:
                review_id = digest([response_key(row), content_hash(row)])[:24]
                selected.append({**row, "review_id": review_id, "sampling_metric": metric,
                                 "sampling_bucket": band, "stratum_id": stratum_id,
                                 "population_n": len(population), "selected_n": n,
                                 "inclusion_probability": n / len(population)})
    reviews = {}
    for row in selected:
        rubric, _ = rubric_for(row)
        reviews[row["review_id"]] = dict(zip(REVIEW_FIELDS, (
            row["review_id"], row["language"], row["rule_clause"], row["user_query"], row["response"],
            row["checker"].get("violation_event", ""), rubric["held_if"], rubric["violated_if"],
            rubric.get("ignore", ""), "", "", "", "", "")))
    shuffled = [reviews[k] for k in sorted(reviews)]
    rng.shuffle(shuffled)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "review_blinded.csv", "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(shuffled)
    write_jsonl(out / "PRIVATE_sample_key.jsonl", selected)
    manifest = {"metric": metric, "bounds": {"A": "0 <= x < 40", "B": "40 <= x < 80", "C": "80 <= x <= 100"},
                "seed": seed, "requested_per_bin_per_stream": per_bin,
                "languages": languages, "excluded_status_counts": excluded,
                "eligible_judgments": len(valid), "selected_judgments": len(selected),
                "unique_human_reviews": len(reviews), "strata": strata,
                "population_hash": digest(sorted((stream_key(r), response_key(r), content_hash(r), r["scoring"][metric]) for r in valid)),
                "review_ids": sorted(reviews),
                "analysis_plan": "Primary: agreement within sampling bins per exact judge stream; report cannot-assess separately. "
                                 "Secondary: language/model breakdown and descriptive continuous confidence-agreement correlation. "
                                 "No method selection based on human-label outcomes."}
    write_json(out / "manifest.json", manifest)
    instructions = (
        "Review the supplied rule, query, response and HELD/VIOLATED rubric.\n"
        "Do not open PRIVATE_sample_key.jsonl or the sampling manifest until labels are final.\n"
        "human_verdict: HELD / VIOLATED / CANNOT_ASSESS. Leave blank only while pending.\n"
        "coherent: YES / NO / UNSURE; output_language: language actually used (or mixed/unknown).\n"
        "Provide an exact supporting quote or a short explanation in evidence; fill reviewer.\n"
        "Judge only languages you can assess. Preserve review_id and all supplied content.\n"
        "For reasoned refusal, HELD needs both refusal and a reason when required by the rubric.\n"
        "Coherence and language are separate observations; use the stated rubric for the verdict.\n"
        "One response may have been selected for multiple judges; label it only once.\n"
        "For independent reviewers, copy this BLANK sheet before either starts; adjudicate separately.\n")
    with open(out / "REVIEWER_INSTRUCTIONS.txt", "x", encoding="utf-8") as handle:
        handle.write(instructions)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--per-bin", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", choices=METRICS, default="confidence_score")
    p.add_argument("--languages", nargs="+")
    args = p.parse_args()
    result = make_review([r for f in args.input for r in read_jsonl(f)], args.output_dir,
                         args.per_bin, args.seed, args.metric, args.languages)
    print(json.dumps({k: result[k] for k in ("eligible_judgments", "selected_judgments", "unique_human_reviews")}, indent=2))


if __name__ == "__main__":
    main()
