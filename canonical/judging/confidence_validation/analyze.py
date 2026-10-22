"""Agreement by score band from actual completed human review sheets."""

import argparse
from collections import defaultdict, Counter
import csv
import json
import math
from pathlib import Path
import random

from .core import response_key
from .data import file_hash, read_jsonl, write_json
from .review import REVIEW_FIELDS, stream_key


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, mid - half), min(1, mid + half)]


def correlation(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else None


def cluster_interval(rows, seed=42, repeats=1000):
    """Prompt-cluster bootstrap preserves correlation of repeated sample_idx.

    Base-rule/task clustering is not inferred from opaque IDs; variants may
    still share a base rule. This interval is conditional on the sampled set.
    """
    clusters = defaultdict(list)
    for r in rows:
        clusters[response_key(r)[:3]].append(int(r["agrees"]))
    if len(clusters) < 2:
        return [None, None]
    values = [clusters[k] for k in sorted(clusters)]
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        drawn = [rng.choice(values) for _ in values]
        draws.append(sum(sum(v) for v in drawn) / sum(len(v) for v in drawn))
    draws.sort()
    return [draws[int(0.025 * (repeats - 1))], draws[int(0.975 * (repeats - 1))]]


def analyze(packet, labels_file, output, make_plot=True, synthetic=False):
    packet, output = Path(packet), Path(output)
    samples = read_jsonl(packet / "PRIVATE_sample_key.jsonl")
    manifest = json.loads((packet / "manifest.json").read_text())
    with open(packet / "review_blinded.csv", encoding="utf-8", newline="") as handle:
        original = {r["review_id"]: r for r in csv.DictReader(handle)}
    with open(labels_file, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(REVIEW_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("human-label sheet lacks expected columns")
        labels = {}
        for row in reader:
            rid = row["review_id"]
            if rid in labels or rid not in original:
                raise ValueError("duplicate or unknown human review_id")
            if any(row[f] != original[rid][f] for f in REVIEW_FIELDS[:9]):
                raise ValueError("review input content was altered")
            verdict = row["human_verdict"].strip().upper()
            if verdict not in ("", "HELD", "VIOLATED", "CANNOT_ASSESS"):
                raise ValueError("invalid human verdict")
            if verdict and (not row["reviewer"].strip() or not row["evidence"].strip()):
                raise ValueError("completed human labels need reviewer and supporting evidence")
            if row["coherent"].strip().upper() not in ("", "YES", "NO", "UNSURE"):
                raise ValueError("invalid coherence label")
            labels[rid] = {**row, "human_verdict": verdict}
    if set(labels) != set(original):
        raise ValueError("label sheet must preserve all review rows, including pending ones")
    groups = defaultdict(list)
    for sample in samples:
        human = labels[sample["review_id"]]
        row = {**sample, "human_verdict": human["human_verdict"], "evidence": human["evidence"]}
        row["agrees"] = row["verdict"] == row["human_verdict"]
        groups[sample["stratum_id"]].append(row)
    summaries, subgroups, wrong = [], [], []
    for stratum in manifest["strata"]:
        rows = groups[stratum["stratum_id"]]
        reviewed = [r for r in rows if r["human_verdict"] in ("HELD", "VIOLATED")]
        n = len(reviewed)
        k = sum(r["agrees"] for r in reviewed)
        summaries.append({**stratum, "reviewed_n": n, "agreement_n": k,
                          "cannot_assess_n": sum(r["human_verdict"] == "CANNOT_ASSESS" for r in rows),
                          "pending_n": sum(not r["human_verdict"] for r in rows),
                          "agreement_rate": k / n if n else None,
                          "wilson_95_independent_row_assumption": wilson(k, n),
                          "prompt_cluster_bootstrap_95": cluster_interval(reviewed),
                          "sampled_verdict_counts": dict(Counter(r["verdict"] for r in rows)),
                          "point_biserial_confidence_agreement_descriptive": correlation(
                              [r["scoring"][manifest["metric"]] for r in reviewed], [int(r["agrees"]) for r in reviewed])})
        sub = defaultdict(list)
        for r in reviewed:
            sub[(r["model_id"], r["language"])].append(r)
            if not r["agrees"]:
                wrong.append({"review_id": r["review_id"], "stream": list(stream_key(r)),
                              "bucket": r["sampling_bucket"], "score": r["scoring"][manifest["metric"]],
                              "verdict": r["verdict"], "human_verdict": r["human_verdict"],
                              "evidence": r["evidence"]})
        for (model, lang), members in sorted(sub.items()):
            subgroups.append({"stratum_id": stratum["stratum_id"], "model_id": model, "language": lang,
                              "reviewed_n": len(members), "agreement_rate": sum(r["agrees"] for r in members) / len(members)})
    # Only a design-weighted estimate is meaningful when bins are sampled unequally.
    # Abstention/nonresponse prevents an unconditional population accuracy claim.
    overall = []
    for stream in sorted({tuple(s["stream"]) for s in summaries}):
        rows = [r for members in groups.values() for r in members if stream_key(r) == stream]
        known = [r for r in rows if r["human_verdict"] in ("HELD", "VIOLATED")]
        den = sum(1 / r["inclusion_probability"] for r in known)
        overall.append({"stream": list(stream), "reviewed_n": len(known),
                        "ipw_agreement_among_assessable": sum(int(r["agrees"]) / r["inclusion_probability"] for r in known) / den if den else None,
                        "all_selected_assessed": len(known) == len(rows),
                        "point_biserial_sample_descriptive": correlation(
                            [r["scoring"][manifest["metric"]] for r in known], [int(r["agrees"]) for r in known])})
    result = {"synthetic_test_only": synthetic, "sampling_metric": manifest["metric"],
              "labels_sha256": file_hash(labels_file), "strata": summaries,
              "model_language_breakdown": subgroups, "overall": overall,
              "disagreements": sorted(wrong, key=lambda r: -r["score"]),
              "notes": ["Confidence is uncalibrated; no causal mechanism is established.",
                        "Intervals describe the reviewed sample. Prompt bootstrap clusters repeated samples, not inferred base-rule variants.",
                        "Wilson intervals assume independent rows and are secondary to the prompt-cluster intervals.",
                        "Correlations are descriptive of stratified samples, without population p-values.",
                        "Inverse-probability weighting uses per-stream/bin selection; cannot-assess and pending rows remain explicit."]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "agreement.json", result)
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        streams = sorted({tuple(s["stream"]) for s in summaries})
        for index, stream in enumerate(streams):
            labelled = False
            for nband, band in enumerate("ABC"):
                s = next(v for v in summaries if tuple(v["stream"]) == stream and v["bucket"] == band)
                if s["agreement_rate"] is None:
                    continue
                x = nband + (index - (len(streams) - 1) / 2) * 0.09
                y = s["agreement_rate"]
                low, high = s["prompt_cluster_bootstrap_95"]
                label = stream[0] if stream[0] == stream[1] else f"{stream[0]} / {stream[1]}"
                ax.plot(x, y, "o", color=f"C{index % 10}", label=label if not labelled else None)
                labelled = True
                if low is not None:
                    ax.vlines(x, low, high, color=f"C{index % 10}")
                ax.annotate(f"n={s['reviewed_n']}", (x, y), xytext=(0, 9 + 10 * index),
                            textcoords="offset points", fontsize=7, ha="center", color=f"C{index % 10}")
        ax.set(xticks=[0,1,2], xticklabels=["A: 0–<40", "B: 40–<80", "C: 80–100"], ylim=(-0.03, 1.16),
               xlabel=manifest["metric"].replace("_", " "), ylabel="Human–judge agreement",
               title=("SYNTHETIC TEST — " if synthetic else "") + "Agreement by score band")
        if any(s["reviewed_n"] for s in summaries):
            ax.legend(fontsize=8, loc="lower right")
        ax.spines[["right", "top"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output / "agreement.png", dpi=180)
        plt.close(fig)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--synthetic-test-only", action="store_true")
    args = p.parse_args()
    result = analyze(args.packet, args.labels, args.output_dir, not args.no_plot, args.synthetic_test_only)
    print(json.dumps({"synthetic_test_only": result["synthetic_test_only"], "strata": len(result["strata"]),
                      "reviewed_judgments": sum(s["reviewed_n"] for s in result["strata"])}, indent=2))


if __name__ == "__main__":
    main()
