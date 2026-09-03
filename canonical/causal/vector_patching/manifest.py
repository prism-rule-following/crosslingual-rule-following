"""Builds the frozen all-language experiment manifest (plan exp2-all-langs §1).

One clean 25-ID prompt manifest, shared by every recipient: 5 IDs from each
of 5 categories, selected from English-held Qwen L0 candidates with one
seeded RNG. Recipient verdicts are preserved as metadata, never hardcoded.
Runs a hard preflight: every selected ID must exist for every recipient
with matching category.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from canonical.causal.vector_patching import pair_selection as ps
from canonical.causal.vector_patching.config import CANONICAL_DATASET_REPO

MODEL_ID = "Qwen/Qwen3-8B"
DONOR_LANGUAGE = "en"
PRESSURE = "L0"
RECIPIENTS = ["de", "hi", "ig", "it", "ko", "ru", "tr", "ur", "yo"]
CATEGORIES = ["ack_invert", "mandatory_referral", "no_dosage", "refuse_with_reason", "scope_lock"]
IDS_PER_CATEGORY = 5
SEED = 0


def load_dataset_text(language):
    path = hf_hub_download(CANONICAL_DATASET_REPO, f"data/{language}/test.jsonl", repo_type="dataset")
    return pd.read_json(path, lines=True).set_index("id")


def select_ids(collapsed, category):
    cands = collapsed[
        (collapsed["model_id"] == MODEL_ID)
        & (collapsed["pressure_level"] == PRESSURE)
        & (collapsed["language"] == DONOR_LANGUAGE)
        & (collapsed["held"])
        & (collapsed["category"] == category)
    ]
    ids = sorted(cands["canonical_id"].unique())
    if len(ids) < IDS_PER_CATEGORY:
        raise ValueError(f"category {category}: only {len(ids)} English-held candidates, need 5")
    rng = np.random.default_rng(SEED)
    return [str(x) for x in rng.choice(ids, size=IDS_PER_CATEGORY, replace=False)]


def verdict_meta(collapsed, language, canonical_id):
    row = collapsed[
        (collapsed["language"] == language)
        & (collapsed["canonical_id"] == canonical_id)
        & (collapsed["model_id"] == MODEL_ID)
        & (collapsed["pressure_level"] == PRESSURE)
    ]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {"held": bool(r["held"]), "low_confidence": bool(r["low_confidence"])}


def preflight(selected_ids, id_category):
    problems = []
    for lang in RECIPIENTS + [DONOR_LANGUAGE]:
        text = load_dataset_text(lang)
        for cid in selected_ids:
            if cid not in text.index:
                problems.append(f"{lang}: missing {cid}")
                continue
            if text.loc[cid, "category"] != id_category[cid]:
                problems.append(
                    f"{lang}: {cid} category {text.loc[cid, 'category']} != {id_category[cid]}"
                )
    if problems:
        raise ValueError("PREFLIGHT FAILED:\n" + "\n".join(problems))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "manifests"))
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading judge verdicts...", flush=True)
    verdicts = ps.load_judge_verdicts()
    collapsed = ps.collapse_verdicts(verdicts)

    id_category = {}
    selected = []
    for category in CATEGORIES:
        ids = select_ids(collapsed, category)
        print(f"[{time.strftime('%H:%M:%S')}] {category}: {ids}", flush=True)
        selected += ids
        for cid in ids:
            id_category[cid] = category
    if len(selected) != len(set(selected)):
        raise ValueError(f"duplicate IDs selected: {selected}")

    print(f"[{time.strftime('%H:%M:%S')}] running preflight over {len(RECIPIENTS) + 1} languages...", flush=True)
    preflight(selected, id_category)
    print("preflight OK: all IDs present with matching category in every language", flush=True)

    manifest = {
        "manifest_id": f"exp2_all_langs_{time.strftime('%Y%m%d_%H%M%S')}",
        "model_id": MODEL_ID,
        "donor_language": DONOR_LANGUAGE,
        "recipients": RECIPIENTS,
        "pressure_level": PRESSURE,
        "selection_seed": SEED,
        "category_order": CATEGORIES,
        "selected_ids": selected,
        "id_category": id_category,
        "en_verdicts": {cid: verdict_meta(collapsed, DONOR_LANGUAGE, cid) for cid in selected},
        "recipient_verdicts": {
            lang: {cid: verdict_meta(collapsed, lang, cid) for cid in selected} for lang in RECIPIENTS
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['manifest_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] manifest written to {out_path}", flush=True)
    print(f"manifest_id: {manifest['manifest_id']}", flush=True)


if __name__ == "__main__":
    main()