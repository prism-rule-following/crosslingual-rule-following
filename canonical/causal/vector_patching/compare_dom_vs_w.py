"""Stage A feasibility comparison: dom vs w, isolated by model.
4 cells: Qwen+dom (8 donors), Qwen+w (en only, only probe available),
Llama+dom (8 donors), Llama+w (8 donors). Recipients yo/ig for both.
CPU only. Downloads one language's activations at a time and frees the
blob before moving on -- disk is tight.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from canonical.causal.vector_patching import feasibility as feas
from canonical.causal.vector_patching import pair_selection as ps
from canonical.causal.vector_patching import probe_vectors as pv
from canonical.causal.vector_patching import vectors as vec
from canonical.causal.vector_patching.config import ACTIVATIONS_REPO, HOOK_NAME, MODEL_SLUGS

DONORS = ["en", "de", "hi", "it", "ko", "ru", "tr", "ur"]
RECIPIENTS = ["yo", "ig"]
QWEN = "Qwen/Qwen3-8B"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
N_LAYERS = {QWEN: 36, LLAMA: 32}
# Maria's probe repository has only English probes for Qwen.  For Llama it
# has probes for en/de/hi/it/ru/ur plus the recipients, but none for ko/tr.
# Keep the full dom donor set; restrict only the w cell to available donors.
W_DONORS = {
    QWEN: ["en"],
    LLAMA: ["en", "de", "hi", "it", "ru", "ur"],
}
PRESSURE = "L0"
OUT_DIR = "/private/tmp/claude-501/-Users-ayesha-Projects-crosslingual-rule-following/882cd551-0560-4a1a-9663-8909a10ca44e/scratchpad"
LOG_PATH = "/Users/ayesha/Projects/crosslingual-rule-following/.github/progress/02-09-26.md"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_blob(model_id, lang):
    slug = MODEL_SLUGS[model_id]
    p = hf_hub_download(ACTIVATIONS_REPO, f"{slug}/{lang}/{HOOK_NAME}.fp16.npy", repo_type="dataset")
    real = os.path.realpath(p)
    if os.path.exists(real):
        os.remove(real)


def load_all(model_id, langs, subset_verdicts):
    Xs, ys, doms = {}, {}, {}
    for lang in langs:
        acts = vec.load_activations(model_id, lang)
        acts = acts[acts["pressure_level"] == PRESSURE].reset_index(drop=True)
        lv = subset_verdicts[subset_verdicts["language"] == lang].set_index("canonical_id")["held"]
        acts = acts[acts["canonical_id"].isin(lv.index)].copy()
        acts["held"] = acts["canonical_id"].map(lv)
        X = np.stack(acts["activation"].to_numpy())
        y = acts["held"].to_numpy().astype(int)
        held_ids = set(lv[lv].index)
        failed_ids = set(lv[~lv].index)
        doms[lang] = vec.dom_vector(acts, held_ids, failed_ids)
        Xs[lang], ys[lang] = X, y
        free_blob(model_id, lang)
        log(f"  {lang}: {X.shape}, held={int(y.sum())}/{len(y)}")
    return Xs, ys, doms


def run_cell(model_id, vector_type, donors, Xs, ys, w_vectors=None, doms=None):
    rows = []
    n_layers = N_LAYERS[model_id]
    for recipient in RECIPIENTS:
        for donor in donors:
            if vector_type == "dom":
                direction_all_layers = doms[donor]
            else:
                direction_all_layers = w_vectors[donor]
            for layer in range(n_layers):
                d = direction_all_layers[layer]
                result = feas.patch_feasibility(
                    Xs[donor][:, layer, :], ys[donor], Xs[recipient][:, layer, :], ys[recipient],
                    {"direction": d},
                )
                m = result["direction"]
                rows.append({
                    "model": model_id, "vector_type": vector_type, "donor": donor,
                    "recipient": recipient, "layer": layer, **m,
                })
    return rows


def main():
    verdicts = ps.load_judge_verdicts()
    collapsed = ps.collapse_verdicts(verdicts)
    all_rows = []

    for model_id in [QWEN, LLAMA]:
        subset = collapsed[(collapsed["model_id"] == model_id) & (collapsed["pressure_level"] == PRESSURE)]
        langs = DONORS + RECIPIENTS
        log(f"=== {model_id}: loading {len(langs)} languages ===")
        Xs, ys, doms = load_all(model_id, langs, subset)

        log(f"=== {model_id}: dom cell ===")
        all_rows += run_cell(model_id, "dom", DONORS, Xs, ys, doms=doms)

        w_donors = W_DONORS[model_id]
        log(f"=== {model_id}: w cell, donors={w_donors} ===")
        w_vectors = pv.language_probe_vectors(model_id, w_donors, N_LAYERS[model_id])
        all_rows += run_cell(model_id, "w", w_donors, Xs, ys, w_vectors=w_vectors)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "compare_dom_vs_w.parquet")
    df.to_parquet(out_path)
    log(f"saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
