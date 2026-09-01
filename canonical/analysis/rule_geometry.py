"""Cross-lingual geometry of rule status at the last prompt token (LP4FM §5).

Port of experimental/analysis/DeepCKA_SMDS_BannedWord.ipynb onto the canonical
activation cache. Differences from the notebook:
  - position is the last prompt token, cached for every row, instead of a
    decision position located inside the response by string search
  - n = 2340 paired items per language instead of 9-13
  - the five lexical realizations of rule status (active/cancelled, on/off,
    true/false, valid/invalid, enabled/disabled) give a leave-one-realization-out
    control: a direction that only reads the literal status token cannot survive it
  - a category-decoding positive control separates "this language does not encode
    rule status" from "nothing is decodable from this language at all"

Stages:
  extract   per (model, language): download activations, compute per-layer status
            decodability, cache a small npz, delete the download
  combine   all-pairs CKA, EN-direction transfer, transport, judge join
  plot      the figures

Run:
  python canonical/analysis/rule_geometry.py extract --model Qwen__Qwen3-8B
  python canonical/analysis/rule_geometry.py combine --model Qwen__Qwen3-8B
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = "crosslingual-rule-following/model-inference-activations"
ACTIVE_STATUSES = {"active", "on", "true", "valid", "enabled"}
MODELS = ["Qwen__Qwen3-8B", "meta-llama__Llama-3.1-8B-Instruct"]
LANGS = ["en", "ru", "de", "it", "ko", "tr", "hi", "ur", "yo", "ig"]
SEED = 42
CKA_SUBSAMPLE = 600


# ---------------------------------------------------------------- data loading

def fetch_lang(model_slug, lang, hook, work_dir, token):
    from huggingface_hub import hf_hub_download

    dest = work_dir / f"{model_slug}__{lang}"
    paths = {}
    for name in ("index.parquet", f"{hook}.fp16.npy"):
        paths[name] = hf_hub_download(
            REPO, f"{model_slug}/{lang}/{name}", repo_type="dataset",
            local_dir=str(dest), token=token,
        )
    return paths["index.parquet"], paths[f"{hook}.fp16.npy"], dest


def paired_views(index_path, array_path):
    """Rows aligned so index i of each return value is one active/revoked pair."""
    idx = pd.read_parquet(index_path)
    idx["base"] = idx["id"].str.replace(r"_(clean|revoked)$", "", regex=True)
    idx["is_active"] = idx["rule_status"].isin(ACTIVE_STATUSES)

    act = idx[idx.is_active].set_index("base")
    rev = idx[~idx.is_active].set_index("base")
    common = pd.Index(sorted(act.index.intersection(rev.index)))

    acts = np.load(array_path, mmap_mode="r")
    meta = act.loc[common, ["pair_type", "grammar_type", "category", "pressure_level"]]
    return (common.to_numpy(), act.loc[common, "row_idx"].to_numpy(),
            rev.loc[common, "row_idx"].to_numpy(), meta, acts)


def layer_pair(acts, act_rows, rev_rows, layer):
    return (np.asarray(acts[act_rows, layer, :], dtype=np.float32),
            np.asarray(acts[rev_rows, layer, :], dtype=np.float32))


# -------------------------------------------------------------------- metrics

def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dom_auc(a, r, train_mask):
    """Diff-of-means status direction fitted on train_mask, scored on its complement."""
    test = ~train_mask
    if train_mask.sum() < 20 or test.sum() < 20:
        return float("nan"), None
    v = unit(a[train_mask].mean(0) - r[train_mask].mean(0))
    scores = np.concatenate([a[test] @ v, r[test] @ v])
    labels = np.concatenate([np.ones(test.sum()), np.zeros(test.sum())])
    return auc(scores, labels), v


def nearest_mean_accuracy(X, labels, train_mask):
    """Positive control: can anything at all be read off this language's activations?"""
    classes = sorted(set(labels))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    means = np.stack([unit(Xn[train_mask & (labels == c)].mean(0)) for c in classes])
    pred = np.asarray(classes)[np.argmax(Xn[~train_mask] @ means.T, axis=1)]
    return float((pred == labels[~train_mask]).mean()), 1.0 / len(classes)


def within_category_coherence(diffs, categories):
    """E3: do different instances of the same rule point the same way in this language?"""
    D = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-9)
    vals = []
    for c in sorted(set(categories)):
        M = D[categories == c]
        if len(M) < 3:
            continue
        G = M @ M.T
        iu = np.triu_indices_from(G, k=1)
        vals.append(float(G[iu].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def _hsic_unbiased(K, L):
    n = K.shape[0]
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    ones = np.ones(n)
    t1 = np.sum(K * L)
    t2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    t3 = 2 * (ones @ K @ (L @ ones)) / (n - 2)
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_debiased(X, Y):
    K, L = X @ X.T, Y @ Y.T
    d = np.sqrt(max(_hsic_unbiased(K, K), 0) * max(_hsic_unbiased(L, L), 0))
    return float(_hsic_unbiased(K, L) / d) if d > 0 else 0.0


def procrustes_fit(src, dst):
    U, _, Vt = np.linalg.svd(src.T @ dst, full_matrices=False)
    return U @ Vt


# --------------------------------------------------------------------- extract

def extract(model_slug, lang, hook, work_dir, cache_dir, token, common_layer=None):
    index_path, array_path, dest = fetch_lang(model_slug, lang, hook, work_dir, token)
    try:
        base_ids, act_rows, rev_rows, meta, acts = paired_views(index_path, array_path)
        n, n_layers, d_model = len(base_ids), acts.shape[1], acts.shape[2]

        rng = np.random.default_rng(SEED)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[rng.permutation(n)[: int(0.6 * n)]] = True

        realizations = meta["pair_type"].to_numpy()
        categories = meta["category"].to_numpy()
        uniq_real = sorted(set(realizations))

        within = np.full(n_layers, np.nan)
        xreal = np.full(n_layers, np.nan)
        dirs = np.zeros((n_layers, d_model), dtype=np.float32)

        for layer in range(n_layers):
            a, r = layer_pair(acts, act_rows, rev_rows, layer)
            score, v = dom_auc(a, r, train_mask)
            within[layer] = score
            if v is not None:
                dirs[layer] = v
            folds = [dom_auc(a, r, realizations != held)[0] for held in uniq_real]
            folds = [f for f in folds if not np.isnan(f)]
            xreal[layer] = float(np.mean(folds)) if folds else np.nan

        own_peak = int(np.nanargmax(xreal))
        L = own_peak if common_layer is None else int(common_layer)
        aL, rL = layer_pair(acts, act_rows, rev_rows, L)
        diffs = aL - rL

        cat_acc, cat_chance = nearest_mean_accuracy(aL, categories, train_mask)
        coherence = within_category_coherence(diffs, categories)

        out = cache_dir / f"{model_slug}__{lang}.npz"
        np.savez_compressed(
            out,
            lang=lang, model=model_slug, hook=hook,
            n_pairs=n, n_layers=n_layers, own_peak=own_peak, common_layer=L,
            status_auc_within=within, status_auc_xreal=xreal,
            dirs=dirs.astype(np.float16),
            act_L=aL.astype(np.float16), rev_L=rL.astype(np.float16),
            base_ids=base_ids, categories=categories, realizations=realizations,
            grammar=meta["grammar_type"].to_numpy(),
            train_mask=train_mask,
            category_decode_acc=cat_acc, category_decode_chance=cat_chance,
            within_category_coherence=coherence,
        )
        print(f"  {lang}: peak L{own_peak}/{n_layers} xreal AUC {np.nanmax(xreal):.3f} | "
              f"category decode {cat_acc:.3f} (chance {cat_chance:.3f}) | "
              f"coherence {coherence:.3f}", flush=True)
        return own_peak
    finally:
        shutil.rmtree(dest, ignore_errors=True)


# --------------------------------------------------------------------- combine

def combine(model_slug, cache_dir, out_dir, judge_path):
    files = {l: cache_dir / f"{model_slug}__{l}.npz" for l in LANGS}
    files = {l: p for l, p in files.items() if p.exists()}
    data = {l: np.load(p, allow_pickle=True) for l, p in files.items()}
    langs = [l for l in LANGS if l in data]

    shared = None
    for l in langs:
        ids = data[l]["base_ids"]
        shared = ids if shared is None else np.intersect1d(shared, ids)
    pos = {l: pd.Index(data[l]["base_ids"]).get_indexer(shared) for l in langs}

    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(shared), size=min(CKA_SUBSAMPLE, len(shared)), replace=False)

    diffs = {}
    for l in langs:
        d = (data[l]["act_L"][pos[l]].astype(np.float32)
             - data[l]["rev_L"][pos[l]].astype(np.float32))
        diffs[l] = d

    cka = np.zeros((len(langs), len(langs)))
    for i, a in enumerate(langs):
        Xa = diffs[a][sub] - diffs[a][sub].mean(0)
        for j, b in enumerate(langs):
            if j < i:
                cka[i, j] = cka[j, i]
                continue
            Xb = diffs[b][sub] - diffs[b][sub].mean(0)
            cka[i, j] = 1.0 if i == j else cka_debiased(Xa, Xb)

    L = int(data["en"]["common_layer"])
    en_dir = data["en"]["dirs"][L].astype(np.float32)
    en_act = data["en"]["act_L"][pos["en"]].astype(np.float32)
    en_rev = data["en"]["rev_L"][pos["en"]].astype(np.float32)
    tr = data["en"]["train_mask"][pos["en"]]

    per_lang = {}
    for l in langs:
        a = data[l]["act_L"][pos[l]].astype(np.float32)
        r = data[l]["rev_L"][pos[l]].astype(np.float32)
        sign_agree = float((diffs[l] @ en_dir > 0).mean())

        src = np.vstack([a[tr], r[tr]])
        dst = np.vstack([en_act[tr], en_rev[tr]])
        R = procrustes_fit(src - src.mean(0), dst - dst.mean(0))

        def sc(A, B):
            return auc(np.concatenate([A @ en_dir, B @ en_dir]),
                       np.concatenate([np.ones(len(A)), np.zeros(len(B))]))

        te = ~tr
        mapped = lambda X: (X - src.mean(0)) @ R + dst.mean(0)
        per_lang[l] = {
            "n_pairs": int(data[l]["n_pairs"]),
            "n_layers": int(data[l]["n_layers"]),
            "own_peak_layer": int(data[l]["own_peak"]),
            "common_layer": L,
            "status_auc_xreal": [float(x) for x in data[l]["status_auc_xreal"]],
            "status_auc_within": [float(x) for x in data[l]["status_auc_within"]],
            "status_auc_xreal_peak": float(np.nanmax(data[l]["status_auc_xreal"])),
            "status_auc_xreal_at_common": float(data[l]["status_auc_xreal"][L]),
            "category_decode_acc": float(data[l]["category_decode_acc"]),
            "category_decode_chance": float(data[l]["category_decode_chance"]),
            "within_category_coherence": float(data[l]["within_category_coherence"]),
            "cka_vs_en": float(cka[langs.index(l), langs.index("en")]),
            "en_direction_sign_agreement": sign_agree,
            "transport_direct": sc(a[te], r[te]),
            "transport_procrustes": sc(mapped(a[te]), mapped(r[te])),
            "in_language_ceiling": dom_auc(a, r, tr)[0],
        }

    judge = json.loads(Path(judge_path).read_text()) if Path(judge_path).exists() else {}
    key = {"Qwen__Qwen3-8B": "Qwen/Qwen3-8B",
           "meta-llama__Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"}[model_slug]
    for l in langs:
        stats = judge.get(key, {}).get(l, {})
        per_lang[l]["held_overall"] = stats.get("held_overall")
        per_lang[l]["held_L0"] = stats.get("held_L0")

    out = {
        "model": model_slug, "langs": langs, "common_layer": L,
        "n_shared_items": int(len(shared)),
        "cka_matrix": cka.tolist(),
        "per_lang": per_lang,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model_slug}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / (model_slug + '.json')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "combine"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--hook", default="hook_resid_post")
    ap.add_argument("--work", default="/tmp/rule_geometry_work")
    ap.add_argument("--cache", default="canonical/results/rule_geometry/cache")
    ap.add_argument("--out", default="canonical/results/rule_geometry")
    ap.add_argument("--judge", default="canonical/results/rule_geometry/judge_summary.json")
    args = ap.parse_args()

    load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))
    token = os.getenv("HF_TOKEN")
    work, cache = Path(args.work), Path(args.cache)
    work.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    models = [args.model] if args.model else MODELS
    langs = [l for l in args.langs.split(",") if l]

    if args.stage == "extract":
        for model in models:
            print(f"[{model}]", flush=True)
            en_cache = cache / f"{model}__en.npz"
            if en_cache.exists():
                common = int(np.load(en_cache, allow_pickle=True)["own_peak"])
            else:
                common = extract(model, "en", args.hook, work, cache, token, None)
            for lang in langs:
                if lang == "en" or (cache / f"{model}__{lang}.npz").exists():
                    continue
                extract(model, lang, args.hook, work, cache, token, common_layer=common)
    else:
        for model in models:
            combine(model, cache, Path(args.out), args.judge)


if __name__ == "__main__":
    main()
