"""Cheap pre-flight check for whether a candidate patch direction
could plausibly work, before spending GPU time generating. Pure numpy -- no
model, no GPU.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def patch_feasibility(
    X_don: np.ndarray,
    y_don: np.ndarray,
    X_rec: np.ndarray,
    y_rec: np.ndarray,
    directions: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """X_*: (n, d_model) at one layer; y_*: 1 = held. directions: {name: vector}.
    Ranked by |cohens_d| descending. Inputs are cast to float32 -- the cached
    activations are fp16, and residual-stream norms at deep layers overflow
    fp16 in dot products/norms."""
    X_don = X_don.astype(np.float32)
    X_rec = X_rec.astype(np.float32)
    amb = np.linalg.norm(X_rec, axis=1).mean()
    out = {}
    for name, v in directions.items():
        u = v.astype(np.float32) / np.linalg.norm(v.astype(np.float32))
        ca, cb = X_don @ u, X_rec @ u

        auc_rec = roc_auc_score(y_rec, cb) if len(set(y_rec)) > 1 else float("nan")
        src, tgt = ca[y_don == 1], cb[y_rec == 0]
        gap = src.mean() - tgt.mean()
        pooled = np.sqrt((src.var(ddof=1) + tgt.var(ddof=1)) / 2)
        already = float((tgt >= src.mean()).mean())
        out[name] = {
            "auc_in_recipient": round(float(auc_rec), 3),
            "gap": round(float(gap), 4),
            "cohens_d": round(float(gap / pooled), 3) if pooled > 0 else float("nan"),
            "gap_rel_ambient": round(float(abs(gap) / amb), 5),
            "frac_already_above": round(already, 3),
        }
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1]["cohens_d"])))


def _language_xy(
    activations: pd.DataFrame, collapsed_verdicts: pd.DataFrame, model_id: str, lang: str,
    pressure_level: str,
) -> "tuple[np.ndarray, np.ndarray]":
    """Full (n, n_layers, d_model) activations and real held/failed labels
    for one language -- not restricted to any particular donor/recipient
    pair, so auc_in_recipient reflects genuine within-language separation."""
    verdicts = collapsed_verdicts[
        (collapsed_verdicts["model_id"] == model_id)
        & (collapsed_verdicts["language"] == lang)
        & (collapsed_verdicts["pressure_level"] == pressure_level)
    ].set_index("canonical_id")["held"]
    acts = activations[activations["canonical_id"].isin(verdicts.index)].copy()
    acts["held"] = acts["canonical_id"].map(verdicts)
    X = np.stack(acts["activation"].to_numpy())
    y = acts["held"].to_numpy().astype(int)
    return X, y


def run_feasibility_grid(
    activations_by_lang: Dict[str, pd.DataFrame],
    collapsed_verdicts: pd.DataFrame,
    model_id: str,
    dom_vectors: Dict[str, np.ndarray],
    donor_recipient_pairs: List[tuple],
    n_layers: int,
    pressure_level: str = "L0",
    extra_directions: Dict[str, np.ndarray] = None,
) -> pd.DataFrame:
    """patch_feasibility for every (donor, recipient) language pair, every
    layer, against every candidate direction. Uses each language's full
    held/failed activation set (not the narrow pair-table id intersection --
    that's for Stage B's actual generation targets, not this check).
    Returns one row per (donor, recipient, layer, direction)."""
    extra_directions = extra_directions or {}
    xy_cache: Dict[str, tuple] = {}
    rows = []
    for donor, recipient in donor_recipient_pairs:
        for lang in (donor, recipient):
            if lang not in xy_cache:
                xy_cache[lang] = _language_xy(
                    activations_by_lang[lang], collapsed_verdicts, model_id, lang, pressure_level
                )
        X_don, y_don = xy_cache[donor]
        X_rec, y_rec = xy_cache[recipient]
        if len(set(y_don)) < 2 or len(set(y_rec)) < 2:
            continue

        directions = {"dom_donor": dom_vectors[donor], "dom_recipient": dom_vectors[recipient]}
        directions.update(extra_directions)

        for layer in range(n_layers):
            layer_directions = {name: v[layer] for name, v in directions.items()}
            result = patch_feasibility(
                X_don[:, layer, :], y_don, X_rec[:, layer, :], y_rec, layer_directions
            )
            for direction_name, metrics in result.items():
                rows.append(
                    {
                        "language_donor": donor,
                        "language_recipient": recipient,
                        "layer": layer,
                        "direction": direction_name,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def top_k_layers(feasibility: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-k layers by |cohens_d|, per (donor, recipient, direction)."""
    ranked = feasibility.assign(abs_d=feasibility["cohens_d"].abs())
    return (
        ranked.sort_values("abs_d", ascending=False)
        .groupby(["language_donor", "language_recipient", "direction"])
        .head(k)
        .drop(columns="abs_d")
    )
