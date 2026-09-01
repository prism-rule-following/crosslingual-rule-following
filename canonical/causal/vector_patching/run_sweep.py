"""Orchestrates Stage A (feasibility, CPU) and Stage B (generation, GPU).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from canonical.causal.vector_patching import feasibility as feas
from canonical.causal.vector_patching import pair_selection as ps
from canonical.causal.vector_patching import vectors as vec
from canonical.causal.vector_patching.config import HOOK_NAME
from canonical.causal.vector_patching.intervene import make_patch_hook, make_steer_hook, run_intervention


def run_stage_a(
    model_id: str,
    languages: List[str],
    donor_recipient_pairs: List[tuple],
    n_layers: int,
    pressure_level: str = "L0",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Stage 0 + vector construction + feasibility grid, all CPU. Returns
    tiers, the pair table, and the top-k-layers-per-pair feasibility ranking
    that Stage B should actually generate for."""
    verdicts = ps.load_judge_verdicts()
    collapsed = ps.collapse_verdicts(verdicts)
    rates = ps.compute_adherence_rates(collapsed, model_id, pressure_level)
    tiers = ps.classify_tiers(rates)
    pairs = ps.build_pair_table(collapsed, model_id, pressure_level)

    activations_by_lang = {}
    for lang in languages:
        acts = vec.load_activations(model_id, lang)
        activations_by_lang[lang] = acts[acts["pressure_level"] == pressure_level].reset_index(
            drop=True
        )

    dom_vectors = vec.language_dom_vectors(model_id, languages, collapsed, pressure_level)
    hr_langs = [lang for lang in languages if tiers.get(lang) == "high"]
    extra_directions = {
        "hr_avg": vec.average_vector(dom_vectors, hr_langs) if hr_langs else None,
        "all_avg": vec.average_vector(dom_vectors, languages),
    }
    extra_directions = {k: v for k, v in extra_directions.items() if v is not None}

    grid = feas.run_feasibility_grid(
        activations_by_lang,
        collapsed,
        model_id,
        dom_vectors,
        donor_recipient_pairs,
        n_layers,
        pressure_level,
        extra_directions,
    )
    top_layers = feas.top_k_layers(grid, k=top_k)

    return {
        "tiers": tiers,
        "rates": rates,
        "pairs": pairs,
        "dom_vectors": dom_vectors,
        "extra_directions": extra_directions,
        "feasibility": grid,
        "top_layers": top_layers,
    }


def load_model(model_id: str, device: str = "cpu"):
    """Mirrors ModelGenerationRunner.load() in canonical/evaluation/inference.py."""
    from transformer_lens.model_bridge import TransformerBridge

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = TransformerBridge.boot_transformers(model_id, device=device, dtype=dtype)
    model.enable_compatibility_mode(disable_warnings=True, no_processing=True)
    model.original_model.eval()
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    return model


def format_chat_prompt(
    tokenizer, system: str, user: str, supports_system_role: bool = True, enable_thinking: bool = False
) -> str:
    """Mirrors ModelGenerationRunner.format_chat_prompt(). enable_thinking=False
    matches the canonical Qwen3-8B run config."""
    if supports_system_role:
        chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        chat = [{"role": "user", "content": f"{system}\n\n{user}"}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def donor_coordinate(donor_activation: np.ndarray, layer: int, direction_vector: np.ndarray) -> float:
    """c_donor = donor's activation at `layer`, projected onto the (layer's)
    direction -- the coordinate make_patch_hook swaps into the recipient.
    donor_activation: (n_layers, d_model) for one donor row; direction_vector
    already sliced to that layer, shape (d_model,)."""
    x = donor_activation[layer].astype(np.float32)
    v = direction_vector.astype(np.float32)
    u = v / np.linalg.norm(v)
    return float(x @ u)


def run_stage_b_row(
    model,
    tokenizer,
    rule_clause: str,
    user_query: str,
    layer: int,
    direction_vector: np.ndarray,
    c_donor: Optional[float] = None,
    alpha: Optional[float] = None,
    mode: str = "patch",
    max_new_tokens: int = 128,
    supports_system_role: bool = True,
) -> str:
    """One patched/steered generation. mode="patch" needs c_donor (the
    donor's coordinate along the direction); mode="steer" needs alpha."""
    prompt = format_chat_prompt(tokenizer, rule_clause, user_query, supports_system_role)
    tokens = model.to_tokens(prompt)
    u = torch.tensor(direction_vector, dtype=torch.float32)

    if mode == "patch":
        hook_fn = make_patch_hook(u, c_donor=c_donor)
    elif mode == "steer":
        hook_fn = make_steer_hook(u, alpha=alpha)
    else:
        raise ValueError(f"mode must be 'patch' or 'steer', got {mode!r}")

    return run_intervention(
        model, tokens, layer=layer, hook_fn=hook_fn, hook_name=HOOK_NAME, max_new_tokens=max_new_tokens
    )
