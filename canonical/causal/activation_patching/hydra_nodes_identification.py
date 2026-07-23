"""Script to identify potential backup Hydra nodes in the identified EAP circuit."""


from typing import List, Dict, Tuple, Optional, Any, Callable
import torch
from crosslingual_rule_following.canonical.causal.activation_patching.ablation import (
    apply_ablation_hooks
)

@torch.no_grad()
def identify_backups(model, dataset: List[Dict[str, Any]], circuit: Dict[str, Any],
                     candidate_nodes: List[str],
                     node_table: Dict[str, Tuple[str, Optional[int]]],
                     ablate_value: Callable[[str, torch.Tensor], torch.Tensor],
                     
                     tokenize_fn: Callable[[Dict[str, Any]], torch.Tensor],
                     top_k: int = 10) -> List[str]:
    """Find backups by CONDITIONAL scoring: how much each candidate's
    activity changes once the primary circuit is ablated.

    The CoAx paper (arxiv 2607.01940) calls this "conditional causal effect".
    """
    circuit_nodes = list(circuit.get("nodes", []))
    # baseline activity (intact model)
    base_activity: Dict[str, float] = {n: 0.0 for n in candidate_nodes}
    cond_activity: Dict[str, float] = {n: 0.0 for n in candidate_nodes}
    n_rows = 0
    for row in dataset:
        tokens = tokenize_fn(row)
        # run 1: model INTACT -- baseline activity of each candidate
        _, cache_intact = model.run_with_cache(tokens, names_filter=lambda n: True, return_type=None)
        for n in candidate_nodes:
            base, head = node_table[n]
            a = cache_intact[base]
            a = a if head is None else a[:, :, head, :]
            base_activity[n] += float(a.norm().cpu())
        # primary ablated
        apply_ablation_hooks(model, ablate_circuit=True,
                                       circuit_nodes=circuit_nodes,
                                       node_table=node_table,
                                       ablate_value=ablate_value)
        # run 2: SAME input, but with the primary circuit ablated. The
        # difference between the two runs is the "wake-up" signal -- a backup
        # is dormant in run 1 and active in run 2.
        _, cache_ablated = model.run_with_cache(tokens, names_filter=lambda n: True, return_type=None)
        model.reset_hooks()
        for n in candidate_nodes:
            base, head = node_table[n]
            a = cache_ablated[base]
            a = a if head is None else a[:, :, head, :]
            cond_activity[n] += float(a.norm().cpu())
        n_rows += 1

    # wake-up = conditional activity - baseline activity (positive => backup)
    wake = {n: (cond_activity[n] - base_activity[n]) / max(n_rows, 1)
            for n in candidate_nodes}
    ranked = sorted(candidate_nodes, key=lambda n: wake[n], reverse=True)
    return ranked[:top_k]