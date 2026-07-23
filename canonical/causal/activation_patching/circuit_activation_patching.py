"""Main script to run activation patching on a circuit."""

from typing import List, Dict, Tuple, Optional, Any, Callable
import random
import torch
from crosslingual_rule_following.canonical.causal.activation_patching.ablation import (
    apply_ablation_hooks, ablate_and_score
)

def run_sufficiency(model, dataset: List[Dict[str, Any]], circuit: Dict[str, Any],
                    node_table: Dict[str, Tuple[str, Optional[int]]], ablate_value: Callable[[str, torch.Tensor], torch.Tensor],
                    generate_fn: Callable[[Any, torch.Tensor], str],
                    adherence_fn: Callable[[Dict[str, Any], str], bool],
                    tokenize_fn: Callable[[Dict[str, Any]], torch.Tensor]) -> Dict[str, Any]:
    """Keep only the circuit; ablate everything else; generate; evaluate.

    Returns a JSON-able dict of per-row responses + adherence, plus the
    aggregate sufficiency rate. High rate => the circuit is SUFFICIENT.
    """
    results = []
    circuit_nodes = list(circuit.get("nodes", []))
    for row in dataset:
        tokens = tokenize_fn(row)
        apply_ablation_hooks(
            model, ablate_circuit=False,
                                       circuit_nodes=circuit_nodes,
                                       node_table=node_table,
                                       ablate_value=ablate_value)
        text = generate_fn(model, tokens)
        model.reset_hooks()
        ok = bool(adherence_fn(row, text))
        results.append({"id": row.get("id"), "category": row.get("category"),
                        "topic": row.get("topic"), "language": row.get("language"),
                        "response": text, "adheres": ok})
    rate = sum(r["adheres"] for r in results) / max(len(results), 1)
    return {"experiment": "sufficiency", "ablation_kind": getattr(ablate_value, 'kind', 'unknown'),
            "n": len(results), "adherence_rate": rate, "responses": results}


def run_necessity(model, dataset: List[Dict[str, Any]], circuit: Dict[str, Any],
                  node_table: Dict[str, Tuple[str, Optional[int]]],
                  ablate_value: Callable[[str, torch.Tensor], torch.Tensor],
                  generate_fn: Callable[[Any, torch.Tensor], str],
                  adherence_fn: Callable[[Dict[str, Any], str], bool],
                  tokenize_fn: Callable[[Dict[str, Any]], torch.Tensor],
                  ablate_edges: bool = False) -> Dict[str, Any]:
    """Ablate the circuit; keep the rest clean; generate; evaluate.

    Low adherence (big drop vs full model) => the circuit is NECESSARY.
    If adherence STAYS HIGH despite ablation => suspect self-repair; go to
    completeness (Section 5). (arxiv 2607.01940: self-repair is a completeness
    failure that muffles necessity.)

    NODE vs EDGE:
      "Most circuit studies evaluate at the level of components ... preserving
       all outgoing connections ... we reserve edge-level evaluation for tests
       that preserve only the specific connections identified." (2606.06267)
      => default = node-level. Edge-level is the stricter variant.
    """
    if ablate_edges:
        # CHOICE: edge-level ablation needs the edge-aware forward graph that
        # the EAP repo (hannamw/EAP-IG) already provides. Flagged rather than
        # shipped as a hand-rolled version that would likely be subtly wrong.
        raise NotImplementedError(
            "Edge-level ablation: use the EAP repo's edge-masked forward "
            "(hannamw/EAP-IG). Node-level is implemented. (2606.06267)")

    out = ablate_and_score(model, dataset, list(circuit.get("nodes", [])),
                           node_table, ablate_value, generate_fn, adherence_fn,
                           tokenize_fn, tag="necessity")
    out["experiment"] = "necessity"
    out["ablation_kind"] = getattr(ablate_value, "kind", "unknown")
    out["n"] = len(out["responses"])
    return out


def run_completeness(model, dataset: List[Dict[str, Any]], circuit: Dict[str, Any],
                     backups: List[str],
                     node_table: Dict[str, Tuple[str, Optional[int]]],
                     ablate_value: Callable[[str, torch.Tensor], torch.Tensor],
                     generate_fn: Callable[[Any, torch.Tensor], str],
                    adherence_fn: Callable[[Dict[str, Any], str], bool],
                     tokenize_fn: Callable[[Dict[str, Any]], torch.Tensor],
                     random_control: bool = True, seed: int = 0) -> Dict[str, Any]:
    """Co-ablate circuit + backups; compare the drop to (a) circuit-only and
    (b) circuit + matched-RANDOM completion.

    Complete circuit => co-ablating circuit+backups drops adherence MORE than
    circuit-only, and MORE than a random completion of the same size.
      "Completing it with the CoAx backups closes the gap ... whereas a
       matched-random completion does not."  (arxiv 2607.01940)
    """
    circuit_nodes = list(circuit.get("nodes", []))

    out = {"experiment": "completeness", "ablation_kind": getattr(ablate_value, 'kind', 'unknown')}
    out["circuit_only"] = ablate_and_score(
        model, dataset, circuit_nodes, node_table, ablate_value,
        generate_fn, adherence_fn, tokenize_fn, tag="circuit_only")
    out["circuit_plus_backups"] = ablate_and_score(
        model, dataset, circuit_nodes + backups, node_table, ablate_value,
        generate_fn, adherence_fn, tokenize_fn, tag="circuit_plus_backups")

    if random_control:
        # matched-random completion: same NUMBER of extra nodes, drawn at
        # random from outside the circuit (control per 2607.01940 & the
        # "compare against random baselines" standard, 2606.06267).
        rng = random.Random(seed)
        pool = [n for n in node_table if n not in set(circuit_nodes) and n not in set(backups)]
        rand_extra = rng.sample(pool, min(len(backups), len(pool)))
        out["circuit_plus_random"] = ablate_and_score(
            model, dataset, circuit_nodes + rand_extra, node_table, ablate_value,
            generate_fn, adherence_fn, tokenize_fn, tag="circuit_plus_random")

    out["backups_used"] = backups
    return out


def run_minimality(model, dataset: List[Dict[str, Any]], circuit: Dict[str, Any],
                   node_table: Dict[str, Tuple[str, Optional[int]]], ablate_value: Callable[[str, torch.Tensor], torch.Tensor],
                   generate_fn: Callable[[Any, torch.Tensor], str],
                    adherence_fn: Callable[[Dict[str, Any], str], bool],
                   tokenize_fn: Callable[[Dict[str, Any]], torch.Tensor]) -> Dict[str, Any]:
    """For each node: run SUFFICIENCY with that node removed from the kept
    circuit. A big drop when node removed => node is necessary (good,
    minimal). Little drop => node may be redundant.
    """
    circuit_nodes = list(circuit.get("nodes", []))
    # reference: full-circuit sufficiency
    ref = run_sufficiency(model, dataset, {"nodes": circuit_nodes}, node_table,
                          ablator, generate_fn, adherence_fn, tokenize_fn)
    ref_rate = ref["adherence_rate"]

    per_node = []
    for node in circuit_nodes:
        reduced = [n for n in circuit_nodes if n != node]
        r = run_sufficiency(model, dataset, {"nodes": reduced}, node_table,
                            ablator, generate_fn, adherence_fn, tokenize_fn)
        per_node.append({
            "dropped_node": node,
            "adherence_without_node": r["adherence_rate"],
            "delta_vs_full": r["adherence_rate"] - ref_rate,  # negative => node mattered
        })
    # a node is "necessary" if removing it drops adherence appreciably.
    per_node.sort(key=lambda d: d["delta_vs_full"])  # most-necessary first
    return {"experiment": "minimality", "ablation_kind": getattr(ablate_value, 'kind', 'unknown'),
            "full_circuit_adherence": ref_rate,
            "per_node": per_node}


def run_statistical_comparison(act_patching_results) -> dict:
    """Function to answer the following (and many more) questions:
    
    1. How similar were the experiments on original vs held-out data?
    2. How similar were the experiments across different languages?"""
    pass


def run_activation_patching(circuit: dict):
    """Function to co-ordinate all of the above."""
    # Run sufficiency verification
    # Run on held-out
    # Run on different languages
