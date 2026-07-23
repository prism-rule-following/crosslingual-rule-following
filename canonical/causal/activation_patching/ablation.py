"""Code related to ablation: precompute values, build the hooks."""

from typing import Optional, Dict, List, Callable, Any, Tuple
import random
import torch


def make_ablation_value(kind: str,
                        mean_acts: Optional[Dict[str, torch.Tensor]] = None,
                        resample_pool: Optional[Dict[str, List[torch.Tensor]]] = None,
                        seed: int = 0) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """Build the replacement-value function used when a node is ablated.

    Returns `value_for(node_name, like) -> tensor shaped like `like``.

    Example for resample_pool:  {node_name: [activation, activation, ...]} -- a list of
                    activations THAT NODE took on OTHER inputs, one entry per
                    source input. Required for "resample". Example, for two
                    nodes and three corrupt prompts:

                        resample_pool = {
                            "blocks.5.attn.hook_z@h3": [
                                t_from_prompt_A,   # [batch, pos, d_head]
                                t_from_prompt_B,
                                t_from_prompt_C,
                            ],
                            "blocks.7.hook_mlp_out": [
                                t_from_prompt_A,   # [batch, pos, d_model]
                                t_from_prompt_B,
                                t_from_prompt_C,
                            ],
                        }

                    Needs to be pre-built.
    """
    if kind not in {"zero", "mean", "resample", "random"}:
        raise ValueError(f"unknown ablation kind {kind!r}; "
                         "expected zero | mean | resample | random")
    if kind == "mean" and not mean_acts:
        raise ValueError("mean ablation requires precomputed mean_acts "
                         "(mean activation per node over a distribution) "
                         "-- arxiv 2407.08734")
    if kind == "resample" and not resample_pool:
        raise ValueError("resample ablation requires a resample_pool "
                         "(activations from other/corrupt inputs) "
                         "-- arxiv 2407.08734 / causal scrubbing")

    rng = random.Random(seed)

    def value_for(name: str, like: torch.Tensor) -> torch.Tensor:
        if kind == "zero":
            return torch.zeros_like(like)
        if kind == "mean":
            return mean_acts[name].to(like.dtype).expand_as(like).clone()
        if kind == "resample":
            pool = resample_pool[name]
            chosen = pool[rng.randrange(len(pool))]
            return chosen.to(like.dtype).expand_as(like).clone()
        # random
        v = torch.randn_like(like)
        return v / (v.norm() + 1e-8) * like.norm()

    value_for.kind = kind          # so results can record which was used
    return value_for


def precompute_mean_acts(model, tokens_iter, names: List[str]) -> Dict[str, torch.Tensor]:
    """Mean activation per node over a distribution of inputs (for mean ablation)."""
    sums: Dict[str, torch.Tensor] = {}
    count = 0
    for tokens in tokens_iter:
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names, return_type=None)
        for n in names:
            a = cache[n].detach()
            sums[n] = a if n not in sums else sums[n] + a
        count += 1
    return {n: (sums[n] / max(count, 1)) for n in names}


def precompute_resample_pool(model, tokens_iter, names: List[str]) -> Dict[str, List[torch.Tensor]]:
    """Cache each node's activation on SEVERAL OTHER INPUTS (=system_non_rule or Rule status: inactive),
    for resample ablation.

    Returns {node_name: [act_from_input_1, act_from_input_2, ...]} -- exactly
    the `resample_pool` shape `make_ablation_value` expects.
    """
    pool: Dict[str, List[torch.Tensor]] = {n: [] for n in names}
    for tokens in tokens_iter:
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names, return_type=None)
        for n in names:
            pool[n].append(cache[n].detach().clone())
    return pool


def ablate_and_score(model, dataset: List[Dict[str, Any]], ablate_nodes: List[str],
                     node_table: Dict[str, Tuple[str, Optional[int]]],
                     ablate_value: Callable[[str, torch.Tensor], torch.Tensor],
                     generate_fn: Callable[[Any, torch.Tensor], str],
                     adherence_fn: Callable[[Dict[str, Any], str], bool],
                     tokenize_fn: Callable[[Dict[str, Any]], torch.Tensor],
                     tag: str = "") -> Dict[str, Any]:
    """Ablate `ablate_nodes`, generate on every row, score adherence.

    The shared workhorse: necessity, completeness and the co-ablation controls
    all reduce to "ablate this specific set of nodes and see what the adherence
    rate does", differing only in WHICH set. Kept top-level (not nested inside
    a caller) so each experiment is a one-line call and the function is
    independently testable.
    """
    results = []
    for row in dataset:
        tokens = tokenize_fn(row)
        apply_ablation_hooks(model, ablate_circuit=True,
                             circuit_nodes=ablate_nodes,
                             node_table=node_table,
                             ablate_value=ablate_value)
        text = generate_fn(model, tokens)
        model.reset_hooks()
        results.append({"id": row.get("id"), "category": row.get("category"),
                        "topic": row.get("topic"), "language": row.get("language"),
                        "response": text, "adheres": bool(adherence_fn(row, text))})
    rate = sum(r["adheres"] for r in results) / max(len(results), 1)
    return {"tag": tag, "adherence_rate": rate, "responses": results}


def parse_nodes(nodes: List[str]) -> Dict[str, Tuple[str, Optional[int]]]:
    # TODO: probably needs to be adapted to whatever EAP returns (should be TL)
    """Parse every node name once: 'blocks.5.attn.hook_z@h3' -> ('blocks.5.attn.hook_z', 3)."""
    table: Dict[str, Tuple[str, Optional[int]]] = {}
    for node in nodes:
        if "@h" in node:
            base, h = node.split("@h")
            table[node] = (base, int(h))
        else:
            table[node] = (node, None)
    return table


def group_by_hook(node_table: Dict[str, Tuple[str, Optional[int]]]) -> Dict[str, List[str]]:
    """Invert the table: {hook_name: [node names living on that hook]}."""
    grouped: Dict[str, List[str]] = {}
    for node, (hook_name, _) in node_table.items():
        grouped.setdefault(hook_name, []).append(node)
    return grouped


def apply_ablation_hooks(model, ablate_circuit: bool, circuit_nodes: List[str],
                         node_table: Dict[str, Tuple[str, Optional[int]]],
                         ablate_value: Callable[[str, torch.Tensor], torch.Tensor]):
    """Register the ablation hooks. Call `model.reset_hooks()` when done.

    ablate_circuit=False -> ablate everything NOT in the circuit
                            (denoising; tests SUFFICIENCY)
    ablate_circuit=True  -> ablate everything IN the circuit
                            (noising;   tests NECESSITY)
    """
    circuit_set = set(circuit_nodes)
    grouped = group_by_hook(node_table)

    def make_hook(nodes_on_this_tensor: List[str]):
        def hook(act, hook):  # noqa: ARG001
            # `# noqa: ARG001` silences the linter's unused-argument warning:
            # TransformerLens requires the (act, hook) signature even when the
            # body never touches `hook`.
            for node in nodes_on_this_tensor:
                in_circuit = node in circuit_set
                # ablate this node only when its membership matches the mode:
                #   ablate_circuit=True  -> ablate the ones IN the circuit
                #   ablate_circuit=False -> ablate the ones OUTSIDE it
                if in_circuit != ablate_circuit:
                    continue
                _, head = node_table[node]
                if head is None:
                    # Whole-tensor node: replace everything this hook carries.
                    act = ablate_value(node, act)
                else:
                    # Per-head node: replace ONLY this head's slice, so the
                    # other heads sharing this tensor keep their real values.
                    act = act.clone()
                    act[:, :, head, :] = ablate_value(node, act[:, :, head, :])
            return act
        return hook

    for hook_name, nodes_here in grouped.items():
        model.add_hook(hook_name, make_hook(nodes_here))