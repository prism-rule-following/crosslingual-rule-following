"""The actual hook: patch (swap, Exp 2) or steer (nudge, Exp 3) the residual
stream at one layer, then generate. Requires a live model -- CPU-testable
against a tiny TransformerLens model, real runs need the GPU pod.
"""

from typing import Callable, Optional

import torch

from canonical.causal.vector_patching.config import HOOK_NAME, POSITION


def make_patch_hook(u: torch.Tensor, c_donor: float, position: int = POSITION) -> Callable:
    """x' = x - (x . u) u + c_donor . u, i.e. swap out the component along u
    for the donor's. Fires once -- the hook's first call within a generate()
    is always the full-prompt forward pass (there's no decode step before
    the prompt has been seen at all), so a single `fired` flag is sufficient
    to scope this to "patch the prompt's last token only"."""
    fired = False

    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        nonlocal fired
        if fired:
            return resid
        fired = True
        u_dev = u.to(resid.device, resid.dtype)
        x = resid[:, position, :]
        coord = (x @ u_dev).unsqueeze(-1)
        resid[:, position, :] = x - coord * u_dev + c_donor * u_dev
        return resid

    return hook_fn


def make_steer_hook(u: torch.Tensor, alpha: float, position: int = POSITION) -> Callable:
    """x' = x + alpha * u. Fires once, same scope as make_patch_hook."""
    fired = False

    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        nonlocal fired
        if fired:
            return resid
        fired = True
        u_dev = u.to(resid.device, resid.dtype)
        resid[:, position, :] = resid[:, position, :] + alpha * u_dev
        return resid

    return hook_fn


def run_intervention(
    model,
    tokens: torch.Tensor,
    layer: int,
    hook_fn: Callable,
    hook_name: str = HOOK_NAME,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    stop_at_eos: bool = True,
) -> str:
    """Registers hook_fn on blocks.{layer}.{hook_name}, generates, returns
    the completion text (prompt stripped)."""
    hook_point = f"blocks.{layer}.{hook_name}"
    n_prompt = tokens.shape[-1]
    with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            stop_at_eos=stop_at_eos,
            verbose=False,
        )
    return model.to_string(out[0, n_prompt:])
