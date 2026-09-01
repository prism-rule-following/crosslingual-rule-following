import torch

from canonical.causal.vector_patching.intervene import (
    make_patch_hook,
    make_steer_hook,
    run_intervention,
)


def test_patch_hook_only_touches_last_position(tiny_model):
    d_model = tiny_model.cfg.d_model
    u = torch.randn(d_model)
    u = u / u.norm()
    hook_fn = make_patch_hook(u, c_donor=3.0)

    resid = torch.randn(2, 5, d_model)
    before = resid.clone()
    out = hook_fn(resid, hook=None)

    assert torch.allclose(out[:, :-1, :], before[:, :-1, :])
    assert not torch.allclose(out[:, -1, :], before[:, -1, :])


def test_patch_hook_zeroes_u_component_and_sets_donor_coordinate():
    d_model = 8
    u = torch.zeros(d_model)
    u[0] = 1.0
    hook_fn = make_patch_hook(u, c_donor=5.0)

    resid = torch.zeros(1, 1, d_model)
    resid[0, 0, 0] = 2.0
    resid[0, 0, 1] = 7.0
    out = hook_fn(resid, hook=None)

    assert torch.isclose(out[0, 0, 0], torch.tensor(5.0))
    assert torch.isclose(out[0, 0, 1], torch.tensor(7.0))


def test_patch_hook_fires_once():
    d_model = 4
    u = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hook_fn = make_patch_hook(u, c_donor=9.0)

    resid = torch.zeros(1, 3, d_model)
    out1 = hook_fn(resid, hook=None)
    assert torch.isclose(out1[0, -1, 0], torch.tensor(9.0))

    # second call with seq_len=1 (single-token decode step) must no-op
    resid2 = torch.zeros(1, 1, d_model)
    out2 = hook_fn(resid2, hook=None)
    assert torch.allclose(out2, resid2)


def test_steer_hook_is_pure_additive():
    d_model = 4
    u = torch.tensor([0.0, 1.0, 0.0, 0.0])
    hook_fn = make_steer_hook(u, alpha=2.0)

    resid = torch.ones(1, 2, d_model)
    before = resid.clone()
    out = hook_fn(resid, hook=None)

    assert torch.allclose(out[:, 0, :], before[:, 0, :])
    assert torch.allclose(out[:, 1, :], before[:, 1, :] + 2.0 * u)


def test_run_intervention_generates_against_real_toy_model(tiny_model):
    d_model = tiny_model.cfg.d_model
    n_layers = tiny_model.cfg.n_layers
    u = torch.randn(d_model)
    u = u / u.norm()

    tokens = tiny_model.to_tokens("The cat sat on the")
    hook_fn = make_patch_hook(u, c_donor=1.0)

    text = run_intervention(tiny_model, tokens, layer=n_layers - 1, hook_fn=hook_fn, max_new_tokens=5)

    assert isinstance(text, str)
    assert len(text) > 0
