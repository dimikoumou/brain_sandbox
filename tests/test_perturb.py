"""Every perturbation must do exactly what it says, and nothing else."""

import numpy as np
import pytest
import torch

from brain_sandbox.config import ModelConfig, ModuleConfig, TaskConfig
from brain_sandbox.model import ModularRNN, to_torch
from brain_sandbox.perturb import (
    Condition, CutPathway, Gain, Inhibition, LesionModule, LesionUnits, Noise, Prune,
    ScaleWeights, TimeConstant, from_spec, perturbed, rehabilitate, to_spec,
)
from brain_sandbox.tasks import TaskSet

DT = 20.0


@pytest.fixture
def setup():
    ts = TaskSet(TaskConfig(names=["go", "delaygo"]), DT)
    m = ModularRNN(ModelConfig(), ts.layout.in_groups, ts.layout.n_out, seed=0)
    x = to_torch(ts.generate("delaygo", 8, np.random.default_rng(0)).x)
    return m, ts, x


def test_original_is_never_changed(setup):
    m, _, _ = setup
    before = {k: v.clone() for k, v in m.state_dict().items()}
    perturbed(m, [LesionModule("sensory"), Noise(5.0), CutPathway("memory", "executive"), Prune(0.5)])
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_whole_region_lesion_gives_exactly_zero_output(setup):
    m, _, x = setup
    p = perturbed(m, [LesionModule("conceptual")])
    _, rates = p(x)
    assert torch.all(rates[:, :, p.units_of("conceptual")] == 0.0)
    assert torch.any(rates[:, :, p.units_of("memory")] != 0.0)


def test_partial_lesion_count(setup):
    m, _, x = setup
    p = perturbed(m, [LesionModule("memory", fraction=0.5, seed=3)])
    sl = p.units_of("memory")
    assert int((p.unit_mask[sl] == 0).sum()) == 32
    assert int((p.unit_mask == 0).sum()) == 32  # nothing outside memory touched
    _, rates = p(x)
    dead = torch.nonzero(p.unit_mask == 0).flatten()
    assert torch.all(rates[:, :, dead] == 0.0)


def test_lesion_units(setup):
    m, _, x = setup
    p = perturbed(m, [LesionUnits([0, 5, 200], "a cluster")])
    _, rates = p(x)
    assert torch.all(rates[:, :, [0, 5, 200]] == 0.0)


def test_cut_pathway_zeroes_only_that_block(setup):
    m, _, _ = setup
    p = perturbed(m, [CutPathway("conceptual", "sensory")])
    W0, W1 = m.effective_W_rec().detach(), p.effective_W_rec().detach()
    s, c = m.units_of("sensory"), m.units_of("conceptual")
    assert (W0[s, c] != 0).any() and torch.all(W1[s, c] == 0)
    W0[s, c] = 0
    assert torch.equal(W0, W1)  # everything else identical


def test_partial_cut(setup):
    m, _, _ = setup
    p = perturbed(m, [CutPathway("sensory", "memory", fraction=0.5)])
    before = (m.effective_W_rec()[m.units_of("memory"), m.units_of("sensory")] != 0).sum().item()
    after = (p.effective_W_rec()[m.units_of("memory"), m.units_of("sensory")] != 0).sum().item()
    assert 0.35 * before < after < 0.65 * before


def test_noise_and_gain_and_tau_are_local(setup):
    m, _, _ = setup
    p = perturbed(m, [Noise(3.0, "sensory"), Gain(0.5, "memory"), TimeConstant(2.0, "executive")])
    s, mem, ex = m.units_of("sensory"), m.units_of("memory"), m.units_of("executive")
    assert torch.allclose(p.sigma[s], 3 * m.sigma[s]) and torch.equal(p.sigma[mem], m.sigma[mem])
    assert torch.allclose(p.gain[mem], 0.5 * m.gain[mem]) and torch.equal(p.gain[s], m.gain[s])
    assert torch.allclose(p.alpha[ex], m.alpha[ex] / 2) and torch.equal(p.alpha[s], m.alpha[s])


def test_global_noise(setup):
    m, _, _ = setup
    p = perturbed(m, [Noise(2.0, include_inputs=True)])
    assert torch.allclose(p.sigma, 2 * m.sigma) and p.input_noise == 2 * m.input_noise


def test_inhibition_scales_only_inhibitory_outputs(setup):
    m, _, _ = setup
    p = perturbed(m, [Inhibition(0.5)])
    W0, W1 = m.effective_W_rec().detach(), p.effective_W_rec().detach()
    assert torch.allclose(W1[:, ~m.is_exc], 0.5 * W0[:, ~m.is_exc])
    assert torch.equal(W1[:, m.is_exc], W0[:, m.is_exc])


def test_inhibition_needs_dale():
    ts = TaskSet(TaskConfig(names=["go"]), DT)
    m = ModularRNN(ModelConfig(dale=False), ts.layout.in_groups, ts.layout.n_out)
    with pytest.raises(ValueError):
        perturbed(m, [Inhibition(0.5)])


def test_prune_removes_the_weakest(setup):
    m, _, _ = setup
    p = perturbed(m, [Prune(0.3)])
    W0 = m.effective_W_rec().detach().abs()
    W1 = p.effective_W_rec().detach().abs()
    exists = m.conn_mask > 0
    removed = exists & (W1 == 0) & (W0 > 0)
    frac = removed.sum().item() / exists.sum().item()
    assert 0.28 < frac < 0.32
    assert W0[removed].max() <= W0[exists & ~removed].min()  # only the weakest ones went


def test_scale_weights_into_module(setup):
    m, _, _ = setup
    p = perturbed(m, [ScaleWeights(0.5, "executive")])
    W0, W1 = m.effective_W_rec().detach(), p.effective_W_rec().detach()
    ex = m.units_of("executive")
    assert torch.allclose(W1[ex], 0.5 * W0[ex])
    assert torch.equal(W1[: ex.start], W0[: ex.start])


def test_specs_round_trip():
    for p in [LesionModule("memory", 0.25, 1), CutPathway("a", "b"), Noise(2.0, "sensory"), Prune(0.1)]:
        assert from_spec(to_spec(p)) == p
    c = Condition.from_dict({"label": "x", "stack": [{"type": "gain", "factor": 2.0}]})
    assert c.stack == [Gain(2.0)] and Condition.from_dict(c.to_dict()) == c


def test_rehabilitation_keeps_the_damage(setup):
    m, ts, x = setup
    r, hist = rehabilitate(m, ts, [LesionModule("memory")], steps=3, eval_every=3, eval_trials=16)
    _, rates = r(x)
    assert torch.all(rates[:, :, r.units_of("memory")] == 0.0)
    assert [h["step"] for h in hist] == [0, 3]
