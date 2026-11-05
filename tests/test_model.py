"""The model must respect its own structure: Dale's law, connectivity, lesions."""

import numpy as np
import pytest
import torch

from brain_sandbox.config import ModelConfig, ModuleConfig, TaskConfig
from brain_sandbox.metrics import trial_correct
from brain_sandbox.model import ModularRNN, to_torch
from brain_sandbox.tasks import TaskSet

DT = 20.0


def make(dale=True, **kw):
    ts = TaskSet(TaskConfig(names=["go", "delaygo"]), DT)
    cfg = ModelConfig(dale=dale, **kw)
    return ModularRNN(cfg, ts.layout.in_groups, ts.layout.n_out, seed=0), ts


def test_dale_signs():
    """Every outgoing weight of an E unit is >= 0, of an I unit <= 0."""
    m, _ = make(dale=True)
    W = m.effective_W_rec().detach()
    assert (W[:, m.is_exc] >= 0).all()
    assert (W[:, ~m.is_exc] <= 0).all()
    assert (W[:, ~m.is_exc] < 0).any()  # inhibition actually exists


def test_dale_signs_survive_training_step():
    m, ts = make(dale=True)
    opt = torch.optim.SGD(m.parameters(), lr=10.0)  # huge step to try to flip signs
    b = ts.generate("go", 8, np.random.default_rng(0))
    out, _ = m(to_torch(b.x))
    ((out - to_torch(b.y)) ** 2).mean().backward()
    opt.step()
    W = m.effective_W_rec().detach()
    assert (W[:, m.is_exc] >= 0).all() and (W[:, ~m.is_exc] <= 0).all()


def test_missing_pathways_have_zero_weight():
    """Default wiring has no sensory<->executive or memory<->conceptual pathway."""
    m, _ = make()
    W = m.effective_W_rec().detach()
    for a, b in [("sensory", "executive"), ("executive", "sensory"), ("memory", "conceptual"), ("conceptual", "memory")]:
        assert (W[m.units_of(b), m.units_of(a)] == 0).all(), f"{a}->{b} should not exist"
    assert (W[m.units_of("memory"), m.units_of("sensory")] != 0).any()


def test_long_range_connections_are_excitatory():
    m, _ = make(dale=True, long_range_excitatory_only=True)
    W = m.effective_W_rec().detach()
    s, x = m.units_of("conceptual"), m.units_of("sensory")
    assert (W[x, s] >= 0).all()  # conceptual -> sensory feedback is purely excitatory


def test_input_routing():
    """Stimuli reach only sensory; the rule reaches only executive."""
    m, ts = make()
    W_in = m.effective_W_in().detach()
    L = ts.layout
    others = [n for n in m.module_names if n != "sensory"]
    for n in others:
        assert (W_in[m.units_of(n), L.in_slices["mod1"]] == 0).all()
    for n in [n for n in m.module_names if n != "executive"]:
        assert (W_in[m.units_of(n), L.in_slices["rule"]] == 0).all()


def test_lesioned_units_output_exactly_zero():
    """Spec requirement: a lesioned unit has exactly zero output at every time step."""
    m, ts = make()
    lesioned = torch.arange(m.units_of("conceptual").start, m.units_of("conceptual").start + 20)
    m.unit_mask[lesioned] = 0.0
    b = ts.generate("delaygo", 16, np.random.default_rng(0))
    _, rates = m(to_torch(b.x))
    assert torch.all(rates[:, :, lesioned] == 0.0)
    assert torch.any(rates[:, :, m.units_of("sensory")] != 0.0)


def test_lesion_removes_influence_on_others():
    """Zeroing output (not only input) means downstream units cannot tell the lesioned units' weights."""
    m, ts = make()
    b = ts.generate("go", 4, np.random.default_rng(0))
    x = to_torch(b.x)
    m.unit_mask[m.units_of("memory")] = 0.0
    _, r1 = m(x, noise=False)
    with torch.no_grad():  # change memory's outgoing weights; should make no difference
        m.W_rec_raw[:, m.units_of("memory")] *= 3.0
    _, r2 = m(x, noise=False)
    assert torch.equal(r1, r2)


def test_flat_model_runs():
    ts = TaskSet(TaskConfig(names=["go"]), DT)
    cfg = ModelConfig(
        modules=[ModuleConfig("flat", 128)],
        pathways=[],
        input_routes={g: ["flat"] for g in ts.layout.in_groups},
        output_module="flat",
    )
    m = ModularRNN(cfg, ts.layout.in_groups, ts.layout.n_out)
    b = ts.generate("go", 4, np.random.default_rng(0))
    out, rates = m(to_torch(b.x))
    assert out.shape == (b.x.shape[0], 4, ts.layout.n_out)
    assert rates.shape[-1] == 128


def test_initial_spectral_radius():
    m, _ = make(init_spectral_radius=0.9)
    radius = torch.linalg.eigvals(m.effective_W_rec().detach()).abs().max().item()
    assert radius == pytest.approx(0.9, rel=1e-4)


def test_per_module_time_constant():
    mods = [ModuleConfig("sensory", tau_ms=50), ModuleConfig("memory", tau_ms=400),
            ModuleConfig("conceptual"), ModuleConfig("executive")]
    m, _ = make(modules=mods)
    assert torch.allclose(m.alpha[m.units_of("sensory")], torch.tensor(DT / 50))
    assert torch.allclose(m.alpha[m.units_of("memory")], torch.tensor(DT / 400))


def test_noise_off_is_deterministic():
    m, ts = make()
    x = to_torch(ts.generate("go", 4, np.random.default_rng(0)).x)
    assert torch.equal(m(x, noise=False)[1], m(x, noise=False)[1])


def test_scoring():
    """A perfect output scores 1; breaking fixation early scores 0."""
    ts = TaskSet(TaskConfig(names=["go"]), DT)
    b = ts.generate("go", 16, np.random.default_rng(0))
    assert trial_correct(b.y, b, ts.layout, DT).all()
    bad = b.y.copy()
    bad[b.response_start - 3 :, :, 0] = 0.0  # drops fixation 3 steps too early
    assert not trial_correct(bad, b, ts.layout, DT).any()
    wrong = b.y.copy()
    wrong[:, :, ts.layout.out_slices["ring"]] = np.roll(wrong[:, :, ts.layout.out_slices["ring"]], 8, axis=-1)
    assert not trial_correct(wrong, b, ts.layout, DT).any()  # 180 degrees off
