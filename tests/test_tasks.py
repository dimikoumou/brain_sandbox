"""Task generators must produce the right inputs and ground truth."""

import numpy as np
import pytest

from brain_sandbox.config import TaskConfig
from brain_sandbox.tasks import TaskSet
from brain_sandbox.tasks.common import FIX_OFF_TARGET, FIX_ON_TARGET, circular_distance

DT = 20.0


@pytest.fixture
def taskset():
    return TaskSet(TaskConfig(names=["go", "delaygo"], n_ring=16), DT)


def test_layout_sizes(taskset):
    L = taskset.layout
    assert L.n_in == 1 + 16 + 16 + 0 + 2
    assert L.n_out == 1 + 16 + 0


def test_ring_decode_recovers_angle(taskset):
    L = taskset.layout
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    decoded = L.decode_ring(L.ring_bump(theta))
    assert circular_distance(decoded, theta).max() < 1e-3


@pytest.mark.parametrize("task", ["go", "delaygo"])
def test_rule_input_is_one_hot(taskset, task):
    b = taskset.generate(task, 8, np.random.default_rng(0))
    rule = b.x[:, :, taskset.layout.in_slices["rule"]]
    assert np.all(rule.sum(-1) == 1.0)
    assert np.all(rule[..., taskset.layout.rule_index(task)] == 1.0)


@pytest.mark.parametrize("task", ["go", "delaygo"])
def test_fixation_cue_and_targets(taskset, task):
    L = taskset.layout
    b = taskset.generate(task, 8, np.random.default_rng(1))
    r0 = b.response_start
    fix_in = b.x[:, :, L.in_slices["fixation"]][..., 0]
    assert np.all(fix_in[:r0] == 1.0) and np.all(fix_in[r0:] == 0.0)
    fix_out = b.y[:, :, L.out_slices["fixation"]][..., 0]
    assert np.allclose(fix_out[:r0], FIX_ON_TARGET) and np.allclose(fix_out[r0:], FIX_OFF_TARGET)


@pytest.mark.parametrize("task", ["go", "delaygo"])
def test_target_direction_equals_stimulus(taskset, task):
    """For Go and Delay-Go, the correct response is the stimulus direction."""
    L = taskset.layout
    b = taskset.generate(task, 32, np.random.default_rng(2))
    s0, _ = b.epochs["stimulus"]
    stim = b.x[s0, :, L.in_slices["mod1"]] + b.x[s0, :, L.in_slices["mod2"]]
    assert circular_distance(L.decode_ring(stim), b.target_dir).max() < 1e-3
    target_ring = b.y[-1, :, L.out_slices["ring"]]
    assert circular_distance(L.decode_ring(target_ring), b.target_dir).max() < 1e-3


@pytest.mark.parametrize("task", ["go", "delaygo"])
def test_stimulus_on_the_right_modality(taskset, task):
    L = taskset.layout
    b = taskset.generate(task, 32, np.random.default_rng(3))
    s0, _ = b.epochs["stimulus"]
    for m, other in ((1, 2), (2, 1)):
        sel = b.meta["modality"] == m
        assert np.all(b.x[s0, sel, L.in_slices[f"mod{m}"]].max(-1) > 0.5)
        assert np.all(b.x[s0, sel, L.in_slices[f"mod{other}"]] == 0.0)


def test_go_stimulus_stays_on_during_response(taskset):
    L = taskset.layout
    b = taskset.generate("go", 8, np.random.default_rng(4))
    stim = b.x[:, :, L.in_slices["mod1"]].max(-1) + b.x[:, :, L.in_slices["mod2"]].max(-1)
    s0 = b.epochs["stimulus"][0]
    assert np.all(stim[:s0] == 0.0) and np.all(stim[s0:] > 0.5)


def test_delaygo_stimulus_off_during_delay(taskset):
    """The whole point of Delay-Go: nothing on the screen during delay and response."""
    L = taskset.layout
    b = taskset.generate("delaygo", 8, np.random.default_rng(5))
    stim = b.x[:, :, L.in_slices["mod1"]].max(-1) + b.x[:, :, L.in_slices["mod2"]].max(-1)
    s0, s1 = b.epochs["stimulus"]
    assert np.all(stim[s0:s1] > 0.5)
    assert np.all(stim[s1:] == 0.0)


def test_loss_mask_grace_period(taskset):
    b = taskset.generate("go", 4, np.random.default_rng(6))
    r0 = b.response_start
    grace = int(100 / DT)
    assert np.all(b.loss_mask[r0 : r0 + grace] == 0.0)
    assert np.all(b.loss_mask[r0 + grace :] > 0.0)


def test_config_override_changes_timing():
    ts = TaskSet(TaskConfig(names=["delaygo"], params={"delaygo": {"delay_ms": 1000}}), DT)
    b = ts.generate("delaygo", 2, np.random.default_rng(0))
    s, e = b.epochs["delay"]
    assert (e - s) * DT == 1000
