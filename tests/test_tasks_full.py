"""Ground truth of the full task set, the pitch task and the probe trials."""

import numpy as np
import pytest

from brain_sandbox.config import TaskConfig
from brain_sandbox.metrics import trial_outcomes
from brain_sandbox.tasks import TASKS, TaskSet
from brain_sandbox.tasks.common import circular_distance
from brain_sandbox.tasks.pitch import probe_batch

DT = 20.0


@pytest.fixture(scope="module")
def ts():
    return TaskSet(TaskConfig(names=list(TASKS), n_tono=36, n_choice=2), DT)


def ring_at(b, ts, t, mod):
    return b.x[t, :, ts.layout.in_slices[f"mod{mod}"]]


@pytest.mark.parametrize("task", list(TASKS))
def test_every_task_scores_perfect_on_its_own_target(ts, task):
    """Feeding the target back in as the 'output' must score 100% correct."""
    b = ts.generate(task, 64, np.random.default_rng(0))
    assert (trial_outcomes(b.y, b, ts.layout, DT) == "correct").all()


def test_anti_target_is_opposite(ts):
    b = ts.generate("anti", 32, np.random.default_rng(1))
    expected = np.mod(b.meta["stim_dir"] + np.pi, 2 * np.pi)
    assert circular_distance(b.target_dir, expected).max() < 1e-9


@pytest.mark.parametrize("task,attend", [("dm1", 1), ("dm2", 2), ("ctxdm1", 1), ("ctxdm2", 2)])
def test_dm_target_is_stronger_stimulus_in_attended_modality(ts, task, attend):
    b = ts.generate(task, 64, np.random.default_rng(2))
    m = b.meta
    expected = np.where(m["coh"] > 0, m["theta1"], m["theta2"])
    assert circular_distance(b.target_dir, expected).max() < 1e-9
    # The attended modality shows exactly: bump(theta1)*(mean+coh) + bump(theta2)*(mean-coh).
    L = ts.layout
    s0 = b.epochs["stimulus"][0]
    expected_input = (L.ring_bump(m["theta1"]) * (m["mean_strength"] + m["coh"])[:, None]
                      + L.ring_bump(m["theta2"]) * (m["mean_strength"] - m["coh"])[:, None])
    assert np.allclose(ring_at(b, ts, s0, attend), expected_input, atol=1e-6)


def test_ctxdm_ignores_other_modality(ts):
    """Trials where the two modalities disagree must follow the attended one."""
    b = ts.generate("ctxdm1", 256, np.random.default_rng(3))
    m = b.meta
    disagree = np.sign(m["coh"]) != np.sign(m["coh_other"])
    assert disagree.sum() > 50
    expected = np.where(m["coh"] > 0, m["theta1"], m["theta2"])
    assert circular_distance(b.target_dir[disagree], expected[disagree]).max() < 1e-9


@pytest.mark.parametrize("task,respond_on_match", [("dms", True), ("dnms", False)])
def test_matching_tasks(ts, task, respond_on_match):
    b = ts.generate(task, 256, np.random.default_rng(4))
    m = b.meta
    respond = m["match"] if respond_on_match else ~m["match"]
    assert np.all(~np.isnan(b.target_dir) == respond)
    assert circular_distance(m["sample_dir"][m["match"]], m["test_dir"][m["match"]]).max() < 1e-9
    assert circular_distance(m["sample_dir"][~m["match"]], m["test_dir"][~m["match"]]).min() >= np.pi / 3 - 1e-9
    assert 0.35 < m["match"].mean() < 0.65
    # nothing on the screen during the delay
    d0, d1 = b.epochs["delay"]
    L = ts.layout
    assert np.all(b.x[d0:d1, :, L.in_slices["mod1"]] == 0) and np.all(b.x[d0:d1, :, L.in_slices["mod2"]] == 0)


def test_relpitch_ground_truth(ts):
    b = ts.generate("relpitch", 512, np.random.default_rng(5))
    a, bb, same = b.meta["melody_a"], b.meta["melody_b"], b.meta["same"]
    ia, ib = np.diff(a, axis=1), np.diff(bb, axis=1)
    assert np.all((ia == ib).all(axis=1) == same)  # same tune <=> same intervals
    assert np.all(b.meta["shift"] != 0)  # always a new key
    assert np.all(b.target_choice == np.where(same, 0, 1))
    n = ts.layout.n_semitones
    assert a.min() >= 0 and bb.min() >= 0 and a.max() <= n and bb.max() <= n
    assert 0.4 < same.mean() < 0.6


def test_relpitch_absolute_pitch_never_gives_the_answer(ts):
    """Melody B never repeats A's notes exactly, so absolute pitch can't reveal 'same'."""
    b = ts.generate("relpitch", 512, np.random.default_rng(6))
    assert not np.any((b.meta["melody_a"] == b.meta["melody_b"]).all(axis=1))


def test_relpitch_tones_are_presented(ts):
    """At each tone's onset, the tonotopic input peaks at the right note."""
    b = ts.generate("relpitch", 32, np.random.default_rng(7))
    L = ts.layout
    tono = b.x[:, :, L.in_slices["tono"]]
    note_steps = int(round(200 / DT))
    for epoch, mel in (("melody1", b.meta["melody_a"]), ("melody2", b.meta["melody_b"])):
        t0 = b.epochs[epoch][0]
        for k in range(mel.shape[1]):
            peak = L.tono_pref[tono[t0 + k * note_steps].argmax(-1)]
            assert np.abs(peak - mel[:, k]).max() <= 0.5
    # nothing on the tonotopic input during the delay between melodies
    d0, d1 = b.epochs["delay"]
    assert np.all(tono[d0:d1] == 0)


def test_probe_batch(ts):
    L = ts.layout
    notes = np.arange(25)
    b = probe_batch(L, notes, DT)
    assert np.all(b.meta["pitch_class"] == notes % 12)
    assert np.all(b.x[:, :, L.in_slices["rule"].start + L.rule_index("relpitch")] == 1)
    p0, p1 = b.epochs["post"]
    assert np.all(b.x[p0:p1, :, L.in_slices["tono"]] == 0)  # silence after the tone


def test_note_frequencies(ts):
    f = ts.layout.note_frequency(np.array([0, 12, 24]))
    assert np.allclose(f, [261.63, 523.26, 1046.52])


def test_outcome_types(ts):
    b = ts.generate("dms", 64, np.random.default_rng(8))
    L = ts.layout
    out = b.y.copy()
    out[:, :, L.out_slices["fixation"]] = 0.85  # never respond
    o = trial_outcomes(out, b, L, DT)
    respond = b.should_respond
    assert np.all(o[respond] == "no_response") and np.all(o[~respond] == "correct")
    early = b.y.copy()
    early[b.response_start - 5 :, :, 0] = 0.0
    assert np.all(trial_outcomes(early, b, L, DT) == "broke_fixation")


def test_pitchdir_ground_truth(ts):
    b = ts.generate("pitchdir", 512, np.random.default_rng(9))
    step = b.meta["second"] - b.meta["first"]
    assert np.all(step != 0)
    assert np.all(b.target_choice == np.where(step > 0, 0, 1))
    n = ts.layout.n_semitones
    assert b.meta["first"].min() >= 0 and b.meta["second"].max() <= n
    assert 0.4 < (step > 0).mean() < 0.6
    # The first note varies widely, so absolute pitch can't predict the answer.
    assert len(np.unique(b.meta["first"])) > 15
    for c in (0, 1):
        assert abs(np.corrcoef(b.meta["first"], b.target_choice)[0, 1]) < 0.35
