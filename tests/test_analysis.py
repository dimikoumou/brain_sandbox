"""The analysis tools must find effects that are there, and not find effects that aren't."""

import numpy as np
import pytest

from brain_sandbox.analysis.clusters import cluster_units, match_to_reference, centroids
from brain_sandbox.analysis.lesion_matrix import find_dissociations, matrix_stats
from brain_sandbox.analysis.pitch_probe import CHANCE, decode_accuracy, few_shot_accuracy, pitch_report, shuffled_accuracy
from brain_sandbox.config import ModelConfig, TaskConfig
from brain_sandbox.model import ModularRNN
from brain_sandbox.stats import holm, t_test
from brain_sandbox.tasks import TASKS, TaskSet


def test_holm_matches_hand_calculation():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_t_test_zero_spread():
    assert t_test([0.2, 0.2, 0.2], "greater") == 0.0
    assert t_test([0.0, 0.0, 0.0]) == 1.0


def _records(effects, seeds=5, noise=0.02, seed=0):
    """effects[condition][task] = true change in accuracy; intact accuracy 0.9."""
    rng = np.random.default_rng(seed)
    tasks = ["A", "B", "C"]
    recs = []
    for s in range(seeds):
        for c, eff in effects.items():
            for t in tasks:
                acc = np.clip(0.9 + eff.get(t, 0.0) + noise * rng.standard_normal(), 0, 1)
                recs.append({"seed": s, "condition": c, "task": t, "accuracy": acc, "broke_fixation": 0.0})
    return recs


def test_finds_planted_double_dissociation():
    recs = _records({"Intact": {}, "X": {"A": -0.5}, "Y": {"B": -0.5}, "Z": {}})
    found, n = find_dissociations(recs)
    pairs = {(d["X"], d["A"], d["Y"], d["B"]) for d in found}
    assert ("X", "A", "Y", "B") in pairs
    assert all({d["X"], d["Y"]} == {"X", "Y"} for d in found)  # Z is never involved
    stats = matrix_stats(recs)
    verdict = {(c["condition"], c["task"]): c["verdict"] for c in stats["cells"]}
    assert verdict[("X", "A")] == "hurts" and verdict[("X", "B")] == "no clear effect"


def test_no_false_dissociations_from_noise():
    recs = _records({"Intact": {}, "X": {}, "Y": {}, "Z": {}}, noise=0.05)
    found, _ = find_dissociations(recs)
    assert found == []


def test_small_effects_are_not_flagged():
    recs = _records({"Intact": {}, "X": {"A": -0.05}, "Y": {"B": -0.05}}, noise=0.001)
    found, _ = find_dissociations(recs, min_effect=0.10)
    assert found == []


def test_decoder_on_synthetic_data():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(12), 30)
    X = np.eye(12)[y] + 0.2 * rng.standard_normal((len(y), 12))
    assert decode_accuracy(X, y) > 0.9
    assert abs(shuffled_accuracy(X, y) - CHANCE) < 0.06
    assert few_shot_accuracy(X, y, 1) < few_shot_accuracy(X, y, 10)


def test_lesioned_region_decodes_at_chance_level():
    y = np.repeat(np.arange(12), 10)
    assert decode_accuracy(np.zeros((len(y), 5)), y) < 0.2


def test_pitch_report_controls_on_untrained_network():
    ts = TaskSet(TaskConfig(names=list(TASKS), n_tono=36, n_choice=2), 20.0)
    m = ModularRNN(ModelConfig(), ts.layout.in_groups, ts.layout.n_out, seed=0)
    rep = pitch_report(m, ts.layout, 20.0, n_reps=8, few_shot=(1,))
    assert rep["input_during"] > 0.9  # the stimulus itself is perfectly informative
    assert rep["input_post"] < 0.2  # nothing left in the input after the tone
    # Only sensory receives tones; regions with no path from the input during
    # the tone are not tested here, but every number must be a valid accuracy.
    for entry in rep["modules"].values():
        assert 0 <= entry["post"] <= 1 and 0 <= entry["post_shuffled"] <= 1


def test_clustering_recovers_planted_groups():
    rng = np.random.default_rng(1)
    profiles = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=float)
    true = np.repeat([0, 1, 2], 30)
    V = profiles[true] * rng.uniform(0.5, 2, (90, 1)) + 0.02 * rng.random((90, 4))
    res = cluster_units(V, k_range=range(2, 6))
    assert res["k"] == 3
    for c in range(3):  # every planted group ends up in a single cluster
        assert len(np.unique(res["labels"][true == c])) == 1


def test_cluster_matching_across_seeds():
    rng = np.random.default_rng(2)
    profiles = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    true = np.repeat([0, 1, 2], 20)
    V = profiles[true] + 0.01 * rng.random((60, 3))
    ref_labels = true
    perm = np.array([2, 0, 1])[true]  # same clusters, different numbering
    new = match_to_reference(centroids(V, ref_labels), centroids(V, perm), perm)
    assert np.array_equal(new, ref_labels)
