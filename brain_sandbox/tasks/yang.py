"""Cognitive tasks reimplemented from Yang et al. 2019 (github.com/gyyang/multitask).

Every task has the signature

    task(layout, batch_size, rng, dt_ms, params) -> TrialBatch

and its default parameters are in a DEFAULTS dict right above it, so you can
see every number the task uses. Durations are in ms; a [low, high] pair means
"sample uniformly per batch". A config can override any of them.

Plain-language summary (the "eye" is the network's response):

    go       a light appears; when told "go", look at it
    anti     a light appears; when told "go", look the OPPOSITE way
    delaygo  a light flashes and disappears; after a wait, look where it was
    dm1      two lights on channel 1; look at the brighter one
    dm2      same, on channel 2
    ctxdm1   two lights on BOTH channels; judge using channel 1 only, ignore channel 2
    ctxdm2   same, but use channel 2 only
    dms      a light, a wait, a second light: if both were in the same place, look
             at it; if not, don't move
    dnms     the reverse: look only if the two lights were in DIFFERENT places
"""

from __future__ import annotations

import numpy as np

from .common import Layout, TrialBatch, TrialBuilder, sample_ms


def _directions_and_modality(rng, B):
    theta = rng.uniform(0, 2 * np.pi, size=B)
    modality = rng.integers(1, 3, size=B)  # stimulus on modality 1 or 2, at random
    return theta, modality


def _two_directions(rng, B):
    """Two directions at least 90 degrees apart (as in Yang et al.)."""
    theta1 = rng.uniform(0, 2 * np.pi, size=B)
    theta2 = np.mod(theta1 + rng.uniform(0.5 * np.pi, 1.5 * np.pi, size=B), 2 * np.pi)
    return theta1, theta2


# --------------------------------------------------------------------------
# Go / Anti: a stimulus appears and stays on; when the fixation cue turns off,
# respond toward it (Go) or away from it (Anti).
# --------------------------------------------------------------------------
GO_DEFAULTS = {
    "fix_ms": [200, 600],  # fixation alone
    "stim_ms": [200, 1000],  # stimulus on, still fixating
    "response_ms": 400,  # fixation off: respond (stimulus still on)
    "strength": [0.8, 1.2],  # stimulus amplitude range
}


def _go_like(task: str, anti: bool, layout, B, rng, dt_ms, params):
    p = {**GO_DEFAULTS, **params}
    epochs = {
        "fixation": sample_ms(rng, p["fix_ms"]),
        "stimulus": sample_ms(rng, p["stim_ms"]),
        "response": sample_ms(rng, p["response_ms"]),
    }
    tb = TrialBuilder(layout, task, B, epochs, dt_ms)
    theta, modality = _directions_and_modality(rng, B)
    strength = rng.uniform(*p["strength"], size=B)

    tb.fixation_input(until="response")
    tb.ring_input(modality, theta, strength, tb.span("stimulus", "response"))
    target = np.mod(theta + np.pi, 2 * np.pi) if anti else theta
    tb.standard_targets(target_dir=target)
    return tb.build(target, {"stim_dir": theta, "modality": modality, "strength": strength})


def go(layout: Layout, B: int, rng: np.random.Generator, dt_ms: float, params: dict) -> TrialBatch:
    return _go_like("go", False, layout, B, rng, dt_ms, params)


def anti(layout: Layout, B: int, rng: np.random.Generator, dt_ms: float, params: dict) -> TrialBatch:
    return _go_like("anti", True, layout, B, rng, dt_ms, params)


# --------------------------------------------------------------------------
# Delay-Go: the stimulus is shown briefly, then disappears for a delay. When
# the fixation cue turns off, respond to the remembered direction.
# --------------------------------------------------------------------------
DELAYGO_DEFAULTS = {
    "fix_ms": [200, 600],
    "stim_ms": [200, 600],
    "delay_ms": [200, 1600],  # nothing on screen; the direction must be held in memory
    "response_ms": 400,
    "strength": [0.8, 1.2],
}


def delaygo(layout: Layout, B: int, rng: np.random.Generator, dt_ms: float, params: dict) -> TrialBatch:
    p = {**DELAYGO_DEFAULTS, **params}
    epochs = {
        "fixation": sample_ms(rng, p["fix_ms"]),
        "stimulus": sample_ms(rng, p["stim_ms"]),
        "delay": sample_ms(rng, p["delay_ms"]),
        "response": sample_ms(rng, p["response_ms"]),
    }
    tb = TrialBuilder(layout, "delaygo", B, epochs, dt_ms)
    theta, modality = _directions_and_modality(rng, B)
    strength = rng.uniform(*p["strength"], size=B)

    tb.fixation_input(until="response")
    tb.ring_input(modality, theta, strength, tb.ep("stimulus"))
    tb.standard_targets(target_dir=theta)
    return tb.build(theta, {"stim_dir": theta, "modality": modality, "strength": strength})


# --------------------------------------------------------------------------
# Decision making. Two stimuli at different directions, one slightly stronger.
# Respond toward the stronger one. "Coherence" is how different they are:
#     strength1 = mean + coh,   strength2 = mean - coh
# so coh = 0 is a coin flip and large coh is easy. This gives the
# psychometric curve (probability of choosing stimulus 1 vs coh).
#
# dm1 / dm2:       both stimuli on modality 1 (or 2).
# ctxdm1 / ctxdm2: stimuli on BOTH modalities with independent coherences;
#                  only one modality counts (context-dependent decision).
# --------------------------------------------------------------------------
DM_DEFAULTS = {
    "fix_ms": [200, 600],
    "stim_ms": [400, 1000],  # stimuli on, still fixating (evidence accumulation)
    "response_ms": 400,  # stimuli stay on during the response
    "mean_strength": [0.8, 1.2],
    # Coherence magnitudes used in training; the sign (which stimulus is
    # stronger) is random. Harder values (0.02, 0.0) are used only when
    # measuring psychometric curves: with them in training, the tasks could
    # not reach the training criterion (README, "Phase 2 notes").
    "coherences": [0.04, 0.08, 0.16, 0.32],
}


def _dm(task: str, layout, B, rng, dt_ms, params, modalities: tuple[int, ...], attend: int):
    p = {**DM_DEFAULTS, **params}
    epochs = {
        "fixation": sample_ms(rng, p["fix_ms"]),
        "stimulus": sample_ms(rng, p["stim_ms"]),
        "response": sample_ms(rng, p["response_ms"]),
    }
    tb = TrialBuilder(layout, task, B, epochs, dt_ms)
    theta1, theta2 = _two_directions(rng, B)
    mean = rng.uniform(*p["mean_strength"], size=B)
    cohs = np.asarray(p["coherences"], dtype=float)
    on = tb.span("stimulus", "response")

    coh_by_mod = {}
    for m in modalities:
        # A config may pass "coh" explicitly (used for psychometric curves).
        coh = np.asarray(params["coh"], dtype=float) * np.ones(B) if "coh" in params else rng.choice(cohs, size=B) * rng.choice([-1, 1], size=B)
        coh_by_mod[m] = coh
        mod = np.full(B, m)
        tb.ring_input(mod, theta1, mean + coh, on)
        tb.ring_input(mod, theta2, mean - coh, on)

    coh = coh_by_mod[attend]
    # coh == 0: there is no correct answer; pick a target at random. (coh = 0
    # only appears in psychometric tests, which look at the raw choice.)
    pick1 = np.where(coh == 0, rng.random(B) < 0.5, coh > 0)
    target = np.where(pick1, theta1, theta2)
    tb.fixation_input(until="response")
    tb.standard_targets(target_dir=target)
    meta = {"theta1": theta1, "theta2": theta2, "coh": coh, "mean_strength": mean,
            "coh_other": coh_by_mod.get(3 - attend), "attend": attend}
    return tb.build(target, meta)


def dm1(layout, B, rng, dt_ms, params):
    return _dm("dm1", layout, B, rng, dt_ms, params, modalities=(1,), attend=1)


def dm2(layout, B, rng, dt_ms, params):
    return _dm("dm2", layout, B, rng, dt_ms, params, modalities=(2,), attend=2)


def ctxdm1(layout, B, rng, dt_ms, params):
    return _dm("ctxdm1", layout, B, rng, dt_ms, params, modalities=(1, 2), attend=1)


def ctxdm2(layout, B, rng, dt_ms, params):
    return _dm("ctxdm2", layout, B, rng, dt_ms, params, modalities=(1, 2), attend=2)


# --------------------------------------------------------------------------
# Delayed match-to-sample (dms) / non-match (dnms).
# Sample stimulus, delay, test stimulus (same modality). Half the trials match.
#   dms:  match -> respond toward the test stimulus; non-match -> keep fixating.
#   dnms: non-match -> respond toward the test stimulus; match -> keep fixating.
# --------------------------------------------------------------------------
DMS_DEFAULTS = {
    "fix_ms": [200, 600],
    "sample_ms": 300,
    "delay_ms": [400, 1200],
    "test_ms": 300,  # test on, still fixating
    "response_ms": 400,  # test stays on during the response
    "strength": [0.8, 1.2],
    "p_match": 0.5,
    # A non-matching test is at least this far from the sample (radians).
    "min_nonmatch_distance": np.pi / 3,
}


def _matching(task: str, respond_on_match: bool, layout, B, rng, dt_ms, params):
    p = {**DMS_DEFAULTS, **params}
    epochs = {
        "fixation": sample_ms(rng, p["fix_ms"]),
        "sample": sample_ms(rng, p["sample_ms"]),
        "delay": sample_ms(rng, p["delay_ms"]),
        "test": sample_ms(rng, p["test_ms"]),
        "response": sample_ms(rng, p["response_ms"]),
    }
    tb = TrialBuilder(layout, task, B, epochs, dt_ms)
    sample, modality = _directions_and_modality(rng, B)
    match = rng.random(B) < p["p_match"]
    d = p["min_nonmatch_distance"]
    offset = rng.uniform(d, 2 * np.pi - d, size=B)
    test = np.where(match, sample, np.mod(sample + offset, 2 * np.pi))
    strength = rng.uniform(*p["strength"], size=B)

    tb.fixation_input(until="response")
    tb.ring_input(modality, sample, strength, tb.ep("sample"))
    tb.ring_input(modality, test, strength, tb.span("test", "response"))
    respond = match if respond_on_match else ~match
    target = np.where(respond, test, np.nan)
    tb.standard_targets(target_dir=target)
    return tb.build(target, {"sample_dir": sample, "test_dir": test, "match": match, "modality": modality})


def dms(layout, B, rng, dt_ms, params):
    return _matching("dms", True, layout, B, rng, dt_ms, params)


def dnms(layout, B, rng, dt_ms, params):
    return _matching("dnms", False, layout, B, rng, dt_ms, params)
