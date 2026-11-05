"""Cognitive tasks reimplemented from Yang et al. 2019 (github.com/gyyang/multitask).

Phase 1 implements Go and Delay-Go. Each task function has the signature

    task(layout, batch_size, rng, dt_ms, params) -> TrialBatch

and its default parameters are in the DEFAULTS dict right above it, so you can
see every number the task uses. Durations are in ms; a [low, high] pair means
"sample uniformly per batch". A config can override any of them.
"""

from __future__ import annotations

import numpy as np

from .common import Layout, TrialBatch, TrialBuilder, sample_ms


def _directions_and_modality(rng, B):
    theta = rng.uniform(0, 2 * np.pi, size=B)
    modality = rng.integers(1, 3, size=B)  # stimulus on modality 1 or 2, at random
    return theta, modality


# --------------------------------------------------------------------------
# Go  (Yang's "fdgo"): a stimulus appears and stays on; when the fixation cue
# turns off, saccade (respond) to the stimulus direction.
# --------------------------------------------------------------------------
GO_DEFAULTS = {
    "fix_ms": [200, 600],  # fixation alone
    "stim_ms": [200, 1000],  # stimulus on, still fixating
    "response_ms": 400,  # fixation off: respond (stimulus still on)
    "strength": [0.8, 1.2],  # stimulus amplitude range
}


def go(layout: Layout, B: int, rng: np.random.Generator, dt_ms: float, params: dict) -> TrialBatch:
    p = {**GO_DEFAULTS, **params}
    epochs = {
        "fixation": sample_ms(rng, p["fix_ms"]),
        "stimulus": sample_ms(rng, p["stim_ms"]),
        "response": sample_ms(rng, p["response_ms"]),
    }
    tb = TrialBuilder(layout, "go", B, epochs, dt_ms)
    theta, modality = _directions_and_modality(rng, B)
    strength = rng.uniform(*p["strength"], size=B)

    tb.fixation_input(until="response")
    tb.ring_input(modality, theta, strength, tb.span("stimulus", "response"))
    tb.standard_targets(target_dir=theta)
    return tb.build(theta, {"stim_dir": theta, "modality": modality, "strength": strength})


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
