"""Scoring network outputs.

Every trial gets exactly one of four outcomes:

* ``correct``
* ``broke_fixation`` -- answered too early: the fixation output dropped below
  0.5 before the go cue (ignoring the first 100 ms, while the network settles
  from its zero initial state). An "impulsive" error.
* ``no_response``    -- should have answered but kept fixating to the end.
* ``wrong``          -- answered, but with the wrong answer (direction more
  than 0.2*pi = 36 degrees off, or the wrong choice), or answered on a trial
  where the right thing was to keep fixating.

A trial is correct when fixation was held until the go cue AND the final time
step shows the right behaviour. Difference from Yang et al. 2019: their score
only checks the last time step, so a network that responds too early still
counts as correct. We count early responses as errors, and report them
separately, because "responds impulsively" is exactly the kind of failure a
lesion might cause and we don't want to hide it -- or let it masquerade as a
more specific deficit.
"""

from __future__ import annotations

import numpy as np

from .tasks.common import Layout, TrialBatch, circular_distance

ANGLE_TOLERANCE = 0.2 * np.pi
SETTLE_MS = 100.0
OUTCOMES = ("correct", "broke_fixation", "no_response", "wrong")


def trial_outcomes(out: np.ndarray, batch: TrialBatch, layout: Layout, dt_ms: float) -> np.ndarray:
    """Array (B,) of outcome strings (see OUTCOMES)."""
    fix = out[:, :, layout.out_slices["fixation"]][..., 0]  # (T, B)
    r0 = batch.response_start
    settle = int(round(SETTLE_MS / dt_ms))
    B = fix.shape[1]

    held = (fix[settle:r0] > 0.5).all(axis=0)
    answered = fix[-1] < 0.5
    respond = batch.should_respond

    right_answer = np.zeros(B, dtype=bool)
    has_dir = ~np.isnan(batch.target_dir)
    if has_dir.any():
        decoded = layout.decode_ring(out[-1, :, layout.out_slices["ring"]])
        err = circular_distance(decoded, np.nan_to_num(batch.target_dir))
        right_answer |= has_dir & (err < ANGLE_TOLERANCE)
    if batch.target_choice is not None:
        chosen = out[-1, :, layout.out_slices["choice"]].argmax(-1)
        right_answer |= (batch.target_choice >= 0) & (chosen == batch.target_choice)

    outcome = np.full(B, "wrong", dtype=object)
    ok = np.where(respond, answered & right_answer, ~answered)
    outcome[ok] = "correct"
    outcome[respond & ~answered] = "no_response"
    outcome[~held] = "broke_fixation"  # takes precedence: it happened first
    return outcome


def trial_correct(out: np.ndarray, batch: TrialBatch, layout: Layout, dt_ms: float) -> np.ndarray:
    """Boolean (B,) array: was each trial performed correctly?"""
    return trial_outcomes(out, batch, layout, dt_ms) == "correct"


def accuracy(out: np.ndarray, batch: TrialBatch, layout: Layout, dt_ms: float) -> float:
    return float(trial_correct(out, batch, layout, dt_ms).mean())


def outcome_rates(outcomes: np.ndarray) -> dict[str, float]:
    return {k: float(np.mean(outcomes == k)) for k in OUTCOMES}
