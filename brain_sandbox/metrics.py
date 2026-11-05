"""Scoring network outputs.

A trial counts as correct only if BOTH hold:

1. **Fixation held**: the fixation output stayed above 0.5 for the whole
   pre-response period (ignoring the first 100 ms, while the network settles
   from its zero initial state).
2. **Correct response** at the last time step:
   * if the task asks for a direction: fixation output < 0.5 and the decoded
     direction is within 0.2*pi (36 degrees) of the target;
   * if the task asks to keep fixating: fixation output still > 0.5.

Difference from Yang et al. 2019: their score only checks the last time step,
so a network that responds too early still counts as correct. We also require
fixation to be held, because "responds impulsively" is exactly the kind of
failure a lesion might cause and we don't want to hide it.
"""

from __future__ import annotations

import numpy as np

from .tasks.common import Layout, TrialBatch, circular_distance

ANGLE_TOLERANCE = 0.2 * np.pi
SETTLE_MS = 100.0


def trial_correct(out: np.ndarray, batch: TrialBatch, layout: Layout, dt_ms: float) -> np.ndarray:
    """Boolean (B,) array: was each trial performed correctly?"""
    fix = out[:, :, layout.out_slices["fixation"]][..., 0]  # (T, B)
    ring = out[:, :, layout.out_slices["ring"]]  # (T, B, n_ring)
    r0 = batch.response_start
    settle = int(round(SETTLE_MS / dt_ms))

    held = (fix[settle:r0] > 0.5).all(axis=0)

    should_respond = ~np.isnan(batch.target_dir)
    final_fix = fix[-1]
    decoded = layout.decode_ring(ring[-1])
    err = circular_distance(decoded, np.nan_to_num(batch.target_dir))
    responded_correctly = (final_fix < 0.5) & (err < ANGLE_TOLERANCE)
    kept_fixating = final_fix > 0.5

    return held & np.where(should_respond, responded_correctly, kept_fixating)


def accuracy(out: np.ndarray, batch: TrialBatch, layout: Layout, dt_ms: float) -> float:
    return float(trial_correct(out, batch, layout, dt_ms).mean())
