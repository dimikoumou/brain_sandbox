"""Shared machinery for all tasks: input/output layout, population codes, trial batches.

Conventions
-----------
* Arrays are **time-major**: shape (T, B, channels).
* All trials in one batch share the same timing (as in Yang et al. 2019).
  Timing still varies from batch to batch, and evaluation uses many batches.
* Directions live on a ring [0, 2*pi). A direction is presented to the network
  as a bump of activity over `n_ring` units with evenly spaced preferred
  directions (a population code, like direction-tuned neurons).
* Tones are presented on a *tonotopic* map: `n_tono` units whose preferred
  pitches are evenly spaced on a log-frequency (semitone) axis, like the
  cochlea / auditory cortex. A note is a bump on that axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import TaskConfig

# Output target levels, following Yang et al. 2019.
FIX_ON_TARGET = 0.85  # fixation output while the network should hold fixation
FIX_OFF_TARGET = 0.05  # fixation output during the response
RING_BASELINE = 0.05  # ring / choice outputs when nothing is being reported
RING_PEAK = 0.8  # added on top of the baseline at the reported direction / choice

# Loss weights (Yang et al. 2019): errors during the response period count 5x,
# and the fixation output counts 2x, because these are what define the task.
W_PRE_RESPONSE = 1.0
W_RESPONSE = 5.0
W_FIXATION_OUTPUT = 2.0
GRACE_MS = 100.0  # no loss right after the go cue: the network may take time to switch

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Layout:
    """Where each input/output group lives in the channel dimension.

    Inputs:  [fixation(1) | mod1(n_ring) | mod2(n_ring) | tono(n_tono) | rule(n_tasks)]
    Outputs: [fixation(1) | ring(n_ring) | choice(n_choice)]

    Keeping one fixed layout for every task means the same network can be
    trained on all of them, and only the one-hot rule input tells them apart.
    """

    def __init__(self, cfg: TaskConfig):
        self.n_ring = cfg.n_ring
        self.n_tono = cfg.n_tono
        self.n_choice = cfg.n_choice
        self.task_names = list(cfg.names)
        self.tuning_width = cfg.tuning_width or (2 * np.pi / cfg.n_ring)
        self.pref_dirs = 2 * np.pi * np.arange(cfg.n_ring) / cfg.n_ring

        # Tonotopic axis, in semitones above the lowest note (note 0).
        # Notes are integers 0..n_semitones. The preferred pitches extend one
        # semitone beyond both ends so the edge notes are coded like the others.
        self.n_semitones = cfg.n_semitones
        self.f_low_hz = cfg.f_low_hz
        self.tono_width = cfg.tono_width
        self.tono_pref = np.linspace(-1.0, cfg.n_semitones + 1.0, cfg.n_tono) if cfg.n_tono else np.zeros(0)

        self.in_groups = {
            "fixation": 1,
            "mod1": cfg.n_ring,
            "mod2": cfg.n_ring,
            "tono": cfg.n_tono,
            "rule": len(self.task_names),
        }
        self.out_groups = {"fixation": 1, "ring": cfg.n_ring, "choice": cfg.n_choice}
        self.in_slices = _slices(self.in_groups)
        self.out_slices = _slices(self.out_groups)
        self.n_in = sum(self.in_groups.values())
        self.n_out = sum(self.out_groups.values())

    def rule_index(self, task: str) -> int:
        return self.task_names.index(task)

    # --- directions ---------------------------------------------------------
    def ring_bump(self, theta: np.ndarray) -> np.ndarray:
        """Gaussian bump over the ring units for each direction in `theta` (B,) -> (B, n_ring)."""
        d = circular_distance(theta[:, None], self.pref_dirs[None, :])
        return np.exp(-0.5 * (d / self.tuning_width) ** 2)

    def decode_ring(self, ring_activity: np.ndarray) -> np.ndarray:
        """Population-vector readout: (..., n_ring) -> angle in [0, 2*pi)."""
        a = ring_activity - ring_activity.min(axis=-1, keepdims=True)
        x = (a * np.cos(self.pref_dirs)).sum(-1)
        y = (a * np.sin(self.pref_dirs)).sum(-1)
        return np.mod(np.arctan2(y, x), 2 * np.pi)

    # --- pitch --------------------------------------------------------------
    def tone_bump(self, note: np.ndarray) -> np.ndarray:
        """Gaussian bump on the tonotopic axis for each note (B,) -> (B, n_tono)."""
        d = note[:, None] - self.tono_pref[None, :]
        return np.exp(-0.5 * (d / self.tono_width) ** 2)

    def note_frequency(self, note: np.ndarray) -> np.ndarray:
        """Equal temperament: each semitone multiplies frequency by 2^(1/12)."""
        return self.f_low_hz * 2.0 ** (np.asarray(note) / 12.0)

    @staticmethod
    def pitch_class(note: np.ndarray) -> np.ndarray:
        """0..11 = C..B, assuming note 0 is a C (the default f_low_hz is C4)."""
        return np.mod(np.asarray(note), 12).astype(int)


def _slices(groups: dict[str, int]) -> dict[str, slice]:
    out, start = {}, 0
    for name, size in groups.items():
        out[name] = slice(start, start + size)
        start += size
    return out


def circular_distance(a, b):
    """Smallest absolute angle between a and b (radians), in [0, pi]."""
    d = np.abs(np.mod(a - b, 2 * np.pi))
    return np.minimum(d, 2 * np.pi - d)


@dataclass
class TrialBatch:
    """One batch of trials of a single task, ready to feed to the network."""

    task: str
    x: np.ndarray  # (T, B, n_in) inputs
    y: np.ndarray  # (T, B, n_out) target outputs
    loss_mask: np.ndarray  # (T, B, n_out) per-element loss weights
    # Epoch boundaries in time steps: name -> (start, end), end exclusive.
    epochs: dict[str, tuple[int, int]]
    # Ground-truth direction to report (B,); NaN means "no direction to report".
    target_dir: np.ndarray
    meta: dict = field(default_factory=dict)  # task-specific details (stimuli, modality...)
    # Ground-truth choice (B,) for tasks answered with the choice outputs;
    # -1 means "not a choice trial".
    target_choice: np.ndarray | None = None

    @property
    def response_start(self) -> int:
        return self.epochs["response"][0]

    @property
    def should_respond(self) -> np.ndarray:
        """True where the correct behaviour is to break fixation and answer."""
        resp = ~np.isnan(self.target_dir)
        if self.target_choice is not None:
            resp |= self.target_choice >= 0
        return resp


class TrialBuilder:
    """Helper that task functions use to fill in a TrialBatch step by step."""

    def __init__(self, layout: Layout, task: str, batch_size: int, epoch_ms: dict[str, float], dt_ms: float,
                 rule_task: str | None = None):
        self.layout = layout
        self.task = task
        self.B = batch_size
        self.dt = dt_ms
        # Convert epoch durations (ms, in order) into time-step boundaries.
        self.epochs: dict[str, tuple[int, int]] = {}
        t = 0
        for name, dur in epoch_ms.items():
            n = max(1, int(round(dur / dt_ms)))
            self.epochs[name] = (t, t + n)
            t += n
        self.T = t
        L = layout
        self.x = np.zeros((self.T, self.B, L.n_in), dtype=np.float32)
        self.y = np.zeros((self.T, self.B, L.n_out), dtype=np.float32)
        self.mask = np.zeros((self.T, self.B, L.n_out), dtype=np.float32)
        # The rule input is on for the whole trial. (A probe trial may borrow
        # the rule of a trained task via `rule_task`.)
        self.x[:, :, L.in_slices["rule"].start + L.rule_index(rule_task or task)] = 1.0

    def ep(self, name: str) -> slice:
        s, e = self.epochs[name]
        return slice(s, e)

    def span(self, first: str, last: str) -> slice:
        """Time steps from the start of epoch `first` to the end of epoch `last`."""
        return slice(self.epochs[first][0], self.epochs[last][1])

    def fixation_input(self, until: str) -> None:
        """Fixation cue on from trial start until the start of epoch `until` (the go cue)."""
        L = self.layout
        self.x[: self.epochs[until][0], :, L.in_slices["fixation"]] = 1.0

    def ring_input(self, modality: np.ndarray, theta: np.ndarray, strength: np.ndarray, when: slice) -> None:
        """Present direction `theta` on modality 1 or 2 (per trial) during `when`."""
        L = self.layout
        bump = L.ring_bump(theta) * strength[:, None]  # (B, n_ring)
        for m in (1, 2):
            sel = modality == m
            sl = L.in_slices[f"mod{m}"]
            self.x[when, sel, sl] += bump[sel][None]

    def tone_input(self, note: np.ndarray, when: slice, strength: float = 1.0) -> None:
        """Play `note` (B,) on the tonotopic input during `when`."""
        L = self.layout
        self.x[when, :, L.in_slices["tono"]] += strength * L.tone_bump(note)[None]

    def standard_targets(self, target_dir: np.ndarray | None = None, target_choice: np.ndarray | None = None,
                         response: str = "response") -> None:
        """Fixate until the response epoch, then answer.

        * `target_dir` (B,): report this direction on the ring outputs (NaN -> keep fixating).
        * `target_choice` (B,): pick choice output 0 or 1 (-1 -> keep fixating).

        Also fills in the loss mask: pre-response errors weight 1, response
        errors weight 5 (after a short grace period), fixation output 2x.
        """
        L = self.layout
        r0 = self.epochs[response][0]
        fix = L.out_slices["fixation"]
        ring = L.out_slices["ring"]
        choice = L.out_slices["choice"]
        respond = np.zeros(self.B, dtype=bool)

        self.y[:, :, ring] = RING_BASELINE
        self.y[:, :, choice] = RING_BASELINE
        if target_dir is not None:
            has_dir = ~np.isnan(target_dir)
            respond |= has_dir
            if has_dir.any():
                bump = L.ring_bump(target_dir[has_dir])
                self.y[r0:, has_dir, ring] = RING_BASELINE + RING_PEAK * bump[None]
        if target_choice is not None:
            for c in range(L.n_choice):
                sel = target_choice == c
                respond |= sel
                self.y[r0:, sel, choice.start + c] = RING_BASELINE + RING_PEAK

        self.y[:r0, :, fix] = FIX_ON_TARGET
        self.y[r0:, :, fix] = np.where(respond, FIX_OFF_TARGET, FIX_ON_TARGET)[None, :, None]

        grace = int(round(GRACE_MS / self.dt))
        self.mask[:r0] = W_PRE_RESPONSE
        self.mask[r0 + grace :] = W_RESPONSE
        self.mask[:, :, fix] *= W_FIXATION_OUTPUT

    def build(self, target_dir: np.ndarray | None, meta: dict, target_choice: np.ndarray | None = None) -> TrialBatch:
        if target_dir is None:
            target_dir = np.full(self.B, np.nan)
        return TrialBatch(self.task, self.x, self.y, self.mask, dict(self.epochs), target_dir, meta, target_choice)


def sample_ms(rng: np.random.Generator, spec) -> float:
    """A duration in ms: either a fixed number or a [low, high] uniform range."""
    if isinstance(spec, (list, tuple)):
        return float(rng.uniform(spec[0], spec[1]))
    return float(spec)
