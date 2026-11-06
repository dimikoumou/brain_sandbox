"""Pitch tasks on the tonotopic input.

relpitch (trained)
------------------
Hear melody A (3 tones), a pause, then melody B (3 tones). Answer with the two
choice outputs:

    choice 0 = "same tune": B is A *transposed* (every note shifted by the
               same amount, so the intervals between notes are identical)
    choice 1 = "different tune": one note of B was changed

B is ALWAYS shifted to a different key (shift != 0), in both kinds of trial,
so the absolute pitch of the notes never tells you the answer. Only the
intervals matter. This is what should give the network a reason to discard
absolute pitch -- whether it actually does is measured, not assumed.

Absolute-pitch probe (NOT trained)
----------------------------------
`probe_batch` makes single-tone trials (one note, then silence) that are fed
to an already-trained, frozen network. Phase 4 then asks whether a simple
linear readout can tell which of the 12 notes (C, C#, ... B) was played from
each region's activity *after the tone has ended*.
"""

from __future__ import annotations

import numpy as np

from .common import Layout, TrialBatch, TrialBuilder, sample_ms

RELPITCH_DEFAULTS = {
    "n_tones": 3,
    "fix_ms": [200, 400],
    "tone_ms": 150,
    "gap_ms": 50,  # silence after each tone
    "delay_ms": [300, 700],  # pause between the two melodies
    "response_ms": 400,
    "max_interval": 4,  # consecutive notes differ by 1..4 semitones, up or down
    "max_shift": 7,  # melody B is transposed by 1..7 semitones, up or down
    "change_sizes": [1, 2],  # a "different" melody has one note moved by this many semitones
    "p_same": 0.5,
}


def _random_melody(rng, n_tones, max_interval, n_semitones):
    """A melody whose notes stay inside 0..n_semitones."""
    while True:
        steps = rng.integers(1, max_interval + 1, size=n_tones - 1) * rng.choice([-1, 1], size=n_tones - 1)
        shape = np.concatenate([[0], np.cumsum(steps)])
        lo, hi = -shape.min(), n_semitones - shape.max()
        if lo <= hi:
            return shape + rng.integers(lo, hi + 1)


def make_melody_pair(rng, p, n_semitones):
    """Return (melody_a, melody_b, same: bool, shift)."""
    a = _random_melody(rng, p["n_tones"], p["max_interval"], n_semitones)
    shifts = [k for k in range(-p["max_shift"], p["max_shift"] + 1)
              if k != 0 and a.min() + k >= 0 and a.max() + k <= n_semitones]
    shift = int(rng.choice(shifts))
    b = a + shift
    same = rng.random() < p["p_same"]
    if not same:
        # Move one note up or down; keep it in range. Any change to one note
        # changes at least one interval, so B is no longer a transposition of A.
        while True:
            i = rng.integers(p["n_tones"])
            delta = int(rng.choice(p["change_sizes"])) * int(rng.choice([-1, 1]))
            if 0 <= b[i] + delta <= n_semitones:
                b = b.copy()
                b[i] += delta
                break
    return a, b, same, shift


def relpitch(layout: Layout, B: int, rng: np.random.Generator, dt_ms: float, params: dict) -> TrialBatch:
    p = {**RELPITCH_DEFAULTS, **params}
    n = p["n_tones"]
    note_ms = p["tone_ms"] + p["gap_ms"]
    epochs = {
        "fixation": sample_ms(rng, p["fix_ms"]),
        "melody1": n * note_ms,
        "delay": sample_ms(rng, p["delay_ms"]),
        "melody2": n * note_ms,
        "response": sample_ms(rng, p["response_ms"]),
    }
    tb = TrialBuilder(layout, "relpitch", B, epochs, dt_ms)

    pairs = [make_melody_pair(rng, p, layout.n_semitones) for _ in range(B)]
    mel_a = np.stack([q[0] for q in pairs])  # (B, n)
    mel_b = np.stack([q[1] for q in pairs])
    same = np.array([q[2] for q in pairs])
    shift = np.array([q[3] for q in pairs])

    tone_steps = int(round(p["tone_ms"] / dt_ms))
    note_steps = int(round(note_ms / dt_ms))
    for epoch, mel in (("melody1", mel_a), ("melody2", mel_b)):
        start = tb.epochs[epoch][0]
        for k in range(n):
            t0 = start + k * note_steps
            tb.tone_input(mel[:, k], slice(t0, t0 + tone_steps))

    tb.fixation_input(until="response")
    choice = np.where(same, 0, 1)
    tb.standard_targets(target_choice=choice)
    return tb.build(None, {"melody_a": mel_a, "melody_b": mel_b, "same": same, "shift": shift}, target_choice=choice)


# --------------------------------------------------------------------------
# Probe trials for latent absolute pitch (never used for training)
# --------------------------------------------------------------------------
PROBE_DEFAULTS = {
    "fix_ms": 300,
    "tone_ms": 150,
    "post_ms": 400,  # silence after the tone; the decoding window lies inside this
}


def probe_batch(layout: Layout, notes: np.ndarray, dt_ms: float, rule_task: str = "relpitch",
                params: dict | None = None) -> TrialBatch:
    """One trial per entry of `notes`: fixation, a single tone, then silence.

    The rule input is set to `rule_task` (by default the relative-pitch task,
    i.e. the network is "in pitch mode"). There is no response and no target:
    these trials are only used to record activity.
    """
    p = {**PROBE_DEFAULTS, **(params or {})}
    epochs = {"fixation": p["fix_ms"], "tone": p["tone_ms"], "post": p["post_ms"]}
    tb = TrialBuilder(layout, "pitch_probe", len(notes), epochs, dt_ms, rule_task=rule_task)
    tb.x[:, :, layout.in_slices["fixation"]] = 1.0  # keep fixating throughout
    tb.tone_input(np.asarray(notes), tb.ep("tone"))
    return tb.build(None, {"note": np.asarray(notes), "pitch_class": layout.pitch_class(notes)})
