"""Latent absolute pitch: can a simple readout tell which note was played?

Procedure (the network is frozen; nothing here trains it):

1. Play single tones -- every note of the two-octave range, many times each --
   with the network "in pitch mode" (the relative-pitch rule input on).
   The network's own noise makes every repetition a little different.
2. Record each region's activity in a window AFTER the tone has ended
   (default 100-300 ms after offset). The input is silent then, so whatever
   pitch information remains is information the network *kept*.
3. Train a logistic-regression readout to name the note's pitch class
   (C, C#, ..., B: 12 classes, chance = 1/12 = 8.3%), and test it on held-out
   trials (5-fold cross-validation).

Controls, reported next to every number:

* **shuffled labels**: same activity, notes randomly reassigned. This is what
  "no information" scores (about chance, but measured, not assumed).
* **input during the tone**: decoding from the raw tonotopic input while the
  tone plays (with the same input noise the network receives). The ceiling.
* **input after the tone**: the raw input in the post-tone window is silent,
  so this is chance by construction -- it shows the post-tone window really
  contains no stimulus.
* **untrained network**: (computed by the caller) the same architecture before
  training. Activity naturally fades after a tone (time constant 100 ms), so
  some pitch information lingers even in an untrained network. Comparing to it
  tells us whether *training* removed or added pitch information.

Few-shot version: train the readout on only N examples per note (N = 1, 2, 5,
10) and test on the rest -- a rough analogue of "how easily could it learn to
name notes".
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..model import to_torch
from ..tasks.pitch import probe_batch

CHANCE = 1 / 12


@torch.no_grad()
def probe_features(model, layout, dt_ms: float, n_reps: int = 24, window_ms=(100, 300), seed: int = 0):
    """Record activity for single-tone probe trials.

    Returns dict with:
      labels         (n_trials,) pitch class
      post[module]   (n_trials, units) mean rate in the post-tone window
      during[module] (n_trials, units) mean rate while the tone plays
      input_post, input_during  (n_trials, n_tono) raw tonotopic input (+ input noise)
    """
    notes = np.tile(np.arange(layout.n_semitones + 1), n_reps)
    batch = probe_batch(layout, notes, dt_ms)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        _, rates = model(to_torch(batch.x))
    rates = rates.numpy()
    t0, t1 = batch.epochs["tone"]
    w0 = t1 + int(round(window_ms[0] / dt_ms))
    w1 = t1 + int(round(window_ms[1] / dt_ms))
    assert w1 <= rates.shape[0], "post-tone window is longer than the probe trial"

    rng = np.random.default_rng(seed)
    tono = batch.x[:, :, layout.in_slices["tono"]]
    noisy = tono + model.input_noise * rng.standard_normal(tono.shape)
    out = {
        "labels": batch.meta["pitch_class"],
        "post": {}, "during": {},
        "input_post": noisy[w0:w1].mean(0),
        "input_during": noisy[t0:t1].mean(0),
    }
    for name, sl in model.module_slices.items():
        out["post"][name] = rates[w0:w1, :, sl].mean(0)
        out["during"][name] = rates[t0:t1, :, sl].mean(0)
    return out


def _readout():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def decode_accuracy(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated accuracy of a linear readout."""
    if np.allclose(X, X[0]):  # e.g. a fully lesioned region: no information at all
        return float(np.mean(y == np.bincount(y).argmax()))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(cross_val_score(_readout(), X, y, cv=cv).mean())


def shuffled_accuracy(X, y, seed: int = 0) -> float:
    return decode_accuracy(X, np.random.default_rng(seed).permutation(y), seed=seed)


def few_shot_accuracy(X, y, n_per_class: int, repeats: int = 20, seed: int = 0) -> float:
    """Train on n examples per pitch class, test on all the rest; average over repeats."""
    if np.allclose(X, X[0]):
        return CHANCE
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    scores = []
    for _ in range(repeats):
        train = np.concatenate([rng.choice(np.where(y == c)[0], n_per_class, replace=False) for c in classes])
        test = np.setdiff1d(np.arange(len(y)), train)
        clf = _readout().fit(X[train], y[train])
        scores.append(clf.score(X[test], y[test]))
    return float(np.mean(scores))


def pitch_report(model, layout, dt_ms: float, n_reps: int = 24, window_ms=(100, 300), seed: int = 0,
                 few_shot=(1, 2, 5, 10), include_controls: bool = True) -> dict:
    """All pitch-decoding numbers for one network."""
    f = probe_features(model, layout, dt_ms, n_reps, window_ms, seed)
    y = f["labels"]
    rep = {"modules": {}, "chance": CHANCE}
    for name in model.module_names:
        X = f["post"][name]
        entry = {"post": decode_accuracy(X, y, seed=seed)}
        if include_controls:
            entry["post_shuffled"] = shuffled_accuracy(X, y, seed=seed)
            entry["during"] = decode_accuracy(f["during"][name], y, seed=seed)
        entry["few_shot"] = {n: few_shot_accuracy(X, y, n, seed=seed) for n in few_shot}
        rep["modules"][name] = entry
    if include_controls:
        rep["input_during"] = decode_accuracy(f["input_during"], y, seed=seed)
        rep["input_post"] = decode_accuracy(f["input_post"], y, seed=seed)
    return rep
