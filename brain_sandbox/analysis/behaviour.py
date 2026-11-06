"""Behaviour of a (possibly perturbed) network: accuracy, error types, psychometric curves."""

from __future__ import annotations

import numpy as np
import torch

from ..metrics import OUTCOMES, trial_outcomes
from ..model import to_torch
from ..tasks import TASKS
from ..tasks.common import circular_distance

TEST_SEED = 2024  # test trials: different from training and from in-training evaluation


@torch.no_grad()
def evaluate_detailed(model, tasks, n_trials: int = 512, seed: int = TEST_SEED, batch_size: int = 16,
                      task_names=None) -> dict[str, dict[str, float]]:
    """Per task: accuracy plus the rate of each error type.

    The trials (and the network's noise) depend only on `seed`, so an intact
    and a perturbed network are tested on exactly the same trials. That makes
    their comparison *paired*, which removes a lot of chance variation.
    """
    out = {}
    with torch.random.fork_rng():
        for name in task_names or tasks.names:
            rng = np.random.default_rng([seed, tasks.names.index(name)])
            torch.manual_seed(seed + 1000 * tasks.names.index(name))
            counts = dict.fromkeys(OUTCOMES, 0)
            done = 0
            while done < n_trials:
                b = min(batch_size, n_trials - done)
                batch = tasks.generate(name, b, rng)
                y, _ = model(to_torch(batch.x))
                o = trial_outcomes(y.numpy(), batch, tasks.layout, tasks.dt_ms)
                for k in OUTCOMES:
                    counts[k] += int((o == k).sum())
                done += b
            rates = {k: v / n_trials for k, v in counts.items()}
            out[name] = {"accuracy": rates["correct"], **rates}
    return out


@torch.no_grad()
def psychometric(model, tasks, task: str = "dm1", cohs=None, n_per_coh: int = 128, seed: int = TEST_SEED):
    """Probability of choosing stimulus 1, as a function of signed coherence.

    Only trials where the network answered (dropped fixation in time, and
    pointed near one of the two stimuli) count toward the probability; the
    fraction that did is returned too.
    """
    if cohs is None:
        cohs = np.array([-0.16, -0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08, 0.16])
    L = tasks.layout
    p_choose1, p_valid = [], []
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for i, c in enumerate(cohs):
            rng = np.random.default_rng([seed, i])
            chose1, valid = 0, 0
            for _ in range(n_per_coh // 16):
                batch = TASKS[task](L, 16, rng, tasks.dt_ms, {**tasks.cfg.params.get(task, {}), "coh": float(c)})
                y, _ = model(to_torch(batch.x))
                y = y.numpy()
                answered = y[-1, :, L.out_slices["fixation"]][:, 0] < 0.5
                ang = L.decode_ring(y[-1, :, L.out_slices["ring"]])
                d1 = circular_distance(ang, batch.meta["theta1"])
                d2 = circular_distance(ang, batch.meta["theta2"])
                ok = answered & (np.minimum(d1, d2) < np.pi / 4)
                chose1 += int((ok & (d1 < d2)).sum())
                valid += int(ok.sum())
            p_choose1.append(chose1 / valid if valid else np.nan)
            p_valid.append(valid / (16 * (n_per_coh // 16)))
    return np.asarray(cohs), np.asarray(p_choose1), np.asarray(p_valid)
