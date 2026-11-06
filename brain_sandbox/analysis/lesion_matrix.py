"""Lesion matrix (tasks x perturbations) and automatic dissociation detection.

Input: "records", one per (seed, condition, task), with accuracy and error
rates, where the condition labelled `baseline` is the intact network.

Effect of a perturbation = perturbed accuracy - intact accuracy, computed per
seed on the SAME test trials (paired), then averaged over seeds.

A perturbation **hurts** (or **helps**) a task when
  * the change is at least `min_effect` (default 10 percentage points), and
  * it is significant over seeds after Holm correction across every cell of
    the matrix (so testing many cells does not manufacture "findings").

A **double dissociation** between perturbations X, Y and tasks A, B means:
  X hurts A more than B   AND   Y hurts B more than A.
We test both halves with paired one-sided t-tests over seeds, take the larger
p-value (both must hold), correct with Holm across every (X, Y, A, B)
combination tested, and also require both differences >= `min_effect`.
That is strict on purpose: with 5 seeds and hundreds of combinations, only
large, consistent effects survive.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..stats import holm, mean_ci, t_test


def to_arrays(records, baseline: str = "Intact"):
    """records -> (seeds, conditions, tasks, values[metric] array of shape (S, C, T))."""
    seeds = sorted({r["seed"] for r in records})
    conds = []
    for r in records:  # keep the order in which conditions were given
        if r["condition"] not in conds:
            conds.append(r["condition"])
    if baseline in conds:
        conds.remove(baseline)
        conds.insert(0, baseline)
    tasks = []
    for r in records:
        if r["task"] not in tasks:
            tasks.append(r["task"])
    metrics = [k for k in records[0] if k not in ("seed", "condition", "task")]
    vals = {m: np.full((len(seeds), len(conds), len(tasks)), np.nan) for m in metrics}
    for r in records:
        i, j, k = seeds.index(r["seed"]), conds.index(r["condition"]), tasks.index(r["task"])
        for m in metrics:
            vals[m][i, j, k] = r[m]
    return seeds, conds, tasks, vals


def matrix_stats(records, baseline: str = "Intact", min_effect: float = 0.10, alpha: float = 0.05):
    """Per (condition, task): mean change in accuracy, 95% CI, Holm-corrected p, and a verdict."""
    seeds, conds, tasks, vals = to_arrays(records, baseline)
    acc = vals["accuracy"]
    change = acc[:, 1:, :] - acc[:, :1, :]  # (S, C-1, T), perturbed - intact
    cells, pvals = [], []
    for j, c in enumerate(conds[1:]):
        for k, t in enumerate(tasks):
            d = change[:, j, k]
            m, lo, hi = mean_ci(d)
            cells.append({"condition": c, "task": t, "change": m, "ci_low": lo, "ci_high": hi,
                          "intact": float(np.nanmean(acc[:, 0, k])),
                          "perturbed": float(np.nanmean(acc[:, j + 1, k])),
                          "broke_fixation": float(np.nanmean(vals["broke_fixation"][:, j + 1, k])) if "broke_fixation" in vals else np.nan,
                          "n_seeds": int((~np.isnan(d)).sum())})
            pvals.append(t_test(d))
    padj = holm(pvals)
    for cell, p, pa in zip(cells, pvals, padj):
        cell["p"] = p
        cell["p_holm"] = pa
        sig = (pa < alpha) if not np.isnan(pa) else False
        if sig and cell["change"] <= -min_effect:
            cell["verdict"] = "hurts"
        elif sig and cell["change"] >= min_effect:
            cell["verdict"] = "helps"
        else:
            cell["verdict"] = "no clear effect"
    return {"seeds": seeds, "conditions": conds, "tasks": tasks, "cells": cells, "change": change}


def find_dissociations(records, baseline: str = "Intact", min_effect: float = 0.10, alpha: float = 0.05):
    """All double dissociations that survive the strict criteria (see module docstring)."""
    seeds, conds, tasks, vals = to_arrays(records, baseline)
    acc = vals["accuracy"]
    drop = acc[:, :1, :] - acc[:, 1:, :]  # (S, C-1, T), positive = worse
    pert = conds[1:]
    tests = []
    for (x, cx), (y, cy) in combinations(enumerate(pert), 2):
        for (a, ta), (b, tb) in combinations(enumerate(tasks), 2):
            # Two orientations: X hurts A & Y hurts B, or X hurts B & Y hurts A.
            for (ia, na), (ib, nb) in (((a, ta), (b, tb)), ((b, tb), (a, ta))):
                d1 = drop[:, x, ia] - drop[:, x, ib]  # X: A worse than B
                d2 = drop[:, y, ib] - drop[:, y, ia]  # Y: B worse than A
                p = max(t_test(d1, "greater"), t_test(d2, "greater"))
                tests.append({"X": cx, "Y": cy, "A": na, "B": nb,
                              "X_effect": float(np.nanmean(d1)), "Y_effect": float(np.nanmean(d2)),
                              "drop_XA": float(np.nanmean(drop[:, x, ia])), "drop_XB": float(np.nanmean(drop[:, x, ib])),
                              "drop_YA": float(np.nanmean(drop[:, y, ia])), "drop_YB": float(np.nanmean(drop[:, y, ib])),
                              "p": p})
    padj = holm([t["p"] for t in tests])
    found = []
    for t, pa in zip(tests, padj):
        t["p_holm"] = pa
        if not np.isnan(pa) and pa < alpha and t["X_effect"] >= min_effect and t["Y_effect"] >= min_effect:
            found.append(t)
    found.sort(key=lambda t: -min(t["X_effect"], t["Y_effect"]))
    return found, len(tests)


def describe_dissociation(d, labels=None) -> str:
    """One sentence. Numbers are changes in accuracy (negative = worse)."""
    lab = (lambda t: labels.get(t, t)) if labels else (lambda t: t)
    return (f"'{d['X']}' hurts \"{lab(d['A'])}\" ({-d['drop_XA']:+.0%}) more than \"{lab(d['B'])}\" ({-d['drop_XB']:+.0%}), "
            f"while '{d['Y']}' does the reverse: \"{lab(d['B'])}\" {-d['drop_YB']:+.0%} vs \"{lab(d['A'])}\" {-d['drop_YA']:+.0%}.")
