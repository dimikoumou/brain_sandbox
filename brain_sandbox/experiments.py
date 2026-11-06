"""Running perturbation experiments from a YAML file.

An experiment file looks like this (see configs/experiments/ for real ones):

    name: lesion_matrix
    run: runs/<folder of a trained model>
    preregistration: |
      Written BEFORE running: what I expect and why.
    predictions:                       # optional, checked automatically
      - {condition: "Switch off memory", task: delaygo, direction: decrease}
      - {condition: "Cut feedback", pitch_module: sensory, direction: increase}
    n_trials: 512                      # test trials per task, per seed
    conditions:
      - label: Switch off memory
        stack: [{type: lesion_module, module: memory}]
    sweep:                             # optional: generate conditions automatically
      label: "Noise in sensory x{value}"
      type: noise
      param: factor
      values: [1.5, 2, 4]
      fixed: {module: sensory}
    cluster_lesions: false             # add one condition per emergent cluster
    pitch: false                       # also measure latent absolute pitch

The intact network is always included as the condition "Intact". Every
condition is tested on every seed of the trained run, on the same test
trials, and all results (raw per-seed numbers, statistics, figures and a
plain-language report) are saved to a new timestamped folder.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .analysis.behaviour import evaluate_detailed
from .analysis.clusters import (centroids, cluster_units, describe_cluster, match_to_reference,
                                region_composition, task_variance)
from .analysis.lesion_matrix import describe_dissociation, find_dissociations, matrix_stats
from .analysis.pitch_probe import pitch_report
from .config import load_config
from .perturb import Condition, LesionUnits, from_spec, perturbed
from .runs import git_info
from .stats import holm, mean_ci, t_test
from .tasks import TASK_LABELS
from .train import build, load_trained

BASELINE = "Intact"


@dataclass
class ExperimentSpec:
    name: str
    run: str
    conditions: list[dict] = field(default_factory=list)
    preregistration: str = ""
    predictions: list[dict] = field(default_factory=list)
    n_trials: int = 512
    seeds: list[int] | None = None  # default: every seed of the trained run
    sweep: dict | None = None
    cluster_lesions: bool = False
    pitch: bool = False
    pitch_reps: int = 24
    pitch_few_shot: list[int] = field(default_factory=lambda: [1, 2, 5, 10])
    pitch_untrained_baseline: bool = True
    min_effect: float = 0.10  # smallest change in task accuracy we call an effect
    min_pitch_effect: float = 0.05  # same, for pitch-decoding accuracy
    notes: str = ""

    @classmethod
    def load(cls, path) -> "ExperimentSpec":
        with open(path) as f:
            data = yaml.safe_load(f)
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown experiment key(s): {sorted(unknown)}")
        return cls(**data)

    def all_conditions(self) -> list[Condition]:
        conds = [Condition(BASELINE, [])] + [Condition.from_dict(c) for c in self.conditions]
        if self.sweep:
            s = self.sweep
            for v in s["values"]:
                spec = {"type": s["type"], **s.get("fixed", {}), s["param"]: v}
                conds.append(Condition(s["label"].format(value=v), [from_spec(spec)]))
        return conds


# --------------------------------------------------------------------------
def run_experiment(spec: ExperimentSpec, out_root: str | Path = "runs", verbose: bool = True) -> Path:
    run_dir = Path(spec.run)
    cfg = load_config(run_dir / "config.yaml")
    seeds = spec.seeds or cfg.seeds
    conditions = spec.all_conditions()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / f"{stamp}_exp_{spec.name}"
    out.mkdir(parents=True)
    with open(out / "experiment.yaml", "w") as f:
        yaml.safe_dump(spec.__dict__, f, sort_keys=False)
    (out / "meta.json").write_text(json.dumps({"created": stamp, "git": git_info(), "trained_run": str(run_dir),
                                               "seeds": seeds}, indent=2))

    # Emergent clusters, matched across seeds (only if asked for).
    cluster_info = _clusters_for_seeds(run_dir, seeds, verbose) if spec.cluster_lesions else None

    behaviour, pitch_rows, controls = [], [], []
    for seed in seeds:
        model, tasks, _ = load_trained(run_dir, seed)
        seed_conditions = list(conditions)
        if cluster_info:
            labels = cluster_info["labels"][seed]
            for c in range(cluster_info["k"]):
                units = np.where(labels == c)[0].tolist()
                seed_conditions.append(Condition(cluster_info["names"][c], [LesionUnits(units, f"cluster {c + 1}")]))
        for cond in seed_conditions:
            m = perturbed(model, cond.stack)
            res = evaluate_detailed(m, tasks, spec.n_trials)
            for task, r in res.items():
                behaviour.append({"seed": seed, "condition": cond.label, "task": task, **r})
            if spec.pitch:
                rep = pitch_report(m, tasks.layout, tasks.dt_ms, n_reps=spec.pitch_reps, seed=seed,
                                   few_shot=tuple(spec.pitch_few_shot), include_controls=(cond.label == BASELINE))
                for mod, e in rep["modules"].items():
                    row = {"seed": seed, "condition": cond.label, "module": mod, "post": e["post"]}
                    row.update({f"few_shot_{n}": v for n, v in e["few_shot"].items()})
                    pitch_rows.append(row)
                    if cond.label == BASELINE:
                        controls.append({"seed": seed, "network": "trained", "module": mod,
                                         "post": e["post"], "post_shuffled": e["post_shuffled"], "during": e["during"]})
                if cond.label == BASELINE:
                    controls.append({"seed": seed, "network": "input", "module": "input",
                                     "post": rep["input_post"], "post_shuffled": np.nan, "during": rep["input_during"]})
            if verbose:
                acc = np.mean([r["accuracy"] for r in res.values()])
                print(f"[seed {seed}] {cond.label:<45} mean accuracy {acc:.2f}", flush=True)
        if spec.pitch and spec.pitch_untrained_baseline:
            untrained, t2 = build(cfg, seed)  # same seed => same initial weights as before training
            rep = pitch_report(untrained, t2.layout, t2.dt_ms, n_reps=spec.pitch_reps, seed=seed, few_shot=(1,))
            for mod, e in rep["modules"].items():
                controls.append({"seed": seed, "network": "untrained", "module": mod,
                                 "post": e["post"], "post_shuffled": e["post_shuffled"], "during": e["during"]})

    _write_csv(out / "behaviour.csv", behaviour)
    if pitch_rows:
        _write_csv(out / "pitch.csv", pitch_rows)
        _write_csv(out / "pitch_controls.csv", controls)
    if cluster_info:
        _save_clusters(out, cluster_info, run_dir, seeds)

    results = analyse(out, spec)
    write_report(out, spec, results)
    if verbose:
        print(f"Results in {out}")
    return out


def _write_csv(path, rows):
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = int(v) if k == "seed" else float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


# --------------------------------------------------------------------------
def _clusters_for_seeds(run_dir, seeds, verbose):
    """Cluster each seed's units; use the reference seed's k for all; match clusters across seeds."""
    Vs, models = {}, {}
    for s in seeds:
        model, tasks, _ = load_trained(run_dir, s)
        Vs[s] = task_variance(model, tasks)
        models[s] = model
    ref = seeds[0]
    ref_res = cluster_units(Vs[ref])
    k = ref_res["k"]
    ref_c = centroids(Vs[ref], ref_res["labels"])
    labels = {ref: ref_res["labels"]}
    for s in seeds[1:]:
        res = cluster_units(Vs[s], k=k)
        labels[s] = match_to_reference(ref_c, centroids(Vs[s], res["labels"]), res["labels"])
    names = [f"Switch off cluster {c + 1} ({describe_cluster(ref_c[c], tasks.names, TASK_LABELS)})" for c in range(k)]
    if verbose:
        print(f"Clusters: k = {k} (silhouette scores {ref_res['silhouette']})")
    return {"k": k, "labels": labels, "V": Vs, "names": names, "silhouette": ref_res["silhouette"],
            "task_names": tasks.names, "module_slices": models[ref].module_slices}


def _save_clusters(out, info, run_dir, seeds):
    from .analysis.clusters import normalise
    from .plotting import plot_clusters

    np.savez(out / "clusters.npz", **{f"labels_seed{s}": info["labels"][s] for s in seeds},
             **{f"task_variance_seed{s}": info["V"][s] for s in seeds})
    comp = {s: region_composition(info["labels"][s], info["module_slices"]).tolist() for s in seeds}
    (out / "clusters.json").write_text(json.dumps({"k": info["k"], "names": info["names"],
                                                   "silhouette": info["silhouette"],
                                                   "regions": list(info["module_slices"]),
                                                   "composition_per_seed": comp}, indent=2))
    s0 = seeds[0]
    plot_clusters(normalise(info["V"][s0]), info["labels"][s0], info["task_names"], info["module_slices"],
                  out / f"clusters_seed{s0}.png")


# --------------------------------------------------------------------------
def analyse(out: Path, spec: ExperimentSpec) -> dict:
    """Statistics, dissociations, prediction checks and figures from the saved CSVs."""
    from .plotting import plot_lesion_matrix, plot_pitch

    behaviour = read_csv(out / "behaviour.csv")
    stats = matrix_stats(behaviour, BASELINE, spec.min_effect)
    dissoc, n_tests = find_dissociations(behaviour, BASELINE, spec.min_effect)
    plot_lesion_matrix(stats, TASK_LABELS, out / "lesion_matrix.png", title=spec.name)

    results: dict[str, Any] = {"cells": stats["cells"], "dissociations": dissoc, "n_dissociation_tests": n_tests,
                               "conditions": stats["conditions"], "tasks": stats["tasks"], "seeds": stats["seeds"]}
    if (out / "pitch.csv").exists():
        pitch = read_csv(out / "pitch.csv")
        controls = read_csv(out / "pitch_controls.csv")
        results["pitch"] = pitch_stats(pitch, spec.min_pitch_effect)
        results["pitch_controls"] = control_stats(controls)
        plot_pitch(results["pitch"], results["pitch_controls"], out / "pitch.png", title=spec.name)
    results["predictions"] = check_predictions(spec, results)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def pitch_stats(rows, min_effect):
    """Per (condition, module): mean post-tone decoding, CI, and change vs intact (paired over seeds)."""
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    modules = list(dict.fromkeys(r["module"] for r in rows))
    seeds = sorted({r["seed"] for r in rows})
    get = {(r["seed"], r["condition"], r["module"]): r for r in rows}
    cells, pvals = [], []
    for c in conds:
        for m in modules:
            vals = np.array([get[(s, c, m)]["post"] for s in seeds])
            base = np.array([get[(s, BASELINE, m)]["post"] for s in seeds])
            mean, lo, hi = mean_ci(vals)
            d = vals - base
            dm, dlo, dhi = mean_ci(d)
            few = {k[len("few_shot_"):]: float(np.mean([get[(s, c, m)][k] for s in seeds]))
                   for k in get[(seeds[0], c, m)] if k.startswith("few_shot_")}
            cells.append({"condition": c, "module": m, "post": mean, "ci_low": lo, "ci_high": hi,
                          "change": dm, "change_ci_low": dlo, "change_ci_high": dhi, "few_shot": few})
            pvals.append(t_test(d) if c != BASELINE else np.nan)
    padj = holm(pvals)
    for cell, p, pa in zip(cells, pvals, padj):
        cell["p"], cell["p_holm"] = p, pa
        if cell["condition"] == BASELINE:
            cell["verdict"] = "reference"
        elif pa < 0.05 and cell["change"] >= min_effect:
            cell["verdict"] = "more pitch information"
        elif pa < 0.05 and cell["change"] <= -min_effect:
            cell["verdict"] = "less pitch information"
        else:
            cell["verdict"] = "no clear change"
    return cells


def control_stats(rows):
    out = []
    for net in dict.fromkeys(r["network"] for r in rows):
        for m in dict.fromkeys(r["module"] for r in rows if r["network"] == net):
            sel = [r for r in rows if r["network"] == net and r["module"] == m]
            entry = {"network": net, "module": m}
            for key in ("post", "post_shuffled", "during"):
                entry[key], entry[key + "_ci_low"], entry[key + "_ci_high"] = mean_ci([r[key] for r in sel])
            out.append(entry)
    return out


def check_predictions(spec, results) -> list[dict]:
    """Compare each pre-registered prediction with the outcome.

    supported      the effect is significant (Holm-corrected) and big enough, in the predicted direction
    contradicted   significant and big enough, in the OPPOSITE direction
    no change      for "no change" predictions: the whole 95% CI lies within +-min_effect
    inconclusive   anything else (e.g. small, or too variable across seeds to tell)
    """
    checked = []
    for pr in spec.predictions:
        if "task" in pr:
            cell = next((c for c in results["cells"] if c["condition"] == pr["condition"] and c["task"] == pr["task"]), None)
            eff, tol = "hurts", spec.min_effect
            what = TASK_LABELS.get(pr["task"], pr["task"])
            if cell is None:
                checked.append({**pr, "outcome": "not measured"})
                continue
            change, lo, hi = cell["change"], cell["ci_low"], cell["ci_high"]
            sig = cell["p_holm"] < 0.05
        else:
            cell = next((c for c in results.get("pitch", []) if c["condition"] == pr["condition"]
                         and c["module"] == pr["pitch_module"]), None)
            tol = spec.min_pitch_effect
            what = f"absolute pitch in {pr['pitch_module']}"
            if cell is None:
                checked.append({**pr, "outcome": "not measured"})
                continue
            change, lo, hi = cell["change"], cell["change_ci_low"], cell["change_ci_high"]
            sig = cell["p_holm"] < 0.05
        big = abs(change) >= tol
        d = pr["direction"]
        if d == "no change":
            outcome = "supported" if (lo > -tol and hi < tol) else ("contradicted" if sig and big else "inconclusive")
        else:
            sign_ok = (change < 0) if d == "decrease" else (change > 0)
            if sig and big:
                outcome = "supported" if sign_ok else "contradicted"
            else:
                outcome = "inconclusive"
        checked.append({**pr, "what": what, "change": change, "ci_low": lo, "ci_high": hi, "outcome": outcome})
    return checked


# --------------------------------------------------------------------------
def write_report(out: Path, spec: ExperimentSpec, results: dict) -> None:
    """A plain-language markdown report."""
    L = []
    L.append(f"# Experiment: {spec.name}\n")
    L.append(f"Trained model: `{spec.run}` · seeds: {results['seeds']} · {spec.n_trials} test trials per task per seed.\n")
    L.append("Numbers are averages over the seeds (independently trained networks), with 95% confidence "
             "intervals in brackets. A change counts as real only if it is at least "
             f"{spec.min_effect:.0%} and survives a correction for testing many things at once.\n")
    if spec.preregistration:
        L.append("## What I predicted before running it\n")
        L.append("> " + spec.preregistration.strip().replace("\n", "\n> ") + "\n")
    if results["predictions"]:
        L.append("## Predictions vs. outcome\n")
        L.append("| Perturbation | Measure | Predicted | Measured change | Outcome |")
        L.append("|---|---|---|---|---|")
        for p in results["predictions"]:
            if p["outcome"] == "not measured":
                L.append(f"| {p['condition']} | {p.get('task', p.get('pitch_module'))} | {p['direction']} | – | not measured |")
                continue
            L.append(f"| {p['condition']} | {p['what']} | {p['direction']} | "
                     f"{p['change']:+.1%} [{p['ci_low']:+.1%}, {p['ci_high']:+.1%}] | **{p['outcome']}** |")
        L.append("")

    L.append("## Effect of each change on each task\n")
    L.append("![lesion matrix](lesion_matrix.png)\n")
    tasks = results["tasks"]
    L.append("| Change | " + " | ".join(TASK_LABELS.get(t, t) for t in tasks) + " |")
    L.append("|---|" + "---|" * len(tasks))
    intact = {c["task"]: c["intact"] for c in results["cells"]}
    L.append(f"| *{BASELINE} (accuracy)* | " + " | ".join(f"{intact[t]:.0%}" for t in tasks) + " |")
    for cond in results["conditions"][1:]:
        row = []
        for t in tasks:
            c = next(x for x in results["cells"] if x["condition"] == cond and x["task"] == t)
            mark = {"hurts": " ▼", "helps": " ▲"}.get(c["verdict"], "")
            row.append(f"{c['change']:+.0%}{mark}")
        L.append(f"| {cond} | " + " | ".join(row) + " |")
    L.append("\n▼ = clearly worse, ▲ = clearly better (big enough and statistically reliable). "
             "Unmarked numbers may just be chance.\n")

    hurt = [c for c in results["cells"] if c["verdict"] == "hurts"]
    fix = [c for c in hurt if c["broke_fixation"] > 0.5 * (1 - c["perturbed"])]
    if fix:
        L.append(f"Of the {len(hurt)} clear impairments, {len(fix)} were mostly *impulsive* errors (answering before "
                 "the go signal) rather than wrong answers; see `behaviour.csv` for the error types.\n")

    L.append("## Double dissociations\n")
    if results["dissociations"]:
        L.append(f"Found {len(results['dissociations'])} (out of {results['n_dissociation_tests']} combinations tested). "
                 "The strongest:\n")
        for d in results["dissociations"][:10]:
            L.append("* " + describe_dissociation(d, TASK_LABELS))
        L.append("")
    else:
        L.append(f"None survived the strict criteria ({results['n_dissociation_tests']} combinations tested).\n")

    if "pitch" in results:
        L.append("## Hidden absolute pitch (can a simple readout name the note after it ended?)\n")
        L.append("![pitch](pitch.png)\n")
        L.append("Chance = 8.3% (1 in 12 notes).\n")
        L.append("| Change | Region | Note decoded correctly | Change vs intact | Verdict |")
        L.append("|---|---|---|---|---|")
        for c in results["pitch"]:
            ch = "–" if c["condition"] == BASELINE else f"{c['change']:+.1%} [{c['change_ci_low']:+.1%}, {c['change_ci_high']:+.1%}]"
            L.append(f"| {c['condition']} | {c['module']} | {c['post']:.1%} [{c['ci_low']:.1%}, {c['ci_high']:.1%}] | {ch} | {c['verdict']} |")
        L.append("\n**Controls** (intact network):\n")
        L.append("| Network | Region | After the tone | Same, labels shuffled | While the tone plays |")
        L.append("|---|---|---|---|---|")
        for c in results["pitch_controls"]:
            sh = "–" if np.isnan(c["post_shuffled"]) else f"{c['post_shuffled']:.1%}"
            L.append(f"| {c['network']} | {c['module']} | {c['post']:.1%} | {sh} | {c['during']:.1%} |")
        L.append("")
    L.append("---\nRaw numbers: `behaviour.csv`" + (", `pitch.csv`, `pitch_controls.csv`" if "pitch" in results else "")
             + "; statistics: `results.json`; exact settings: `experiment.yaml`.\n")
    (out / "report.md").write_text("\n".join(L))
