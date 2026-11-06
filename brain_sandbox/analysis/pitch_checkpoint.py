"""The phase-2 checkpoint: does training make the network discard absolute pitch?

Compares hidden absolute pitch (decoded 100-300 ms after a tone ends) across:

* each trained model type (e.g. main, no-pitch control, high pressure), and
* the SAME networks before training (their random initial weights), which is
  a paired comparison: seed k trained vs. seed k untrained.

"Discarded" here means: trained networks keep LESS note information than they
did before training. "Discarded because of the pitch game" additionally needs
the main model to keep less than the no-pitch control. Both comparisons are
reported; with 3-5 seeds per model only large differences will be detectable.

Run:  python -m brain_sandbox pitch-compare --runs runs/A runs/B runs/C --labels main "no pitch" pressure
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as _st

from ..config import load_config
from ..stats import holm, mean_ci, t_test
from ..train import build, load_trained
from .behaviour import evaluate_detailed
from .pitch_probe import pitch_report


def measure(run_dir, n_reps: int = 24) -> list[dict]:
    """Per seed and region: post-tone decoding when trained and when untrained (+ controls)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    rows = []
    for seed in cfg.seeds:
        if not (run_dir / f"seed{seed}" / "model.pt").exists():
            continue
        trained, tasks, _ = load_trained(run_dir, seed)
        untrained, _ = build(cfg, seed)
        pitch_task = "pitchdir" if "pitchdir" in tasks.names else "relpitch"
        task_acc = evaluate_detailed(trained, tasks, 512, task_names=[pitch_task])[pitch_task]["accuracy"]
        rt = pitch_report(trained, tasks.layout, tasks.dt_ms, n_reps=n_reps, seed=seed, few_shot=(1, 5))
        ru = pitch_report(untrained, tasks.layout, tasks.dt_ms, n_reps=n_reps, seed=seed, few_shot=(1, 5))
        for mod in rt["modules"]:
            rows.append({"seed": seed, "module": mod, "pitch_task_accuracy": task_acc,
                         "trained": rt["modules"][mod]["post"], "untrained": ru["modules"][mod]["post"],
                         "shuffled": rt["modules"][mod]["post_shuffled"],
                         "during_tone": rt["modules"][mod]["during"],
                         "few_shot_1": rt["modules"][mod]["few_shot"][1], "few_shot_5": rt["modules"][mod]["few_shot"][5]})
    return rows


def compare(runs: dict[str, str], out_root="runs", n_reps: int = 24, reference: str | None = None) -> Path:
    """Measure every run, compare, and write a report + figure into a new folder."""
    from ..plotting import plot_pitch_checkpoint

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / f"{stamp}_pitch_checkpoint"
    out.mkdir(parents=True)
    data = {label: measure(path, n_reps) for label, path in runs.items()}
    (out / "raw.json").write_text(json.dumps({"runs": runs, "data": data}, indent=2, default=float))
    reference = reference or next(iter(runs))
    modules = list(dict.fromkeys(r["module"] for rows in data.values() for r in rows))

    summary, tests = [], []
    for label, rows in data.items():
        for m in modules:
            sel = [r for r in rows if r["module"] == m]
            tr = [r["trained"] for r in sel]
            un = [r["untrained"] for r in sel]
            entry = {"model": label, "module": m, "n_seeds": len(sel)}
            for key, vals in (("trained", tr), ("untrained", un), ("shuffled", [r["shuffled"] for r in sel])):
                entry[key], entry[key + "_lo"], entry[key + "_hi"] = mean_ci(vals)
            d = np.array(tr) - np.array(un)
            entry["change_vs_untrained"], entry["change_lo"], entry["change_hi"] = mean_ci(d)
            entry["p_vs_untrained"] = t_test(d)
            entry["pitch_task_accuracy"] = float(np.mean([r["pitch_task_accuracy"] for r in sel]))
            summary.append(entry)
            tests.append(entry["p_vs_untrained"])
    # Between models (unpaired: different networks): Welch t-test vs the reference model.
    for label in runs:
        if label == reference:
            continue
        for m in modules:
            a = [r["trained"] for r in data[reference] if r["module"] == m]
            b = [r["trained"] for r in data[label] if r["module"] == m]
            p = float(_st.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else np.nan
            summary.append({"model": f"{reference} vs {label}", "module": m, "difference": float(np.mean(a) - np.mean(b)),
                            "p_between": p})
            tests.append(p)
    padj = holm(tests)
    i = 0
    for e in summary:
        if "p_vs_untrained" in e:
            e["p_holm"] = padj[i]
            i += 1
        elif "p_between" in e:
            e["p_holm"] = padj[i]
            i += 1
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot_pitch_checkpoint(summary, list(runs), modules, out / "pitch_checkpoint.png")
    _write_report(out, summary, runs, modules, reference)
    return out


def _write_report(out, summary, runs, modules, reference):
    L = ["# Checkpoint: does training make the network discard absolute pitch?\n",
         "Hidden absolute pitch = how often a simple decoder names the note (1 of 12) from a region's activity "
         "100–300 ms **after** the tone ended. Chance = 8.3%. Mean over seeds [95% CI].\n",
         "![checkpoint](pitch_checkpoint.png)\n",
         "| Model | Region | Before training | After training | Change | Reliable? (Holm p<0.05) | Pitch game accuracy |",
         "|---|---|---|---|---|---|---|"]
    for e in summary:
        if "trained" not in e:
            continue
        sig = "yes" if e["p_holm"] < 0.05 else "no"
        L.append(f"| {e['model']} | {e['module']} | {e['untrained']:.1%} | {e['trained']:.1%} [{e['trained_lo']:.1%}, {e['trained_hi']:.1%}] | "
                 f"{e['change_vs_untrained']:+.1%} [{e['change_lo']:+.1%}, {e['change_hi']:+.1%}] | {sig} | {e['pitch_task_accuracy']:.0%} |")
    L.append("\n**Between models** (after training; Welch t-test, unpaired):\n")
    L.append("| Comparison | Region | Difference | Reliable? |")
    L.append("|---|---|---|---|")
    for e in summary:
        if "p_between" in e:
            L.append(f"| {e['model']} | {e['module']} | {e['difference']:+.1%} | {'yes' if e['p_holm'] < 0.05 else 'no'} |")
    L.append("\nRuns: " + ", ".join(f"{k} = `{v}`" for k, v in runs.items()))
    (out / "report.md").write_text("\n".join(L) + "\n")
