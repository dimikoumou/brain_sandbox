"""Plots. Kept separate so the core code does not depend on matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no window needed
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(history: list[dict], path: str | Path, title: str = "") -> None:
    iters = [h["iter"] for h in history]
    tasks = [k[4:] for k in history[0] if k.startswith("acc_")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(iters, [h["loss"] for h in history], color="k")
    ax1.set_yscale("log")
    ax1.set_xlabel("training iteration")
    ax1.set_ylabel("task loss (masked MSE)")
    for t in tasks:
        ax2.plot(iters, [h[f"acc_{t}"] for h in history], label=t)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("training iteration")
    ax2.set_ylabel("accuracy")
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_learning_curves_multi(histories: dict[int, list[dict]], path: str | Path, title: str = "") -> None:
    """One panel per task, one line per seed (shows seed-to-seed variability)."""
    tasks = [k[4:] for k in next(iter(histories.values()))[0] if k.startswith("acc_")]
    fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(4 * (len(tasks) + 1), 3.6))
    for seed, hist in histories.items():
        it = [h["iter"] for h in hist]
        axes[0].plot(it, [h["loss"] for h in hist], label=f"seed {seed}", lw=1)
        for ax, t in zip(axes[1:], tasks):
            ax.plot(it, [h[f"acc_{t}"] for h in hist], lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("task loss")
    axes[0].legend(frameon=False, fontsize=8)
    for ax, t in zip(axes[1:], tasks):
        ax.set_title(t)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("accuracy")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example_trial(batch, out: np.ndarray, rates: np.ndarray, module_slices: dict, layout, dt_ms: float, path, trial: int = 0) -> None:
    """Inputs, outputs and per-module activity for one trial."""
    T = batch.x.shape[0]
    t = np.arange(T) * dt_ms
    n = 2 + len(module_slices)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.6 * n), sharex=True)
    L = layout
    axes[0].plot(t, batch.x[:, trial, L.in_slices["fixation"]], label="fixation in", color="k")
    axes[0].plot(t, batch.x[:, trial, L.in_slices["mod1"]].max(1), label="mod1 max", color="C0")
    axes[0].plot(t, batch.x[:, trial, L.in_slices["mod2"]].max(1), label="mod2 max", color="C1")
    axes[0].set_ylabel("inputs")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[1].plot(t, out[:, trial, L.out_slices["fixation"]], label="fixation out", color="k")
    axes[1].plot(t, batch.y[:, trial, L.out_slices["fixation"]], "--", color="grey", label="target")
    axes[1].set_ylabel("output")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    for ax, (name, sl) in zip(axes[2:], module_slices.items()):
        ax.plot(t, rates[:, trial, sl], lw=0.5, alpha=0.6)
        ax.set_ylabel(name, fontsize=8)
    for name, (s, e) in batch.epochs.items():
        for ax in axes:
            ax.axvline(s * dt_ms, color="grey", lw=0.5, ls=":")
        axes[0].text(s * dt_ms, 1.05, name, fontsize=7, transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(f"{batch.task}: example trial", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Shared style for the analysis figures
# ==========================================================================
# Colours follow a validated colour-blind-safe palette. Each brain region
# always gets the same colour, in every figure.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
REGION_COLORS = {"sensory": "#2a78d6", "memory": "#eb6834", "conceptual": "#1baf7a",
                 "executive": "#eda100", "flat": "#2a78d6", "input": MUTED}
# Diverging scale for "change in accuracy": red = worse, gray = no change, blue = better.
DIVERGING = ["#a82e2e", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)


def plot_lesion_matrix(stats, labels: dict, path, title: str = "") -> None:
    """Heatmap: rows = changes made to the network, columns = tasks, colour = change in accuracy."""
    from matplotlib.colors import LinearSegmentedColormap

    conds, tasks = stats["conditions"][1:], stats["tasks"]
    cell = {(c["condition"], c["task"]): c for c in stats["cells"]}
    M = np.array([[cell[(c, t)]["change"] for t in tasks] for c in conds])
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    h = 0.42 * len(conds) + 2.6
    fig, ax = plt.subplots(figsize=(1.0 * len(tasks) + 4.5, h), facecolor=SURFACE)
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i, c in enumerate(conds):
        for j, t in enumerate(tasks):
            v = M[i, j]
            verdict = cell[(c, t)]["verdict"]
            txt = f"{v:+.0%}".replace("+0%", "0%").replace("-0%", "0%")
            strong = verdict in ("hurts", "helps")
            ax.text(j, i, txt + ("*" if strong else ""), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(v) > 0.55 else INK, fontweight="bold" if strong else "normal")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([labels.get(t, t) for t in tasks], rotation=35, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds, fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("change in accuracy vs. intact network", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    ax.set_title((title + "\n" if title else "") + "Red = worse, blue = better. * = clear effect (≥10 points, reliable across seeds)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pitch(cells, controls, path, title: str = "") -> None:
    """Grouped bars: post-tone absolute-pitch decoding per region, for each condition (+ untrained network)."""
    modules = list(dict.fromkeys(c["module"] for c in cells))
    conds = list(dict.fromkeys(c["condition"] for c in cells))
    untrained = {c["module"]: c for c in controls if c["network"] == "untrained"}
    shuffled = [c["post_shuffled"] for c in controls if c["network"] == "trained"]
    groups = (["Untrained network"] if untrained else []) + conds
    n = len(groups)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(modules) + 0.5 * n), 4.2), facecolor=SURFACE)
    _style(ax)
    shades = ["#c3c2b7"] + ["#0d366b", "#256abf", "#5598e7", "#86b6ef", "#b7d3f6", "#e34948", "#eb6834", "#eda100"]
    for gi, g in enumerate(groups):
        xs, ys, lo, hi = [], [], [], []
        for mi, m in enumerate(modules):
            if g == "Untrained network":
                c = untrained[m]
                y, l, h = c["post"], c["post_ci_low"], c["post_ci_high"]
            else:
                c = next(x for x in cells if x["condition"] == g and x["module"] == m)
                y, l, h = c["post"], c["ci_low"], c["ci_high"]
            xs.append(mi - 0.4 + width * (gi + 0.5))
            ys.append(y)
            lo.append(y - l if np.isfinite(l) else 0)
            hi.append(h - y if np.isfinite(h) else 0)
        color = shades[0] if g == "Untrained network" else shades[1 + (gi - (1 if untrained else 0)) % (len(shades) - 1)]
        ax.bar(xs, ys, width * 0.92, color=color, label=g, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=0)
    ax.axhline(1 / 12, color=INK, lw=1, ls="--")
    ax.text(len(modules) - 0.45, 1 / 12 + 0.01, "chance (1 in 12)", fontsize=7.5, color=INK_2, ha="right")
    if shuffled:
        ax.axhline(np.mean(shuffled), color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, fontsize=9, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("note named correctly\n(100–300 ms after the tone ended)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    ax.set_title((title + ": " if title else "") + "hidden absolute pitch per region (bars: mean of seeds, lines: 95% CI)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_clusters(Vn, labels, task_names, module_slices, path) -> None:
    """Left: each unit's task profile, units grouped by cluster. Right: which regions each cluster's units come from."""
    from .tasks import TASK_LABELS

    act = labels >= 0
    order = np.argsort(labels[act], kind="stable")
    X = Vn[act][order]
    lab = labels[act][order]
    k = labels.max() + 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.3, 1]}, facecolor=SURFACE)
    ax1.imshow(X, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for b in np.where(np.diff(lab))[0]:
        ax1.axhline(b + 0.5, color=INK, lw=0.6)
    ax1.set_xticks(range(len(task_names)))
    ax1.set_xticklabels([TASK_LABELS.get(t, t) for t in task_names], rotation=40, ha="right", fontsize=7.5, color=INK_2)
    mids = [np.mean(np.where(lab == c)[0]) for c in range(k)]
    ax1.set_yticks(mids)
    ax1.set_yticklabels([f"cluster {c + 1}" for c in range(k)], fontsize=8, color=INK)
    ax1.set_title("Each row is one unit: how much it is used by each task", fontsize=9, color=INK, loc="left")
    ax1.tick_params(length=0)

    regions = list(module_slices)
    comp = np.zeros((k, len(regions)))
    for j, sl in enumerate(module_slices.values()):
        for c in range(k):
            comp[c, j] = np.sum(labels[sl] == c)
    _style(ax2)
    left = np.zeros(k)
    for j, r in enumerate(regions):
        ax2.barh(range(k), comp[:, j], left=left, color=REGION_COLORS.get(r, MUTED), label=r,
                 edgecolor=SURFACE, linewidth=1.5, height=0.7)
        left += comp[:, j]
    ax2.set_yticks(range(k))
    ax2.set_yticklabels([f"cluster {c + 1}" for c in range(k)], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("number of units")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.set_title("Which designed region each cluster's units come from", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_psychometric(curves: dict, path, title: str = "") -> None:
    """curves[label] = (coh, p_choose1). One line per condition."""
    fig, ax = plt.subplots(figsize=(5, 3.8), facecolor=SURFACE)
    _style(ax)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for i, (lab, (coh, p)) in enumerate(curves.items()):
        ax.plot(coh, p, "-o", color=colors[i % 4], lw=2, ms=4, label=lab)
    ax.axhline(0.5, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("how much brighter light 1 is (coherence)")
    ax.set_ylabel("probability of choosing light 1")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep(out_dir, values, knob: str) -> None:
    """Accuracy of every task as one knob is turned (one line per task, mean over seeds)."""
    import csv
    from pathlib import Path

    from .tasks import TASK_LABELS

    out_dir = Path(out_dir)
    rows = list(csv.DictReader(open(out_dir / "behaviour.csv")))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    xs = list(range(len(conds)))
    cats = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for i, t in enumerate(tasks):
        ys = [np.mean([float(r["accuracy"]) for r in rows if r["condition"] == c and r["task"] == t]) for c in conds]
        style = "-" if i < len(cats) else "--"
        ax.plot(xs, ys, style, marker="o", ms=4, lw=1.6, color=cats[i % len(cats)], label=TASK_LABELS.get(t, t))
    ax.set_xticks(xs)
    ax.set_xticklabels(["intact"] + [str(v) for v in values], fontsize=8)
    ax.set_xlabel(knob)
    ax.set_ylabel("accuracy (mean of seeds)")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pitch_checkpoint(summary, models, modules, path) -> None:
    """Per region: hidden absolute pitch before training (grey) and after, for each model type."""
    fig, ax = plt.subplots(figsize=(max(7, 2.0 * len(modules)), 4.0), facecolor=SURFACE)
    _style(ax)
    cats = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    groups = ["before training"] + models
    w = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        xs, ys, lo, hi = [], [], [], []
        for mi, m in enumerate(modules):
            if g == "before training":
                es = [e for e in summary if e.get("module") == m and "untrained" in e]
                y = float(np.mean([e["untrained"] for e in es])) if es else np.nan
                l = h = 0.0
            else:
                e = next((e for e in summary if e.get("model") == g and e.get("module") == m and "trained" in e), None)
                if e is None:
                    continue
                y, l, h = e["trained"], e["trained"] - e["trained_lo"], e["trained_hi"] - e["trained"]
            xs.append(mi - 0.4 + w * (gi + 0.5))
            ys.append(y)
            lo.append(l if np.isfinite(l) else 0)
            hi.append(h if np.isfinite(h) else 0)
        color = "#c3c2b7" if gi == 0 else cats[(gi - 1) % len(cats)]
        ax.bar(xs, ys, w * 0.92, color=color, label=g, edgecolor=SURFACE, linewidth=1)
        if gi:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=0.8)
    ax.axhline(1 / 12, color=INK, ls="--", lw=1)
    ax.text(len(modules) - 0.45, 1 / 12 + 0.015, "chance", fontsize=7.5, color=INK_2, ha="right")
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules)
    ax.set_ylim(0, 1)
    ax.set_ylabel("note named correctly\n(100–300 ms after the tone)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Hidden absolute pitch before and after training (grey = same networks before training, mean of all)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
