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
