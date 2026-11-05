"""Training loop.

Each iteration: pick one task at random, generate a batch of its trials, run
the network, and take one Adam step on the masked mean-squared error between
output and target (Yang et al. 2019 also train this way, one task per batch).

Loss = weighted MSE                        (the task)
     + l2_rate   * mean(rate^2)            (pressure: keep activity low)
     + l2_weight * mean(W_rec^2 + W_in^2)  (pressure: keep weights small)

The two L2 terms are the "pressure knobs" from the spec: they make it costly
for the network to keep information it doesn't need, which matters for the
question of whether absolute pitch gets discarded (phase 2).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .metrics import accuracy
from .model import ModularRNN, to_torch
from .runs import make_run_dir, seed_dir
from .tasks import TaskSet

EVAL_SEED = 12345  # evaluation trials are the same at every checkpoint, so curves are comparable


def set_seed(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def build(cfg: ExperimentConfig, seed: int) -> tuple[ModularRNN, TaskSet]:
    tasks = TaskSet(cfg.tasks, cfg.model.dt_ms)
    model = ModularRNN(cfg.model, tasks.layout.in_groups, tasks.layout.n_out, seed=seed)
    return model, tasks


def masked_mse(out: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (mask * (out - y) ** 2).sum() / mask.sum()


@torch.no_grad()
def evaluate(model: ModularRNN, tasks: TaskSet, n_trials: int, seed: int = EVAL_SEED, batch_size: int = 16) -> dict[str, float]:
    """Accuracy per task on a fixed set of evaluation trials.

    Uses its own torch RNG state so evaluating does not change the training
    noise sequence (otherwise eval_every would change the training result).
    """
    rng = np.random.default_rng(seed)
    results = {}
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for name in tasks.names:
            correct, total = 0.0, 0
            remaining = n_trials
            while remaining > 0:
                b = min(batch_size, remaining)
                batch = tasks.generate(name, b, rng)
                out, _ = model(to_torch(batch.x))
                correct += accuracy(out.numpy(), batch, tasks.layout, tasks.dt_ms) * b
                total += b
                remaining -= b
            results[name] = correct / total
    return results


def train_one(cfg: ExperimentConfig, seed: int, out_dir: Path | None = None, verbose: bool = True) -> dict:
    """Train one network with one seed. Returns its training history."""
    tc = cfg.train
    torch.set_num_threads(tc.num_threads)
    rng = set_seed(seed)
    model, tasks = build(cfg, seed)
    if verbose:
        print(model.describe())
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)

    history: list[dict] = []
    t0 = time.time()
    running_loss = []
    hits = 0
    for it in range(1, tc.n_iters + 1):
        task = tasks.sample_task(rng)
        batch = tasks.generate(task, tc.batch_size, rng)
        out, rates = model(to_torch(batch.x))
        task_loss = masked_mse(out, to_torch(batch.y), to_torch(batch.loss_mask))
        loss = (
            task_loss
            + tc.l2_rate * (rates**2).mean()
            + tc.l2_weight * ((model.W_rec_raw**2).mean() + (model.W_in**2).mean())
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        running_loss.append(task_loss.item())

        if it % tc.eval_every == 0 or it == tc.n_iters:
            acc = evaluate(model, tasks, tc.eval_batch)
            row = {"iter": it, "time_s": round(time.time() - t0, 1), "loss": float(np.mean(running_loss))}
            row.update({f"acc_{k}": v for k, v in acc.items()})
            history.append(row)
            running_loss = []
            if verbose:
                accs = "  ".join(f"{k}={v:.2f}" for k, v in acc.items())
                print(f"[seed {seed}] iter {it:5d}  loss {row['loss']:.4f}  {accs}  ({row['time_s']} s)", flush=True)
            # Stop once every task is above criterion at two evaluations in a row
            # (one lucky evaluation is not enough).
            if tc.stop_at_accuracy is not None and min(acc.values()) >= tc.stop_at_accuracy:
                hits += 1
                if hits >= 2:
                    if verbose:
                        print(f"[seed {seed}] reached criterion {tc.stop_at_accuracy} on all tasks; stopping.")
                    break
            else:
                hits = 0

    if out_dir is not None:
        save_seed_results(out_dir, model, history)
    return {"model": model, "tasks": tasks, "history": history}


def save_seed_results(out_dir: Path, model: ModularRNN, history: list[dict]) -> None:
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    from .plotting import plot_learning_curve

    plot_learning_curve(history, out_dir / "learning_curve.png")


def train_experiment(cfg: ExperimentConfig, runs_root: str | Path = "runs") -> Path:
    run_dir = make_run_dir(cfg, runs_root)
    print(f"Run folder: {run_dir}")
    for seed in cfg.seeds:
        train_one(cfg, seed, seed_dir(run_dir, seed))
    return run_dir


def load_trained(run_dir: str | Path, seed: int) -> tuple[ModularRNN, TaskSet, ExperimentConfig]:
    from .config import load_config

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    model, tasks = build(cfg, seed)
    model.load_state_dict(torch.load(run_dir / f"seed{seed}" / "model.pt"))
    model.eval()
    return model, tasks, cfg
