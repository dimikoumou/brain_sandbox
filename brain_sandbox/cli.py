"""Command-line interface.

    python -m brain_sandbox train --config configs/phase1_modular.yaml
    python -m brain_sandbox train --config configs/phase1_modular.yaml --seeds 0 1 2 --set train.n_iters=500
    python -m brain_sandbox evaluate --run runs/<folder>

`perturb`, `sweep` and `report` arrive in phase 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .config import config_from_dict, config_to_dict, load_config


def apply_overrides(cfg, overrides: list[str]):
    """--set a.b=value overrides, with YAML parsing of the value."""
    d = config_to_dict(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"--set: unknown key {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    return config_from_dict(d)


def cmd_train(args):
    from .plotting import plot_learning_curves_multi
    from .train import train_experiment
    import csv

    cfg = load_config(args.config)
    if args.seeds is not None:
        cfg.seeds = args.seeds
    cfg = apply_overrides(cfg, args.set or [])
    run_dir = train_experiment(cfg, args.runs_root, parallel=args.parallel)

    histories = {}
    for s in cfg.seeds:
        with open(run_dir / f"seed{s}" / "metrics.csv") as f:
            histories[s] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    plot_learning_curves_multi(histories, run_dir / "learning_curves.png", title=cfg.name)
    print(f"Done. Results in {run_dir}")


def cmd_evaluate(args):
    from .plotting import plot_example_trial
    from .train import evaluate, load_trained
    from .model import to_torch
    from .stats import mean_ci

    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    per_seed = {}
    for s in cfg.seeds:
        model, tasks, _ = load_trained(run_dir, s)
        per_seed[s] = evaluate(model, tasks, args.n_trials, seed=args.eval_seed)
        if args.plots:
            rng = np.random.default_rng(args.eval_seed)
            for name in tasks.names:
                batch = tasks.generate(name, 8, rng)
                out, rates = model(to_torch(batch.x))
                plot_example_trial(
                    batch, out.detach().numpy(), rates.detach().numpy(), model.module_slices,
                    tasks.layout, tasks.dt_ms, run_dir / f"seed{s}" / f"example_{name}.png",
                )
    print(f"Accuracy on {args.n_trials} held-out trials per task (eval seed {args.eval_seed}):")
    summary = {}
    for name in cfg.tasks.names:
        vals = [per_seed[s][name] for s in cfg.seeds]
        m, lo, hi = mean_ci(vals)
        summary[name] = {"per_seed": vals, "mean": m, "ci95": [lo, hi]}
        print(f"  {name:<10} mean {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(vals)} seeds)")
    (run_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="brain_sandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train one network per seed")
    t.add_argument("--config", required=True)
    t.add_argument("--seeds", type=int, nargs="+")
    t.add_argument("--set", nargs="*", help="override config values, e.g. train.n_iters=500")
    t.add_argument("--runs-root", default="runs")
    t.add_argument("--parallel", type=int, default=1, help="train this many seeds at once (1 CPU core each)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a trained run on fresh trials")
    e.add_argument("--run", required=True)
    e.add_argument("--n-trials", type=int, default=1000)
    e.add_argument("--eval-seed", type=int, default=999)  # differs from the in-training eval seed
    e.add_argument("--plots", action="store_true", help="also save example-trial plots")
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
