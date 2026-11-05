"""Run folders: every run saves enough to be reproduced exactly.

Layout of a run folder:

    runs/<YYYYmmdd-HHMMSS>_<name>/
        config.yaml          the full resolved config (all defaults filled in)
        meta.json            git hash, dirty flag, library versions, command
        seed<k>/
            model.pt         trained weights + buffers
            metrics.csv      loss and per-task accuracy during training
            learning_curve.png
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, save_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        # A dirty tree means the code that produced the result is not exactly
        # the commit above. The report flags this.
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def make_run_dir(cfg: ExperimentConfig, root: str | Path = "runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{stamp}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(cfg, run_dir / "config.yaml")
    meta = {
        "created": stamp,
        "git": git_info(),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seeds": cfg.seeds,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def seed_dir(run_dir: Path, seed: int) -> Path:
    d = Path(run_dir) / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d
