"""Task registry. To add a task, write a function in a task file and register it here."""

from __future__ import annotations

import numpy as np

from ..config import TaskConfig
from .common import Layout, TrialBatch, circular_distance
from . import yang

TASKS = {
    "go": yang.go,
    "delaygo": yang.delaygo,
}


class TaskSet:
    """The tasks one network is trained on, plus the shared input/output layout."""

    def __init__(self, cfg: TaskConfig, dt_ms: float):
        unknown = [n for n in cfg.names if n not in TASKS]
        if unknown:
            raise KeyError(f"Unknown task(s) {unknown}. Available: {sorted(TASKS)}")
        self.cfg = cfg
        self.dt_ms = dt_ms
        self.layout = Layout(cfg)
        self.names = list(cfg.names)
        w = np.array([cfg.weights.get(n, 1.0) for n in self.names], dtype=float)
        self.probs = w / w.sum()

    def generate(self, task: str, batch_size: int, rng: np.random.Generator) -> TrialBatch:
        return TASKS[task](self.layout, batch_size, rng, self.dt_ms, self.cfg.params.get(task, {}))

    def sample_task(self, rng: np.random.Generator) -> str:
        return str(rng.choice(self.names, p=self.probs))


__all__ = ["TASKS", "TaskSet", "Layout", "TrialBatch", "circular_distance"]
