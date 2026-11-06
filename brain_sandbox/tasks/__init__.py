"""Task registry. To add a task, write a function in a task file and register it here."""

from __future__ import annotations

import numpy as np

from ..config import TaskConfig
from .common import Layout, TrialBatch, circular_distance
from . import pitch, yang

TASKS = {
    "go": yang.go,
    "anti": yang.anti,
    "delaygo": yang.delaygo,
    "dm1": yang.dm1,
    "dm2": yang.dm2,
    "ctxdm1": yang.ctxdm1,
    "ctxdm2": yang.ctxdm2,
    "dms": yang.dms,
    "dnms": yang.dnms,
    "relpitch": pitch.relpitch,
}

# Plain-language names, used in plots, the report and the dashboard.
TASK_LABELS = {
    "go": "Look at the light",
    "anti": "Look away from the light",
    "delaygo": "Remember where the light was",
    "dm1": "Pick the brighter light (channel 1)",
    "dm2": "Pick the brighter light (channel 2)",
    "ctxdm1": "Pick the brighter light, ignoring channel 2",
    "ctxdm2": "Pick the brighter light, ignoring channel 1",
    "dms": "Same place twice? Then look",
    "dnms": "Different place? Then look",
    "relpitch": "Same tune in a new key?",
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


__all__ = ["TASKS", "TASK_LABELS", "TaskSet", "Layout", "TrialBatch", "circular_distance"]
