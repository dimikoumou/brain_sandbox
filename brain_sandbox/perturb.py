"""Perturbations: the knobs for tuning and damaging a trained network.

Every perturbation is a small object with

    .apply(model)   edit the model's "knob" buffers in place
    .describe()     one plain-language sentence

You never apply a perturbation to the original network: `perturbed(model, ...)`
makes a deep copy first, so the trained network is never changed. Perturbations
stack: `perturbed(model, [LesionModule("conceptual", 0.5), Noise(2.0, "sensory")])`.

All of them act on test-time buffers (see model.py), not on the trained
weights, so no retraining is involved -- except `rehabilitate`, which is the
explicit "recovery" experiment.

To add a new knob: write a dataclass with `apply` and `describe`, and add it
to REGISTRY at the bottom so YAML configs can use it.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .model import ModularRNN


def _units(model: ModularRNN, module: str | None) -> torch.Tensor:
    """Indices of a module's units, or of all units if module is None."""
    if module is None:
        return torch.arange(model.N)
    sl = model.units_of(module)
    return torch.arange(sl.start, sl.stop)


# --------------------------------------------------------------------------
# 1. Lesions
# --------------------------------------------------------------------------
@dataclass
class LesionModule:
    """Silence a random `fraction` of a region's units (1.0 = the whole region)."""

    module: str
    fraction: float = 1.0
    seed: int = 0  # which units are picked is random but reproducible

    def apply(self, model):
        idx = _units(model, self.module)
        n = int(round(self.fraction * len(idx)))
        gen = torch.Generator().manual_seed(self.seed)
        chosen = idx[torch.randperm(len(idx), generator=gen)[:n]]
        model.unit_mask[chosen] = 0.0

    def describe(self):
        if self.fraction >= 1:
            return f"Switch off the whole {self.module} region"
        return f"Switch off {self.fraction:.0%} of the {self.module} region"


@dataclass
class LesionUnits:
    """Silence a given list of units (e.g. an emergent cluster)."""

    units: list[int]
    label: str = "selected units"

    def apply(self, model):
        model.unit_mask[torch.as_tensor(self.units, dtype=torch.long)] = 0.0

    def describe(self):
        return f"Switch off {self.label} ({len(self.units)} units)"


# --------------------------------------------------------------------------
# 2. Disconnection
# --------------------------------------------------------------------------
@dataclass
class CutPathway:
    """Remove the connections from region `src` to region `dst` (a fraction of them, or all)."""

    src: str
    dst: str
    fraction: float = 1.0
    seed: int = 0

    def apply(self, model):
        s, d = model.units_of(self.src), model.units_of(self.dst)
        block = model.w_rec_scale[d, s]
        if self.fraction >= 1:
            block.zero_()
        else:
            gen = torch.Generator().manual_seed(self.seed)
            keep = (torch.rand(block.shape, generator=gen) >= self.fraction).float()
            block.mul_(keep)

    def describe(self):
        amount = "Cut" if self.fraction >= 1 else f"Cut {self.fraction:.0%} of"
        return f"{amount} the connection from {self.src} to {self.dst}"


# --------------------------------------------------------------------------
# 3-4. Noise and gain
# --------------------------------------------------------------------------
@dataclass
class Noise:
    """Multiply the random noise in a region (or everywhere) by `factor`."""

    factor: float
    module: str | None = None
    include_inputs: bool = False  # also scale the noise on the input channels

    def apply(self, model):
        model.sigma[_units(model, self.module)] *= self.factor
        if self.include_inputs:
            model.input_noise *= self.factor

    def describe(self):
        where = f"the {self.module} region" if self.module else "the whole network"
        return f"Make {where} {self.factor:g}x noisier"


@dataclass
class Gain:
    """Multiply the activation slope ('excitability') by `factor` -- a drug/neuromodulator analogue."""

    factor: float
    module: str | None = None

    def apply(self, model):
        model.gain[_units(model, self.module)] *= self.factor

    def describe(self):
        where = f"the {self.module} region" if self.module else "the whole network"
        word = "more" if self.factor > 1 else "less"
        return f"Make {where} {word} excitable (gain x{self.factor:g})"


# --------------------------------------------------------------------------
# 5. E/I balance (needs Dale's law)
# --------------------------------------------------------------------------
@dataclass
class Inhibition:
    """Scale the strength of all connections *from* inhibitory units (in a region, or everywhere)."""

    factor: float
    module: str | None = None

    def apply(self, model):
        if not model.cfg.dale:
            raise ValueError("Inhibition needs a model trained with Dale's law (model.dale: true)")
        idx = _units(model, self.module)
        inh = idx[~model.is_exc[idx]]
        model.w_rec_scale[:, inh] *= self.factor

    def describe(self):
        where = f" in {self.module}" if self.module else ""
        word = "Strengthen" if self.factor > 1 else "Weaken"
        return f"{word} inhibition{where} (x{self.factor:g})"


# --------------------------------------------------------------------------
# 6. Time constant
# --------------------------------------------------------------------------
@dataclass
class TimeConstant:
    """Multiply a region's time constant tau by `factor` (>1 = slower units)."""

    factor: float
    module: str | None = None

    def apply(self, model):
        idx = _units(model, self.module)
        # alpha = dt / tau; it must stay <= 1 for the Euler step to make sense.
        model.alpha[idx] = torch.clamp(model.alpha[idx] / self.factor, max=1.0)

    def describe(self):
        where = f"the {self.module} region" if self.module else "the whole network"
        word = "slower" if self.factor > 1 else "faster"
        return f"Make {where} {word} (time constant x{self.factor:g})"


# --------------------------------------------------------------------------
# 7. Weight scaling and synaptic pruning
# --------------------------------------------------------------------------
@dataclass
class ScaleWeights:
    """Multiply all connections *into* a region (or all connections) by `factor`."""

    factor: float
    module: str | None = None

    def apply(self, model):
        model.w_rec_scale[_units(model, self.module), :] *= self.factor

    def describe(self):
        where = f"into {self.module}" if self.module else "everywhere"
        return f"Scale connections {where} by x{self.factor:g}"


@dataclass
class Prune:
    """Delete the weakest `fraction` of existing connections (into a region, or everywhere)."""

    fraction: float
    module: str | None = None

    def apply(self, model):
        rows = _units(model, self.module)
        W = (model.effective_W_rec().detach().abs())[rows]
        exists = model.conn_mask[rows] > 0
        values = W[exists]
        if len(values) == 0 or self.fraction <= 0:
            return
        k = int(round(self.fraction * len(values)))
        if k == 0:
            return
        threshold = torch.kthvalue(values, k).values
        weak = exists & (W <= threshold)
        sub = model.w_rec_scale[rows]
        sub[weak] = 0.0
        model.w_rec_scale[rows] = sub

    def describe(self):
        where = f" into {self.module}" if self.module else ""
        return f"Delete the weakest {self.fraction:.0%} of connections{where}"


# --------------------------------------------------------------------------
# Composition helpers
# --------------------------------------------------------------------------
REGISTRY = {
    "lesion_module": LesionModule,
    "lesion_units": LesionUnits,
    "cut_pathway": CutPathway,
    "noise": Noise,
    "gain": Gain,
    "inhibition": Inhibition,
    "time_constant": TimeConstant,
    "scale_weights": ScaleWeights,
    "prune": Prune,
}
_NAME_OF = {cls: name for name, cls in REGISTRY.items()}


def from_spec(spec: dict):
    """{"type": "lesion_module", "module": "conceptual", "fraction": 0.5} -> LesionModule(...)."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in REGISTRY:
        raise KeyError(f"Unknown perturbation type '{kind}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[kind](**spec)


def to_spec(p) -> dict:
    return {"type": _NAME_OF[type(p)], **asdict(p)}


def perturbed(model: ModularRNN, perturbations) -> ModularRNN:
    """A perturbed COPY of `model`. The original is untouched."""
    m = copy.deepcopy(model)
    for p in perturbations:
        p.apply(m)
    return m


def describe_stack(perturbations) -> str:
    return " + ".join(p.describe() for p in perturbations) if perturbations else "No change (intact network)"


@dataclass
class Condition:
    """A named stack of perturbations (one column of the lesion matrix)."""

    label: str
    stack: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        return cls(d["label"], [from_spec(s) for s in d.get("stack", [])])

    def to_dict(self) -> dict:
        return {"label": self.label, "stack": [to_spec(p) for p in self.stack]}


# --------------------------------------------------------------------------
# 8. Recovery ("rehabilitation")
# --------------------------------------------------------------------------
def rehabilitate(model: ModularRNN, tasks, perturbations, steps: int, lr: float = 1e-3, batch_size: int = 64,
                 eval_every: int = 100, eval_trials: int = 128, seed: int = 0, l2_rate: float = 1e-4):
    """Apply a perturbation, then retrain the damaged copy for `steps` iterations.

    The damage stays in place during retraining (lesioned units stay silent,
    cut pathways stay cut), so the network has to find a way around it --
    like rehabilitation after a stroke. Returns (retrained copy, history),
    where history lists the accuracy per task over the retraining.
    """
    from .model import to_torch
    from .train import evaluate, masked_mse

    m = perturbed(model, perturbations)
    m.train()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    history = [{"step": 0, **evaluate(m, tasks, eval_trials)}]
    for step in range(1, steps + 1):
        batch = tasks.generate(tasks.sample_task(rng), batch_size, rng)
        out, rates = m(to_torch(batch.x))
        loss = masked_mse(out, to_torch(batch.y), to_torch(batch.loss_mask)) + l2_rate * (rates**2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            history.append({"step": step, **evaluate(m, tasks, eval_trials)})
    m.eval()
    return m, history
