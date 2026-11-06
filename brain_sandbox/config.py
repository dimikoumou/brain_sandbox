"""Configuration objects, loaded from YAML.

Every run is described completely by one `ExperimentConfig`. We use plain
dataclasses (rather than raw dicts) so that a typo in a YAML file raises an
error instead of being silently ignored -- a silent typo in a noise level would
invalidate an experiment without anyone noticing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class ModuleConfig:
    """One named "region": a block of units with its own dynamics."""

    name: str
    size: int = 64
    tau_ms: float = 100.0  # membrane-like time constant of the rate units
    noise: float = 0.05  # sigma_rec: std of private recurrent noise
    gain: float = 1.0  # slope multiplier inside the activation function
    frac_exc: float = 0.8  # fraction of excitatory units (only used with Dale's law)


@dataclass
class PathwayConfig:
    """A directed connection between two modules, src -> dst."""

    src: str
    dst: str
    density: float = 0.2  # fraction of possible synapses that exist


@dataclass
class ModelConfig:
    modules: list[ModuleConfig] = field(
        default_factory=lambda: [
            ModuleConfig("sensory"),
            ModuleConfig("memory"),
            ModuleConfig("conceptual"),
            ModuleConfig("executive"),
        ]
    )
    # Default wiring from the spec. "A<->B" is written as two directed pathways.
    # conceptual -> sensory is the top-down feedback pathway.
    pathways: list[PathwayConfig] = field(
        default_factory=lambda: [
            PathwayConfig("sensory", "memory"),
            PathwayConfig("memory", "sensory"),
            PathwayConfig("sensory", "conceptual"),
            PathwayConfig("conceptual", "sensory"),
            PathwayConfig("conceptual", "executive"),
            PathwayConfig("executive", "conceptual"),
            PathwayConfig("memory", "executive"),
            PathwayConfig("executive", "memory"),
        ]
    )
    within_density: float = 1.0  # density of connections inside a module
    # Which modules each input group projects to. Groups: fixation, mod1,
    # mod2, tono, rule (see tasks/layout.py).
    input_routes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "fixation": ["sensory"],
            "mod1": ["sensory"],
            "mod2": ["sensory"],
            "tono": ["sensory"],
            "rule": ["executive"],
        }
    )
    output_module: str = "executive"  # the readout only sees this module
    dale: bool = True
    # With Dale's law: only excitatory units send connections *between*
    # modules (as for long-range cortical projections). Inhibition stays local.
    long_range_excitatory_only: bool = True
    activation: str = "softplus"  # "softplus" or "relu"
    dt_ms: float = 20.0
    input_noise: float = 0.01  # std of noise added to every input channel
    init_spectral_radius: float = 1.0
    seed_offset: int = 0  # added to the run seed for weight/mask init


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
@dataclass
class TaskConfig:
    names: list[str] = field(default_factory=lambda: ["go", "delaygo"])
    n_ring: int = 16  # units per direction-tuned ("ring") input modality
    n_tono: int = 0  # tonotopic input units (pitch tasks; phase 2)
    n_choice: int = 0  # extra binary-choice output units (pitch "same/different")
    # Pitch: notes are integers 0..n_semitones above f_low_hz (default C4 =
    # 261.63 Hz, so 24 semitones = two octaves, C4..C6).
    n_semitones: int = 24
    f_low_hz: float = 261.63
    tono_width: float = 0.75  # width of a tone's bump on the tonotopic map, in semitones
    tuning_width: float | None = None  # radians; default 2*pi/n_ring
    # Per-task overrides of the timing/stimulus parameters defined at the top
    # of each task file, e.g. {"delaygo": {"delay_ms": [400, 1600]}}.
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Relative sampling weight of each task during training (default uniform).
    weights: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    n_iters: int = 3000
    batch_size: int = 64
    lr: float = 3e-3  # 1e-3 was too slow for the modular model (README, "Phase 1 notes")
    grad_clip: float = 1.0
    # "Pressure" knobs (see README): L2 penalties on firing rates and weights.
    l2_rate: float = 1e-4
    l2_weight: float = 1e-5
    eval_every: int = 100
    eval_batch: int = 256  # trials per task per evaluation
    stop_at_accuracy: float | None = 0.95  # criterion: every task at or above this...
    # ...then keep practising this many extra iterations, so the network is not
    # left only *just* passing (phase 1 showed a thin fixation margin).
    practice_after_criterion: int = 0
    num_threads: int = 4


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seeds: list[int] = field(default_factory=lambda: [0])
    model: ModelConfig = field(default_factory=ModelConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""


# --------------------------------------------------------------------------
# YAML <-> dataclass conversion (strict: unknown keys are an error)
# --------------------------------------------------------------------------
def _build(cls, data: dict[str, Any]):
    if data is None:
        return cls()
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        if cls is ModelConfig and key == "modules":
            value = [_build(ModuleConfig, m) for m in value]
        elif cls is ModelConfig and key == "pathways":
            value = [_build(PathwayConfig, p) for p in value]
        elif cls is ExperimentConfig and key == "model":
            value = _build(ModelConfig, value)
        elif cls is ExperimentConfig and key == "tasks":
            value = _build(TaskConfig, value)
        elif cls is ExperimentConfig and key == "train":
            value = _build(TrainConfig, value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return _build(ExperimentConfig, yaml.safe_load(f))


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, data)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
