"""Same seed -> identical training; different seed -> different network."""

import torch

from brain_sandbox.config import ExperimentConfig, ModelConfig, ModuleConfig, TrainConfig
from brain_sandbox.train import train_one


def small_cfg():
    return ExperimentConfig(
        name="test",
        model=ModelConfig(modules=[ModuleConfig(n, size=16) for n in ("sensory", "memory", "conceptual", "executive")]),
        train=TrainConfig(n_iters=15, batch_size=8, eval_every=5, eval_batch=16, stop_at_accuracy=None, num_threads=1),
    )


def test_same_seed_same_result():
    a = train_one(small_cfg(), seed=3, verbose=False)
    b = train_one(small_cfg(), seed=3, verbose=False)
    assert [h["loss"] for h in a["history"]] == [h["loss"] for h in b["history"]]
    for (k, va), (_, vb) in zip(a["model"].state_dict().items(), b["model"].state_dict().items()):
        assert torch.equal(va, vb), k


def test_different_seed_different_result():
    a = train_one(small_cfg(), seed=3, verbose=False)
    b = train_one(small_cfg(), seed=4, verbose=False)
    assert not torch.equal(a["model"].W_rec_raw, b["model"].W_rec_raw)


def test_evaluation_does_not_change_training():
    """Evaluating more or less often must not change the trained weights."""
    c1, c2 = small_cfg(), small_cfg()
    c2.train.eval_every = 3
    a = train_one(c1, seed=0, verbose=False)
    b = train_one(c2, seed=0, verbose=False)
    assert torch.equal(a["model"].W_rec_raw, b["model"].W_rec_raw)
