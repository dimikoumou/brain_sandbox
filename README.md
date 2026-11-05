# Brain Sandbox

A small artificial "brain" to train on several cognitive tasks, then **tune**
(change parameters) and **damage** (remove regions, cut connections, add
noise), to see which abilities break, which survive, and whether any get better.

> **What this is and isn't.** This is a *hypothesis generator*. It is a small
> recurrent neural network with four groups of units that we *call* regions.
> Results here say "what this model needs, or does when perturbed". They do
> not say what humans can do. Any interesting result is at most a reason to
> look for the same thing in real data.

## Status

| Phase | Content | State |
|---|---|---|
| 1 | Model, Go + Delay-Go, training loop, tests | **done** |
| 2 | Full task set, pitch tasks, multi-seed training, the "is absolute pitch discarded?" checkpoint | next |
| 3 | Perturbation library, lesion matrix, dissociation detection | |
| 4 | Emergent clusters, latent-pitch probes | |
| 5 | Streamlit dashboard | |
| 6 | Worked "release" experiment notebook | |

## The science in plain language

**Why a recurrent network?** Yang et al. (2019, *Nature Neuroscience*) trained
one recurrent neural network (RNN) on about 20 cognitive tasks, the kind used
in monkey experiments: look toward a light, look away from it, remember where
it was, decide which of two stimuli is stronger, and so on. Without being told
to, the network split itself into groups of units that each served a subset of
the tasks, a bit like brain areas. Driscoll, Shenoy & Sussillo (2024) went
further: switching off ("lesioning") one group only hurt the tasks that use
the computation that group performs.

**What we add.** Instead of only letting groups emerge, we also *design*
regions and can compare the two:

* `sensory`: receives every stimulus (including tones, from phase 2)
* `memory`: holds information across delays
* `conceptual`: a top-down, abstraction module. It is the model's analogue
  of the anterior temporal lobe, by *role in the wiring*, nothing more.
* `executive`: receives the task instruction and drives the output

A "flat" model (one unstructured block of the same size) is the control.

**The motivating question.** Snyder's "release" hypothesis (Snyder 2009,
*Phil. Trans. R. Soc. B*) proposes that inhibiting the left anterior temporal
lobe can expose raw low-level detail that top-down processing normally
discards, possibly including absolute pitch. We can test a model version of
this: train a network on *relative* pitch (recognise a melody in any key),
then weaken `conceptual` or its feedback to `sensory`, and measure whether
absolute pitch becomes *more decodable* from the network's activity. A
null result is a result, and we will report it as one.

## The model

Each unit has a firing rate *r* that follows

    tau * dr/dt = -r + f( gain * (W_rec r + W_in u + b + noise) )

* `f` is softplus (a smooth ReLU: rates are never negative).
* `tau` (default 100 ms) sets how fast a unit responds; it can differ per
  region. Simulated with Euler steps of `dt = 20 ms`.
* `W_rec` (unit → unit) is masked: only connections listed as pathways exist.
  Default wiring: sensory↔memory, sensory↔conceptual, conceptual↔executive,
  memory↔executive. conceptual→sensory is the top-down feedback.

### Design choices you may be asked to defend

* **Dale's law (on by default).** Each unit is either excitatory (80%) or
  inhibitory (20%): all its outgoing weights share one sign. Real neurons
  work this way, and it makes "weaken inhibition" a meaningful experiment.
  Implemented as `W = |W_raw| * sign(presynaptic unit)`, which training cannot
  break (tested).
* **Long-range connections are excitatory only.** Between regions only
  excitatory units project, as with long-range cortical projections;
  inhibition is local. As a consequence, cutting a pathway removes only
  excitatory drive.
* **Balanced initialisation.** Inhibitory weights start 4× stronger than
  excitatory ones (there are 4× fewer inhibitory units), so the untrained
  network is not runaway-excitatory. The whole matrix is then scaled to
  spectral radius 1.0.
* **Noise scaled by √(2/α).** This keeps the noise level the same if you
  change `dt` (the standard way to discretise continuous noise).
* **Inputs are routed.** Stimuli enter `sensory` only; the task rule enters
  `executive` only; the readout reads `executive` only. So every task *has* to
  route information through the region graph, otherwise lesions would be
  meaningless. In the flat model everything connects to the single block.
* **Lesion = zero output.** A lesioned unit's rate is forced to exactly 0 at
  every step, so it sends nothing anywhere (tested). Zeroing only its input
  would leave the bias and noise still driving it.
* **Knobs are buffers, not weights.** Lesion masks, τ, noise, gain and a
  weight multiplier are stored separately from the trained weights, so any
  perturbation can be applied to (and removed from) a trained network without
  retraining.
* **Stricter scoring than Yang et al.** A trial counts as correct only if
  the network *held fixation until the go cue* and then reported the right
  direction (within 36°). Yang et al. only checked the last time step, which
  would hide "responds too early", a likely lesion effect.
* **Seeds are the unit of replication.** Every result is averaged over
  independently trained networks (default 5 seeds), with 95% confidence
  intervals over seeds. Trials from one network are not independent evidence.
* **Pressure knobs.** `train.l2_rate` and `train.l2_weight` penalise high
  activity and large weights. Without some pressure a network has no reason to
  throw information away, which matters for the absolute-pitch question.

## Tasks (phase 1)

Directions are shown as a bump of activity over 16 direction-tuned input units
(on one of two "modalities", chosen at random). Before the go cue the network
must keep its fixation output high; when the fixation cue disappears it must
drop fixation and report a direction on 16 output units.

* **Go**: the stimulus appears and stays on; respond to it.
* **Delay-Go**: the stimulus flashes, disappears for 0.2–1.6 s, then respond
  to the remembered direction.

Timings and stimulus ranges are at the top of `brain_sandbox/tasks/yang.py` and
can be overridden in the config.

## How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                    # run the tests (~15 s)

# train 5 seeds of the modular model (results go to runs/<timestamp>_phase1_modular/)
python -m brain_sandbox train --config configs/phase1_modular.yaml

# quick try: one seed, fewer iterations
python -m brain_sandbox train --config configs/phase1_modular.yaml --seeds 0 --set train.n_iters=1000

# evaluate on fresh trials (mean and 95% CI over seeds), plus example-trial plots
python -m brain_sandbox evaluate --run runs/<folder> --plots
```

Every run folder stores the full resolved config, the git commit (and whether
the working tree had uncommitted changes), library versions and, per seed, the
weights, a metrics CSV and the learning curve.

## Code map

```
brain_sandbox/
  config.py      all settings (dataclasses; unknown YAML keys are an error)
  model.py       the modular rate RNN
  tasks/         common.py (layouts, ring codes, trial builder), yang.py (tasks)
  metrics.py     trial scoring
  train.py       training loop, evaluation, loading
  runs.py        timestamped run folders
  stats.py       mean and 95% CI over seeds
  plotting.py    learning curves, example trials
  cli.py         command line
configs/         YAML experiment configs
tests/           pytest
```

## Limitations

* A rate model: no spikes, no dendrites, no synaptic dynamics. Each unit
  stands for a population, not a neuron.
* No development, no ongoing plasticity after training (except the explicit
  "rehabilitation" retraining in phase 3), no neuromodulator systems (the
  gain knob is a crude stand-in), no glia, no body.
* "Regions" are defined by wiring only. Calling one `conceptual` does not make
  it conceptual; whether it ends up doing anything abstract is something to
  *measure*, not assume.
* Trained with backpropagation through time, which brains do not do. What we
  study is the solution the network ends up with, not how it was learned.
* Small (256 units) and trained on a handful of tasks. Effects that need scale
  or richer inputs will not appear here.

## References

* Yang, Joglekar, Song, Newsome & Wang (2019). Task representations in neural
  networks trained to perform many cognitive tasks. *Nature Neuroscience* 22, 297–306.
* Driscoll, Shenoy & Sussillo (2024). Flexible multitask computation in
  recurrent networks utilizes shared dynamical motifs. *Nature Neuroscience* 27, 1349–1363.
* Snyder (2009). Explaining and inducing savant skills: privileged access to
  lower level, less-processed information. *Phil. Trans. R. Soc. B* 364, 1399–1405.
* Song, Yang & Wang (2016). Training excitatory-inhibitory recurrent neural
  networks for cognitive tasks. *PLoS Comput. Biol.* 12, e1004792.
