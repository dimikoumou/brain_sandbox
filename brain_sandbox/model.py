"""The modular, continuous-time leaky rate RNN.

Continuous-time equation, for the vector of firing rates r:

    tau * dr/dt = -r + f( gain * (W_rec r + W_in u + b + noise) )

Euler discretisation with step dt and alpha = dt / tau (per unit):

    r[t+1] = (1 - alpha) * r[t] + alpha * f( gain * (W_rec r[t] + W_in u[t] + b + sqrt(2/alpha) * sigma * xi) )

where xi ~ N(0, 1) fresh at every step. The sqrt(2/alpha) factor makes the
noise level independent of dt (it is the Euler-Maruyama discretisation of
white noise, as in Yang et al. 2019 / Song et al. 2016).

Design: "trained parameters" vs. "state knobs"
-----------------------------------------------
Trained parameters (nn.Parameter): W_rec_raw, W_in, b, W_out, b_out.

Everything a perturbation might change is stored as a *buffer* -- a tensor
that is saved with the model but not trained:

    unit_mask        (N,)   1 = unit alive, 0 = lesioned (its rate is forced to 0)
    w_rec_scale      (N,N)  multiplies the recurrent weights (pathway cuts, pruning, E/I scaling)
    alpha            (N,)   dt / tau per unit
    sigma            (N,)   recurrent noise std per unit
    gain             (N,)   activation slope per unit

A perturbation (phase 3) will be a small object that edits these buffers on a
copy of a trained model. Nothing about the trained weights changes, so the
same trained network can be tested under any stack of perturbations.

The connectivity mask (`conn_mask`) is also a buffer, but it is part of the
architecture (which synapses exist at all), not a perturbation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .config import ModelConfig


class ModularRNN(nn.Module):
    def __init__(self, cfg: ModelConfig, input_groups: dict[str, int], n_out: int, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        gen = torch.Generator().manual_seed(seed + cfg.seed_offset)

        # ---- unit bookkeeping: which units belong to which module ----------
        self.module_names = [m.name for m in cfg.modules]
        if len(set(self.module_names)) != len(self.module_names):
            raise ValueError("Module names must be unique")
        self.module_slices: dict[str, slice] = {}
        start = 0
        for m in cfg.modules:
            self.module_slices[m.name] = slice(start, start + m.size)
            start += m.size
        self.N = start

        # ---- E/I identity (Dale's law) -------------------------------------
        # Within each module, the first frac_exc of units are excitatory.
        is_exc = torch.ones(self.N, dtype=torch.bool)
        if cfg.dale:
            for m in cfg.modules:
                sl = self.module_slices[m.name]
                n_exc = int(round(m.frac_exc * m.size))
                is_exc[sl.start + n_exc : sl.stop] = False
        self.register_buffer("is_exc", is_exc)
        # Sign of each unit's *outgoing* weights: +1 excitatory, -1 inhibitory.
        self.register_buffer("dale_sign", torch.where(is_exc, 1.0, -1.0))

        # ---- connectivity mask (which synapses exist) ----------------------
        self.register_buffer("conn_mask", self._build_conn_mask(gen))

        # ---- per-unit dynamical knobs --------------------------------------
        def per_unit(attr):
            v = torch.empty(self.N)
            for m in cfg.modules:
                v[self.module_slices[m.name]] = getattr(m, attr)
            return v

        self.register_buffer("alpha", cfg.dt_ms / per_unit("tau_ms"))
        self.register_buffer("sigma", per_unit("noise"))
        self.register_buffer("gain", per_unit("gain"))
        self.register_buffer("unit_mask", torch.ones(self.N))
        self.register_buffer("w_rec_scale", torch.ones(self.N, self.N))
        self.input_noise = cfg.input_noise
        if (self.alpha > 1).any():
            raise ValueError("dt must be smaller than every tau (alpha = dt/tau <= 1)")

        # ---- trained parameters --------------------------------------------
        self.register_buffer("in_mask", self._build_input_mask(input_groups))
        self.n_in = self.in_mask.shape[1]
        self.W_rec_raw = nn.Parameter(self._init_recurrent(gen))
        self.W_in = nn.Parameter(torch.randn(self.N, self.n_in, generator=gen) / math.sqrt(self.n_in))
        self.b = nn.Parameter(torch.zeros(self.N))

        out_sl = self.module_slices[cfg.output_module]
        self.out_slice = out_sl
        n_read = out_sl.stop - out_sl.start
        self.W_out = nn.Parameter(torch.randn(n_out, n_read, generator=gen) / math.sqrt(n_read))
        self.b_out = nn.Parameter(torch.zeros(n_out))

        if cfg.activation == "softplus":
            self.act = nn.functional.softplus
        elif cfg.activation == "relu":
            self.act = torch.relu
        else:
            raise ValueError(f"Unknown activation {cfg.activation}")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_conn_mask(self, gen: torch.Generator) -> torch.Tensor:
        """mask[post, pre] = 1 if the synapse pre -> post exists."""
        cfg = self.cfg
        mask = torch.zeros(self.N, self.N)

        def random_block(n_post, n_pre, density):
            return (torch.rand(n_post, n_pre, generator=gen) < density).float()

        for m in cfg.modules:  # within-module recurrence
            sl = self.module_slices[m.name]
            mask[sl, sl] = random_block(m.size, m.size, cfg.within_density)

        for p in cfg.pathways:  # between-module pathways
            for name in (p.src, p.dst):
                if name not in self.module_slices:
                    raise KeyError(f"Pathway {p.src}->{p.dst} names unknown module '{name}'")
            src, dst = self.module_slices[p.src], self.module_slices[p.dst]
            block = random_block(dst.stop - dst.start, src.stop - src.start, p.density)
            if cfg.dale and cfg.long_range_excitatory_only:
                block = block * self.is_exc[src].float()[None, :]  # only E units project out
            mask[dst, src] = block

        mask.fill_diagonal_(0.0)  # no self-connections (autapses)
        return mask

    def _build_input_mask(self, input_groups: dict[str, int]) -> torch.Tensor:
        """in_mask[unit, input channel] = 1 if that input reaches that unit."""
        n_in = sum(input_groups.values())
        mask = torch.zeros(self.N, n_in)
        start = 0
        for group, size in input_groups.items():
            for module in self.cfg.input_routes.get(group, []):
                mask[self.module_slices[module], start : start + size] = 1.0
            start += size
        return mask

    def _init_recurrent(self, gen: torch.Generator) -> torch.Tensor:
        """Random initial weights, scaled so the effective W_rec has a chosen spectral radius.

        With Dale's law the raw parameter is stored as a magnitude and the sign
        comes from the presynaptic unit (see `effective_W_rec`). Inhibitory
        weights start 4x larger (for 80/20) so that, on average, each unit
        receives balanced excitation and inhibition -- otherwise the network
        starts out runaway-excitatory.
        """
        W = torch.randn(self.N, self.N, generator=gen)
        if self.cfg.dale:
            W = W.abs()
            for m in self.cfg.modules:
                sl = self.module_slices[m.name]
                inh = ~self.is_exc[sl]
                if inh.any():
                    W[:, sl.start + torch.nonzero(inh).flatten()] *= m.frac_exc / (1 - m.frac_exc)
        W_eff = self._apply_structure(W)
        radius = torch.linalg.eigvals(W_eff).abs().max().item()
        if radius > 0:
            W = W * (self.cfg.init_spectral_radius / radius)
        return W

    def _apply_structure(self, W_raw: torch.Tensor) -> torch.Tensor:
        """Raw parameter -> actual weights: enforce sign (Dale) and existing synapses."""
        if self.cfg.dale:
            # abs() rather than relu(): a weight that hits zero can still recover
            # under training instead of being stuck at zero forever.
            W_raw = W_raw.abs() * self.dale_sign[None, :]
        return W_raw * self.conn_mask

    # ------------------------------------------------------------------
    # Effective weights (what the dynamics actually use)
    # ------------------------------------------------------------------
    def effective_W_rec(self) -> torch.Tensor:
        return self._apply_structure(self.W_rec_raw) * self.w_rec_scale

    def effective_W_in(self) -> torch.Tensor:
        # Inputs are not neurons of the network, so Dale's law does not apply to them.
        return self.W_in * self.in_mask

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network on inputs x (T, B, n_in).

        Returns (outputs (T, B, n_out), rates (T, B, N)).
        `noise=False` switches off input and recurrent noise (used in tests).
        """
        T, B, _ = x.shape
        W_rec = self.effective_W_rec()
        W_in = self.effective_W_in()
        alpha, gain, mask = self.alpha, self.gain, self.unit_mask
        noise_std = torch.sqrt(2.0 / alpha) * self.sigma  # dt-independent noise level

        if noise and self.input_noise > 0:
            x = x + self.input_noise * torch.randn_like(x)
        drive_in = x @ W_in.T + self.b  # input currents for all time steps at once

        r = torch.zeros(B, self.N, device=x.device)
        rates = []
        for t in range(T):
            current = r @ W_rec.T + drive_in[t]
            if noise:
                current = current + noise_std * torch.randn_like(current)
            r = (1 - alpha) * r + alpha * self.act(gain * current)
            # Lesion: a masked unit has exactly zero output at every step, so it
            # sends nothing to any other unit or to the readout.
            r = r * mask
            rates.append(r)
        rates = torch.stack(rates)  # (T, B, N)
        out = rates[..., self.out_slice] @ self.W_out.T + self.b_out
        return out, rates

    # Convenience --------------------------------------------------------
    def units_of(self, module: str) -> slice:
        return self.module_slices[module]

    def describe(self) -> str:
        lines = [f"ModularRNN: {self.N} units, Dale={self.cfg.dale}, activation={self.cfg.activation}"]
        W = self.effective_W_rec().detach()
        for m in self.cfg.modules:
            sl = self.module_slices[m.name]
            n_e = int(self.is_exc[sl].sum())
            lines.append(
                f"  {m.name:<11} {m.size:4d} units ({n_e} E / {m.size - n_e} I), tau={m.tau_ms} ms, noise={m.noise}, gain={m.gain}"
            )
        lines.append("  pathways (number of existing synapses):")
        for p in self.cfg.pathways:
            n = int(self.conn_mask[self.module_slices[p.dst], self.module_slices[p.src]].sum())
            lines.append(f"    {p.src:>11} -> {p.dst:<11} {n}")
        radius = torch.linalg.eigvals(W).abs().max().item()
        lines.append(f"  spectral radius of W_rec: {radius:.2f}")
        return "\n".join(lines)


def to_torch(a: np.ndarray, device="cpu") -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float32, device=device)
