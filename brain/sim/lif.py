"""Leaky-integrate-and-fire simulator over a weighted connectome graph.

This runs directly on whatever weighted adjacency matrix it's handed - a
synthetic random graph for now (see `demo.py`), the real hemibrain graph once
`brain/connectome/graph.py` exists (M2). The model itself is deliberately
plain, standard neuroscience: no claim to novelty here, the "real fly brain"
part of this project is the connectivity data, not the neuron model.

Convention: `weights[i, j]` is the synaptic weight from pre-synaptic neuron
`j` onto post-synaptic neuron `i` (signed: positive = excitatory, negative =
inhibitory), so synaptic input to every neuron for a given spike vector is
`weights @ spikes`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LIFParams:
    """Parameters shared by every neuron in the network.

    Defaults are in dimensionless "sim units" (dt=1.0 tick), not calibrated
    to biophysical units - M3 is about validating stability and plausible
    relative firing rates, not matching real membrane time constants.
    """

    dt: float = 1.0
    tau_mem: float = 10.0  # membrane time constant, in ticks
    v_rest: float = 0.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    refractory_steps: int = 2
    syn_scale: float = 1.0  # global scale on synaptic current, tunable knob

    # Spike-frequency adaptation: each spike bumps a per-neuron adaptation
    # current by `b_adapt`, which decays with time constant `tau_adapt` and
    # is subtracted from the neuron's drive. Without this, small recurrent
    # excitatory/inhibitory networks like the synthetic test graph tend to
    # be bistable - either silent, or (once recurrent gain is high enough
    # to propagate a signal at all) stuck in permanent self-sustained
    # firing regardless of input. Adaptation is also just real neurophysi-
    # ology (most real neurons adapt), not merely a stability patch.
    b_adapt: float = 0.5
    tau_adapt: float = 30.0

    # Short-term synaptic depression: distinct from (and complementary to)
    # spike-frequency adaptation above. Adaptation suppresses a neuron's own
    # excitability after it fires; depression instead weakens a neuron's
    # *outgoing* synapses after it fires, recovering with time constant
    # `tau_depression`. This matters for networks with strong reverberating
    # loops between populations (found necessary for the real hemibrain
    # graph, much denser than the synthetic test graph - see
    # connectome/README.md): per-neuron adaptation alone wasn't enough to
    # stop those loops from settling into permanent self-sustained activity,
    # since depression acts directly on the connections doing the
    # reverberating rather than only on the neurons receiving it.
    depression_frac: float = 0.0  # 0 = disabled (backwards compatible default)
    tau_depression: float = 50.0


class LIFNetwork:
    """A leaky-integrate-and-fire network over a fixed weighted graph."""

    def __init__(self, weights: np.ndarray | sp.spmatrix, params: LIFParams | None = None):
        if weights.shape[0] != weights.shape[1]:
            raise ValueError(f"weights must be square, got {weights.shape}")
        self.weights = weights
        self.n = weights.shape[0]
        self.params = params or LIFParams()
        self.reset()

    def reset(self) -> None:
        p = self.params
        self.v = np.full(self.n, p.v_rest, dtype=np.float64)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.last_spikes = np.zeros(self.n, dtype=np.float64)
        self.adaptation = np.zeros(self.n, dtype=np.float64)
        self.avail = np.ones(self.n, dtype=np.float64)

    def step(self, ext_current: np.ndarray) -> np.ndarray:
        """Advance the network by one tick given external injected current.

        Returns the binary spike vector (shape (n,), dtype float64) for
        this tick.
        """
        p = self.params
        if ext_current.shape != (self.n,):
            raise ValueError(f"ext_current must have shape ({self.n},), got {ext_current.shape}")

        syn_current = self.weights @ (self.last_spikes * self.avail) * p.syn_scale
        syn_current = np.asarray(syn_current).reshape(self.n)

        not_refractory = self.refractory <= 0
        dv = (p.dt / p.tau_mem) * (
            -(self.v - p.v_rest) + syn_current + ext_current - self.adaptation
        )
        self.v = np.where(not_refractory, self.v + dv, p.v_reset)

        spikes = (self.v >= p.v_thresh) & not_refractory
        self.v[spikes] = p.v_reset
        self.refractory[spikes] = p.refractory_steps
        self.refractory[~spikes & ~not_refractory] -= 1
        self.refractory = np.maximum(self.refractory, 0)

        self.adaptation += (p.dt / p.tau_adapt) * (-self.adaptation)
        self.adaptation[spikes] += p.b_adapt

        self.avail += (p.dt / p.tau_depression) * (1.0 - self.avail)
        self.avail[spikes] *= 1.0 - p.depression_frac

        self.last_spikes = spikes.astype(np.float64)
        return self.last_spikes

    def run(self, ext_current_trace: np.ndarray) -> np.ndarray:
        """Run for T ticks given a (T, n) external-current trace.

        Returns the (T, n) spike trace.
        """
        t_steps, n = ext_current_trace.shape
        if n != self.n:
            raise ValueError(f"ext_current_trace must have {self.n} columns, got {n}")
        spikes = np.zeros((t_steps, self.n), dtype=np.float64)
        for t in range(t_steps):
            spikes[t] = self.step(ext_current_trace[t])
        return spikes
