"""Synthetic stand-in connectome for validating the simulator (M3).

This is NOT the real fly brain - it's a random sparse directed graph used
only to sanity-check that `LIFNetwork` is stable and that a signal injected
into an "input pool" can actually reach an "output pool". Once
`brain/connectome/graph.py` (M2) can build the real weighted hemibrain graph,
callers switch to that instead; this module goes away or stays only for
tests.

Neurons obey Dale's law (each neuron is either excitatory or inhibitory on
all of its outgoing synapses), matching real neurobiology and giving the
synthetic graph at least that much resemblance to a real one.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def make_synthetic_graph(
    n_neurons: int = 200,
    n_input: int = 20,
    n_output: int = 20,
    p_connect: float = 0.03,
    inhibitory_frac: float = 0.2,
    inhibitory_strength: float = 2.0,
    weight_scale: float = 0.5,
    seed: int | None = 0,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build a random sparse directed graph with Dale's-law signed weights.

    `inhibitory_strength` multiplies the magnitude of inhibitory synapses
    relative to excitatory ones. A plain 50/50 magnitude split with only
    ~20% inhibitory neurons (roughly the real excitatory/inhibitory ratio)
    leaves net recurrent drive excitation-dominated, which pushes a random
    recurrent network into runaway self-sustained firing that no longer
    tracks its input. Real fast inhibition (e.g. GABAergic) is disproportion-
    ately strong per synapse, which is what keeps balanced cortical-like
    networks from doing that - so we model it that way too, rather than
    hand-tuning the fraction to a razor's-edge value that would be fragile
    to any other parameter change.

    Returns (weights, input_idx, output_idx) where `weights[i, j]` is the
    synapse strength from pre-synaptic neuron j to post-synaptic neuron i,
    `input_idx` is the index array for the first `n_input` neurons, and
    `output_idx` is the index array for the last `n_output` neurons.
    """
    if n_input + n_output > n_neurons:
        raise ValueError("n_input + n_output must not exceed n_neurons")

    rng = np.random.default_rng(seed)

    adjacency = rng.random((n_neurons, n_neurons)) < p_connect
    np.fill_diagonal(adjacency, False)  # no self-synapses

    magnitudes = rng.lognormal(mean=0.0, sigma=0.5, size=(n_neurons, n_neurons))
    is_inhibitory_pre = rng.random(n_neurons) < inhibitory_frac
    sign_per_pre = np.where(is_inhibitory_pre, -1.0, 1.0)
    strength_per_pre = np.where(is_inhibitory_pre, inhibitory_strength, 1.0)

    weights = adjacency * magnitudes * (sign_per_pre * strength_per_pre)[np.newaxis, :] * weight_scale
    weights_sparse = sp.csr_matrix(weights)

    input_idx = np.arange(0, n_input)
    output_idx = np.arange(n_neurons - n_output, n_neurons)
    return weights_sparse, input_idx, output_idx
