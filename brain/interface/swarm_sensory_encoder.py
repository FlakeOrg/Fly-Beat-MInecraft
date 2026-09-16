"""A sensory encoder variant for FlyBrainMain only (see agent/fly_brain_main.py).

Every other bot in this project earns everything itself: its senses cover
only what it can personally observe right now, and its progress lives or
dies on its own trial and error. FlyBrainMain is a deliberate, disclosed
exception to that - the same connectome, neuron pools, and base senses as
everyone else, plus a handful of extra input channels carrying a live
summary of how the rest of the training swarm is doing. It is not part of
ES training/population sampling, so this doesn't touch what the actual
trained swarm perceives or how it's scored - it's one extra, separately
running bot with a wider view.
"""

from __future__ import annotations

import numpy as np

from interface.sensory_encoder import FEATURE_NAMES, SensoryEncoder, extract_features

SWARM_FEATURE_NAMES = [
    "swarm_progress_avg",
    "swarm_progress_best",
    "swarm_danger_frac",
]


def extract_swarm_features(swarm_summary: dict | None) -> np.ndarray:
    """`swarm_summary` is whatever agent/fly_brain_main.py's background
    poller last computed - a plain dict, not a live connection, so this
    stays a pure function. Defaults to all-zero (no swarm signal) when
    unavailable, e.g. before the first poll completes."""
    if not swarm_summary:
        return np.zeros(len(SWARM_FEATURE_NAMES))
    return np.array(
        [
            swarm_summary.get("progress_avg", 0.0),
            swarm_summary.get("progress_best", 0.0),
            swarm_summary.get("danger_frac", 0.0),
        ],
        dtype=np.float64,
    )


class SwarmSensoryEncoder(SensoryEncoder):
    """Same receptive-field scheme as the base encoder (see
    SensoryEncoder._default_weights), just over a longer feature vector -
    the base FEATURE_NAMES columns plus SWARM_FEATURE_NAMES. Deliberately
    does not call super().__init__(): the parent hardcodes
    `n_features = len(FEATURE_NAMES)` before building default weights, and
    this needs the larger count in place first."""

    def __init__(self, input_idx: np.ndarray, weights: np.ndarray | None = None, gain: float = 3.0, seed: int = 0):
        self.input_idx = np.asarray(input_idx)
        self.n_features = len(FEATURE_NAMES) + len(SWARM_FEATURE_NAMES)
        self.gain = gain
        self.weights = weights if weights is not None else self._default_weights(seed)

    def encode(self, observation: dict, n_neurons_total: int, swarm_summary: dict | None = None) -> np.ndarray:
        features = np.concatenate([extract_features(observation), extract_swarm_features(swarm_summary)])
        ext_current = np.zeros(n_neurons_total, dtype=np.float64)
        ext_current[self.input_idx] = self.gain * (self.weights @ features)
        return ext_current
