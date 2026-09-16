"""Descending-neuron firing rates -> a discrete Minecraft action.

Trainable boundary, mirror image of sensory_encoder.py. For M4 the weights
are hand-built (an even, arbitrary partition of the output pool across
actions) just to validate that spikes reaching the output pool turn into
sensible-looking bot commands; M5 replaces `_default_weights` with
something learned.
"""

from __future__ import annotations

import numpy as np

ACTIONS = ["forward", "left", "right", "jump", "attack", "mine_ahead", "place_ahead"]


class MotorDecoder:
    def __init__(self, output_idx: np.ndarray, weights: np.ndarray | None = None, threshold: float = 0.15, seed: int = 0):
        self.output_idx = np.asarray(output_idx)
        self.threshold = threshold
        self.weights = weights if weights is not None else self._default_weights(seed)

    def _default_weights(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.output_idx))
        groups = np.array_split(order, len(ACTIONS))
        weights = np.zeros((len(ACTIONS), len(self.output_idx)))
        for action_i, idxs in enumerate(groups):
            if len(idxs) > 0:
                weights[action_i, idxs] = 1.0 / len(idxs)
        return weights

    def decode(self, spike_window: np.ndarray) -> dict[str, bool]:
        """`spike_window`: (T, n_total) binary spike trace over the last T
        sim ticks. Returns which of ACTIONS are "on" for this control step.
        """
        rates = spike_window[:, self.output_idx].mean(axis=0)
        scores = self.weights @ rates
        return {action: bool(scores[i] > self.threshold) for i, action in enumerate(ACTIONS)}
