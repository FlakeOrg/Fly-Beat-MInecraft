"""Descending-neuron firing rates -> discrete Minecraft and speech actions.

Speech outputs are part of the decoder interface, so the simulated brain can
select a speech intent.  SpeechController turns those intents into text while
keeping cooldowns and server-facing policy outside the connectome.
"""

from __future__ import annotations

import numpy as np

ACTIONS = [
    "forward", "left", "right", "jump", "attack", "mine_ahead", "place_ahead",
    "say_hello", "say_hungry", "say_hurt", "say_help", "say_found", "say_made",
]


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
        """Return motor and speech intents from a (T, n_total) spike trace."""
        rates = spike_window[:, self.output_idx].mean(axis=0)
        scores = self.weights @ rates
        return {action: bool(scores[i] > self.threshold) for i, action in enumerate(ACTIONS)}
