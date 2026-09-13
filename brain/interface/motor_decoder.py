"""Descending-neuron firing rates -> a discrete Minecraft action.

Trainable boundary, mirror image of sensory_encoder.py. For M4 the weights
are hand-built (an even, arbitrary partition of the output pool across
actions) just to validate that spikes reaching the output pool turn into
sensible-looking bot commands; M5 replaces `_default_weights` with
something learned.
"""

from __future__ import annotations

import numpy as np

# Movement/physical primitives, then the task verbs.
#
# The task verbs are what make self-directed play possible at all: with only
# movement and mining in the action space, a bot cannot reach even the first
# rung of the progression ladder (a crafting table) no matter how long it
# trains, because "craft" is not something it is able to attempt. Each verb
# is only an attempt - it fails harmlessly when the materials or a nearby
# table aren't there. *When* to use them is not encoded anywhere; that's
# what the reward ladder has to teach.
MOVEMENT_ACTIONS = ["forward", "left", "right", "jump", "attack", "mine_ahead"]

TASK_ACTIONS = [
    "craft_planks",
    "craft_sticks",
    "craft_table",
    "place_table",
    "craft_wooden_pickaxe",
    "craft_stone_pickaxe",
    "craft_furnace",
    "place_furnace",
    "smelt_iron",
    "craft_iron_pickaxe",
]

ACTIONS = MOVEMENT_ACTIONS + TASK_ACTIONS


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

    def scores(self, spike_window: np.ndarray) -> np.ndarray:
        rates = spike_window[:, self.output_idx].mean(axis=0)
        return self.weights @ rates

    def decode(self, spike_window: np.ndarray) -> dict[str, bool]:
        """`spike_window`: (T, n_total) binary spike trace over the last T
        sim ticks. Returns which of ACTIONS are "on" for this control step.
        """
        scores = self.scores(spike_window)
        return {action: bool(scores[i] > self.threshold) for i, action in enumerate(ACTIONS)}

    def decode_task_action(self, spike_window: np.ndarray) -> str | None:
        """The single strongest task verb above threshold, or None.

        Only one is allowed per control step: task verbs cost a real bridge
        round trip each, and firing every one that happens to cross
        threshold would spend most of an episode on failing craft attempts
        instead of playing. Forcing a choice also makes the credit
        assignment cleaner - exactly one deliberate act per step."""
        scores = self.scores(spike_window)
        best, best_score = None, self.threshold
        for i, action in enumerate(ACTIONS):
            if action in TASK_ACTIONS and scores[i] > best_score:
                best, best_score = action, scores[i]
        return best
