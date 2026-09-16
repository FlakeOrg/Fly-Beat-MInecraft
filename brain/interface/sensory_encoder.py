"""Game state -> injected current into the connectome's input neuron pool.

This is the trainable boundary between Minecraft and the fixed connectome
(see ../../docs/architecture.md). For M4 the weights are hand-built, just to
validate that a real observation can drive the network at all; M5 replaces
`_default_weights` with something learned (evolution strategy) against a
survival/progress reward.
"""

from __future__ import annotations

import math

import numpy as np

FEATURE_NAMES = [
    "bias",
    "health_frac",
    "food_frac",
    "hostile_near",
    "entity_near",
    "block_ahead",
    "has_placeable",
    "is_falling",
    "on_ground",
]

HOSTILE_MOB_NAMES = {
    "zombie",
    "skeleton",
    "creeper",
    "spider",
    "cave_spider",
    "enderman",
    "witch",
    "phantom",
    "drowned",
    "husk",
    "stray",
    "pillager",
    "vindicator",
    "ravager",
    "blaze",
    "ghast",
    "piglin",
    "hoglin",
    "wither_skeleton",
    "ender_dragon",
}

ENTITY_NEAR_RADIUS = 8.0
HOSTILE_NEAR_RADIUS = 6.0

PLACEABLE_ITEM_NAMES = {
    "dirt", "grass_block", "cobblestone", "stone", "sand", "gravel",
    "oak_planks", "spruce_planks", "birch_planks", "jungle_planks",
    "acacia_planks", "dark_oak_planks", "mangrove_planks", "cherry_planks",
    "netherrack", "end_stone", "bricks", "glass", "crafting_table",
}


def _facing_offset(yaw: float) -> tuple[int, int]:
    """Rounds the bot's horizontal facing direction to a unit grid offset,
    using Mineflayer's yaw convention (yaw=0 faces +Z, increasing yaw turns
    towards -X)."""
    dx = -math.sin(yaw)
    dz = math.cos(yaw)
    return int(round(dx)), int(round(dz))


def extract_features(observation: dict) -> np.ndarray:
    """Turns a bot-bridge observation dict into a fixed-size feature vector."""
    health_frac = observation["health"] / 20.0
    food_frac = observation["food"] / 20.0

    entities = observation["nearbyEntities"]
    entity_near = 1.0 if any(e["distance"] <= ENTITY_NEAR_RADIUS for e in entities) else 0.0
    hostile_near = 1.0 if any(
        e["name"] in HOSTILE_MOB_NAMES and e["distance"] <= HOSTILE_NEAR_RADIUS for e in entities
    ) else 0.0

    ahead_x, ahead_z = _facing_offset(observation["yaw"])
    block_ahead = 1.0 if any(
        b["x"] == ahead_x and b["z"] == ahead_z and b["y"] in (0, 1) for b in observation["nearbyBlocks"]
    ) else 0.0
    has_placeable = 1.0 if any(
        item.get("name") in PLACEABLE_ITEM_NAMES and item.get("count", 0) > 0
        for item in observation.get("inventory", [])
    ) else 0.0

    velocity_y = observation["velocity"]["y"]
    is_falling = 1.0 if (velocity_y < -0.1 and not observation["onGround"]) else 0.0
    on_ground = 1.0 if observation["onGround"] else 0.0

    return np.array(
        [1.0, health_frac, food_frac, hostile_near, entity_near, block_ahead, has_placeable, is_falling, on_ground],
        dtype=np.float64,
    )


class SensoryEncoder:
    def __init__(self, input_idx: np.ndarray, weights: np.ndarray | None = None, gain: float = 3.0, seed: int = 0):
        self.input_idx = np.asarray(input_idx)
        self.n_features = len(FEATURE_NAMES)
        self.gain = gain
        self.weights = weights if weights is not None else self._default_weights(seed)

    def _default_weights(self, seed: int) -> np.ndarray:
        """Hand-built receptive fields: each input neuron listens to a
        small random subset of features (real sensory neurons don't each
        see every feature dimension either) rather than one dense
        all-to-all projection. Replaced by a learned projection in M5.
        """
        rng = np.random.default_rng(seed)
        weights = np.zeros((len(self.input_idx), self.n_features))
        for i in range(len(self.input_idx)):
            k = rng.integers(1, min(3, self.n_features) + 1)
            chosen = rng.choice(self.n_features, size=k, replace=False)
            weights[i, chosen] = rng.uniform(0.5, 1.5, size=k)
        return weights

    def encode(self, observation: dict, n_neurons_total: int) -> np.ndarray:
        features = extract_features(observation)
        ext_current = np.zeros(n_neurons_total, dtype=np.float64)
        ext_current[self.input_idx] = self.gain * (self.weights @ features)
        return ext_current
