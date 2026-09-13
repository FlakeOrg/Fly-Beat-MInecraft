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
    # physical state
    "bias",
    "health_frac",
    "food_frac",
    "hostile_near",
    "entity_near",
    "block_ahead",
    "is_falling",
    "on_ground",
    # what's around worth acting on
    "wood_ahead",
    "wood_visible",
    "stone_ahead",
    "stone_visible",
    "iron_visible",
    "table_near",
    "furnace_near",
    # what it's carrying
    "has_logs",
    "has_planks",
    "has_sticks",
    "has_table",
    "has_wood_pickaxe",
    "has_stone_pickaxe",
    "has_furnace",
    "has_raw_iron",
    "has_iron_ingot",
    # how deep it is (ore is a function of depth)
    "depth_frac",
]

# Without the inventory and block-identity features above, a self-learning
# bot is structurally incapable of ever learning to craft: it cannot tell
# "I am holding wood" from "I am holding nothing", or a tree from a rock,
# so no amount of trial and error can associate a craft attempt with the
# state that makes it succeed. Adding senses is not the same as being told
# what to do - it still has to discover the whole sequence from reward.

LOG_NAMES = (
    "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
)
PLANK_NAMES = tuple(name.replace("_log", "_planks") for name in LOG_NAMES)
STONE_BLOCK_NAMES = ("stone", "cobblestone", "deepslate", "cobbled_deepslate", "granite", "andesite", "diorite", "tuff")
IRON_ORE_NAMES = ("iron_ore", "deepslate_iron_ore")

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


def _facing_offset(yaw: float) -> tuple[int, int]:
    """Rounds the bot's horizontal facing direction to a unit grid offset,
    using Mineflayer's yaw convention (yaw=0 faces +Z, increasing yaw turns
    towards -X)."""
    dx = -math.sin(yaw)
    dz = math.cos(yaw)
    return int(round(dx)), int(round(dz))


def _has(observation: dict, names) -> float:
    carried = {i["name"] for i in observation.get("inventory", [])}
    if isinstance(names, str):
        return 1.0 if names in carried else 0.0
    return 1.0 if carried & set(names) else 0.0


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
    blocks = observation.get("nearbyBlocks", [])
    ahead_blocks = [b for b in blocks if b["x"] == ahead_x and b["z"] == ahead_z and b["y"] in (0, 1)]
    block_ahead = 1.0 if ahead_blocks else 0.0

    def ahead_is(names) -> float:
        return 1.0 if any(b["name"] in names for b in ahead_blocks) else 0.0

    def visible(names) -> float:
        return 1.0 if any(b["name"] in names for b in blocks) else 0.0

    velocity_y = observation["velocity"]["y"]
    is_falling = 1.0 if (velocity_y < -0.1 and not observation["onGround"]) else 0.0
    on_ground = 1.0 if observation["onGround"] else 0.0

    # Normalized so the network sees a bounded signal: 0 at bedrock-ish
    # depth, 1 at build height.
    y = observation.get("position", {}).get("y", 64.0)
    depth_frac = max(0.0, min(1.0, (y + 64.0) / 384.0))

    return np.array(
        [
            1.0, health_frac, food_frac, hostile_near, entity_near, block_ahead, is_falling, on_ground,
            ahead_is(LOG_NAMES), visible(LOG_NAMES),
            ahead_is(STONE_BLOCK_NAMES), visible(STONE_BLOCK_NAMES),
            visible(IRON_ORE_NAMES),
            visible(("crafting_table",)), visible(("furnace",)),
            _has(observation, LOG_NAMES), _has(observation, PLANK_NAMES), _has(observation, "stick"),
            _has(observation, "crafting_table"), _has(observation, "wooden_pickaxe"),
            _has(observation, "stone_pickaxe"), _has(observation, "furnace"),
            _has(observation, "raw_iron"), _has(observation, "iron_ingot"),
            depth_frac,
        ],
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
