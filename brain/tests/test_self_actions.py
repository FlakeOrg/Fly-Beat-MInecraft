"""Tests for the self-directed play layer: the fly brain's own task verbs.

The point of these is to pin down the boundary that makes self-play honest:
self_actions.py must translate a chosen verb into a game action and nothing
more - it must never contain strategy about what the bot ought to do next.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.self_actions import build_command, perform  # noqa: E402
from interface.motor_decoder import ACTIONS, MOVEMENT_ACTIONS, TASK_ACTIONS, MotorDecoder  # noqa: E402
from interface.sensory_encoder import FEATURE_NAMES, extract_features  # noqa: E402


def obs(inventory=None, blocks=None, **overrides):
    base = {
        "position": {"x": 0.0, "y": 64.0, "z": 0.0},
        "health": 20,
        "food": 20,
        "yaw": 0.0,
        "onGround": True,
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "nearbyEntities": [],
        "nearbyBlocks": blocks or [],
        "inventory": [
            {"slot": i, "name": n, "count": c} for i, (n, c) in enumerate((inventory or {}).items())
        ],
    }
    base.update(overrides)
    return base


class RecordingClient:
    def __init__(self, fail=False):
        self.commands = []
        self.fail = fail

    def do_action(self, command):
        self.commands.append(command)
        if self.fail:
            raise RuntimeError("rejected")
        return {"ok": True}


# --- the action space is actually capable of the goal ----------------------


def test_task_verbs_cover_the_iron_pickaxe_chain():
    """Every step of the chain has to be *expressible* by the network, or no
    amount of trial and error could ever reach it."""
    for needed in ("craft_planks", "craft_sticks", "craft_table", "place_table",
                   "craft_wooden_pickaxe", "craft_stone_pickaxe", "craft_furnace",
                   "smelt_iron", "craft_iron_pickaxe"):
        assert needed in TASK_ACTIONS


def test_action_and_movement_sets_are_disjoint_and_complete():
    assert set(ACTIONS) == set(MOVEMENT_ACTIONS) | set(TASK_ACTIONS)
    assert not set(MOVEMENT_ACTIONS) & set(TASK_ACTIONS)


# --- translation, not strategy --------------------------------------------


def test_craft_planks_uses_whatever_log_is_carried():
    assert build_command("craft_planks", obs({"birch_log": 2}))["item"] == "birch_planks"


def test_craft_planks_is_inexpressible_without_logs():
    assert build_command("craft_planks", obs({})) is None


def test_smelt_prefers_coal_but_falls_back_to_planks():
    assert build_command("smelt_iron", obs({"raw_iron": 1, "coal": 1}))["fuel"] == "coal"
    assert build_command("smelt_iron", obs({"raw_iron": 1, "oak_planks": 4}))["fuel"] == "oak_planks"
    assert build_command("smelt_iron", obs({"raw_iron": 1})) is None


def test_verbs_are_attempted_verbatim_without_precondition_checks():
    """A craft the bot can't afford must still be *attempted* - learning that
    it fails is the bot's job, not something this layer should pre-empt."""
    client = RecordingClient()
    perform("craft_iron_pickaxe", obs({}), client)
    assert client.commands == [{"type": "craft", "item": "iron_pickaxe", "count": 1}]


def test_failed_attempt_is_reported_not_raised():
    client = RecordingClient(fail=True)
    assert perform("craft_table", obs({"oak_planks": 4}), client) is False


# --- perception ------------------------------------------------------------


def test_bot_can_perceive_its_own_inventory():
    """Without this the network cannot tell 'holding wood' from 'holding
    nothing', so crafting is unlearnable in principle."""
    features = dict(zip(FEATURE_NAMES, extract_features(obs({"oak_log": 3}))))
    assert features["has_logs"] == 1.0
    assert features["has_planks"] == 0.0


def test_bot_can_tell_a_tree_from_a_rock():
    wood = dict(zip(FEATURE_NAMES, extract_features(obs(blocks=[{"x": 0, "y": 0, "z": 1, "name": "oak_log"}]))))
    rock = dict(zip(FEATURE_NAMES, extract_features(obs(blocks=[{"x": 0, "y": 0, "z": 1, "name": "stone"}]))))
    assert wood["wood_ahead"] == 1.0 and wood["stone_ahead"] == 0.0
    assert rock["stone_ahead"] == 1.0 and rock["wood_ahead"] == 0.0


def test_depth_is_perceivable():
    deep = dict(zip(FEATURE_NAMES, extract_features(obs(position={"x": 0, "y": 10, "z": 0}))))
    high = dict(zip(FEATURE_NAMES, extract_features(obs(position={"x": 0, "y": 200, "z": 0}))))
    assert deep["depth_frac"] < high["depth_frac"]


# --- decoder ---------------------------------------------------------------


def test_decoder_picks_at_most_one_task_verb():
    decoder = MotorDecoder(np.arange(40), seed=0)
    spikes = np.ones((5, 60))
    chosen = decoder.decode_task_action(spikes)
    assert chosen is None or chosen in TASK_ACTIONS


def test_decoder_returns_none_when_nothing_crosses_threshold():
    decoder = MotorDecoder(np.arange(40), seed=0)
    silent = np.zeros((5, 60))
    assert decoder.decode_task_action(silent) is None
