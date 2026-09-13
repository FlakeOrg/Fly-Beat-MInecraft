"""Tests for live movement command fallbacks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.movement import movement_command_from_actions  # noqa: E402


def make_observation(**overrides) -> dict:
    obs = {
        "yaw": 0.0,
        "onGround": True,
        "nearbyBlocks": [],
    }
    obs.update(overrides)
    return obs


def test_all_false_actions_explore_forward():
    actions = {"forward": False, "left": False, "right": False, "jump": False, "attack": False, "mine_ahead": False}
    command = movement_command_from_actions(actions, make_observation())
    assert command["forward"] is True


def test_all_false_actions_jump_and_sidestep_when_blocked():
    actions = {"forward": False, "left": False, "right": False, "jump": False, "attack": False, "mine_ahead": False}
    observation = make_observation(nearbyBlocks=[{"x": 0, "y": 0, "z": 1, "boundingBox": "block"}])
    command = movement_command_from_actions(actions, observation, step=0)
    assert command["forward"] is True
    assert command["jump"] is True
    assert command["right"] is True


def test_purposeful_non_movement_action_does_not_explore():
    actions = {"forward": False, "left": False, "right": False, "jump": False, "attack": True, "mine_ahead": False}
    command = movement_command_from_actions(actions, make_observation())
    assert command["forward"] is False
    assert command["jump"] is False


def test_existing_movement_is_preserved():
    actions = {"forward": False, "left": True, "right": False, "jump": False, "attack": False, "mine_ahead": False}
    command = movement_command_from_actions(actions, make_observation())
    assert command["left"] is True
    assert command["forward"] is False
