"""Tests for the (explicitly non-biological) task manager, against
fabricated observations - no live bot-bridge or Minecraft server needed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.task_manager import TaskManager, WOOD_TARGET  # noqa: E402


def make_observation(**overrides) -> dict:
    obs = {
        "position": {"x": 0.0, "y": 64.0, "z": 0.0},
        "health": 20,
        "food": 20,
        "inventory": [],
        "nearbyBlocks": [],
        "nearbyEntities": [],
    }
    obs.update(overrides)
    return obs


def test_starts_in_gather_wood_stage():
    tm = TaskManager()
    assert tm.stage == "gather_wood"


def test_defers_to_fly_brain_when_no_log_visible():
    tm = TaskManager()
    actions = tm.decide(make_observation())
    assert actions is None


def test_walks_towards_a_distant_log():
    tm = TaskManager()
    obs = make_observation(nearbyBlocks=[{"x": 3, "y": 0, "z": 0, "name": "oak_log"}])
    actions = tm.decide(obs)
    assert actions is not None
    assert any(a["type"] == "look" for a in actions)
    assert any(a["type"] == "move" and a.get("forward") for a in actions)


def test_digs_an_adjacent_log():
    tm = TaskManager()
    obs = make_observation(nearbyBlocks=[{"x": 1, "y": 0, "z": 0, "name": "oak_log"}])
    actions = tm.decide(obs)
    assert any(a["type"] == "dig" for a in actions)


def test_advances_to_craft_planks_once_wood_target_met():
    tm = TaskManager()
    obs = make_observation(inventory=[{"slot": 0, "name": "oak_log", "count": WOOD_TARGET}])
    actions = tm.decide(obs)
    assert actions is None  # advancing itself doesn't emit an action this tick
    assert tm.stage == "craft_planks"


def test_craft_planks_issues_craft_command():
    tm = TaskManager()
    tm.stage = "craft_planks"
    obs = make_observation(inventory=[{"slot": 0, "name": "oak_log", "count": WOOD_TARGET}])
    actions = tm.decide(obs)
    assert actions == [{"type": "craft", "item": "oak_planks", "count": 8}]


def test_craft_planks_falls_back_to_gather_wood_if_out_of_logs():
    tm = TaskManager()
    tm.stage = "craft_planks"
    actions = tm.decide(make_observation())
    assert actions is None
    assert tm.stage == "gather_wood"


def test_advances_to_craft_table_once_planks_target_met():
    tm = TaskManager()
    tm.stage = "craft_planks"
    obs = make_observation(inventory=[{"slot": 0, "name": "oak_planks", "count": 8}])
    tm.decide(obs)
    assert tm.stage == "craft_table"


def test_craft_tools_crafts_each_wanted_tool_in_order():
    tm = TaskManager()
    tm.stage = "craft_tools"
    obs = make_observation()

    actions = tm.decide(obs)
    assert actions == [{"type": "craft", "item": "wooden_pickaxe", "count": 1}]

    obs["inventory"] = [{"slot": 0, "name": "wooden_pickaxe", "count": 1}]
    actions = tm.decide(obs)
    assert actions == [{"type": "craft", "item": "wooden_axe", "count": 1}]


def test_advances_to_explore_once_all_tools_crafted():
    tm = TaskManager()
    tm.stage = "craft_tools"
    obs = make_observation(
        inventory=[
            {"slot": 0, "name": "wooden_pickaxe", "count": 1},
            {"slot": 1, "name": "wooden_axe", "count": 1},
            {"slot": 2, "name": "wooden_sword", "count": 1},
        ]
    )
    actions = tm.decide(obs)
    assert actions is None
    assert tm.stage == "explore"


def test_explore_stage_always_defers_to_fly_brain():
    tm = TaskManager()
    tm.stage = "explore"
    assert tm.decide(make_observation()) is None


def test_low_health_near_hostile_triggers_flee_regardless_of_stage():
    tm = TaskManager()
    tm.stage = "craft_tools"
    obs = make_observation(
        health=5,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}],
    )
    actions = tm.decide(obs)
    assert actions is not None
    assert any(a["type"] == "look" for a in actions)
    assert any(a["type"] == "move" and a.get("forward") for a in actions)


def test_high_health_near_hostile_does_not_flee():
    tm = TaskManager()
    obs = make_observation(
        health=20,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}],
    )
    actions = tm.decide(obs)
    assert actions is None  # falls through to normal gather_wood logic (no log nearby -> defer)


def test_low_health_far_from_hostile_does_not_flee():
    tm = TaskManager()
    obs = make_observation(
        health=5,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 20.0, "position": {"x": 20.0, "y": 64.0, "z": 0.0}}],
    )
    actions = tm.decide(obs)
    assert actions is None
