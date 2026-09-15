"""Tests for the objective-free task manager.

No progression, no stage machine, and no quest logic are allowed here.
Only immediate self-preservation behavior is tested.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.task_manager import TaskManager  # noqa: E402


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


def test_status_is_safety_only():
    tm = TaskManager()
    assert tm.status() == "safety_only"


def test_no_threat_no_objective_action():
    tm = TaskManager()
    assert tm.decide(make_observation()) is None


def test_low_health_near_hostile_triggers_flee():
    tm = TaskManager()
    obs = make_observation(
        health=5,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}],
    )
    actions = tm.decide(obs)
    assert actions is not None
    assert any(a["type"] == "look" for a in actions)
    assert any(a["type"] == "move" and a.get("forward") for a in actions)


def test_high_health_near_hostile_defers_to_fly_brain():
    tm = TaskManager()
    obs = make_observation(
        health=20,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}],
    )
    assert tm.decide(obs) is None


def test_low_health_far_from_hostile_does_not_flee():
    tm = TaskManager()
    obs = make_observation(
        health=5,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 20.0, "position": {"x": 20.0, "y": 64.0, "z": 0.0}}],
    )
    assert tm.decide(obs) is None
