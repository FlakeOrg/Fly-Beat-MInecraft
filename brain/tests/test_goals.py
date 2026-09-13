"""Tests for the self-play end goal and milestone ladder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.goals import END_GOAL, milestone_score, next_training_step, reached_steps  # noqa: E402


def inv(*names: str) -> list[dict]:
    return [{"slot": i, "name": name, "count": 1} for i, name in enumerate(names)]


def test_end_goal_is_dragon_kill():
    assert END_GOAL == "kill_ender_dragon"


def test_next_step_starts_at_crafting_table():
    step = next_training_step({"inventory": []})
    assert step.id == "crafting_table"


def test_next_step_advances_by_inventory_milestones():
    obs = {"inventory": inv("crafting_table", "wooden_pickaxe")}
    assert next_training_step(obs).id == "stone_pickaxe"


def test_milestone_score_counts_reached_items():
    obs = {"inventory": inv("crafting_table", "wooden_pickaxe")}
    assert milestone_score(obs) == 7.0


def test_reached_steps_returns_known_steps_only():
    obs = {"inventory": inv("crafting_table", "dirt")}
    assert [step.id for step in reached_steps(obs)] == ["crafting_table"]
