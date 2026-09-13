"""End goal and training ladder for self-play.

These steps are for scoring and reporting progress, not for issuing live
commands. In self mode the bot is not told which block to walk to or what
item to craft next; it only gets normal observations, actions, and rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

END_GOAL = "kill_ender_dragon"


@dataclass(frozen=True)
class TrainingStep:
    id: str
    label: str
    item: str | None
    reward: float


TRAINING_STEPS = (
    TrainingStep("crafting_table", "make a crafting table", "crafting_table", 2.0),
    TrainingStep("wooden_pickaxe", "make a wooden pickaxe", "wooden_pickaxe", 5.0),
    TrainingStep("stone_pickaxe", "make a stone pickaxe", "stone_pickaxe", 10.0),
    TrainingStep("furnace", "make a furnace", "furnace", 8.0),
    TrainingStep("raw_iron", "mine iron ore", "raw_iron", 12.0),
    TrainingStep("iron_ingot", "smelt iron", "iron_ingot", 20.0),
    TrainingStep("iron_pickaxe", "make an iron pickaxe", "iron_pickaxe", 50.0),
    TrainingStep("diamond", "find diamonds", "diamond", 90.0),
    TrainingStep("obsidian", "collect obsidian", "obsidian", 120.0),
    TrainingStep("blaze_rod", "get blaze rods", "blaze_rod", 180.0),
    TrainingStep("ender_eye", "make eyes of ender", "ender_eye", 260.0),
    TrainingStep("dragon", "kill the Ender Dragon", None, 1000.0),
)

MILESTONE_REWARDS = {step.item: step.reward for step in TRAINING_STEPS if step.item is not None}


def inventory_names(observation: dict) -> set[str]:
    return {item["name"] for item in observation.get("inventory", [])}


def reached_steps(observation: dict) -> list[TrainingStep]:
    names = inventory_names(observation)
    return [step for step in TRAINING_STEPS if step.item is not None and step.item in names]


def next_training_step(observation: dict) -> TrainingStep:
    names = inventory_names(observation)
    for step in TRAINING_STEPS:
        if step.item is None or step.item not in names:
            return step
    return TRAINING_STEPS[-1]


def milestone_score(observation: dict) -> float:
    names = inventory_names(observation)
    return sum(step.reward for step in TRAINING_STEPS if step.item is not None and step.item in names)
