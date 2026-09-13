"""Executes the fly brain's chosen task verb as a real game action.

This is deliberately *only* a translation layer: "the network fired
craft_table" -> "ask the server to craft a crafting table". It contains no
strategy at all - nothing here decides what the bot should be doing, what
it needs next, or where to go. Every attempt is made exactly when the
network asks for it, and most attempts fail harmlessly (crafting a pickaxe
with no wood simply doesn't work), which is precisely the feedback the
reward ladder in agent/goals.py has to teach it to avoid.

Contrast with agent/task_manager.py, which is the opposite: a hand-written
strategy that knows the whole recipe tree and tells the bot what to do
next. That one is assisted play; this one is self-directed play.
"""

from __future__ import annotations

from agent.bridge_client import BridgeError

LOG_TO_PLANKS = {
    "oak_log": "oak_planks",
    "birch_log": "birch_planks",
    "spruce_log": "spruce_planks",
    "jungle_log": "jungle_planks",
    "acacia_log": "acacia_planks",
    "dark_oak_log": "dark_oak_planks",
    "mangrove_log": "mangrove_planks",
    "cherry_log": "cherry_planks",
}


def _carried(observation: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in observation.get("inventory", []):
        counts[item["name"]] = counts.get(item["name"], 0) + item["count"]
    return counts


def _plank_type(observation: dict) -> str | None:
    """Which plank the bot can actually make, given the logs it holds. This
    is a mechanical detail of the recipe system (oak logs make oak planks),
    not a hint about what to do."""
    carried = _carried(observation)
    for log_name, plank_name in LOG_TO_PLANKS.items():
        if carried.get(log_name, 0) > 0:
            return plank_name
    return None


def _any_plank(observation: dict) -> str | None:
    carried = _carried(observation)
    for plank_name in LOG_TO_PLANKS.values():
        if carried.get(plank_name, 0) > 0:
            return plank_name
    return None


def build_command(action: str, observation: dict) -> dict | None:
    """Maps a task verb to a bridge command, or None if it's not even
    expressible right now (e.g. craft_planks with no logs at all). Returning
    None avoids burning a round trip on a request that cannot possibly
    parse - the attempt still counts as failed from the network's side."""
    if action == "craft_planks":
        plank = _plank_type(observation)
        return {"type": "craft", "item": plank, "count": 4} if plank else None

    if action == "craft_sticks":
        return {"type": "craft", "item": "stick", "count": 4}

    if action == "craft_table":
        return {"type": "craft", "item": "crafting_table", "count": 1}

    if action == "place_table":
        return {"type": "placeNearby", "item": "crafting_table"}

    if action == "craft_wooden_pickaxe":
        return {"type": "craft", "item": "wooden_pickaxe", "count": 1}

    if action == "craft_stone_pickaxe":
        return {"type": "craft", "item": "stone_pickaxe", "count": 1}

    if action == "craft_furnace":
        return {"type": "craft", "item": "furnace", "count": 1}

    if action == "place_furnace":
        return {"type": "placeNearby", "item": "furnace"}

    if action == "smelt_iron":
        fuel = "coal" if _carried(observation).get("coal", 0) > 0 else _any_plank(observation)
        if fuel is None:
            return None
        return {"type": "smelt", "input": "raw_iron", "fuel": fuel, "count": 1}

    if action == "craft_iron_pickaxe":
        return {"type": "craft", "item": "iron_pickaxe", "count": 1}

    return None


def perform(action: str, observation: dict, client) -> bool:
    """Attempts one task verb. True if the server accepted it, False if it
    was impossible or rejected - either way the episode continues."""
    command = build_command(action, observation)
    if command is None:
        return False
    try:
        client.do_action(command)
        return True
    except BridgeError:
        return False
    except Exception:  # noqa: BLE001 - a failed attempt must never end a run
        return False
