"""Tests for the (explicitly non-biological) task manager, against
fabricated observations and a fake bridge client - no live bot-bridge or
Minecraft server needed.

The fake client records every command the task manager issues and serves
canned findBlocks results, which lets these tests assert on real behavior
(what it actually does, in what order, given a world state) rather than
just on internal stage bookkeeping.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.task_manager import (  # noqa: E402
    COBBLE_TARGET,
    RAW_IRON_TARGET,
    FreeWillTaskManager,
    TaskManager,
    WOOD_TARGET,
)


class FakeClient:
    """Records issued commands; answers findBlocks from `blocks`."""

    def __init__(self, blocks=None, fail_types=()):
        self.commands = []
        self.blocks = blocks or {}
        self.fail_types = set(fail_types)

    def do_action(self, command):
        self.commands.append(command)
        kind = command.get("type")
        if kind in self.fail_types:
            raise RuntimeError(f"simulated failure for {kind}")
        if kind == "findBlocks":
            matches = []
            for name in command["names"]:
                for pos in self.blocks.get(name, []):
                    matches.append({"x": pos[0], "y": pos[1], "z": pos[2], "name": name, "distance": 1.0})
            return {"ok": True, "blocks": matches}
        return {"ok": True}

    def types(self):
        return [c["type"] for c in self.commands]

    def of_type(self, kind):
        return [c for c in self.commands if c["type"] == kind]


def make_observation(**overrides) -> dict:
    obs = {
        "position": {"x": 0.0, "y": 64.0, "z": 0.0},
        "health": 20,
        "food": 20,
        "inWater": False,
        "inLava": False,
        "inventory": [],
        "nearbyBlocks": [],
        "nearbyEntities": [],
    }
    obs.update(overrides)
    return obs


def inv(**items):
    return [{"slot": i, "name": name, "count": count} for i, (name, count) in enumerate(items.items())]


# --- the progression chain -------------------------------------------------


def test_starts_by_gathering_wood():
    tm = TaskManager()
    client = FakeClient(blocks={"oak_log": [(5, 64, 0)]})
    assert tm.step(make_observation(), client) is True
    assert "mineBlock" in client.types()
    assert client.of_type("mineBlock")[0]["x"] == 5


def test_defers_to_fly_brain_when_no_wood_is_findable():
    tm = TaskManager()
    client = FakeClient(blocks={})
    # Nothing to mine anywhere - the fly brain should get control to explore.
    assert tm.step(make_observation(), client) is False


def test_crafts_planks_once_it_has_logs():
    tm = TaskManager()
    client = FakeClient()
    tm.step(make_observation(inventory=inv(oak_log=WOOD_TARGET)), client)
    crafts = client.of_type("craft")
    assert crafts and crafts[0]["item"] == "oak_planks"


def test_crafts_sticks_once_it_has_planks():
    tm = TaskManager()
    client = FakeClient()
    tm.step(make_observation(inventory=inv(oak_log=WOOD_TARGET, oak_planks=12)), client)
    crafts = client.of_type("craft")
    assert crafts and crafts[0]["item"] == "stick"


def test_places_a_crafting_table_before_crafting_a_pickaxe():
    tm = TaskManager()
    client = FakeClient(blocks={})  # no crafting table in range
    tm.step(make_observation(inventory=inv(oak_planks=12, stick=6)), client)
    # Must put a table down before it can craft a 3x3 recipe.
    assert "placeNearby" in client.types()


def test_crafts_wooden_pickaxe_when_a_table_is_already_placed():
    tm = TaskManager()
    client = FakeClient(blocks={"crafting_table": [(1, 64, 0)]})
    tm.step(make_observation(inventory=inv(oak_planks=12, stick=6)), client)
    crafts = client.of_type("craft")
    assert crafts and crafts[0]["item"] == "wooden_pickaxe"


def test_gathers_stone_once_it_has_a_wooden_pickaxe():
    tm = TaskManager()
    client = FakeClient(blocks={"stone": [(3, 62, 0)]})
    tm.step(make_observation(inventory=inv(wooden_pickaxe=1, oak_planks=12, stick=6)), client)
    assert "mineBlock" in client.types()


def test_digs_down_when_no_stone_is_reachable():
    tm = TaskManager()
    client = FakeClient(blocks={})
    tm.step(make_observation(inventory=inv(wooden_pickaxe=1, oak_planks=12, stick=6)), client)
    assert "gotoY" in client.types()


def test_crafts_stone_pickaxe_once_it_has_cobble_and_sticks():
    tm = TaskManager()
    client = FakeClient(blocks={"crafting_table": [(1, 64, 0)]})
    tm.step(
        make_observation(inventory=inv(wooden_pickaxe=1, cobblestone=COBBLE_TARGET, stick=6)),
        client,
    )
    crafts = client.of_type("craft")
    assert crafts and crafts[0]["item"] == "stone_pickaxe"


def test_crafts_furnace_after_stone_pickaxe():
    tm = TaskManager()
    client = FakeClient(blocks={"crafting_table": [(1, 64, 0)]})
    tm.step(
        make_observation(inventory=inv(stone_pickaxe=1, wooden_pickaxe=1, cobblestone=COBBLE_TARGET, stick=6)),
        client,
    )
    # With a stone pickaxe and plenty of cobble but no furnace yet, a furnace
    # is the next thing standing between it and smelted iron.
    crafts = client.of_type("craft")
    assert crafts and crafts[0]["item"] == "furnace"


def test_hunts_for_iron_once_equipped_with_stone_pickaxe_and_furnace():
    tm = TaskManager()
    client = FakeClient(blocks={"iron_ore": [(10, 40, 3)]})
    tm.step(
        make_observation(inventory=inv(stone_pickaxe=1, furnace=1, cobblestone=COBBLE_TARGET, stick=6)),
        client,
    )
    mined = client.of_type("mineBlock")
    assert mined and mined[0]["x"] == 10


def test_tunnels_deeper_when_no_iron_is_in_range():
    tm = TaskManager()
    client = FakeClient(blocks={})
    tm.step(
        make_observation(
            position={"x": 0.0, "y": 70.0, "z": 0.0},
            inventory=inv(stone_pickaxe=1, furnace=1, cobblestone=COBBLE_TARGET, stick=6),
        ),
        client,
    )
    assert "gotoY" in client.types()


def test_smelts_raw_iron_when_a_furnace_is_placed():
    tm = TaskManager()
    client = FakeClient(blocks={"furnace": [(1, 40, 0)]})
    tm.step(
        make_observation(inventory=inv(stone_pickaxe=1, raw_iron=RAW_IRON_TARGET, oak_planks=8, stick=6)),
        client,
    )
    smelts = client.of_type("smelt")
    assert smelts and smelts[0]["input"] == "raw_iron"
    assert smelts[0]["fuel"] == "oak_planks"


def test_prefers_coal_as_smelting_fuel():
    tm = TaskManager()
    client = FakeClient(blocks={"furnace": [(1, 40, 0)]})
    tm.step(
        make_observation(inventory=inv(stone_pickaxe=1, raw_iron=3, coal=5, oak_planks=8, stick=6)),
        client,
    )
    assert client.of_type("smelt")[0]["fuel"] == "coal"


def test_crafts_the_iron_pickaxe_once_it_has_ingots_and_sticks():
    tm = TaskManager()
    client = FakeClient(blocks={"crafting_table": [(1, 40, 0)]})
    tm.step(make_observation(inventory=inv(iron_ingot=3, stick=6, stone_pickaxe=1)), client)
    crafts = client.of_type("craft")
    assert crafts and crafts[0]["item"] == "iron_pickaxe"


def test_goal_reached_hands_control_back_to_the_fly_brain():
    tm = TaskManager()
    client = FakeClient()
    obs = make_observation(inventory=inv(iron_pickaxe=1))
    assert tm.step(obs, client) is False
    assert tm.has_goal_item(obs) is True


# --- survival overrides ----------------------------------------------------


def test_escaping_water_outranks_progression():
    tm = TaskManager()
    client = FakeClient(blocks={"oak_log": [(5, 64, 0)]})
    tm.step(make_observation(inWater=True), client)
    # Should be swimming out, not wandering off to chop a tree.
    assert "mineBlock" not in client.types()
    assert client.of_type("move")[0]["jump"] is True


def test_fights_back_when_healthy():
    tm = TaskManager()
    client = FakeClient()
    obs = make_observation(
        health=20,
        nearbyEntities=[
            {"id": 7, "name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}
        ],
    )
    tm.step(obs, client)
    assert client.of_type("attack")[0]["entityId"] == 7


def test_flees_when_badly_hurt():
    tm = TaskManager()
    client = FakeClient()
    obs = make_observation(
        health=5,
        nearbyEntities=[
            {"id": 7, "name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}
        ],
    )
    tm.step(obs, client)
    goto = client.of_type("goto")
    assert goto, "should path away from the threat"
    assert goto[0]["x"] < 0  # away from a hostile standing at +x


def test_ignores_distant_hostiles():
    tm = TaskManager()
    client = FakeClient(blocks={"oak_log": [(5, 64, 0)]})
    obs = make_observation(
        health=20,
        nearbyEntities=[
            {"id": 7, "name": "zombie", "kind": "Hostile mobs", "distance": 30.0, "position": {"x": 30.0, "y": 64.0, "z": 0.0}}
        ],
    )
    tm.step(obs, client)
    assert "attack" not in client.types()
    assert "mineBlock" in client.types()


def test_eats_when_hungry():
    tm = TaskManager()
    client = FakeClient()
    tm.step(make_observation(food=6, inventory=inv(bread=2)), client)
    assert "consume" in client.types()


def test_does_not_try_to_eat_with_no_food():
    tm = TaskManager()
    client = FakeClient(blocks={"oak_log": [(5, 64, 0)]})
    tm.step(make_observation(food=2), client)
    assert "consume" not in client.types()


# --- robustness ------------------------------------------------------------


def test_action_failure_does_not_propagate():
    """A long unattended run must survive individual action failures - a tree
    someone else chopped, a path that got blocked - without crashing."""
    tm = TaskManager()
    client = FakeClient(blocks={"oak_log": [(5, 64, 0)]}, fail_types={"mineBlock"})
    assert tm.step(make_observation(), client) is True  # handled, not raised


def test_find_blocks_failure_is_survivable():
    tm = TaskManager()
    client = FakeClient(fail_types={"findBlocks"})
    assert tm.step(make_observation(), client) is False


def test_does_not_deadlock_when_it_cannot_afford_a_crafting_table():
    """Found live: holding cobble + sticks for a stone pickaxe but only one
    plank and no table, it looped forever on 'placing crafting_table'. It
    must instead fall through to restocking the wood that unblocks it."""
    tm = TaskManager()
    client = FakeClient(blocks={"oak_log": [(5, 64, 0)]})  # no table in range
    tm.step(
        make_observation(inventory=inv(wooden_pickaxe=1, cobblestone=3, stick=6, oak_planks=1)),
        client,
    )
    assert "mineBlock" in client.types(), "should go get wood, not retry an impossible craft"


def test_stockpiles_wood_before_converting_it_to_planks():
    """Found live: crafting planks the moment a single log arrived drained
    the pile faster than it filled, so the wood target was never reached and
    the bot stranded itself a plank short of a crafting table."""
    tm = TaskManager()
    client = FakeClient(blocks={"oak_log": [(5, 64, 0)]})
    tm.step(make_observation(inventory=inv(oak_log=1)), client)
    assert "mineBlock" in client.types()
    assert not client.of_type("craft"), "should keep gathering, not convert its only log"


def test_converts_logs_once_no_more_wood_is_reachable():
    """The wood-first rule must not become its own deadlock: with no tree in
    range it has to work with what it already carries."""
    tm = TaskManager()
    client = FakeClient(blocks={})  # no trees anywhere
    tm.step(make_observation(inventory=inv(oak_log=2)), client)
    crafts = client.of_type("craft")
    assert crafts and crafts[0]["item"] == "oak_planks"


def test_progression_is_rederived_after_losing_everything():
    """Dying drops the whole inventory. Because the objective is derived from
    inventory rather than stored, the same TaskManager instance must simply
    restart the chain instead of staying stuck on a late-game stage."""
    tm = TaskManager()
    client = FakeClient(blocks={"crafting_table": [(1, 40, 0)], "oak_log": [(5, 64, 0)]})
    tm.step(make_observation(inventory=inv(iron_ingot=3, stick=6)), client)
    assert client.of_type("craft")[0]["item"] == "iron_pickaxe"

    # ...now it dies and loses everything.
    fresh = FakeClient(blocks={"oak_log": [(5, 64, 0)]})
    tm.step(make_observation(inventory=[]), fresh)
    assert "mineBlock" in fresh.types()  # back to chopping wood, unprompted


# --- FreeWillTaskManager: safety-only, no progression -----------------------
#
# No progression, no stage machine, and no quest logic are allowed here.
# Only the low-health flee override is tested; everything else must be left
# to the fly-brain and reward system.


def test_free_will_starts_safety_only():
    tm = FreeWillTaskManager()
    assert tm.status == "safety_only"


def test_free_will_no_threat_defers_to_fly_brain():
    tm = FreeWillTaskManager()
    client = FakeClient()
    assert tm.step(make_observation(), client) is False
    assert client.commands == []


def test_free_will_low_health_near_hostile_triggers_flee():
    tm = FreeWillTaskManager()
    client = FakeClient()
    obs = make_observation(
        health=5,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}],
    )
    assert tm.step(obs, client) is True
    assert "look" in client.types()
    moves = client.of_type("move")
    assert moves and moves[0]["forward"] is True
    assert tm.status == "fleeing"


def test_free_will_high_health_near_hostile_defers_to_fly_brain():
    tm = FreeWillTaskManager()
    client = FakeClient()
    obs = make_observation(
        health=20,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 2.0, "position": {"x": 1.0, "y": 64.0, "z": 0.0}}],
    )
    assert tm.step(obs, client) is False
    assert client.commands == []


def test_free_will_low_health_far_from_hostile_does_not_flee():
    tm = FreeWillTaskManager()
    client = FakeClient()
    obs = make_observation(
        health=5,
        nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 20.0, "position": {"x": 20.0, "y": 64.0, "z": 0.0}}],
    )
    assert tm.step(obs, client) is False
    assert client.commands == []
