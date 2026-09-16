"""Non-biological task manager (M6): explicit progression logic for
Minecraft survival/crafting/combat, on top of the fly-brain's low-level
motor control (loop.py). This is the piece that makes no claim to be
biology - a real fly brain has no capacity for crafting trees or multi-
step planning, so this layer exists to do exactly what a fly brain can't.

Division of labor: when there's a specific, precise target (a known block
to walk to and mine, a recipe to craft, ore to smelt), this drives the bot
directly - deterministic logic, not the trained connectome. When there's
nothing concrete to do, `step()` returns False and the caller
(agent/loop.py, training/live_rollout.py) falls back to the fly-brain's
trained reflexes for open-ended movement/threat response. That's
deliberate: the fly brain's job was always reflexive execution under this
plan, not planning - see docs/architecture.md.

Current progression covers wood -> planks/sticks -> crafting table ->
wooden pickaxe -> stone -> stone pickaxe -> furnace -> iron ore -> smelt
-> **iron pickaxe**, plus always-on survival overrides (get out of water,
fight or flee hostiles, eat when starving).

Two deliberate design choices worth knowing:

1. The current objective is *derived from inventory every step* rather
   than stored in a stage variable. Dying drops everything, and episodes
   restart mid-progression, so a stored stage goes stale constantly; a
   derived one self-heals - after a death it simply notices it has no wood
   again and restarts the chain.
2. It issues actions through the bridge client directly instead of
   returning a command list, because almost every step now needs a
   query -> decide -> act round trip ("where is the nearest iron ore" ->
   "walk there" -> "mine it"), which a fire-and-forget command list can't
   express.

`FreeWillTaskManager` below is the deliberate alternative: no progression
stack at all, just the same low-health flee override, so the fly-brain and
reward system are left to earn every bit of progress themselves. Callers
(agent/loop.py, training/live_rollout.py) pick between the two - or neither
- through a mode flag; both share the `step(observation, client) -> bool`
/ `status` interface so they're interchangeable at the call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

LOG_NAMES = [
    "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
]
PLANK_NAMES = [name.replace("_log", "_planks") for name in LOG_NAMES]
STONE_NAMES = ["stone", "cobblestone", "deepslate", "cobbled_deepslate", "granite", "andesite", "diorite", "tuff"]
IRON_ORE_NAMES = ["iron_ore", "deepslate_iron_ore"]
COAL_ORE_NAMES = ["coal_ore", "deepslate_coal_ore"]
FOOD_NAMES = [
    "bread", "cooked_beef", "cooked_porkchop", "cooked_chicken", "cooked_mutton",
    "cooked_cod", "cooked_salmon", "apple", "carrot", "baked_potato", "sweet_berries",
    "melon_slice", "beef", "porkchop", "chicken", "mutton", "rotten_flesh",
]

# Targets are deliberately a bit above the bare minimum: a crafting table
# costs 4 planks and gets left behind whenever the bot wanders off to mine,
# so it needs to be able to afford another one rather than stranding itself
# one plank short of a pickaxe.
WOOD_TARGET = 8
PLANKS_TARGET = 12
STICKS_TARGET = 6
COBBLE_TARGET = 20
RAW_IRON_TARGET = 3

LOW_HEALTH_THRESHOLD = 8
FIGHT_HEALTH_THRESHOLD = 14  # above this, fight back instead of fleeing
HOSTILE_ENGAGE_DISTANCE = 6.0
LOW_FOOD_THRESHOLD = 16

# FreeWillTaskManager's own flee trigger - deliberately separate from
# HOSTILE_ENGAGE_DISTANCE above (TaskManager's fight-or-flee radius), since
# FreeWillTaskManager never fights back and has no health band to fight in.
FREE_WILL_FLEE_DISTANCE = 6.0

# Iron generates well below the surface; if a search at the current depth
# finds nothing, the bot tunnels down toward this level and looks again.
IRON_SEARCH_Y = 40
DEEP_SEARCH_RADIUS = 96
SURFACE_SEARCH_RADIUS = 64


def _inventory_count(observation: dict, item_name: str) -> int:
    return sum(i["count"] for i in observation["inventory"] if i["name"] == item_name)


def _count_any(observation: dict, names: list[str]) -> int:
    return sum(_inventory_count(observation, name) for name in names)


def _first_present(observation: dict, names: list[str]) -> str | None:
    return next((n for n in names if _inventory_count(observation, n) > 0), None)


def _nearest_hostile(observation: dict) -> dict | None:
    hostiles = [e for e in observation["nearbyEntities"] if e.get("kind") == "Hostile mobs"]
    if not hostiles:
        return None
    return min(hostiles, key=lambda e: e["distance"])


def _yaw_towards(dx: float, dz: float) -> float:
    """Mineflayer yaw convention: yaw=0 faces +Z, increasing yaw turns toward -X."""
    return math.atan2(-dx, dz)


@dataclass
class TaskManager:
    """Drives Minecraft progression. `status` is the last objective it acted
    on, for logging."""

    status: str = "starting"
    goal_item: str = "iron_pickaxe"
    _failures: dict = field(default_factory=dict)

    # --- public API -----------------------------------------------------

    def step(self, observation: dict, client) -> bool:
        """Advances progression by one chunk of work. Returns True if it did
        something (so the caller skips the fly-brain reflexes this tick),
        False if there was nothing concrete to do."""
        if self._survival_override(observation, client):
            return True
        return self._progress(observation, client)

    def has_goal_item(self, observation: dict) -> bool:
        return _inventory_count(observation, self.goal_item) > 0

    # --- survival (always-on, outranks progression) ---------------------

    def _survival_override(self, observation: dict, client) -> bool:
        # Drowning was the single most common death in live runs: the bot
        # would path into water, sink, and never prioritise getting out.
        if observation.get("inWater") or observation.get("inLava"):
            self.status = "escaping liquid"
            self._act(client, {"type": "move", "forward": True, "jump": True, "sprint": False})
            return True

        hostile = _nearest_hostile(observation)
        if hostile is not None and hostile["distance"] <= HOSTILE_ENGAGE_DISTANCE:
            health = observation["health"]
            if health >= FIGHT_HEALTH_THRESHOLD:
                # Healthy enough to fight: standing and trading hits beats
                # being chased down while fleeing, which is what actually
                # killed most bots.
                self.status = f"fighting {hostile['name']}"
                self._act(client, {"type": "attack", "entityId": hostile["id"]})
                return True
            if health <= LOW_HEALTH_THRESHOLD:
                self.status = f"fleeing {hostile['name']}"
                pos, hpos = observation["position"], hostile["position"]
                away_x = pos["x"] + (pos["x"] - hpos["x"]) * 4
                away_z = pos["z"] + (pos["z"] - hpos["z"]) * 4
                self._act(client, {"type": "goto", "x": away_x, "z": away_z, "range": 2})
                return True

        if observation["food"] <= LOW_FOOD_THRESHOLD:
            food = _first_present(observation, FOOD_NAMES)
            if food is not None:
                self.status = f"eating {food}"
                self._act(client, {"type": "equip", "item": food, "destination": "hand"})
                self._act(client, {"type": "consume"})
                return True

        return False

    # --- progression ----------------------------------------------------

    def _progress(self, observation: dict, client) -> bool:
        """Works backwards from the goal: each branch checks whether its own
        prerequisites are met, and if not, falls through to the branch that
        produces them. Derived fresh each step so a death (which drops the
        whole inventory) simply restarts the chain."""
        if self.has_goal_item(observation):
            self.status = f"done - has {self.goal_item}"
            return False  # goal reached; hand the bot back to the fly brain

        planks = _count_any(observation, PLANK_NAMES)
        sticks = _inventory_count(observation, "stick")
        cobble = _inventory_count(observation, "cobblestone") + _inventory_count(observation, "cobbled_deepslate")
        logs = _count_any(observation, LOG_NAMES)
        raw_iron = _inventory_count(observation, "raw_iron")
        iron_ingots = _inventory_count(observation, "iron_ingot")
        has_wood_pick = _inventory_count(observation, "wooden_pickaxe") > 0
        has_stone_pick = _inventory_count(observation, "stone_pickaxe") > 0
        has_furnace = _inventory_count(observation, "furnace") > 0

        # Every branch below that crafts a 3x3 recipe is gated on actually
        # being able to get a table down - otherwise the bot commits to a
        # craft it can never perform and loops forever instead of falling
        # through to restock the planks that would unblock it.
        can_craft_big = self._table_ready(client, observation)

        # 0. If the *only* thing standing between the bot and a craft it's
        #    otherwise ready for is a crafting table it can't afford, then
        #    restocking wood outranks everything else. Without this it
        #    cheerfully goes off mining more cobble it has no way to turn
        #    into tools, tunnelling steadily further from the trees that
        #    would have unblocked it.
        crafting_blocked = not can_craft_big and (
            raw_iron >= 1
            or (iron_ingots >= 3 and sticks >= 2)
            or (cobble >= 3 and not has_stone_pick and sticks >= 2)
            or (not has_wood_pick and planks >= 3 and sticks >= 2)
        )
        if crafting_blocked:
            if logs >= 1:
                log_name = _first_present(observation, LOG_NAMES)
                self.status = "crafting planks to afford a table"
                return self._act(client, {"type": "craft", "item": log_name.replace("_log", "_planks"), "count": 4})
            self.status = "need wood to afford a crafting table"
            if self._gather_wood(observation, client):
                return True

        # 1. Final craft.
        if iron_ingots >= 3 and sticks >= 2 and can_craft_big:
            return self._craft_with_table(client, "iron_pickaxe")

        # 2. Smelt raw iron into ingots.
        if raw_iron >= 1 and iron_ingots < 3 and can_craft_big:
            return self._smelt_iron(observation, client)

        # 3. Stone pickaxe - the gateway to iron (mining ore without one
        #    drops literally nothing).
        if not has_stone_pick and cobble >= 3 and sticks >= 2 and can_craft_big:
            return self._craft_with_table(client, "stone_pickaxe")

        # 4. Furnace, crafted *before* descending: cobble and a crafting
        #    table are both already to hand up here, and discovering you
        #    need one only after hauling ore up from y=40 wastes a trip.
        if has_stone_pick and not has_furnace and cobble >= 8 and can_craft_big:
            return self._craft_with_table(client, "furnace")

        # 5. Mine iron ore.
        if has_stone_pick and has_furnace and raw_iron < RAW_IRON_TARGET:
            return self._mine_iron(observation, client)

        # 6. Gather stone (needs a wooden pickaxe, or it drops nothing).
        if has_wood_pick and cobble < COBBLE_TARGET:
            return self._gather_stone(observation, client)

        # 7. Wooden pickaxe.
        if not has_wood_pick and planks >= 3 and sticks >= 2 and can_craft_big:
            return self._craft_with_table(client, "wooden_pickaxe")

        # 8. Basic materials. Wood has to come *before* plank crafting:
        #    converting the instant a single log arrives drains the pile
        #    faster than it fills, so the log target was never actually
        #    reached (found live - it started crafting after ~3 logs and
        #    stranded itself one plank short of a crafting table). If no
        #    wood is reachable at all, fall through and work with what's
        #    already carried rather than idling.
        if logs < WOOD_TARGET and self._gather_wood(observation, client):
            return True
        if sticks < STICKS_TARGET and planks >= 2:
            self.status = "crafting sticks"
            return self._act(client, {"type": "craft", "item": "stick", "count": 4})
        if planks < PLANKS_TARGET and logs >= 1:
            log_name = _first_present(observation, LOG_NAMES)
            self.status = "crafting planks"
            return self._act(client, {"type": "craft", "item": log_name.replace("_log", "_planks"), "count": 4})

        return False

    # --- individual objectives ------------------------------------------

    def _gather_wood(self, observation: dict, client) -> bool:
        self.status = "gathering wood"
        return self._mine_nearest(client, LOG_NAMES, SURFACE_SEARCH_RADIUS)

    def _gather_stone(self, observation: dict, client) -> bool:
        self.status = "gathering stone"
        if self._mine_nearest(client, STONE_NAMES, SURFACE_SEARCH_RADIUS):
            return True
        # No stone in range at the surface - head underground, where there
        # is definitionally nothing but stone.
        self.status = "digging down for stone"
        return self._act(client, {"type": "gotoY", "y": max(IRON_SEARCH_Y, observation["position"]["y"] - 12)})

    def _mine_iron(self, observation: dict, client) -> bool:
        self.status = "looking for iron"
        if self._mine_nearest(client, IRON_ORE_NAMES, DEEP_SEARCH_RADIUS):
            return True
        # Iron generates deep; if none is visible from here, go deeper and
        # look again next step rather than wandering the surface forever.
        if observation["position"]["y"] > IRON_SEARCH_Y:
            self.status = "tunnelling toward iron depth"
            return self._act(client, {"type": "gotoY", "y": IRON_SEARCH_Y})
        # Already deep with nothing in range: mine sideways to expose new
        # chunks instead of standing still in an empty pocket.
        self.status = "exploring for iron"
        return self._mine_nearest(client, STONE_NAMES, 16)

    def _smelt_iron(self, observation: dict, client) -> bool:
        fuel = self._pick_fuel(observation)
        if fuel is None:
            self.status = "no fuel for smelting - getting wood"
            return self._gather_wood(observation, client)

        if not self._ensure_block_nearby(client, "furnace"):
            if _inventory_count(observation, "furnace") == 0:
                cobble = _inventory_count(observation, "cobblestone")
                if cobble >= 8:
                    return self._craft_with_table(client, "furnace")
                self.status = "need cobblestone for a furnace"
                return self._gather_stone(observation, client)
            return True  # placement was attempted this step

        self.status = "smelting iron"
        return self._act(
            client,
            {"type": "smelt", "input": "raw_iron", "fuel": fuel, "count": min(3, _inventory_count(observation, "raw_iron"))},
        )

    def _pick_fuel(self, observation: dict) -> str | None:
        if _inventory_count(observation, "coal") > 0:
            return "coal"
        planks = _first_present(observation, PLANK_NAMES)
        if planks is not None:
            return planks
        return _first_present(observation, LOG_NAMES)

    # --- helpers --------------------------------------------------------

    def _table_ready(self, client, observation: dict) -> bool:
        """True if a 3x3 craft can actually happen: a table is already placed
        within reach, or one can be put down (carrying one, or holding enough
        planks to make one).

        Checked *before* committing to a table-requiring craft. Found live:
        the bot leaves its table behind every time it walks off to mine, and
        after spending its last planks on a wooden pickaxe it sat in a
        permanent "placing crafting_table" loop - it had the cobble for a
        stone pickaxe, no table, and no way to make one, and nothing sent it
        back for wood."""
        if _inventory_count(observation, "crafting_table") > 0:
            return True
        if _count_any(observation, PLANK_NAMES) >= 4:
            return True
        return bool(self._find_blocks(client, ["crafting_table"], 4))

    def _craft_with_table(self, client, item: str) -> bool:
        """Crafts a 3x3 recipe, placing a crafting table first if there isn't
        one within reach. The bot leaves its table behind every time it walks
        off to mine, so this can't assume one is still nearby."""
        if not self._ensure_block_nearby(client, "crafting_table"):
            return True  # spent this step getting a table down
        self.status = f"crafting {item}"
        return self._act(client, {"type": "craft", "item": item, "count": 1})

    def _ensure_block_nearby(self, client, block_name: str) -> bool:
        """True if `block_name` is already placed within reach. Otherwise
        places one (or crafts it first) and returns False, so the caller
        retries its craft on a later step."""
        found = self._find_blocks(client, [block_name], max_distance=4)
        if found:
            return True

        try:
            self.status = f"placing {block_name}"
            client.do_action({"type": "placeNearby", "item": block_name})
            return False
        except Exception:
            # No such item in inventory (or nowhere to put it) - make one.
            # crafting_table and furnace are themselves craftable from
            # material we're already carrying at this point in the chain.
            try:
                client.do_action({"type": "craft", "item": block_name, "count": 1})
            except Exception:
                pass
            return False

    def _mine_nearest(self, client, names: list[str], max_distance: int) -> bool:
        """Finds the nearest matching block and mines it (walking there
        first). False if there's nothing of that kind in range."""
        blocks = self._find_blocks(client, names, max_distance)
        if not blocks:
            return False
        target = blocks[0]
        return self._act(client, {"type": "mineBlock", "x": target["x"], "y": target["y"], "z": target["z"]})

    def _find_blocks(self, client, names: list[str], max_distance: int) -> list[dict]:
        try:
            result = client.do_action(
                {"type": "findBlocks", "names": names, "maxDistance": max_distance, "count": 8}
            )
        except Exception:
            return []
        return result.get("blocks", []) if isinstance(result, dict) else []

    def _act(self, client, command: dict) -> bool:
        """Sends one action, swallowing failures. A failed action is normal
        here (a tree got chopped by another bot, a path got blocked) and must
        not take down a long unattended training run - the next step simply
        re-derives the objective from the new world state."""
        try:
            client.do_action(command)
            return True
        except Exception as exc:  # noqa: BLE001 - see docstring
            key = command.get("type", "?")
            self._failures[key] = self._failures.get(key, 0) + 1
            self.status = f"{self.status} (failed: {exc})"
            return True  # still counts as "acted" - don't thrash the fly brain on a failure


@dataclass
class FreeWillTaskManager:
    """No progression stack, no mission, no explicit objective: the only
    thing it does is flee a nearby threat at low health. Everything else -
    every bit of resource-gathering, crafting, and combat - is left for the
    fly-brain and reward system to earn on their own. Same `step()`/`status`
    surface as TaskManager (see module docstring) so agent/loop.py and
    training/live_rollout.py can swap it in without any other change."""

    status: str = "safety_only"
    steps_in_stage: int = 0

    def step(self, observation: dict, client) -> bool:
        self.steps_in_stage += 1
        commands = self._flee_if_needed(observation)
        if commands is None:
            return False
        self.status = "fleeing"
        for command in commands:
            try:
                client.do_action(command)
            except Exception:
                pass  # best-effort: a failed flee step just tries again next tick
        return True

    def _flee_if_needed(self, observation: dict) -> list[dict] | None:
        if observation["health"] > LOW_HEALTH_THRESHOLD:
            return None
        hostile = _nearest_hostile(observation)
        if hostile is None or hostile["distance"] > FREE_WILL_FLEE_DISTANCE:
            return None
        pos = observation["position"]
        away_dx = pos["x"] - hostile["position"]["x"]
        away_dz = pos["z"] - hostile["position"]["z"]
        yaw = _yaw_towards(away_dx, away_dz)
        return [
            {"type": "look", "yaw": yaw, "pitch": 0.0, "relative": False},
            {"type": "move", "forward": True, "sprint": True, "left": False, "right": False, "jump": False},
        ]
