"""Non-biological task manager (M6): explicit progression logic for
Minecraft survival/crafting/combat, on top of the fly-brain's low-level
motor control (loop.py). This is the piece that makes no claim to be
biology - a real fly brain has no capacity for crafting trees or multi-
step planning, so this layer exists to do exactly what a fly brain can't.

Division of labor: when there's a specific, precise target (a known block
to walk up to and mine, a placed crafting table to use, a recipe to
craft), this issues bot-bridge actions directly - deterministic geometry
and scripted logic, not the trained connectome. When there's no specific
subgoal action to take (nothing nearby worth gathering, no crafting to
do), `decide()` returns None and the caller (agent/loop.py) falls back to
the fly-brain's trained reflexes for open-ended movement/threat response.
That's deliberate: the fly brain's job was always reflexive execution
under this plan, not planning - see docs/architecture.md.

Currently implements the earliest slice of progression (gather wood ->
craft planks -> craft a table -> craft basic wooden tools -> hand off to
open-ended exploration) plus an always-on low-health flee override. Later
stages (stone/iron/diamond tools, Nether travel, stronghold, the Ender
Dragon fight) follow the same stage-machine pattern - not implemented yet,
this is a first working slice to prove the pattern end to end, not the
full game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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

WOOD_TARGET = 4
PLANKS_TARGET = 8
LOW_HEALTH_THRESHOLD = 8
FLEE_DISTANCE = 6.0
MAX_STAGE_STEPS = 400  # safety valve: don't get stuck in one stage forever


def _inventory_count(observation: dict, item_name: str) -> int:
    return sum(i["count"] for i in observation["inventory"] if i["name"] == item_name)


def _total_logs(observation: dict) -> int:
    return sum(_inventory_count(observation, name) for name in LOG_TO_PLANKS)


def _total_planks(observation: dict) -> int:
    return sum(_inventory_count(observation, name) for name in LOG_TO_PLANKS.values())


def _yaw_towards(dx: float, dz: float) -> float:
    """Mineflayer yaw convention: yaw=0 faces +Z, increasing yaw turns toward -X."""
    return math.atan2(-dx, dz)


def _nearest_hostile(observation: dict) -> dict | None:
    hostiles = [e for e in observation["nearbyEntities"] if e.get("kind") == "Hostile mobs"]
    if not hostiles:
        return None
    return min(hostiles, key=lambda e: e["distance"])


@dataclass
class TaskManager:
    stage: str = "gather_wood"
    steps_in_stage: int = 0
    _crafted_tools: set = field(default_factory=set)
    # Absolute (floored) position of the log block currently being pursued,
    # or None. Without this, picking "nearest" fresh from each tick's
    # bot-relative nearbyBlocks made the bot dither between two similarly-
    # distanced trees: a step towards one shifts relative distances enough
    # that the other becomes "nearest" next tick, so it never commits to
    # either - a real live-run failure mode, not hypothetical.
    _wood_target: tuple[int, int, int] | None = None

    def status(self) -> str:
        return f"{self.stage} (step {self.steps_in_stage})"

    def _advance(self, next_stage: str) -> None:
        self.stage = next_stage
        self.steps_in_stage = 0

    def decide(self, observation: dict) -> list[dict] | None:
        """Returns bot-bridge action dicts to run instead of the fly-brain
        decoder this step, or None to defer to the fly brain."""
        self.steps_in_stage += 1

        flee = self._flee_if_needed(observation)
        if flee is not None:
            return flee

        if self.stage == "gather_wood":
            return self._gather_wood(observation)
        if self.stage == "craft_planks":
            return self._craft_planks(observation)
        if self.stage == "craft_table":
            return self._craft_and_place_table(observation)
        if self.stage == "craft_tools":
            return self._craft_tools(observation)
        if self.stage == "explore":
            return None  # hand off to the trained fly-brain reflexes

        return None

    def _flee_if_needed(self, observation: dict) -> list[dict] | None:
        if observation["health"] > LOW_HEALTH_THRESHOLD:
            return None
        hostile = _nearest_hostile(observation)
        if hostile is None or hostile["distance"] > FLEE_DISTANCE:
            return None
        pos = observation["position"]
        away_dx = pos["x"] - hostile["position"]["x"]
        away_dz = pos["z"] - hostile["position"]["z"]
        yaw = _yaw_towards(away_dx, away_dz)
        return [
            {"type": "look", "yaw": yaw, "pitch": 0.0, "relative": False},
            {"type": "move", "forward": True, "sprint": True, "left": False, "right": False, "jump": False},
        ]

    def _locate_wood_target(self, observation: dict) -> tuple[int, int, int] | None:
        """Returns (dx, dy, dz) offset (bot-relative) of the log block to
        pursue this tick, committing to the same absolute block across
        calls until it's reached/dug or leaves view - rather than
        recomputing "nearest" fresh every tick, which caused visible
        dithering between two similarly-close trees."""
        pos = observation["position"]
        ox, oy, oz = math.floor(pos["x"]), math.floor(pos["y"]), math.floor(pos["z"])
        logs = [b for b in observation["nearbyBlocks"] if b["name"] in LOG_TO_PLANKS]
        if not logs:
            self._wood_target = None
            return None

        if self._wood_target is not None:
            tx, ty, tz = self._wood_target
            rel_x, rel_y, rel_z = tx - ox, ty - oy, tz - oz
            if any(b["x"] == rel_x and b["y"] == rel_y and b["z"] == rel_z for b in logs):
                return rel_x, rel_y, rel_z
            self._wood_target = None  # dug or out of view - pick a new one below

        nearest = min(logs, key=lambda b: b["x"] ** 2 + b["y"] ** 2 + b["z"] ** 2)
        self._wood_target = (ox + nearest["x"], oy + nearest["y"], oz + nearest["z"])
        return nearest["x"], nearest["y"], nearest["z"]

    def _gather_wood(self, observation: dict) -> list[dict] | None:
        if _total_logs(observation) >= WOOD_TARGET:
            self._advance("craft_planks")
            self._wood_target = None
            return None

        target = self._locate_wood_target(observation)
        if target is None:
            if self.steps_in_stage > MAX_STAGE_STEPS:
                # nothing findable nearby after a long search - let the fly
                # brain wander somewhere new instead of getting stuck
                self.steps_in_stage = 0
            return None  # let the fly brain explore until something's in range

        dx, dy, dz = target
        if abs(dx) <= 1 and abs(dz) <= 1 and abs(dy) <= 1:
            pos = observation["position"]
            return [
                {"type": "stop"},
                {"type": "dig", "x": pos["x"] + dx, "y": pos["y"] + dy, "z": pos["z"] + dz},
            ]

        yaw = _yaw_towards(dx, dz)
        return [
            {"type": "look", "yaw": yaw, "pitch": 0.0, "relative": False},
            {"type": "move", "forward": True, "left": False, "right": False, "jump": dy > 0},
        ]

    def _craft_planks(self, observation: dict) -> list[dict] | None:
        if _total_planks(observation) >= PLANKS_TARGET:
            self._advance("craft_table")
            return None
        log_name = next((n for n in LOG_TO_PLANKS if _inventory_count(observation, n) > 0), None)
        if log_name is None:
            self._advance("gather_wood")  # ran out somehow; go get more
            self._wood_target = None
            return None
        return [{"type": "craft", "item": LOG_TO_PLANKS[log_name], "count": PLANKS_TARGET}]

    def _craft_and_place_table(self, observation: dict) -> list[dict] | None:
        if _inventory_count(observation, "crafting_table") == 0 and self.steps_in_stage == 1:
            return [{"type": "craft", "item": "crafting_table", "count": 1}]
        if _inventory_count(observation, "crafting_table") > 0:
            pos = observation["position"]
            below = next((b for b in observation["nearbyBlocks"] if b["x"] == 0 and b["z"] == 0 and b["y"] == -1), None)
            if below is None:
                return [{"type": "move", "forward": True}]
            return [
                {"type": "stop"},
                {
                    "type": "place",
                    "x": pos["x"],
                    "y": pos["y"] - 1,
                    "z": pos["z"],
                    "face": "up",
                    "item": "crafting_table",
                },
            ]
        if self.steps_in_stage > MAX_STAGE_STEPS:
            self._advance("craft_tools")  # give up placing, try tools anyway (may fail without a table)
        return None

    def _craft_tools(self, observation: dict) -> list[dict] | None:
        wanted = ["wooden_pickaxe", "wooden_axe", "wooden_sword"]
        for tool in wanted:
            if tool in self._crafted_tools or _inventory_count(observation, tool) > 0:
                self._crafted_tools.add(tool)
                continue
            return [{"type": "craft", "item": tool, "count": 1}]

        self._advance("explore")
        return None
