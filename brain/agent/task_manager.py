"""Objective-free task manager.

This planner contains no progression stack, no mission, and no explicit
objective. It only provides immediate self-preservation behavior: flee
from nearby threats when health is low, otherwise return None so the
fly-brain and reward system can act freely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LOW_HEALTH_THRESHOLD = 8
FLEE_DISTANCE = 6.0


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
    steps_in_stage: int = 0

    def status(self) -> str:
        return "safety_only"

    def decide(self, observation: dict) -> list[dict] | None:
        """Return only survival overrides; otherwise defer completely."""
        self.steps_in_stage += 1

        flee = self._flee_if_needed(observation)
        if flee is not None:
            return flee
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

